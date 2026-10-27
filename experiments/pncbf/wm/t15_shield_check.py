"""T15 GPU check: the WM shield (safety_value.py + wm_shield.py) inside eval rollouts of a fixed checkpoint.

Runs inside the normal runner (`--runner.mode=test --runner.load_model=<checkpoint>`; PPO.test is replaced), with
`--environment.record_wm_state=True` (pre-reset x⁺ for the physical-fall statistic).

Per action-noise probability p (absolute std --gate_noise_std, added to the processed policy action *before* the
shield, like the eval sweep's mode B) and per shield mode (none | gated_f32 | full_f32 | gated_bf16 | full_bf16 |
critic...), the same seed / reset / command start:
  - rollout of --gate_steps steps as a python loop over three jitted pieces (policy + noise, shield, env step; the
    env step is compiled once for all modes), blocking after each; the first 5 steps are not timed
  - the deterministic policy, noise, shield (on the noisy action), env.step
  - per step: interventions, ‖Δu‖, G before / after, violations (G(u⁰) < 0 on non-stale rows) and how many of them the
    gate covers, stale rows; per env: whether the first episode terminated (fall) / physically fell within the window
  - time per env step and of the shield call alone (medians; per-step dispatch included, so slightly above a fused scan)
Writes --gate_out (json) and prints one line per run.

    python pncbf/wm/t15_shield_check.py --gate_wm <wm_final.msgpack | random> --gate_out out.json \
        [--gate_modes none,gated_f32,...] [--gate_probs 0.4,0.8] [--gate_steps 500] [--gate_chunk 100] [--gate_H 25] \
        [--gate_events fall] [--gate_p_c 0.5] [--gate_alpha 0.5] [--gate_relin 1] [--gate_frac 0.25] <runner args>
"""
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_wm import G  # noqa: E402  (pops --gate_* args)

from rl_x.algorithms.ncbf_ppo.flax_full_jit import safety_value as SV  # noqa: E402
from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_shield as WS  # noqa: E402
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model as W  # noqa: E402
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model_eval as WE  # noqa: E402


def _load_wm(env):
    from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_state
    if G["wm"] != "random":
        return WE.load_bundle(G["wm"])
    lay, xl = wm_state.state_layout(env), wm_state.xi_layout(env)
    schema = {"groups": lay.to_json()["groups"], "state_dim": lay.dim, "action_dim": int(env.nr_actuator_joints),
              "xi_dim": xl.dim}
    wm = W.WorldModel(W.WMLayout.from_schema(schema), W.WMConfig())
    return wm, wm.init(jax.random.PRNGKey(0)), {"schema": dict(schema, xi_active=None)}


def run(model, _episodes):
    from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_state
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.eval_sweep import apply_initial_commands, make_agent
    env = model.env
    if not env.record_wm_state:
        raise ValueError("needs --environment.record_wm_state=True")
    if G.get("task", "check") == "bench":
        return bench(model)
    agent = make_agent(model)
    wm, bundle, meta = _load_wm(env)
    xa = meta["schema"].get("xi_active")
    spec = SV.ImaginationSpec.from_env(env, xi_active=xa)
    policy_fn = SV.make_imagined_policy(model.policy.apply, model.get_processed_action)
    params = {"wm": bundle, "policy": model.policy_state.params}
    pcfg = wm_state.PhysicalFallConfig()
    modes = G.get("modes", "none,gated_f32,full_f32,gated_bf16,full_bf16").split(",")
    probs = [float(p) for p in G.get("probs", "0.4,0.8").split(",")]
    T, chunk = int(G.get("steps", 500)), int(G.get("chunk", 100))
    assert T % chunk == 0 and T // chunk >= 2
    H = int(G.get("H", 25))
    events = tuple(G.get("events", "fall").split(","))
    noise_std = float(G.get("noise_std", 1.0))
    nr_envs = int(env.nr_envs)
    A = int(env.nr_actuator_joints)
    seed = int(G.get("seed", 0))
    results = []

    def build(mode):
        if mode == "none":
            return None, 1
        gate, dt = mode.split("_")[:2]
        source = "mjx_oracle" if mode.endswith("_oracle") else "wm"
        sv_cfg = SV.SafetyValueConfig(source=source, H=H, events=events, T_l=(float(G.get("T_l", 1.0)),),
                                      matmul_dtype={"f32": "float32", "bf16": "bfloat16"}[dt],
                                      open_loop=G.get("open_loop", "0") == "1", scan_unroll=int(G.get("unroll", 1)))
        vf = SV.make_safety_value_fn(sv_cfg, spec, wm=wm, policy_fn=policy_fn,
                                     mjx_rollout_fn=SV.make_mjx_oracle_rollout(env, policy_fn, spec))
        sh_cfg = WS.ShieldConfig(p_c=float(G.get("p_c", 0.5)), alpha=float(G.get("alpha", 0.5)),
                                 tau=float(G.get("tau", 0.1)), gate=gate, gate_frac=float(G.get("frac", 0.25)),
                                 relinearize=int(G.get("relin", 1)), max_delta_u=float(G.get("max_delta_u", 0.5)),
                                 gate_vjp=G.get("gate_vjp", "recompute"), relin_mode=G.get("relin_mode", "gradient"),
                                 relin_frac=float(G.get("relin_frac", 0.0)))
        return WS.make_shield_fn(sh_cfg, vf, nr_events=len(events)), len(events)

    STAT_KEYS = ("n_active", "sum_du", "n_resolved", "n_improved", "n_viol", "n_viol_gated", "n_stale", "sum_Gb",
                 "n_nonfinite")

    # Jitted once and shared by all modes (the env step dominates the compile time): policy + noise, env step.
    @jax.jit
    def pre(env_state, key, p):
        key, kb, kn = jax.random.split(key, 3)
        u_pol, _ = agent.policy(env_state)
        on = jax.random.uniform(kb, (nr_envs,)) < p
        return u_pol + noise_std * jax.random.normal(kn, (nr_envs, A)) * on[:, None], key

    @jax.jit
    def post(env_state, u, alive, fell, phys):
        env_state = env.step(env_state, u)
        term = env_state.terminated.astype(bool)
        done = term | env_state.truncated.astype(bool)
        pf = wm_state.margin_fall_physical(env_state.info["wm/x_plus"], env_state.info["wm/xi_next"], pcfg,
                                           spec.lay, spec.xl) < 0
        return env_state, alive & ~done, fell | (alive & term), phys | (alive & pf), jnp.sum(term.astype(jnp.float32))

    def make_shield_step(shield_fn, oracle=False):
        @jax.jit
        def shield_step(params, env_state, u_cmd, v_prev):
            inputs = SV.inputs_from_env_state(env, env_state, spec, extra=env_state if oracle else None)
            u, v_prev, d = shield_fn(params, inputs, u_cmd, v_prev)
            act, ok = d["active"], ~d["stale"]
            viol = (d["G_before"] < 0) & ok
            f = lambda b: jnp.sum(b.astype(jnp.float32))
            st = {"n_active": f(act), "sum_du": jnp.sum(d["delta_u_norm"] * act),
                  "n_resolved": f(act & (d["G_after"] >= 0)), "n_improved": f(act & (d["G_after"] > d["G_before"])),
                  "n_viol": f(viol), "n_viol_gated": f(viol & d["gate_mask"]), "n_stale": f(d["stale"]),
                  "sum_Gb": jnp.sum(jnp.where(ok & jnp.isfinite(d["G_before"]), d["G_before"], 0.0)),
                  "n_nonfinite": f(~jnp.isfinite(d["G_before"]) | ~jnp.all(jnp.isfinite(u), -1))}
            return u, v_prev, st
        return shield_step

    g_c = float(np.log(float(G.get("p_c", 0.5)) / (1 - float(G.get("p_c", 0.5)))))
    for p in probs:
        for mode in modes:
            shield_fn, E = build(mode)
            shield_step = make_shield_step(shield_fn, mode.endswith("_oracle")) if shield_fn is not None else None
            base = jax.random.PRNGKey(seed)
            env_state = env.reset(jax.random.split(jax.random.fold_in(base, 0), nr_envs), True)
            env_state = apply_initial_commands(env, env_state, jax.random.fold_in(base, 3), 0.25)
            key, v_prev = jax.random.fold_in(base, 1), jnp.full((nr_envs, E), g_c, jnp.float32)
            alive, fell, phys = jnp.ones(nr_envs, bool), jnp.zeros(nr_envs, bool), jnp.zeros(nr_envs, bool)
            tot = {k: 0.0 for k in STAT_KEYS + ("n_term",)}
            per_chunk, step_ms, shield_ms = [], [], []
            t_c = time.time()
            n_act_chunk = 0.0
            for t in range(T):
                t0 = time.perf_counter()
                u_cmd, key = pre(env_state, key, p)
                if shield_step is not None:
                    jax.block_until_ready(u_cmd)
                    t1 = time.perf_counter()
                    u, v_prev, st = shield_step(params, env_state, u_cmd, v_prev)
                    jax.block_until_ready(u)
                    t2 = time.perf_counter()
                else:
                    u, st, t1, t2 = u_cmd, None, 0.0, 0.0
                env_state, alive, fell, phys, n_term = post(env_state, u, alive, fell, phys)
                jax.block_until_ready(alive)
                t3 = time.perf_counter()
                if t == 0:
                    compile_s = time.time() - t_c
                elif t >= min(5, T - 1):
                    step_ms.append((t3 - t0) * 1e3)
                    shield_ms.append((t2 - t1) * 1e3)
                st = jax.device_get(st) if st is not None else {}
                for k in STAT_KEYS:
                    tot[k] += float(st.get(k, 0.0))
                tot["n_term"] += float(n_term)
                n_act_chunk += float(st.get("n_active", 0.0))
                if (t + 1) % chunk == 0:
                    per_chunk.append(n_act_chunk / (chunk * nr_envs))
                    n_act_chunk = 0.0
            n_act = max(tot["n_active"], 1.0)
            n_ok = max(T * nr_envs - tot["n_stale"], 1.0)
            r = {"p": p, "mode": mode, "events": list(events), "H": H, "nr_envs": nr_envs, "steps": T,
                 "ms_per_step": float(np.median(step_ms)), "ms_shield": float(np.median(shield_ms)),
                 "ms_per_step_p90": float(np.percentile(step_ms, 90)), "compile_s": compile_s,
                 "intervention_rate": tot["n_active"] / (T * nr_envs),
                 "mean_delta_u_active": tot["sum_du"] / n_act,
                 "frac_active_resolved": tot["n_resolved"] / n_act, "frac_active_improved": tot["n_improved"] / n_act,
                 "violation_rate": tot["n_viol"] / n_ok, "frac_viol_in_gate": tot["n_viol_gated"] / max(tot["n_viol"], 1.0),
                 "stale_rate": tot["n_stale"] / (T * nr_envs), "mean_G_before": tot["sum_Gb"] / n_ok,
                 "nonfinite": tot["n_nonfinite"],
                 "first_episode_fall_rate": float(np.mean(np.asarray(fell))),
                 "first_episode_phys_fall_rate": float(np.mean(np.asarray(phys))),
                 "terminations_per_1k_env_steps": 1e3 * tot["n_term"] / (T * nr_envs),
                 "intervention_rate_per_chunk": per_chunk}
            results.append(r)
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()
                              if k != "intervention_rate_per_chunk"}), flush=True)
            with open(G["out"], "w") as fh:
                json.dump({"wm": G["wm"], "config": G, "results": results}, fh, indent=1)


# ---------------------------------------------------------------------------------------------------------------
# Speed bench (--gate_task bench): the shield alone, in-scan, per variant; references on the same GPU
# ---------------------------------------------------------------------------------------------------------------
BENCH_VARIANTS = [
    # name, SafetyValueConfig overrides, ShieldConfig overrides
    ("default_f32", {"matmul_dtype": "float32"}, {}),
    ("default_bf16", {}, {}),
    ("gatedvjp_only_bf16", {}, {"relinearize": 0, "post_check": False}),   # = T13's gated VJP
    ("forward_only_bf16", {}, None),                                          # value of all N, no gradient
    ("relin0_bf16", {}, {"relinearize": 0}),
    ("residuals_bf16", {}, {"gate_vjp": "residuals"}),
    ("secant_bf16", {}, {"relin_mode": "secant"}),
    ("relinfrac05_bf16", {}, {"relin_frac": 0.05}),
    ("unroll5_bf16", {"scan_unroll": 5}, {}),
    ("unroll25_bf16", {"scan_unroll": 25}, {}),
    ("secant_unroll5_bf16", {"scan_unroll": 5}, {"relin_mode": "secant"}),
    ("secant_residuals_bf16", {}, {"relin_mode": "secant", "gate_vjp": "residuals"}),
    ("secant_residuals_unroll5_bf16", {"scan_unroll": 5}, {"relin_mode": "secant", "gate_vjp": "residuals"}),
    ("secant_nopost_bf16", {}, {"relin_mode": "secant", "post_check": False}),
    ("full_f32", {"matmul_dtype": "float32"}, {"gate": "full"}),
]


def _time(fn, *args, reps=5):
    out = fn(*args)
    jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def bench(model):
    import dataclasses
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.eval_sweep import apply_initial_commands, make_agent
    env = model.env
    agent = make_agent(model)
    wm, bundle, meta = _load_wm(env)
    spec = SV.ImaginationSpec.from_env(env, xi_active=meta["schema"].get("xi_active"))
    policy_fn = SV.make_imagined_policy(model.policy.apply, model.get_processed_action)
    params = {"wm": bundle, "policy": model.policy_state.params}
    nr_envs, A = int(env.nr_envs), int(env.nr_actuator_joints)
    R = int(G.get("reps_in_scan", 20))
    only = [v for v in G.get("variants", "").split(",") if v]
    res = {"device": str(jax.devices()[0]), "nr_envs": nr_envs, "H": int(G.get("H", 25)), "reps_in_scan": R}

    base = jax.random.PRNGKey(0)
    env_state = env.reset(jax.random.split(jax.random.fold_in(base, 0), nr_envs), True)
    env_state = apply_initial_commands(env, env_state, jax.random.fold_in(base, 3), 0.25)

    @jax.jit
    def ppo_steps(env_state, key):   # reference: policy + action noise + env step, fused scan of R steps
        def body(c, _):
            st, k = c
            k, kb, kn = jax.random.split(k, 3)
            u, _ = agent.policy(st)
            on = jax.random.uniform(kb, (nr_envs,)) < 0.8
            st = env.step(st, u + jax.random.normal(kn, (nr_envs, A)) * on[:, None])
            return (st, k), None
        return jax.lax.scan(body, (env_state, key), None, length=R)[0]

    env_state, _ = ppo_steps(env_state, jax.random.fold_in(base, 1))   # also moves to a non-initial state
    env_state, _ = ppo_steps(env_state, jax.random.fold_in(base, 2))
    res["ppo_env_step_ms"] = _time(ppo_steps, env_state, jax.random.fold_in(base, 4)) / R * 1e3
    print(f"ppo env step {res['ppo_env_step_ms']:.2f} ms", flush=True)

    inputs_fn = jax.jit(lambda st: SV.inputs_from_env_state(env, st, spec))
    res["inputs_ms_call"] = _time(inputs_fn, env_state, reps=20) * 1e3
    inputs = inputs_fn(env_state)
    u_pol = jax.jit(lambda st: agent.policy(st)[0])(env_state)
    u0 = u_pol + jax.random.normal(jax.random.PRNGKey(7), u_pol.shape) * 0.5
    v_prev = jnp.zeros((nr_envs, 1))
    print(f"inputs_from_env_state {res['inputs_ms_call']:.2f} ms (jitted call)", flush=True)

    base_sv = SV.SafetyValueConfig(H=int(G.get("H", 25)), matmul_dtype="bfloat16")
    base_sh = WS.ShieldConfig()
    res["variants"] = {}
    for name, sv_kw, sh_kw in BENCH_VARIANTS:
        if only and name not in only:
            continue
        vf = SV.make_safety_value_fn(dataclasses.replace(base_sv, **sv_kw), spec, wm=wm, policy_fn=policy_fn)
        if sh_kw is None:
            call = lambda u, vp: (vf(params, inputs, u)[0][:, 0], vp, {})
        else:
            fn = WS.make_shield_fn(dataclasses.replace(base_sh, **sh_kw), vf)
            call = lambda u, vp, fn=fn: fn(params, inputs, u, vp)

        @jax.jit
        def run(u0, vp, call=call):
            def body(c, i):
                u, vp = c
                us, vp2, d = call(u, vp)
                # next candidate depends on this result (no hoisting), stays ~u0
                return (u0 + 1e-6 * jnp.tanh(us), vp2), jnp.mean(d.get("active", jnp.zeros(1)).astype(jnp.float32))
            (u, vp), act = jax.lax.scan(body, (u0, vp), jnp.arange(R))
            return u, act
        t_c = time.time()
        jax.block_until_ready(run(u0, v_prev))
        compile_s = time.time() - t_c
        ms = _time(run, u0, v_prev) / R * 1e3
        act = float(np.mean(np.asarray(run(u0, v_prev)[1])))
        res["variants"][name] = {"ms": ms, "x_ppo_step": ms / res["ppo_env_step_ms"], "compile_s": compile_s,
                                 "active_frac": act, "sv": sv_kw, "sh": sh_kw}
        print(f"{name}: {ms:.2f} ms = {ms / res['ppo_env_step_ms']:.2f}x ppo step (active {act:.3f})", flush=True)
        with open(G["out"], "w") as fh:
            json.dump(res, fh, indent=1)

    if G.get("wm_updates", "1") == "1":
        from bench_wm_speed import train_update_bench
        for k in (16,):
            for dt in ("float32", "bfloat16"):
                w2 = W.WorldModel(wm.layout, dataclasses.replace(wm.cfg, matmul_dtype=dt))
                res[f"wm_update_ms_b4096_k{k}_{dt}"] = train_update_bench(w2, 4096, k, 12)
                print(f"wm update b4096 k{k} {dt}: {res[f'wm_update_ms_b4096_k{k}_{dt}']:.1f} ms", flush=True)
    with open(G["out"], "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import PPO
    from rl_x.runner.runner import Runner
    PPO.test = run
    Runner().run()
