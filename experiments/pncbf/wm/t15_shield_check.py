"""T15 GPU check: the WM shield (safety_value.py + wm_shield.py) inside eval rollouts of a fixed checkpoint.

Runs inside the normal runner (`--runner.mode=test --runner.load_model=<checkpoint>`; PPO.test is replaced), with
`--environment.record_wm_state=True` (pre-reset x⁺ for the physical-fall statistic).

Per action-noise probability p (absolute std --gate_noise_std, added to the processed policy action *before* the
shield, like the eval sweep's mode B) and per shield mode (none | gated_f32 | full_f32 | gated_bf16 | full_bf16 |
critic...), the same seed / reset / command start:
  - rollout of --gate_steps steps (scan chunks of --gate_chunk steps; chunk 0 includes the compile and is not timed)
  - the deterministic policy, noise, shield (on the noisy action), env.step
  - per step: interventions, ‖Δu‖, G before / after, violations (G(u⁰) < 0 on non-stale rows) and how many of them the
    gate covers, stale rows; per env: whether the first episode terminated (fall) / physically fell within the window
  - time per env step (median over the timed chunks)
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
        gate, dt = mode.split("_")
        sv_cfg = SV.SafetyValueConfig(source="wm", H=H, events=events, T_l=(float(G.get("T_l", 1.0)),),
                                      matmul_dtype={"f32": "float32", "bf16": "bfloat16"}[dt],
                                      open_loop=G.get("open_loop", "0") == "1")
        vf = SV.make_safety_value_fn(sv_cfg, spec, wm=wm, policy_fn=policy_fn)
        sh_cfg = WS.ShieldConfig(p_c=float(G.get("p_c", 0.5)), alpha=float(G.get("alpha", 0.5)),
                                 tau=float(G.get("tau", 0.1)), gate=gate, gate_frac=float(G.get("frac", 0.25)),
                                 relinearize=int(G.get("relin", 1)), max_delta_u=float(G.get("max_delta_u", 0.5)))
        return WS.make_shield_fn(sh_cfg, vf, nr_events=len(events)), len(events)

    for p in probs:
        for mode in modes:
            shield_fn, E = build(mode)
            g_c = float(np.log(float(G.get("p_c", 0.5)) / (1 - float(G.get("p_c", 0.5)))))

            def step(carry, t):
                env_state, v_prev, alive, fell, phys, key = carry
                key, kb, kn = jax.random.split(key, 3)
                u_pol, _ = agent.policy(env_state)
                on = jax.random.uniform(kb, (nr_envs,)) < p
                u_cmd = u_pol + noise_std * jax.random.normal(kn, (nr_envs, A)) * on[:, None]
                if shield_fn is None:
                    u = u_cmd
                    z = jnp.zeros((), jnp.float32)
                    st = {k: z for k in ("n_active", "sum_du", "n_resolved", "n_improved", "n_viol", "n_viol_gated",
                                         "n_stale", "sum_Gb", "n_nonfinite")}
                else:
                    inputs = SV.inputs_from_env_state(env, env_state, spec)
                    u, v_prev, d = shield_fn(params, inputs, u_cmd, v_prev)
                    act, ok = d["active"], ~d["stale"]
                    viol = (d["G_before"] < 0) & ok
                    f = lambda b: jnp.sum(b.astype(jnp.float32))
                    st = {"n_active": f(act), "sum_du": jnp.sum(d["delta_u_norm"] * act),
                          "n_resolved": f(act & (d["G_after"] >= 0)), "n_improved": f(act & (d["G_after"] > d["G_before"])),
                          "n_viol": f(viol), "n_viol_gated": f(viol & d["gate_mask"]), "n_stale": f(d["stale"]),
                          "sum_Gb": jnp.sum(jnp.where(ok & jnp.isfinite(d["G_before"]), d["G_before"], 0.0)),
                          "n_nonfinite": f(~jnp.isfinite(d["G_before"]) | ~jnp.all(jnp.isfinite(u), -1))}
                env_state = env.step(env_state, u)
                term = env_state.terminated.astype(bool)
                done = term | env_state.truncated.astype(bool)
                pf = wm_state.margin_fall_physical(env_state.info["wm/x_plus"], env_state.info["wm/xi_next"], pcfg,
                                                   spec.lay, spec.xl) < 0
                fell = fell | (alive & term)
                phys = phys | (alive & pf)
                alive = alive & ~done
                st["n_term"] = jnp.sum(term.astype(jnp.float32))
                return (env_state, v_prev, alive, fell, phys, key), st

            @jax.jit
            def run_chunk(carry):
                return jax.lax.scan(step, carry, jnp.arange(chunk))

            base = jax.random.PRNGKey(seed)
            env_state = env.reset(jax.random.split(jax.random.fold_in(base, 0), nr_envs), True)
            env_state = apply_initial_commands(env, env_state, jax.random.fold_in(base, 3), 0.25)
            carry = (env_state, jnp.full((nr_envs, E), g_c, jnp.float32), jnp.ones(nr_envs, bool),
                     jnp.zeros(nr_envs, bool), jnp.zeros(nr_envs, bool), jax.random.fold_in(base, 1))
            stats, times = [], []
            t_c = time.time()
            for c in range(T // chunk):
                t0 = time.time()
                carry, st = run_chunk(carry)
                st = jax.device_get(st)
                jax.block_until_ready(carry[2])
                if c == 0:
                    compile_s = time.time() - t_c
                else:
                    times.append((time.time() - t0) / chunk * 1e3)
                stats.append(st)
            st = {k: np.concatenate([s[k] for s in stats]) for k in stats[0]}
            tot = {k: float(v.sum()) for k, v in st.items()}
            n_act = max(tot["n_active"], 1.0)
            n_ok = max(T * nr_envs - tot["n_stale"], 1.0)
            r = {"p": p, "mode": mode, "events": list(events), "H": H, "nr_envs": nr_envs, "steps": T,
                 "ms_per_step": float(np.median(times)), "ms_per_step_all": [float(x) for x in times],
                 "compile_plus_chunk0_s": compile_s,
                 "intervention_rate": tot["n_active"] / (T * nr_envs),
                 "mean_delta_u_active": tot["sum_du"] / n_act,
                 "frac_active_resolved": tot["n_resolved"] / n_act, "frac_active_improved": tot["n_improved"] / n_act,
                 "violation_rate": tot["n_viol"] / n_ok, "frac_viol_in_gate": tot["n_viol_gated"] / max(tot["n_viol"], 1.0),
                 "stale_rate": tot["n_stale"] / (T * nr_envs), "mean_G_before": tot["sum_Gb"] / n_ok,
                 "nonfinite": tot["n_nonfinite"],
                 "first_episode_fall_rate": float(np.mean(np.asarray(carry[3]))),
                 "first_episode_phys_fall_rate": float(np.mean(np.asarray(carry[4]))),
                 "terminations_per_1k_env_steps": 1e3 * tot["n_term"] / (T * nr_envs),
                 "intervention_rate_per_chunk": [float(s["n_active"].sum()) / (chunk * nr_envs) for s in stats]}
            results.append(r)
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()
                              if not k.endswith("_all") and k != "intervention_rate_per_chunk"}), flush=True)
            with open(G["out"], "w") as fh:
                json.dump({"wm": G["wm"], "config": G, "results": results}, fh, indent=1)
            jax.clear_caches()


if __name__ == "__main__":
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import PPO
    from rl_x.runner.runner import Runner
    PPO.test = run
    Runner().run()
