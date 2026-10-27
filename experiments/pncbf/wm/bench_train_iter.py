"""T13 speed, the GB ratio: wall time of a full training iteration (rollout of nr_steps incl. the shield + PPO /
critic update) with a world-model shield inserted, vs the T06b iteration.

Runs the normal training (`--runner.mode=train`, T06b args, no saving / logging) for a few iterations and reads the
per-iteration time from the `time/sps` metric (the first iteration includes compilation and is dropped).
`--bench_wm_mode`:
  none   unchanged algorithm (T06b ppo: shield off; T06b ii_h25: critic shield)
  full   every rollout step also runs the WM H-step rollout + one VJP w.r.t. the action for all envs
  gated  forward rollout for all envs + VJP only for the `--bench_gate_frac` lowest-V envs (static top_k)
The WM shield cost is added on top of the configured critic shield call (wrapping `batched_ncbf_safety_layer`), so run
it with use_safety_layer=False (PPO) to get "PPO + WM shield". Inside the PPO update (anticipation loss) the call
is not wrapped. The WM's own training (50 updates / iteration in plan 4.10) is not included; it is timed by
bench_wm_speed.py (train_update_ms) and added in the report. WM params, the imagined policy and the state are
random / stand-ins with the real sizes (cost only).
"""
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_wm_speed import BENCH, B, STUB_GO2_SCHEMA, safety_value, random_policy  # noqa: E402  (pops --bench_*)
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model as W  # noqa: E402

SPS = []


def main():
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import PPO
    from rl_x.runner.runner import Runner

    orig_log, orig_train = PPO.log, PPO.train

    def log(self, name, value, step):
        if name == "time/sps":
            SPS.append(float(value))
            print(f"iteration {len(SPS)}: sps={float(value):.0f}", flush=True)
        return orig_log(self, name, value, step)

    mode = B("wm_mode", "none")

    def train(self):
        if mode != "none":
            self.batched_ncbf_safety_layer = wrap_shield(self, self.batched_ncbf_safety_layer, mode)
        t = time.time()
        out = orig_train(self)
        report(self, time.time() - t)
        return out

    PPO.log, PPO.train = log, train
    Runner().run()


def wrap_shield(model, orig, mode):
    env = model.env
    N = env.nr_envs
    schema = dict(STUB_GO2_SCHEMA)
    layout = W.WMLayout.from_schema(schema)
    wm = W.WorldModel(layout, W.WMConfig(width=B("width", 512), depth=B("depth", 3), ensemble_size=B("ensemble", 1),
                                         matmul_dtype=B("matmul_dtype", "float32")))
    bundle = wm.init(jax.random.PRNGKey(0))
    H = B("H", 25)
    gi = layout.group_index()
    from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_state
    _, full_obs_fn = wm_state.make_policy_obs_fn(env)
    lay_s, xl = wm_state.state_layout(env), wm_state.xi_layout(env)
    mcfg = wm_state.margin_config_from_env(env)
    obs_dim = env.single_observation_space.shape[0]
    pol = random_policy(obs_dim, layout.action_dim)        # T06b-sized policy (cost of the imagined policy)
    k = max(1, int(round(B("gate_frac", 0.25) * N)))
    S = layout.state_dim

    def vfn(u, x0, xi_full, command):
        xi = xi_full[..., :layout.xi_dim]
        pf = lambda x: pol(full_obs_fn(x, command, xi_full))
        traj = W.rollout(wm, bundle, x0, xi, None, u, pf, H)
        xf = jnp.broadcast_to(xi_full, traj.shape[:-1] + (xl.dim,))
        return safety_value(wm_state.margin_fall(traj, xf, mcfg, lay_s, xl))

    if mode == "t15":   # T15: the full WM shield (safety_value + wm_shield), ShieldConfig / SafetyValueConfig overrides
        import dataclasses
        from rl_x.algorithms.ncbf_ppo.flax_full_jit import safety_value as SV
        from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_shield as WS

        def kv(text):
            out = {}
            for item in [t for t in str(text).split(",") if t]:
                key, val = item.split("=")
                out[key] = {"True": True, "False": False}.get(val, val)
                for typ in (int, float):
                    try:
                        out[key] = typ(val)
                        break
                    except ValueError:
                        pass
            return out
        spec = SV.ImaginationSpec.from_env(env, xi_active=np.arange(layout.xi_dim))
        sv_cfg = dataclasses.replace(SV.SafetyValueConfig(H=H, matmul_dtype=B("matmul_dtype", "float32")),
                                     **kv(B("t15_sv", "")))
        sh_cfg = dataclasses.replace(WS.ShieldConfig(p_c=float(B("t15_p_c", 0.9))), **kv(B("t15_sh", "")))
        vf = SV.make_safety_value_fn(sv_cfg, spec, wm=wm, policy_fn=lambda p, o: pol(o))
        shield_fn = WS.make_shield_fn(sh_cfg, vf)
        print(f"t15 shield: {sv_cfg} {sh_cfg}", flush=True)

    def wrapped(raw_action, observation, last_action, last_state, latent_z, coeff, params_stack, first_step):
        out = orig(raw_action, observation, last_action, last_state, latent_z, coeff, params_stack, first_step)
        if raw_action.shape[0] != N:      # PPO-update call (anticipation loss): unchanged
            return out
        x0 = jnp.zeros((N, S)).at[:, :min(S, observation.shape[1])].set(observation[:, :S]) * 0.1
        x0 = x0.at[:, gi["imu_height"]].add(0.3)
        xi_full = jnp.zeros((N, xl.dim)) + 0.01 * jnp.mean(observation)
        command = observation[:, env.goal_velocities_obs_idx]
        u = raw_action
        if mode == "t15":   # stand-in state (cost only); real shapes, gating / relinearization have static shapes
            A_ = layout.action_dim
            inputs = SV.ShieldInputs(x0=x0, xi_full=xi_full, delay_buffer=jnp.zeros((N, 2, A_)) + 0.0 * u[:, None, :],
                                     delay_steps=jnp.zeros((N,), jnp.int32), command=command,
                                     first_step=first_step, stale=jnp.zeros((N,), bool))
            u_safe, v_exec, _ = shield_fn({"wm": bundle, "policy": {}}, inputs, u, jnp.zeros((N, 1)) + 0.5)
            return (out[0] + 1e-12 * (u_safe - u) + 1e-12 * jnp.mean(v_exec),) + tuple(out[1:])
        if mode == "full":
            v, g = jax.value_and_grad(lambda uu: jnp.sum(vfn(uu, x0, xi_full, command)))(u)
        elif mode == "gated":
            v, g, _ = W.gated_value_and_grad(vfn, u, (x0, xi_full, command), k)
        else:
            raise ValueError(mode)
        action = out[0] + 1e-12 * g + 1e-12 * jnp.mean(v)
        return (action,) + tuple(out[1:])

    return wrapped


def report(model, total_s):
    it_s = [model.nr_steps * model.nr_envs / s for s in SPS[1:]]   # drop the first (compilation)
    res = {"mode": B("wm_mode", "none"), "gate_frac": B("gate_frac", 0.25), "H": B("H", 25), "width": B("width", 512),
           "depth": B("depth", 3), "matmul_dtype": B("matmul_dtype", "float32"), "nr_envs": model.nr_envs,
           "nr_steps": model.nr_steps, "use_safety_layer": bool(model.ncbf_use_safety_layer),
           "iterations_timed": len(it_s), "iteration_s_median": float(np.median(it_s)) if it_s else None,
           "iteration_s_all": it_s, "total_s_incl_compile": total_s, "device": str(jax.devices()[0]),
           "bench_args": BENCH}
    print(json.dumps(res, indent=1), flush=True)
    if "bench_out" in BENCH:
        with open(BENCH["bench_out"], "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
