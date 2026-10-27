"""T14 (P1, Q1): per-state predictions of "failure within H" on the oracle decision points of ONE checkpoint's shards.

Runs inside the normal runner (`--runner.mode=test --runner.load_model=<that checkpoint>`; PPO.test is replaced),
which gives the env (policy obs, delay, margin configs), the checkpoint's policy and its safety critic.

States: oracle decision points (oracle_t × oracle_env: every 10 steps, the first 256 envs of each shard), without
the states whose IMU groups are stale (episode step 0, or right after a shock / DR step).
u0 = oracle_u0 = the checkpoint's policy on the noise-free o(x+_t) (the oracle branch's first action).

Predictors (all written as V = logit P̂(no event within H)):
  wm_<event>_H<H>_T<T>   world-model imagination from x+_t (u0 through the env delay, then the policy on o(x̂)),
                          analytic margin per step, log P̂ = Σ_{j≤H} −softplus(−ℓ_j / T)   (plan 4.5)
  pers_<event>_H<H>_T<T> persistence baseline: the margin of x+_t held for H steps
  critic_u0              this checkpoint's T06b safety critic h(s, u0): ensemble-mean P(safe), inputs
                          obs_full[t][ncbf_observation_indices] (noisy, as in deployment), u0, latent z =
                          encoder(history_stack[t]) — exactly PPO._safety_prediction
  critic_stored          the same critic evaluated by wm-data at collection with u_policy[t] (policy on the noisy
                          obs) instead of u0 (oracle_critic_p_safe)
Events: fall = margin_fall (the T06b termination, IMU height < 0 at eval, or the qvel limit);
        phys = margin_fall_physical (IMU height < 0.5 nominal or tilt > 1 rad; wm-data's physical fall).
Labels per event and H: oracle (shock-free MJX branch from the same state with u0) and actual (the recorded episode).
"""
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_wm import G, v_from_margins  # noqa: E402  (pops --gate_* args; reuse its CLI prefix)

from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model as W  # noqa: E402
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model_data as WD  # noqa: E402
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model_eval as WE  # noqa: E402

HORIZONS = (10, 25)
LABEL_H_INDEX = {1: 0, 10: 1, 25: 2, 50: 3}


def first_true(mask, axis=-1, none=None):
    """Index of the first True along axis, `none` (default: length) if there is none."""
    n = mask.shape[axis]
    idx = np.argmax(mask, axis=axis)
    return np.where(mask.any(axis=axis), idx, n if none is None else none)


def run(model, _episodes):
    from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_state
    env = model.env
    wm, bundle, meta = WE.load_bundle(G["wm"])
    act = np.asarray(meta["schema"]["xi_active"])
    T_ls = [float(t) for t in G.get("T_ls", "0.1,0.3,1,3").split(",")]
    Hmax = max(HORIZONS)
    _, full_obs_fn = wm_state.make_policy_obs_fn(env)
    lay, xl = wm_state.state_layout(env), wm_state.xi_layout(env)
    mcfg = wm_state.margin_config_from_env(env)
    shards = WD.find_shard_dirs(G["shards"])
    sch = WD.load_schema(WD.find_schema(shards[0]))
    pm = dict(sch.get("margin_physical", {}))
    pcfg = wm_state.PhysicalFallConfig(**{k: pm[k] for k in ("height_fraction", "tilt_threshold", "height_scale",
                                                             "tilt_scale") if k in pm})

    def policy(obs):
        mean, _ = model.policy.apply(model.policy_state.params, obs, None, None)
        return model.get_processed_action(mean)

    @jax.jit
    def wm_margins(u0, x0, xi_full, buf, cmd):
        ds = wm_state.delay_steps_from_xi(xi_full, xl)
        traj = W.rollout(wm, bundle, x0, xi_full[:, act], buf, u0, lambda x: policy(full_obs_fn(x, cmd, xi_full)), Hmax,
                         delay_step_fn=lambda b, u: wm_state.delay_step(b, u, ds))
        xb = jnp.broadcast_to(xi_full, traj.shape[:-1] + xi_full.shape[-1:])
        return (wm_state.margin_fall(traj, xb, mcfg, lay, xl), wm_state.margin_fall_physical(traj, xb, pcfg, lay, xl),
                wm_state.margin_fall(x0, xi_full, mcfg, lay, xl), wm_state.margin_fall_physical(x0, xi_full, pcfg, lay, xl))

    has_critic = hasattr(model, "ncbf_state") and hasattr(model, "_safety_prediction")

    @jax.jit
    def critic(obs_full, u0, hist):
        z = model.encoder_state.apply_fn(model.encoder_state.params, hist)
        return model._safety_prediction(obs_full, u0, z)[1]

    out = {}
    def add(k, v):
        out.setdefault(k, []).append(np.asarray(v))

    for si, d in enumerate(shards):
        ld = lambda f: np.load(os.path.join(d, f + ".npy"), mmap_mode="r")
        m = WD.shard_meta(d)
        ot, oe = np.asarray(ld("oracle_t")), np.asarray(ld("oracle_env"))
        T, N = ld("done").shape
        table = np.asarray(ld("xi_table"))
        ou0 = np.asarray(ld("oracle_u0"))
        o_fw, o_fm = np.asarray(ld("oracle_fail_within")), np.asarray(ld("oracle_fail_within_mask"))
        o_first = np.asarray(ld("oracle_first_failure"))
        o_mphys = np.asarray(ld("oracle_margin_physical"))
        o_trunc = np.asarray(ld("oracle_truncated"))
        o_obs, o_hist = ld("oracle_obs_full"), ld("oracle_history_stack")
        o_pstored = np.asarray(ld("oracle_critic_p_safe"))
        done_all, term_all = ld("done"), ld("terminated")
        shock = np.asarray(ld("perturbed_next")[:, oe]) | np.asarray(ld("dr_resampled_next")[:, oe])
        es_all = ld("episode_step")
        H_or = o_mphys.shape[-1]
        for p, t in enumerate(ot):
            es = np.asarray(es_all[t, oe])
            ok = (es != 0) & ~((t > 0) & shock[max(t - 1, 0)])
            if not ok.any():
                continue
            x0 = jnp.asarray(ld("x_plus")[t, oe])
            xi_full = jnp.asarray(table[np.asarray(ld("xi_idx")[t, oe])])
            args = (jnp.asarray(ou0[p]), x0, xi_full, jnp.asarray(ld("delay_buffer")[t, oe]), jnp.asarray(ld("command")[t, oe]))
            mf, mp, mf0, mp0 = (np.asarray(a) for a in wm_margins(*args))
            for H in HORIZONS:
                for T_l in T_ls:
                    add(f"wm_fall_H{H}_T{T_l:g}", np.asarray(v_from_margins(jnp.asarray(mf[:H]), T_l))[ok])
                    add(f"wm_phys_H{H}_T{T_l:g}", np.asarray(v_from_margins(jnp.asarray(mp[:H]), T_l))[ok])
                    add(f"pers_fall_H{H}_T{T_l:g}", np.asarray(v_from_margins(jnp.broadcast_to(jnp.asarray(mf0), (H,) + mf0.shape), T_l))[ok])
                    add(f"pers_phys_H{H}_T{T_l:g}", np.asarray(v_from_margins(jnp.broadcast_to(jnp.asarray(mp0), (H,) + mp0.shape), T_l))[ok])
            add("wm_min_margin_fall_H25", mf[:25].min(0)[ok])
            add("margin_fall_t", mf0[ok])
            if has_critic:
                ps = np.clip(np.asarray(critic(jnp.asarray(o_obs[p]), args[0], jnp.asarray(o_hist[p]))), 1e-7, 1 - 1e-7)
                add("critic_u0", (np.log(ps) - np.log1p(-ps))[ok])
            pst = np.clip(o_pstored[p], 1e-7, 1 - 1e-7)
            add("critic_stored", (np.log(pst) - np.log1p(-pst))[ok])
            # ---- labels: oracle -----------------------------------------------------------------------------
            phys_first = first_true(o_mphys[p] < 0, axis=-1)                     # [K], H_or if none
            trunc_first = first_true(o_trunc[p], axis=-1)
            for H in HORIZONS:
                i = LABEL_H_INDEX[H]
                add(f"oracle_fall_H{H}", o_fw[p, :, i][ok]); add(f"oracle_fall_H{H}_mask", o_fm[p, :, i][ok])
                ph = phys_first < H
                add(f"oracle_phys_H{H}", ph[ok]); add(f"oracle_phys_H{H}_mask", (ph | (trunc_first >= H))[ok])
            add("oracle_first_failure", o_first[p][ok])
            add("oracle_first_phys", phys_first[ok])
            # ---- labels: actual (same episode, t .. t+H-1) ---------------------------------------------------
            t1 = min(t + 50, T)
            dn = np.asarray(done_all[t:t1, oe]); tm = np.asarray(term_all[t:t1, oe])
            mph = np.asarray(ld("margin_physical_next")[t:t1, oe])
            end = first_true(dn.T, axis=-1)                                      # rows until the episode ends (incl.)
            alive = np.arange(t1 - t)[None, :] <= end[:, None]                   # [K, L]
            term_at_end = np.take_along_axis(tm.T, np.minimum(end, t1 - t - 1)[:, None], 1)[:, 0] & (end < t1 - t)
            steps_term = np.where(term_at_end, end, 10 ** 6)
            steps_phys = first_true((mph.T < 0) & alive, axis=-1, none=10 ** 6)
            for H in HORIZONS:
                i = LABEL_H_INDEX[H]
                add(f"actual_fall_H{H}", np.asarray(ld("fail_within")[t, oe, i])[ok])
                add(f"actual_fall_H{H}_mask", np.asarray(ld("fail_within_mask")[t, oe, i])[ok])
                ev = steps_phys < H
                known = ev | ((end >= H - 1) & (t + H <= T)) | term_at_end
                add(f"actual_phys_H{H}", ev[ok]); add(f"actual_phys_H{H}_mask", known[ok])
            add("actual_steps_to_term", steps_term[ok])
            add("actual_steps_to_phys", steps_phys[ok])
            # ---- ids ----------------------------------------------------------------------------------------
            add("episode_id", np.asarray(ld("episode_id")[t, oe])[ok]); add("env", oe[ok]); add("t", np.full(ok.sum(), t))
            add("shard", np.full(ok.sum(), si))
        out.setdefault("shard_setting", []).append(m.get("setting", os.path.basename(d)))
        out.setdefault("shard_path", []).append(d)
        print(f"shard {si + 1}/{len(shards)} {m.get('setting')} done", flush=True)
    res = {k: (np.concatenate(v) if k not in ("shard_setting", "shard_path") else np.asarray(v)) for k, v in out.items()}
    m0 = WD.shard_meta(shards[0])
    np.savez_compressed(G["out"], checkpoint=str(m0.get("checkpoint_label", m0.get("checkpoint", ""))),
                        method=str(m0.get("method", "")), has_critic=has_critic, wm=G["wm"], **res)
    print(f"wrote {G['out']}: {len(res['shard'])} states", flush=True)


if __name__ == "__main__":
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import PPO
    from rl_x.runner.runner import Runner
    PPO.test = run
    Runner().run()
