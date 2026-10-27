"""T13 gate check: with the trained world model, how often would a state that needs a shield correction fall outside
the gated VJP's top-k (the k lowest-V envs of a step)?

For sampled time steps t of a shard, all (or --gate_envs) envs at t form one shield batch. For each env:
V = logit P(no fall within H) of the H-step imagination from the recorded x+_t with the recorded deterministic
policy action u_policy_t at t, the checkpoint's policy afterwards, the env's delay function and the analytic fall
margin (plan sec. 4.5; T_l = 0.1). A state "needs a correction" if V < gamma_c (default 0 logit = p_c 0.5, a rough
stand-in for the shield's threshold) and is "in the band" if V < gamma_c + delta. Reported: the share of such states,
how many are outside the top-k (missed by the gate) and how often a step has more than k of them.

Runs inside the normal runner (`--runner.mode=test --runner.load_model=<the shard's checkpoint>`) to get the env
(policy obs, delay, margin config) and the policy; PPO.test is replaced. Example:
  cd <tree>/experiments && python pncbf/wm/gate_wm.py --gate_wm RUN/wm_final.msgpack --gate_shards '<dirs glob>' \
    --gate_out OUT.json <runner args of bench_wm.sbatch with that checkpoint>
"""
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _pop(argv, prefix="--gate_"):
    out, rest = {}, []
    it = iter(argv)
    for a in it:
        if a.startswith(prefix):
            k, v = (a[2:].split("=", 1) if "=" in a else (a[2:], next(it)))
            out[k[len(prefix) - 2:]] = v
        else:
            rest.append(a)
    return out, rest


G, rest = _pop(sys.argv[1:])
sys.argv = [sys.argv[0]] + rest

from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model as W  # noqa: E402
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model_data as WD  # noqa: E402
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model_eval as WE  # noqa: E402


def safety_value(margins, T_l):
    logp = jnp.sum(-jax.nn.softplus(-margins / T_l), axis=0)
    return logp - jnp.log(-jnp.expm1(jnp.minimum(logp, -1e-12)))


def _auc(score, label):
    """P(score of a positive > score of a negative), ties count 1/2."""
    if label.all() or not label.any():
        return None
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = label.sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * (~label).sum()))


def run(model, _episodes):
    import dataclasses
    from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_state
    env = model.env
    wm, bundle, meta = WE.load_bundle(G["wm"])
    if G.get("matmul_dtype"):
        wm = W.WorldModel(wm.layout, dataclasses.replace(wm.cfg, matmul_dtype=G["matmul_dtype"]))
    act = np.asarray(meta["schema"]["xi_active"])
    H, T_l = int(G.get("H", 25)), float(G.get("T_l", 0.1))
    gamma_c = float(G.get("gamma_c", 0.0))
    deltas = [float(d) for d in G.get("deltas", "0,1,2").split(",")]
    fracs = [float(f) for f in G.get("fracs", "0.25,0.5").split(",")]
    n_times, n_envs = int(G.get("times", 100)), int(G.get("envs", 1024))
    _, full_obs_fn = wm_state.make_policy_obs_fn(env)
    lay, xl = wm_state.state_layout(env), wm_state.xi_layout(env)
    mcfg = wm_state.margin_config_from_env(env)

    def policy(obs):
        mean, _ = model.policy.apply(model.policy_state.params, obs, None, None)
        return model.get_processed_action(mean)

    @jax.jit
    def values(x0, xi_full, buf, u0, cmd):
        ds = wm_state.delay_steps_from_xi(xi_full, xl)
        traj = W.rollout(wm, bundle, x0, xi_full[:, act], buf, u0, lambda x: policy(full_obs_fn(x, cmd, xi_full)), H,
                         delay_step_fn=lambda b, u: wm_state.delay_step(b, u, ds))
        m = wm_state.margin_fall(traj, jnp.broadcast_to(xi_full, traj.shape[:-1] + xi_full.shape[-1:]), mcfg, lay, xl)
        return safety_value(m, T_l), jnp.min(m, axis=0)

    rng = np.random.default_rng(0)
    res = {"gamma_c": gamma_c, "T_l": T_l, "H": H, "deltas": deltas, "fracs": fracs, "shards": {}}
    allv = []
    for d in WD.find_shard_dirs(G["shards"]):
        ld = lambda f: np.load(os.path.join(d, f + ".npy"), mmap_mode="r")
        T, N = ld("done").shape
        envs = np.sort(rng.choice(N, min(n_envs, N), replace=False))
        times = np.sort(rng.choice(np.arange(1, T), min(n_times, T - 1), replace=False))
        table = np.asarray(ld("xi_table"))
        counts = {f"need_d{dl:g}": 0 for dl in deltas}
        missed = {f"d{dl:g}_frac{fr:g}": 0 for dl in deltas for fr in fracs}
        overflow = {f"d{dl:g}_frac{fr:g}": 0 for dl in deltas for fr in fracs}
        n_states = 0
        vs, fails, step_need = [], [], []
        for t in times:
            ok = np.asarray(ld("episode_step")[t, envs]) != 0
            x0 = jnp.asarray(ld("x_plus")[t, envs])
            xi_full = jnp.asarray(table[np.asarray(ld("xi_idx")[t, envs])])
            buf = jnp.asarray(ld("delay_buffer")[t, envs])
            u0 = jnp.asarray(ld("u_policy")[t, envs])
            cmd = jnp.asarray(ld("command")[t, envs])
            v, mmin = jax.device_get(values(x0, xi_full, buf, u0, cmd))
            v = np.where(ok, v, np.inf)        # states at episode step 0 (stale IMU) never need a correction here
            order = np.argsort(v)
            for dl in deltas:
                need = v < gamma_c + dl
                counts[f"need_d{dl:g}"] += int(need.sum())
                for fr in fracs:
                    k = max(1, int(round(fr * len(envs))))
                    sel = np.zeros(len(envs), bool)
                    sel[order[:k]] = True
                    missed[f"d{dl:g}_frac{fr:g}"] += int((need & ~sel).sum())
                    overflow[f"d{dl:g}_frac{fr:g}"] += int(need.sum() > k)
            n_states += int(ok.sum())
            step_need.append(float((v < gamma_c).sum() / max(ok.sum(), 1)))
            vs.append(v[ok])
            fw = np.asarray(ld("fail_within")[t, envs])        # actual failure within H in {1,10,25,50}
            fails.append(fw[ok, 2] if fw.ndim == 2 else fw[ok])
        vcat, fcat = np.concatenate(vs), np.concatenate(fails).astype(bool)
        allv.append(vcat)
        res["shards"][d] = {
            "states": n_states, "steps": len(times), "envs_per_step": len(envs),
            "need_frac": {k: c / max(n_states, 1) for k, c in counts.items()},
            "missed_frac_of_needing": {k: missed[k] / max(counts[f"need_{k.split('_')[0]}"], 1) for k in missed},
            "steps_with_overflow_frac": {k: c / len(times) for k, c in overflow.items()},
            "V_quantiles": {q: float(np.quantile(vcat, q)) for q in (0.01, 0.05, 0.1, 0.25, 0.5)},
            "actual_fail25_rate": float(fcat.mean()),
            "need_frac_per_step_max": float(np.max(step_need)), "need_frac_per_step_p99": float(np.quantile(step_need, 0.99)),
            "auc_V_vs_actual_fail25": _auc(-vcat, fcat),
            "mean_V_failing25": float(vcat[fcat].mean()) if fcat.any() else None,
            "mean_V_not_failing25": float(vcat[~fcat].mean()),
        }
        print(d, json.dumps(res["shards"][d]), flush=True)
    with open(G["out"], "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import PPO
    from rl_x.runner.runner import Runner
    PPO.test = run
    Runner().run()
