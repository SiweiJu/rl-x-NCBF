"""T13 gate / calibration check of a trained world model on the shards of ONE checkpoint (feeds T14 and the shield).

States = the oracle decision points of each shard (oracle_t × oracle_env: every 10 steps, the first 256 envs), with
the oracle's first action u0 = policy on the noise-free o(x+_t). For each state: the H-step imagination from the
recorded x+_t (u0 at t through the env's delay, then the checkpoint's deterministic policy on o(x̂), recorded ξ) and
the analytic fall margin ℓ = wm_state.margin_fall (min of height / s_z with s_z = 0.05 m and qvel / 10 m/s; at eval
the curriculum coefficient is 1, so the height threshold is 0 and ℓ ≈ IMU height / 0.05 m).
V_T = logit P̂, log P̂ = Σ_j −softplus(−ℓ_j / T) (plan sec. 4.5), for every T in --gate_T_ls.

Reported per T: AUC and Brier of P̂ = sigmoid(V) against the oracle (no shock / noise after t) and the actual 25-step
labels, V quantiles, median |∇_u V| (L2 over the action) for states with |V − γ_c| < 2 vs the others, and the gate
coverage: per decision step the 256 envs form one shield batch; a state needs a correction if V < γ_c (γ_c = 0
logit = p_c 0.5, a rough stand-in) and is missed if it is not among the top-k lowest V (k = 25 % / 50 %).

Runs inside the normal runner (`--runner.mode=test --runner.load_model=<checkpoint>`); PPO.test is replaced.
"""
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp


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

LABEL_H_INDEX = 2   # label_horizons (1, 10, 25, 50) -> H = 25


def v_from_margins(m, T_l):
    """m (H, ...) margins -> V (...), numerically stable."""
    logp = jnp.sum(-jax.nn.softplus(-m / T_l), axis=0)
    return logp - jnp.log(-jnp.expm1(jnp.minimum(logp, -1e-12)))


def auc(score, label):
    """P(score of a positive > score of a negative), ties 1/2."""
    label = np.asarray(label, bool)
    if label.all() or not label.any():
        return None
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = label.sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * (~label).sum()))


def brier(p_safe, fail):
    return float(np.mean((p_safe - (1.0 - fail.astype(np.float64))) ** 2))


def run(model, _episodes):
    import dataclasses
    from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_state
    env = model.env
    wm, bundle, meta = WE.load_bundle(G["wm"])
    if G.get("matmul_dtype"):
        wm = W.WorldModel(wm.layout, dataclasses.replace(wm.cfg, matmul_dtype=G["matmul_dtype"]))
    act = np.asarray(meta["schema"]["xi_active"])
    H = int(G.get("H", 25))
    T_ls = [float(t) for t in G.get("T_ls", "0.1,0.3,1,3").split(",")]
    gamma_c = float(G.get("gamma_c", 0.0))
    fracs = [float(f) for f in G.get("fracs", "0.25,0.5").split(",")]
    n_grad = int(G.get("grad_states", 128))       # per shard: up to n_grad band states + n_grad others
    _, full_obs_fn = wm_state.make_policy_obs_fn(env)
    lay, xl = wm_state.state_layout(env), wm_state.xi_layout(env)
    mcfg = wm_state.margin_config_from_env(env)

    def policy(obs):
        mean, _ = model.policy.apply(model.policy_state.params, obs, None, None)
        return model.get_processed_action(mean)

    def margins_traj(u0, x0, xi_full, buf, cmd):
        ds = wm_state.delay_steps_from_xi(xi_full, xl)
        traj = W.rollout(wm, bundle, x0, xi_full[:, act], buf, u0, lambda x: policy(full_obs_fn(x, cmd, xi_full)), H,
                         delay_step_fn=lambda b, u: wm_state.delay_step(b, u, ds))
        return wm_state.margin_fall(traj, jnp.broadcast_to(xi_full, traj.shape[:-1] + xi_full.shape[-1:]), mcfg, lay, xl)

    margins_jit = jax.jit(margins_traj)
    grad_jit = jax.jit(lambda u0, x0, xi, buf, cmd, T: jax.grad(
        lambda u: jnp.sum(v_from_margins(margins_traj(u, x0, xi, buf, cmd), T)))(u0))

    rows = {f: [] for f in ("oracle_fail", "oracle_mask", "actual_fail", "actual_mask", "shard")}
    V = {T: [] for T in T_ls}
    gate = {T: {fr: {"need": 0, "missed": 0, "overflow_steps": 0} for fr in fracs} for T in T_ls}
    n_steps = 0
    grads = {T: {"band": [], "other": []} for T in T_ls}
    shards = WD.find_shard_dirs(G["shards"])
    for si, d in enumerate(shards):
        ld = lambda f: np.load(os.path.join(d, f + ".npy"), mmap_mode="r")
        ot, oe = np.asarray(ld("oracle_t")), np.asarray(ld("oracle_env"))
        ou0 = np.asarray(ld("oracle_u0"))
        ofw, ofm = np.asarray(ld("oracle_fail_within"))[..., LABEL_H_INDEX], np.asarray(ld("oracle_fail_within_mask"))[..., LABEL_H_INDEX]
        table = np.asarray(ld("xi_table"))
        cache = []
        for p, t in enumerate(ot):
            ok = np.asarray(ld("episode_step")[t, oe]) != 0
            args = (jnp.asarray(ou0[p]), jnp.asarray(ld("x_plus")[t, oe]), jnp.asarray(table[np.asarray(ld("xi_idx")[t, oe])]),
                    jnp.asarray(ld("delay_buffer")[t, oe]), jnp.asarray(ld("command")[t, oe]))
            m = margins_jit(*args)
            afw = np.asarray(ld("fail_within")[t, oe])[:, LABEL_H_INDEX]
            afm = np.asarray(ld("fail_within_mask")[t, oe])[:, LABEL_H_INDEX]
            n_steps += 1
            for T in T_ls:
                v = np.asarray(v_from_margins(m, T))
                vv = np.where(ok, v, np.inf)
                order = np.argsort(vv)
                need = vv < gamma_c
                for fr in fracs:
                    k = max(1, int(round(fr * len(oe))))
                    sel = np.zeros(len(oe), bool)
                    sel[order[:k]] = True
                    g = gate[T][fr]
                    g["need"] += int(need.sum())
                    g["missed"] += int((need & ~sel).sum())
                    g["overflow_steps"] += int(need.sum() > k)
                V[T].append(v[ok])
            rows["oracle_fail"].append(ofw[p][ok]); rows["oracle_mask"].append(ofm[p][ok])
            rows["actual_fail"].append(afw[ok]); rows["actual_mask"].append(afm[ok])
            rows["shard"].append(np.full(int(ok.sum()), si))
            cache.append((args, ok, np.asarray(v_from_margins(m, 1.0))))
        # gradients: a few hundred states per shard, band states (|V - gamma_c| < 2 at that T) preferred
        for T in T_ls:
            band_left, other_left = n_grad, n_grad
            for args, ok, _ in cache:
                if band_left <= 0 and other_left <= 0:
                    break
                v = np.asarray(v_from_margins(margins_jit(*args), T))
                inb = ok & (np.abs(v - gamma_c) < 2.0)
                oth = ok & ~inb
                if not (inb.any() and band_left > 0) and not (oth.any() and other_left > 0):
                    continue
                g = np.linalg.norm(np.asarray(grad_jit(*args, T)), axis=-1)
                take_b = np.nonzero(inb)[0][:max(band_left, 0)]
                take_o = np.nonzero(oth)[0][:max(min(other_left, 16), 0)]   # spread "other" over many steps
                grads[T]["band"] += g[take_b].tolist()
                grads[T]["other"] += g[take_o].tolist()
                band_left -= len(take_b)
                other_left -= len(take_o)
        print(f"shard {si + 1}/{len(shards)} done: {d}", flush=True)

    cat = {f: np.concatenate(v) for f, v in rows.items()}
    res = {"checkpoint": str(G.get("label", "")), "shards": shards, "H": H, "gamma_c": gamma_c,
           "margin": "wm_state.margin_fall: min(height margin / 0.05 m, qvel margin / 10 m/s); at eval the height "
                     "threshold is 0, so l ~ IMU height in units of 5 cm",
           "n_states": int(len(cat["shard"])), "n_decision_steps": n_steps,
           "oracle_fail_rate": float(cat["oracle_fail"][cat["oracle_mask"]].mean()),
           "actual_fail_rate": float(cat["actual_fail"][cat["actual_mask"]].mean()), "by_T_l": {}}
    for T in T_ls:
        v = np.concatenate(V[T])
        p_safe = 1.0 / (1.0 + np.exp(-np.clip(v, -50, 50)))
        r = {}
        for lab in ("oracle", "actual"):
            m = cat[f"{lab}_mask"]
            f = cat[f"{lab}_fail"][m]
            r[f"auc_{lab}"] = auc(-v[m], f)
            r[f"brier_{lab}"] = brier(p_safe[m], f)
            r[f"brier_{lab}_constant_baseline"] = brier(np.full(m.sum(), 1.0 - f.mean()), f)
        r["V_quantiles"] = {str(q): float(np.quantile(v, q)) for q in (0.001, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.9)}
        r["frac_V_below_gamma_c"] = float((v < gamma_c).mean())
        r["frac_abs_V_minus_gamma_c_lt_2"] = float((np.abs(v - gamma_c) < 2).mean())
        r["grad_norm_median_band"] = float(np.median(grads[T]["band"])) if grads[T]["band"] else None
        r["grad_norm_median_other"] = float(np.median(grads[T]["other"])) if grads[T]["other"] else None
        r["grad_n_band"], r["grad_n_other"] = len(grads[T]["band"]), len(grads[T]["other"])
        r["gate"] = {str(fr): {"need": g["need"], "missed_frac": g["missed"] / max(g["need"], 1),
                               "overflow_steps_frac": g["overflow_steps"] / max(n_steps, 1)}
                     for fr, g in gate[T].items()}
        res["by_T_l"][str(T)] = r
        print(f"T_l={T}: " + json.dumps({k: v_ for k, v_ in r.items() if k != "V_quantiles"}), flush=True)
    with open(G["out"], "w") as f:
        json.dump(res, f, indent=1)
    # per-state arrays for pooling over checkpoints (gate_summary.py)
    np.savez_compressed(os.path.splitext(G["out"])[0] + ".npz",
                        **{f"V_T{T:g}": np.concatenate(V[T]) for T in T_ls}, **cat,
                        **{f"grad_band_T{T:g}": np.asarray(grads[T]["band"]) for T in T_ls},
                        **{f"grad_other_T{T:g}": np.asarray(grads[T]["other"]) for T in T_ls})


if __name__ == "__main__":
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import PPO
    from rl_x.runner.runner import Runner
    PPO.test = run
    Runner().run()
