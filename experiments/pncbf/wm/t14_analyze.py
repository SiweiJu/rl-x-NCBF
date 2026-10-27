"""T14 analysis: pool t14_collect.py outputs (one .npz per held-out checkpoint) and compute, per predictor × label ×
event × H: AUC and Brier (with cluster-bootstrap CIs over episodes), lead time of actual failures, splits by setting
and by checkpoint, and the decision-point-A comparison (WM vs the ii_h25 critic on oracle labels, H = 25).

  python experiments/pncbf/wm/t14_analyze.py OUT_DIR/t14_*.npz --T_main 1 --out OUT_DIR/t14_summary.json
"""
import argparse
import json

import numpy as np

EVENTS = ("fall", "phys")
LABELS = ("oracle", "actual")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


class Scorer:
    """AUC / Brier of a failure score against binary labels, with per-state weights (cluster bootstrap)."""

    def __init__(self, score, y):
        self.order = np.argsort(score, kind="mergesort")
        s = score[self.order]
        self.grp = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])])
        self.ng = int(self.grp[-1]) + 1 if len(s) else 0
        self.y = y[self.order].astype(np.float64)

    def auc(self, w=None):
        w = np.ones(len(self.y)) if w is None else w[self.order]
        wp = np.bincount(self.grp, w * self.y, self.ng)
        wn = np.bincount(self.grp, w * (1 - self.y), self.ng)
        Wp, Wn = wp.sum(), wn.sum()
        if Wp <= 0 or Wn <= 0:
            return np.nan
        below = np.concatenate([[0], np.cumsum(wn)[:-1]])
        return float(np.sum(wp * (below + 0.5 * wn)) / (Wp * Wn))


def brier(p_fail, y, w=None):
    w = np.ones(len(y)) if w is None else w
    return float(np.sum(w * (p_fail - y) ** 2) / np.sum(w))


def load(files):
    zs = [dict(np.load(f, allow_pickle=False)) for f in files]
    keys = [k for k in zs[0] if zs[0][k].ndim == 1 and len(zs[0][k]) == len(zs[0]["shard"])]
    d = {k: np.concatenate([z[k] for z in zs]) for k in keys if all(k in z for z in zs)}
    d["ckpt"] = np.concatenate([np.full(len(z["shard"]), i) for i, z in enumerate(zs)])
    d["setting"] = np.concatenate([np.asarray(z["shard_setting"])[z["shard"]] for z in zs])
    meta = [{"file": f, "checkpoint": str(z["checkpoint"]), "method": str(z["method"]), "has_critic": bool(z["has_critic"]),
             "n": int(len(z["shard"]))} for f, z in zip(files, zs)]
    cl = np.stack([d["ckpt"], d["shard"], d["env"], d["episode_id"]], 1)
    _, d["cluster"] = np.unique(cl, axis=0, return_inverse=True)
    d["cluster"] = d["cluster"].reshape(-1)
    return d, meta


def boot_weights(cluster, B, rng):
    n_cl = int(cluster.max()) + 1
    for _ in range(B):
        c = np.bincount(rng.integers(0, n_cl, n_cl), minlength=n_cl)
        yield c[cluster].astype(np.float64)


def metric_cells(d, preds, sel, B, rng, ci=True):
    """{pred: {label_event_H: {auc, auc_ci, brier, brier_ci, n, pos_rate}}} on the states `sel`."""
    out = {}
    for name, (key_fn, kind) in preds.items():
        out[name] = {}
        for lab in LABELS:
            for ev in EVENTS:
                for H in (10, 25):
                    yk, mk = f"{lab}_{ev}_H{H}", f"{lab}_{ev}_H{H}_mask"
                    vk = key_fn(ev, H)
                    if vk not in d or yk not in d:
                        continue
                    m = sel & d[mk].astype(bool) & np.isfinite(d[vk])
                    y = d[yk][m].astype(np.float64)
                    v = d[vk][m]
                    pf = sigmoid(-v)
                    sc = Scorer(-v, y)
                    cell = {"n": int(m.sum()), "pos_rate": float(y.mean()) if len(y) else None,
                            "auc": sc.auc(), "brier": brier(pf, y), "brier_const": brier(np.full(len(y), y.mean()), y)}
                    if ci and B > 0 and H == 25:
                        cl = np.unique(d["cluster"][m], return_inverse=True)[1]
                        a, b = [], []
                        for w in boot_weights(cl, B, rng):
                            a.append(sc.auc(w)); b.append(brier(pf, y, w))
                        cell["auc_ci"] = [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
                        cell["brier_ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
                    out[name][f"{lab}_{ev}_H{H}"] = cell
    return out


def lead_times(d, vkey, sel, steps_key, neg_key, neg_mask_key, fpr=0.05):
    """Actual events: for each episode with an event, the earliest decision point within 50 steps before it whose
    P̂_fail exceeds the threshold; lead = steps from that point to the event (10-step resolution)."""
    v = d[vkey]
    pf = sigmoid(-v)
    negs = sel & d[neg_mask_key].astype(bool) & ~d[neg_key].astype(bool)
    thr_fpr = float(np.quantile(pf[negs], 1 - fpr)) if negs.any() else np.nan
    res = {"thr_fpr5": thr_fpr}
    steps = d[steps_key]
    idx = np.nonzero(sel & (steps < 50))[0]
    idx = idx[np.lexsort((d["t"][idx], d["cluster"][idx]))]          # by cluster, then time
    cl = d["cluster"][idx]
    starts = np.concatenate([[0], np.nonzero(cl[1:] != cl[:-1])[0] + 1]) if len(idx) else np.zeros(0, int)
    groups = np.split(idx, starts[1:]) if len(idx) else []
    for name, thr in (("p0.5", 0.5), ("fpr5", thr_fpr)):
        lead = []
        for g in groups:
            hit = g[pf[g] > thr]
            if len(hit):
                lead.append(int(steps[hit[0]]))
        res[name] = {"events": int(len(groups)), "detected_frac": len(lead) / max(len(groups), 1),
                     "lead_median": float(np.median(lead)) if lead else None,
                     "lead_q25_q75": [float(np.percentile(lead, 25)), float(np.percentile(lead, 75))] if lead else None}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--T_main", type=float, default=1.0)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    d, meta = load(a.files)
    Ts = sorted({float(k.split("_T")[-1]) for k in d if k.startswith("wm_fall_H25_T")})
    Tm = a.T_main
    preds = {f"WM T_l={T:g}": ((lambda ev, H, T=T: f"wm_{ev}_H{H}_T{T:g}"), "wm") for T in Ts}
    preds[f"persistence T_l={Tm:g}"] = ((lambda ev, H: f"pers_{ev}_H{H}_T{Tm:g}"), "pers")
    preds["critic(u0)"] = ((lambda ev, H: "critic_u0"), "critic")
    preds["critic(stored, u_policy)"] = ((lambda ev, H: "critic_stored"), "critic")
    all_sel = np.ones(len(d["shard"]), bool)
    is_ii = np.isin(d["ckpt"], [i for i, m in enumerate(meta) if "ii_h25" in m["checkpoint"] or m["method"] == "ii_h25"])
    res = {"meta": meta, "n_states": int(len(d["shard"])), "T_main": Tm,
           "all_checkpoints": metric_cells(d, {k: v for k, v in preds.items() if v[1] != "critic"}, all_sel, a.B, rng),
           "ii_h25_checkpoints": metric_cells(d, preds, is_ii, a.B, rng)}
    # decision point A: paired bootstrap of AUC(WM, T_main) - AUC(critic(u0)) on ii_h25, oracle fall H=25
    m = is_ii & d["oracle_fall_H25_mask"].astype(bool)
    y = d["oracle_fall_H25"][m].astype(float)
    s_wm, s_cr = Scorer(-d[f"wm_fall_H25_T{Tm:g}"][m], y), Scorer(-d["critic_u0"][m], y)
    cl = np.unique(d["cluster"][m], return_inverse=True)[1]
    diffs = [s_wm.auc(w) - s_cr.auc(w) for w in boot_weights(cl, a.B, rng)]
    res["decision_A"] = {"auc_wm": s_wm.auc(), "auc_critic_u0": s_cr.auc(), "diff": s_wm.auc() - s_cr.auc(),
                         "diff_ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                         "n": int(m.sum()), "pos": int(y.sum())}
    # lead times (actual events), all checkpoints for WM / persistence, ii_h25 for the critic
    res["lead"] = {}
    for ev, steps_key in (("fall", "actual_steps_to_term"), ("phys", "actual_steps_to_phys")):
        for name, vk, sel in ((f"WM T_l={Tm:g}", f"wm_{ev}_H25_T{Tm:g}", all_sel),
                              (f"persistence T_l={Tm:g}", f"pers_{ev}_H25_T{Tm:g}", all_sel),
                              (f"WM T_l={Tm:g} (ii_h25)", f"wm_{ev}_H25_T{Tm:g}", is_ii),
                              ("critic(u0) (ii_h25)", "critic_u0", is_ii)):
            res["lead"][f"{ev}: {name}"] = lead_times(d, vk, sel, steps_key, f"actual_{ev}_H25", f"actual_{ev}_H25_mask")
    # splits: oracle fall H=25, by setting and by checkpoint
    split_preds = {k: v for k, v in preds.items() if k in (f"WM T_l={Tm:g}", f"persistence T_l={Tm:g}", "critic(u0)")}
    res["by_setting"] = {s: metric_cells(d, split_preds, d["setting"] == s, 0, rng, ci=False) for s in np.unique(d["setting"])}
    res["by_setting_ii_h25"] = {s: metric_cells(d, split_preds, (d["setting"] == s) & is_ii, 0, rng, ci=False)
                                for s in np.unique(d["setting"])}
    res["by_checkpoint"] = {meta[i]["checkpoint"]: metric_cells(d, split_preds, d["ckpt"] == i, 0, rng, ci=False)
                            for i in range(len(meta))}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=float)
    # ---- print the main tables ----
    fmt = lambda c, k: "–" if c is None or c.get(k) is None or (isinstance(c.get(k), float) and np.isnan(c[k])) else f"{c[k]:.3f}"
    ci = lambda c, k: "" if c is None or f"{k}_ci" not in c else f" [{c[f'{k}_ci'][0]:.3f}, {c[f'{k}_ci'][1]:.3f}]"
    for block in ("ii_h25_checkpoints", "all_checkpoints"):
        print(f"\n## {block} (H=25)\n| predictor | label/event | AUC [95% CI] | Brier [CI] (const) | n / pos rate |\n|---|---|---|---|---|")
        for p, cells in res[block].items():
            for key in ("oracle_fall_H25", "actual_fall_H25", "oracle_phys_H25", "actual_phys_H25"):
                c = cells.get(key)
                if c:
                    print(f"| {p} | {key} | {fmt(c, 'auc')}{ci(c, 'auc')} | {fmt(c, 'brier')}{ci(c, 'brier')} ({c['brier_const']:.4f}) | {c['n']} / {c['pos_rate']:.4f} |")
    print("\n## H=10 AUC (ii_h25): " + "; ".join(f"{p}: oracle_fall {fmt(c.get('oracle_fall_H10'), 'auc')}" for p, c in res["ii_h25_checkpoints"].items()))
    print(f"\n## decision A: {json.dumps(res['decision_A'])}")
    print("\n## lead (actual events, H=25 predictions)")
    for k, v in res["lead"].items():
        print(f"{k}: P>0.5 det {v['p0.5']['detected_frac']:.3f} lead {v['p0.5']['lead_median']} | FPR5% (thr {v['thr_fpr5']:.3g}) det {v['fpr5']['detected_frac']:.3f} lead {v['fpr5']['lead_median']} ({v['p0.5']['events']} events)")
    print("\n## by setting (oracle fall H25 AUC / Brier)")
    for s, cells in res["by_setting"].items():
        print(s + ": " + "; ".join(f"{p} {fmt(c.get('oracle_fall_H25'), 'auc')}/{fmt(c.get('oracle_fall_H25'), 'brier')}" for p, c in cells.items()))
    print("\n## by setting, ii_h25 checkpoints only (oracle fall H25 AUC / Brier)")
    for s, cells in res["by_setting_ii_h25"].items():
        print(s + ": " + "; ".join(f"{p} {fmt(c.get('oracle_fall_H25'), 'auc')}/{fmt(c.get('oracle_fall_H25'), 'brier')}" for p, c in cells.items()))
    print("\n## by checkpoint (oracle fall H25 AUC / Brier)")
    for s, cells in res["by_checkpoint"].items():
        print(s + ": " + "; ".join(f"{p} {fmt(c.get('oracle_fall_H25'), 'auc')}/{fmt(c.get('oracle_fall_H25'), 'brier')}" for p, c in cells.items()))


if __name__ == "__main__":
    main()
