"""Pool gate_wm.py outputs (one .npz per checkpoint) into one table per T_l: AUC / Brier vs oracle and actual
25-step labels, V quantiles, median |grad_u V| in the band |V - gamma_c| < 2 vs elsewhere, gate coverage.

  python experiments/pncbf/wm/gate_summary.py OUT_DIR/gate_*.npz [--out summary.json]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_wm import auc, brier  # noqa: E402  (gate_wm only pops --gate_* args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--gamma_c", type=float, default=0.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    zs = [np.load(f) for f in a.files]
    Ts = sorted({k[3:] for z in zs for k in z.files if k.startswith("V_T")}, key=float)
    cat = lambda key: np.concatenate([z[key] for z in zs])
    gates = [json.load(open(os.path.splitext(f)[0] + ".json")) for f in a.files]
    out = {"files": a.files, "n_states": int(len(cat("shard"))), "by_T_l": {}}
    print(f"{out['n_states']} states from {len(zs)} checkpoints; oracle fail rate "
          f"{cat('oracle_fail')[cat('oracle_mask')].mean():.4f}, actual {cat('actual_fail')[cat('actual_mask')].mean():.4f}")
    print("T_l | AUC oracle | AUC actual | Brier oracle (const) | Brier actual (const) | V q1/q5/q50 | "
          "frac |V-g|<2 | med|dV/du| band / other (n) | gate25 missed / overflow")
    for T in Ts:
        v = cat(f"V_T{T}")
        ps = 1 / (1 + np.exp(-np.clip(v, -50, 50)))
        r = {}
        for lab in ("oracle", "actual"):
            m = cat(f"{lab}_mask")
            f = cat(f"{lab}_fail")[m]
            r[f"auc_{lab}"] = auc(-v[m], f)
            r[f"brier_{lab}"] = brier(ps[m], f)
            r[f"brier_{lab}_const"] = brier(np.full(m.sum(), 1 - f.mean()), f)
        r["V_q"] = {q: float(np.quantile(v, q)) for q in (0.01, 0.05, 0.5)}
        r["band_frac"] = float((np.abs(v - a.gamma_c) < 2).mean())
        gb, go = cat(f"grad_band_T{T}"), cat(f"grad_other_T{T}")
        r["grad_med_band"] = float(np.median(gb)) if len(gb) else None
        r["grad_med_other"] = float(np.median(go)) if len(go) else None
        r["grad_n"] = [int(len(gb)), int(len(go))]
        gk = lambda g: next(k for k in g["by_T_l"] if float(k) == float(T))   # json keys "1.0", npz keys "1"
        need = sum(g["by_T_l"][gk(g)]["gate"]["0.25"]["need"] for g in gates)
        missed = sum(g["by_T_l"][gk(g)]["gate"]["0.25"]["missed_frac"] * g["by_T_l"][gk(g)]["gate"]["0.25"]["need"] for g in gates)
        over = np.mean([g["by_T_l"][gk(g)]["gate"]["0.25"]["overflow_steps_frac"] for g in gates])
        r["gate25_missed_frac"], r["gate25_overflow_steps_frac"] = missed / max(need, 1), float(over)
        out["by_T_l"][T] = r
        fmt = lambda x: "–" if x is None else f"{x:.3g}"
        print(f"{T} | {fmt(r['auc_oracle'])} | {fmt(r['auc_actual'])} | {r['brier_oracle']:.4f} ({r['brier_oracle_const']:.4f}) | "
              f"{r['brier_actual']:.4f} ({r['brier_actual_const']:.4f}) | {r['V_q'][0.01]:.3g}/{r['V_q'][0.05]:.3g}/{r['V_q'][0.5]:.3g} | "
              f"{r['band_frac']:.4f} | {fmt(r['grad_med_band'])} / {fmt(r['grad_med_other'])} {r['grad_n']} | "
              f"{r['gate25_missed_frac']:.3f} / {r['gate25_overflow_steps_frac']:.3f}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
