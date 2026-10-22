#!/usr/bin/env python3
"""G0 part 1: aggregate reconstructed-eval pkls exactly like misc/rss/plot_line_plot_one_figure.py and
compare each point with the paper figure values (paper_configs/paper_curves.csv, figure exps.pdf).
Aggregation: per (method, setting, x): mean over that entry's pkls -> entry mean; then mean and
std (ddof=1) over entries. The paper band is std/2.236 (= SE for 5 entries); z = (ours - paper) / paper_se,
with paper_se floored at max(paper_se, our_se, 0.01) so that zero-width paper bands do not explode.
Usage: python compare_g0.py <pkl root> <paper_curves.csv> [--threshold 250] [--out g0.csv]"""
import argparse, csv, glob, os, pickle, re, statistics as st
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("paper"); ap.add_argument("--threshold", type=int, default=250)
ap.add_argument("--out", default=None)
a = ap.parse_args()
LABEL = {"no_safety": "PPO-RL", "NCBF": "PPO-RL-NCBF", "ours": "ours"}
TITLE = {"action_noise_2": "Action Noise - Flat Terrain", "action_noise_hfield": "Action Noise - Heightfield Terrain",
         "perturbation": "Perturbation - Flat Terrain", "perturbation_hfield": "Perturbation - Heightfield Terrain"}
RE = re.compile(r"_sampling_prob_([0-9mp]+)_seed_")
tok = lambda t: float(t.replace("p", "."))
per_entry = defaultdict(list)  # (method, setting, x, metric, entry) -> values
for f in glob.glob(os.path.join(a.root, "**/rollout/*.pkl"), recursive=True):
    parts = f.split(os.sep); i = parts.index("rollout")
    run_root, setting, entry = parts[i - 3], parts[i - 2], parts[i - 1]
    x = tok(RE.search(f).group(1))
    eps = pickle.load(open(f, "rb"))
    L = np.array([len(e["dones"]) for e in eps]); R = np.array([float(np.ravel(e["ret"])[0]) for e in eps])
    per_entry[(run_root, setting, x, "success_rate", entry)].append(float(np.mean(L >= a.threshold)))
    per_entry[(run_root, setting, x, "avg_return", entry)].append(float(R.mean()))
agg = defaultdict(list)
for (rr, s, x, m, e), v in per_entry.items():
    agg[(LABEL.get(rr, rr), TITLE[s], x, m)].append(float(np.mean(v)))
paper = {}
for r in csv.DictReader(open(a.paper)):
    if r["figure"] == "exps.pdf":
        paper[(r["method"], r["column_title"], float(r["x"]), r["row"])] = (float(r["mean"]), float(r["se_band_half_width"] or 0))
rows = []
for k in sorted(agg):
    v = agg[k]; mean = float(np.mean(v)); sd = st.stdev(v) if len(v) > 1 else 0.0; se = sd / 2.236
    pm, pse = paper.get(k, (float("nan"), float("nan")))
    z = (mean - pm) / max(pse, se, 0.01)
    rows.append(dict(method=k[0], setting=k[1], x=k[2], metric=k[3], n_entries=len(v), ours_mean=round(mean, 4),
                     ours_se=round(se, 4), paper_mean=pm, paper_se=pse, diff=round(mean - pm, 4), z=round(z, 2)))
for r in rows:
    print(f"{r['metric']:12s} {r['setting'][:30]:30s} {r['method']:12s} x={r['x']:<5g} n={r['n_entries']} "
          f"repro={r['ours_mean']:.3f}±{r['ours_se']:.3f} paper={r['paper_mean']:.3f}±{r['paper_se']:.3f} z={r['z']:+.1f}")
for m in ("success_rate", "avg_return"):
    zs = np.array([r["z"] for r in rows if r["metric"] == m and np.isfinite(r["z"])])
    if len(zs):
        print(f"{m}: {len(zs)} points, |z|<=1: {np.mean(np.abs(zs) <= 1):.0%}, |z|<=2: {np.mean(np.abs(zs) <= 2):.0%}, median z {np.median(zs):+.2f}")
if a.out:
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
