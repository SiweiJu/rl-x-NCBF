#!/usr/bin/env python3
"""Summarise eval pkls: finished rate (len >= threshold and len >= max len), mean return, mean length.
Usage: python summarize_pkls.py <root> [--threshold 250] [--csv out.csv]"""
import argparse, csv, glob, os, pickle, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("--threshold", type=int, default=250); ap.add_argument("--csv", default=None)
a = ap.parse_args()
RE = re.compile(r"eval_safety_(True|False)_gamma_([0-9mp]+)_sampling_prob_([0-9mp]+)_seed_(\d+)\.pkl$")
tok = lambda t: (-1 if t.startswith("m") else 1) * float(t.lstrip("m").replace("p", "."))
rows = []
for f in sorted(glob.glob(os.path.join(a.root, "**/rollout/*.pkl"), recursive=True)):
    m = RE.search(f)
    eps = pickle.load(open(f, "rb"))
    L = np.array([len(e["dones"]) for e in eps]); R = np.array([float(np.ravel(e["ret"])[0]) for e in eps])
    parts = f.split(os.sep); i = parts.index("rollout")
    rows.append(dict(variant=os.path.relpath(f, a.root).split(os.sep)[0], run_root=parts[i - 3], setting=parts[i - 2],
                     entry=parts[i - 1], safety=m.group(1), gamma=tok(m.group(2)), sampling_prob=tok(m.group(3)),
                     seed=int(m.group(4)), n=len(eps), finished_rate=float(np.mean(L >= a.threshold)),
                     full_len_rate=float(np.mean(L >= L.max())), max_len=int(L.max()), mean_len=float(L.mean()),
                     avg_return=float(R.mean()), file=f))
for r in rows:
    print(f"{r['variant']:18s} {r['run_root']:9s} {r['setting']:20s} {r['entry'][:34]:34s} sp={r['sampling_prob']:<5g} "
          f"n={r['n']} fin250={r['finished_rate']:.2f} full={r['full_len_rate']:.2f} len={r['mean_len']:.0f}/{r['max_len']} ret={r['avg_return']:.3f}")
if a.csv:
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
