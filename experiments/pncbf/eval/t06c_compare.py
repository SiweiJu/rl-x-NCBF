#!/usr/bin/env python3
"""T06c bisect table from the G0 part 2 eval summaries (PPO, mode A).

Metric: per training seed, the mean success over the 22 G0 points (2 terrains x (6 action-noise + 5 push) levels);
then mean and standard error over seeds. Every group is compared with T05 PPO (paper code, the target) and T06b PPO
(pncbf-v2), both on all 5 seeds and on the matched seeds 1042 1142 1242 (the T06c seeds).

    <conda rlx-ncbf>/bin/python t06c_compare.py /home/ju/repo/pncbf/g0_part2/out --out /home/ju/repo/pncbf/g0_part2/results
writes t06c_compare.md.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from g0_compare import load

MATCHED_SEEDS = (1042, 1142, 1242)
REFERENCES = ("T05", "T06b")


def per_seed_means(success, group, mode="A", method="ppo"):
    """{seed: (mean success over the points, nr points)} for one group."""
    by_seed = defaultdict(list)
    for (g, m, md, setting, x), per_ckpt in success.items():
        if g != group or m != method or md != mode:
            continue
        for label, values in per_ckpt.items():
            by_seed[int(label.rsplit("_s", 1)[1])].append(float(np.mean(values)))
    return {seed: (float(np.mean(v)), len(v)) for seed, v in by_seed.items()}


def summarize(seed_means, seeds=None):
    values = np.array([m for s, (m, _) in seed_means.items() if seeds is None or s in seeds])
    if values.size == 0:
        return None
    se = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else float("nan")
    return float(values.mean()), se, int(values.size)


def fmt(stat):
    return "–" if stat is None else f"{stat[0]:.3f} ± {stat[1]:.3f} (n={stat[2]})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix", default="T06c-", help="groups to tabulate (label prefix)")
    args = ap.parse_args()
    success, _ = load(args.root)
    groups = sorted({k[0] for k in success if k[0].startswith(args.prefix)})
    seed_means = {g: per_seed_means(success, g) for g in (*REFERENCES, *groups)}
    ref = {(r, which): summarize(seed_means[r], None if which == "all" else MATCHED_SEEDS)
           for r in REFERENCES for which in ("all", "matched")}
    md = ["# T06c: PPO (shield off), mean success over the G0 part 2 points, mode A", "",
          f"source: `{args.root}`. Per seed: mean over the points; then mean ± se over seeds. "
          f"Matched = seeds {' '.join(map(str, MATCHED_SEEDS))}.", "",
          "| group | points/seed | mean success | Δ vs T05 (matched) | Δ vs T06b (matched) | gap closed |",
          "|---|---|---|---|---|---|"]
    for r in REFERENCES:
        md.append(f"| {r} (all) | {max((n for _, n in seed_means[r].values()), default=0)} | {fmt(ref[(r, 'all')])} | | | |")
        md.append(f"| {r} (matched) | | {fmt(ref[(r, 'matched')])} | | | |")
    t05, t06b = ref[("T05", "matched")], ref[("T06b", "matched")]
    for g in groups:
        stat = summarize(seed_means[g])
        points = max((n for _, n in seed_means[g].values()), default=0)
        d05 = f"{stat[0] - t05[0]:+.3f}" if stat and t05 else "–"
        d06 = f"{stat[0] - t06b[0]:+.3f}" if stat and t06b else "–"
        closed = f"{(stat[0] - t06b[0]) / (t05[0] - t06b[0]):.0%}" if stat and t05 and t06b and t05[0] != t06b[0] else "–"
        md.append(f"| {g} | {points} | {fmt(stat)} | {d05} | {d06} | {closed} |")
    md += ["", "gap closed = (group − T06b) / (T05 − T06b) on the matched seeds; 22 points per seed when complete.", ""]
    out = Path(args.out or args.root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "t06c_compare.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
