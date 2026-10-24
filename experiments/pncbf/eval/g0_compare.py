#!/usr/bin/env python3
"""G0 part 2 comparison tables from the eval_summary.json files of g0_part2.py (no pandas needed).

Per (group, method, setting, x): success rate averaged over eval seeds per checkpoint, then mean and standard error
over the checkpoints (training seeds). Shield methods use mode B (the paper setting), PPO uses mode A (its shield is
off); a second set of tables uses mode A for every method.

Comparisons (rows = method x setting x x):
  T05 vs paper     training reproduction, same MJX eval on both (and T05 vs G0 part 1 = paper checkpoints on the
                   CPU test() path, if --g0_part1 is given)
  T06a vs T05      effect of the fixes (T06a reuses T05's PPO)
  T06b vs T05      pncbf-v2 vs paper code
z = (a - b) / sqrt(se_a^2 + se_b^2), with each se floored at 0.01 (independent sets of trained models).

    python3 g0_compare.py /home/ju/repo/pncbf/g0_part2/out --g0_part1 <recover>/paper_eval/results/g0_part1_ep5.csv \
        --out /home/ju/repo/pncbf/g0_part2/results
writes g0_part2_points.csv (aggregates), g0_part2_compare.csv (all comparisons) and g0_part2_compare.md.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SETTING_TITLES = {
    "action_noise_flat": "Action Noise - Flat Terrain",
    "action_noise_hfield": "Action Noise - Heightfield Terrain",
    "push_flat": "Perturbation - Flat Terrain",
    "push_hfield": "Perturbation - Heightfield Terrain",
}
SETTING_ORDER = list(SETTING_TITLES)
PART1_METHODS = {"PPO-RL": "ppo", "PPO-RL-NCBF": "ncbf_h1", "ours": "ours_h25"}
METHOD_ORDER = ["ppo", "ncbf_h1", "ours_h25"]
COMPARISONS = [("T05", "paper", "training reproduction"), ("T06a", "T05", "effect of the fixes"), ("T06b", "T05", "pncbf-v2 vs paper code")]
SE_FLOOR = 0.01


def load(root):
    """{(group, method, mode, setting, x): {checkpoint_label: [success per eval seed]}} (+ returns)."""
    success, returns = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for path in Path(root).rglob("eval_summary.json"):
        s = json.loads(path.read_text())
        if "setting" not in s:
            continue
        group = s["checkpoint_label"].split("_")[0]
        key = (group, s["method"], s["deployment_mode"], s["setting"], round(float(s["sampling_prob"]), 6))
        success[key][s["checkpoint_label"]].append(s["success_rate"])
        returns[key][s["checkpoint_label"]].append(s["mean_episode_return"])
    return success, returns


def aggregate(per_checkpoint):
    values = np.array([np.mean(v) for v in per_checkpoint.values()], dtype=float)
    n = values.size
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return float(values.mean()), se, n


def paper_setting_mode(method):
    return "A" if method == "ppo" else "B"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--g0_part1", default=None, help="recover's g0_part1_ep5.csv (paper checkpoints, CPU test() path)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or args.root)
    out.mkdir(parents=True, exist_ok=True)

    success, returns = load(args.root)
    points = {}
    for key, per_ckpt in success.items():
        mean, se, n = aggregate(per_ckpt)
        rmean, rse, _ = aggregate(returns[key])
        points[key] = {"success": mean, "success_se": se, "n": n, "return": rmean, "return_se": rse}
    with open(out / "g0_part2_points.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "method", "mode", "setting", "x", "n_checkpoints", "success_mean", "success_se", "return_mean", "return_se"])
        for key in sorted(points):
            p = points[key]
            w.writerow([*key, p["n"], f"{p['success']:.4f}", f"{p['success_se']:.4f}", f"{p['return']:.3f}", f"{p['return_se']:.3f}"])

    references = {}  # ("part1", method, "B"/"A", setting, x) from the CPU path
    if args.g0_part1:
        for row in csv.DictReader(open(args.g0_part1)):
            if row["metric"] != "success_rate" or row["method"] not in PART1_METHODS:
                continue
            setting = {v: k for k, v in SETTING_TITLES.items()}[row["setting"]]
            method = PART1_METHODS[row["method"]]
            references[("G0part1_CPU", method, paper_setting_mode(method), setting, round(float(row["x"]), 6))] = {
                "success": float(row["ours_mean"]), "success_se": float(row["ours_se"]), "n": int(row["n_entries"])}

    def get(group, method, mode, setting, x):
        if group == "G0part1_CPU":
            return references.get((group, method, mode, setting, x))
        if group == "T06a" and method == "ppo":
            group = "T05"  # T06a reuses T05's PPO
        return points.get((group, method, mode, setting, x))

    comparisons = list(COMPARISONS) + ([("T05", "G0part1_CPU", "T05 (MJX) vs paper checkpoints on the CPU path")] if references else [])
    xs = sorted({(k[3], k[4]) for k in list(points) + list(references)}, key=lambda t: (SETTING_ORDER.index(t[0]) if t[0] in SETTING_ORDER else 9, t[1]))
    rows, md = [], ["# G0 part 2 comparisons", "", f"source: `{args.root}`; z = Δ / sqrt(se_a² + se_b²), se floored at {SE_FLOOR}", ""]
    for table_mode in ("paper", "A"):
        for a, b, title in comparisons:
            if table_mode == "A" and "G0part1" in b:
                continue
            md += [f"## {a} vs {b}: {title} ({'mode B for shield methods, A for PPO' if table_mode == 'paper' else 'mode A, all methods'})", "",
                   "| method | setting | x | " + a + " | " + b + " | Δ | z |", "|---|---|---|---|---|---|---|"]
            zs = []
            for method in METHOD_ORDER:
                mode = paper_setting_mode(method) if table_mode == "paper" else "A"
                for setting, x in xs:
                    pa, pb = get(a, method, mode, setting, x), get(b, method, mode, setting, x)
                    if pa is None or pb is None:
                        continue
                    sea, seb = max(pa["success_se"] if np.isfinite(pa["success_se"]) else 0, SE_FLOOR), max(pb["success_se"] if np.isfinite(pb["success_se"]) else 0, SE_FLOOR)
                    delta = pa["success"] - pb["success"]
                    z = delta / np.sqrt(sea ** 2 + seb ** 2)
                    zs.append(z)
                    rows.append([table_mode, a, b, method, mode, setting, x, pa["n"], pb["n"], f"{pa['success']:.4f}", f"{pa['success_se']:.4f}",
                                 f"{pb['success']:.4f}", f"{pb['success_se']:.4f}", f"{delta:+.4f}", f"{z:+.2f}"])
                    md.append(f"| {method} | {setting} | {x:g} | {pa['success']:.3f} ± {pa['success_se']:.3f} (n={pa['n']}) | "
                              f"{pb['success']:.3f} ± {pb['success_se']:.3f} (n={pb['n']}) | {delta:+.3f} | {z:+.1f} |")
            if zs:
                zs = np.array(zs)
                md += ["", f"{len(zs)} points: |z| ≤ 1: {np.mean(np.abs(zs) <= 1):.0%}, |z| ≤ 2: {np.mean(np.abs(zs) <= 2):.0%}, median z {np.median(zs):+.2f}", ""]
            else:
                md += ["", "(no data yet)", ""]
    with open(out / "g0_part2_compare.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table", "a", "b", "method", "mode", "setting", "x", "n_a", "n_b", "success_a", "se_a", "success_b", "se_b", "delta", "z"])
        w.writerows(rows)
    (out / "g0_part2_compare.md").write_text("\n".join(md) + "\n")
    print(f"{len(points)} aggregated points, {len(rows)} comparison rows -> {out}/g0_part2_compare.md")


if __name__ == "__main__":
    main()
