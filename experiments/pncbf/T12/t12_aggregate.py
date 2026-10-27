#!/usr/bin/env python3
"""Aggregate the T12 shard summaries (summary.json + meta.json) into tables for the report.

    python3 t12_aggregate.py /home/ju/repo/pncbf/p1_data/v1 [--md out.md] [--csv out.csv]
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def dir_size(path):
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--md", default="")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    rows = []
    for summary_path in sorted(glob.glob(os.path.join(args.root, "*", "*", "*", "summary.json"))):
        shard = os.path.dirname(summary_path)
        s = json.load(open(summary_path))
        m = json.load(open(os.path.join(shard, "meta.json")))
        pf = s["physical_fall"]
        row = {
            "split": m["split"], "method": m["method"], "seed": m["train_seed"], "setting": m["setting"],
            "gb": dir_size(shard) / 1e9, "transitions": s["nr_transitions"], "terminations": s["nr_terminations"],
            "episodes": pf["episodes"],
            "near_fail_50": s["actual_fail_within_50_count"], "near_fail_25": s["actual_fail_within_25_count"],
            "shock_rate": s["shock_rate"], "dr_rate": s["dr_resample_rate"],
            "term_height": s["termination_causes"]["imu_height_below_threshold"],
            "term_qvel": s["termination_causes"]["qvel_limit_only"], "term_other": s["termination_causes"]["other"],
            "fell_term": pf["fell_long_and_terminated"], "fell_no_term": pf["fell_long_not_terminated"],
            "fell_no_term_trunc": pf["fell_long_not_terminated_truncated"],
            "fallen_steps_no_term": pf["fallen_steps_not_terminated"],
            "median_fallen_steps_before_term": pf["median_fallen_steps_before_termination"],
            "margin_ok": s["margin_consistent"] and s["oracle_margin_consistent"],
        }
        for h in (1, 10, 25, 50):
            o = s[f"oracle_vs_actual_{h}"]
            row[f"agree_{h}"] = o["agree"]
            row[f"o_n_{h}"], row[f"o_act_{h}"], row[f"o_orc_{h}"], row[f"o_both_{h}"] = o["n"], o["actual_fail"], o["oracle_fail"], o["both_fail"]
        rows.append(row)
    if not rows:
        print("no shards")
        return
    keys = list(rows[0])
    lines = []
    tot = defaultdict(float)
    for r in rows:
        for k in ("gb", "transitions", "terminations", "episodes", "near_fail_50", "near_fail_25", "term_height", "term_qvel",
                  "term_other", "fell_term", "fell_no_term", "fell_no_term_trunc", "fallen_steps_no_term"):
            tot[k] += r[k]
    lines.append(f"shards: {len(rows)}, size {tot['gb']:.1f} GB, transitions {int(tot['transitions'])}, terminations {int(tot['terminations'])}, "
                 f"episodes {int(tot['episodes'])}, (t, env) with a failure within 50 / 25 steps: {int(tot['near_fail_50'])} / {int(tot['near_fail_25'])}")
    lines.append(f"termination causes: IMU height < threshold {int(tot['term_height'])}, qvel limit only {int(tot['term_qvel'])}, other {int(tot['term_other'])}")
    lines.append(f"physical falls (>= 0.5 s): terminated episodes {int(tot['fell_term'])}, NOT terminated {int(tot['fell_no_term'])} "
                 f"(of which truncated {int(tot['fell_no_term_trunc'])}); fallen steps without termination {int(tot['fallen_steps_no_term'])}")
    lines.append(f"all margins consistent: {all(r['margin_ok'] for r in rows)}")
    # per method x setting
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["setting"])].append(r)
    lines.append("")
    lines.append("| method | setting | shards | transitions | term. | term/1k steps | near-fail@25 | shock rate | term: height / qvel / other | fell>=0.5s: term / not term | oracle agree @1/10/25/50 | oracle fail / actual fail @25 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (method, setting), rs in sorted(groups.items()):
        sm = lambda k: sum(r[k] for r in rs)
        agree = []
        for h in (1, 10, 25, 50):
            n = sm(f"o_n_{h}")
            agree.append(f"{sum(r[f'agree_{h}'] * r[f'o_n_{h}'] for r in rs if r[f'agree_{h}'] is not None) / n:.3f}" if n else "-")
        lines.append(f"| {method} | {setting} | {len(rs)} | {sm('transitions')} | {sm('terminations')} | {1000 * sm('terminations') / sm('transitions'):.2f} | "
                     f"{sm('near_fail_25')} | {sum(r['shock_rate'] for r in rs) / len(rs):.4f} | {sm('term_height')} / {sm('term_qvel')} / {sm('term_other')} | "
                     f"{sm('fell_term')} / {sm('fell_no_term')} | {' / '.join(agree)} | {sm('o_orc_25')} / {sm('o_act_25')} |")
    text = "\n".join(lines)
    print(text)
    if args.md:
        open(args.md, "w").write(text + "\n")
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
