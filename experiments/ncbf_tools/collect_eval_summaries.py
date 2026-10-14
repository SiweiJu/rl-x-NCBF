"""Collect eval_summary.json files written by `runner.mode=test` into one CSV.

Usage:
    python ncbf_tools/collect_eval_summaries.py runs/<project>/<exp> [--out results.csv]

Each run directory name becomes the `run` column; a trailing `_speed<v>` (as written by
cluster_runner/eval_go2_obstacles_speed_sweep.sh) is parsed into an `obstacle_speed` column.
"""
import argparse
import csv
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for summary_path in sorted(args.root.rglob("eval_summary.json")):
        summary = json.loads(summary_path.read_text())
        run = summary_path.parent.name
        match = re.search(r"_speed(\d+)p?(\d*)$", run)
        speed = float(f"{match.group(1)}.{match.group(2) or 0}") if match else None
        rows.append({"run": run, "obstacle_speed": speed, **summary})

    if not rows:
        raise SystemExit(f"No eval_summary.json found under {args.root}")
    out = args.out or args.root / "eval_summaries.csv"
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
