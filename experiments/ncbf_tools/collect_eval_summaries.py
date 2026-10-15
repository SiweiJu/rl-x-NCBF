"""Collect eval_summary.json files written by `runner.mode=test` into one CSV.

Usage:
    python ncbf_tools/collect_eval_summaries.py runs/<project>/<exp> [--out results.csv] [--file counterfactual_summary.json]

Each run directory name becomes the `run` column; a trailing `_speed<v>` (as written by
cluster_runner/eval_go2_obstacles_speed_sweep.sh) is parsed into an `obstacle_speed` column, and a
trailing obstacle preset (cluster_runner/eval_go2_obstacles_held_out.sh) into an `obstacle_preset` column.
Nested entries become `parent/child` columns; lists are written as JSON.
"""
import argparse
import csv
import json
import re
from pathlib import Path

PRESETS = ("train", "held_out_shape", "held_out_motion", "held_out_size", "held_out_sensor")


def flatten(summary, prefix=""):
    row = {}
    for key, value in summary.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            row.update(flatten(value, f"{name}/"))
        elif isinstance(value, list):
            row[name] = json.dumps(value)
        else:
            row[name] = value
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--file", default="eval_summary.json", help="summary file name to collect")
    args = parser.parse_args()

    rows = []
    for summary_path in sorted(args.root.rglob(args.file)):
        summary = json.loads(summary_path.read_text())
        run = summary_path.parent.name
        match = re.search(r"_speed(\d+)p?(\d*)$", run)
        speed = float(f"{match.group(1)}.{match.group(2) or 0}") if match else None
        preset = next((name for name in PRESETS if run.endswith(f"_{name}")), None)
        rows.append({"run": run, "obstacle_speed": speed, "obstacle_preset": preset, **flatten(summary)})

    if not rows:
        raise SystemExit(f"No {args.file} found under {args.root}")
    out = args.out or args.root / (Path(args.file).stem.replace("summary", "summaries") + ".csv")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
