#!/usr/bin/env python3
"""Read the `eval_summary.json` files written by the full-jit eval sweep (eval_sweep.py) into one DataFrame.

Replaces the paper's pkl -> CSV step (write_evaluation_to_csv.py). Each summary is one
(checkpoint, setting, deployment mode, sampling probability, eval seed) point with 1 episode x nr_envs envs.

Columns: every scalar summary field, `event_rates/<name>`, `protocol/<name>`, plus the paper-script aliases
  experiment   = paper experiment folder name (action_noise_2, action_noise_hfield, perturbation, perturbation_hfield)
  entry_name   = checkpoint_label (one trained model = one "entry" in the paper scripts)
  finished_rate, avg_return = success_rate, mean_episode_return
  summary_path = path of the json file

Usage:
  python misc/rss/eval_summary_reader.py <root> [<root> ...] [--out all.csv]
"""
import argparse
import json
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np

try:
    import pandas as pd
except ImportError:  # load_eval_summary_rows works without pandas
    pd = None

# new setting name -> paper experiment folder name (misc/rss/plot_*.py exp_name_table)
SETTING_TO_PAPER_EXPERIMENT = {
    "action_noise_flat": "action_noise_2",
    "action_noise_hfield": "action_noise_hfield",
    "push_flat": "perturbation",
    "push_hfield": "perturbation_hfield",
}
PAPER_EXPERIMENT_ORDER = ["action_noise_2", "action_noise_hfield", "perturbation", "perturbation_hfield"]


def _flatten(summary: dict, prefix: str = "") -> dict:
    row = {}
    for key, value in summary.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            row.update(_flatten(value, f"{name}/"))
        elif isinstance(value, list):
            row[name] = json.dumps(value)
        else:
            row[name] = value
    return row


def load_eval_summary_rows(roots: Iterable[Union[str, Path]], file_name: str = "eval_summary.json") -> List[dict]:
    """Flattened sweep summaries (one dict per json file) with the paper-script aliases; no pandas needed."""
    roots = list(roots)
    rows: List[dict] = []
    for root in roots:
        root = Path(root).expanduser()
        paths = [root] if root.is_file() else sorted(root.rglob(file_name))
        for path in paths:
            summary = json.loads(path.read_text())
            if "setting" not in summary:  # old multistep test() summary without sweep metadata
                continue
            row = _flatten(summary)
            row["summary_path"] = str(path)
            row["experiment"] = SETTING_TO_PAPER_EXPERIMENT.get(row["setting"], row["setting"])
            row["entry_name"] = row["checkpoint_label"]
            row["finished_rate"] = row["success_rate"]
            row["avg_return"] = row["mean_episode_return"]
            row["method"] = row.get("method") or row["checkpoint_label"]
            rows.append(row)
    if not rows:
        raise SystemExit(f"No sweep {file_name} found under {roots}")
    return rows


def read_eval_summaries(roots: Iterable[Union[str, Path]], file_name: str = "eval_summary.json") -> "pd.DataFrame":
    if pd is None:
        raise ImportError("read_eval_summaries needs pandas; use load_eval_summary_rows without it")
    return pd.DataFrame(load_eval_summary_rows(roots, file_name))


def aggregate_over_checkpoints(df, value: str, keys=("method", "deployment_mode", "experiment", "sampling_prob")):
    """Mean over eval seeds per checkpoint, then mean and standard error over checkpoints (training seeds)."""
    keys = list(keys)
    per_checkpoint = df.groupby(keys + ["checkpoint_label"], dropna=False)[value].mean().reset_index()
    out = per_checkpoint.groupby(keys, dropna=False)[value].agg(mean="mean", std="std", n="count").reset_index()
    out["std"] = out["std"].fillna(0.0)
    out["se"] = out["std"] / np.sqrt(out["n"].clip(lower=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = read_eval_summaries(args.roots)
    out = Path(args.out) if args.out else Path(args.roots[0]) / "eval_summaries_sweep.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
