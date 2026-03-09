#!/usr/bin/env python3
"""
Aggregate rollout evaluation PKLs under a runs folder and write a CSV summary.

You provide a RUNS ROOT folder, e.g.
  /home/siwei/Documents/repos/rl-x-NCBF-q/experiments/runs/

We recursively find:
  **/rollout/eval_safety_*_gamma_*_sampling_prob_*_seed_*.pkl

For each PKL:
  - compute mean finished episodes (len >= threshold) and mean return
  - extract:
      model_name      := parent of 'rollout' (e.g., no_safe_seed_1042)
      experiment_name := parent of model_name (e.g., action_noise_hfield)
      project_folder  := parent of experiment_name (e.g., /.../runs/no_safety)

Then write a CSV into the project folder:
  <project_folder>/eval_summary_<timestamp>.csv

Notes
- "finished episodes" is counted per file as (#episodes with length >= threshold).
  We store both counts and rates (finished_rate = finished/n_episodes).
- Return is taken from ep["ret"] (scalar or [1]) if present, else sum(ep["rewards"]).
"""

import argparse
import csv
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


FNAME_RE = re.compile(
    r"eval_safety_(?P<safety>True|False)"
    r"_gamma_(?P<gamma>[0-9mp]+)"
    r"_sampling_prob_(?P<sp>[0-9mp]+)"
    r"_seed_(?P<seed>\d+)"
    r"\.pkl$"
)


def tok_to_float(tok: Optional[str]) -> Optional[float]:
    """Convert tokens like 0p002, 1p0, m0p405 into floats."""
    if tok is None:
        return None
    sign = -1.0 if tok.startswith("m") else 1.0
    core = tok[1:] if tok.startswith("m") else tok
    core = core.replace("p", ".")
    return sign * float(core)


def parse_from_filename(p: Path) -> Dict[str, Any]:
    m = FNAME_RE.search(p.name)
    if not m:
        return {
            "safety_layer": None,
            "gamma": None,
            "sampling_prob": None,
            "seed": None,
        }
    return {
        "safety_layer": (m.group("safety") == "True"),
        "gamma": tok_to_float(m.group("gamma")),
        "sampling_prob": tok_to_float(m.group("sp")),
        "seed": int(m.group("seed")),
    }


def infer_names_from_path(pkl_path: Path) -> Tuple[str, str, Path]:
    """
    Locate 'rollout' in the path and infer:
      model      = parent of rollout
      experiment = parent of model
      project    = parent of experiment

    Example:
      .../runs/no_safety/action_noise_hfield/no_safe_seed_1042/rollout/eval_...
        => project   .../runs/no_safety
           experiment action_noise_hfield
           model      no_safe_seed_1042
    """
    parts = pkl_path.parts
    if "rollout" in parts:
        idx = parts.index("rollout")
        if idx < 2:
            raise ValueError(f"Path too short to infer experiment/model: {pkl_path}")
        model = parts[idx - 1]
        experiment = parts[idx - 2]
        project = Path(*parts[: idx - 2])  # up to parent of experiment
        return experiment, model, project

    # Fallback: assume .../<experiment>/<model>/<file.pkl>
    model = pkl_path.parent.name
    experiment = pkl_path.parent.parent.name
    project = pkl_path.parent.parent.parent
    return experiment, model, project


def load_rollouts(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"{path} did not contain a list. Got: {type(obj)}")
    return obj


def episode_length(ep: Dict[str, Any]) -> int:
    for k in ("dones", "rewards", "actions", "raw_action", "safe_action", "predictions"):
        if k in ep:
            try:
                return int(np.asarray(ep[k]).shape[0])
            except Exception:
                pass
    for v in ep.values():
        try:
            arr = np.asarray(v)
            if arr.ndim >= 1 and arr.shape[0] > 1:
                return int(arr.shape[0])
        except Exception:
            continue
    return 0


def episode_return(ep: Dict[str, Any]) -> Optional[float]:
    if "ret" in ep:
        try:
            arr = np.asarray(ep["ret"])
            if arr.shape == ():
                return float(arr)
            return float(arr.reshape(-1)[0])
        except Exception:
            pass
    if "episode_return" in ep:
        try:
            return float(ep["episode_return"])
        except Exception:
            pass
    if "rewards" in ep:
        try:
            return float(np.sum(np.asarray(ep["rewards"], dtype=float)))
        except Exception:
            pass
    return None


@dataclass
class FileSummary:
    file: str
    project: str
    experiment: str
    model: str
    safety_layer: Optional[bool]
    gamma: Optional[float]
    sampling_prob: Optional[float]
    seed: Optional[int]

    n_episodes: int
    finished_count: int
    finished_rate: float

    avg_return: float
    std_return: float
    missing_return_count: int

    avg_ep_len: float
    min_ep_len: int
    max_ep_len: int


def summarize_one_pkl(path: Path, threshold: int) -> FileSummary:
    meta = parse_from_filename(path)
    experiment, model, project = infer_names_from_path(path)

    rollouts = load_rollouts(path)
    lengths = np.array([episode_length(ep) for ep in rollouts], dtype=int)

    n_eps = int(len(rollouts))
    finished = int(np.sum(lengths >= threshold))
    finished_rate = float(finished / n_eps) if n_eps > 0 else float("nan")

    rets: List[float] = []
    missing_ret = 0
    for ep in rollouts:
        r = episode_return(ep)
        if r is None:
            missing_ret += 1
        else:
            rets.append(r)
    rets_arr = np.array(rets, dtype=float) if rets else np.array([], dtype=float)

    avg_ret = float(np.mean(rets_arr)) if rets_arr.size else float("nan")
    std_ret = float(np.std(rets_arr)) if rets_arr.size else float("nan")

    return FileSummary(
        file=str(path),
        project=str(project),
        experiment=experiment,
        model=model,
        safety_layer=meta["safety_layer"],
        gamma=meta["gamma"],
        sampling_prob=meta["sampling_prob"],
        seed=meta["seed"],
        n_episodes=n_eps,
        finished_count=finished,
        finished_rate=finished_rate,
        avg_return=avg_ret,
        std_return=std_ret,
        missing_return_count=int(missing_ret),
        avg_ep_len=float(np.mean(lengths)) if lengths.size else float("nan"),
        min_ep_len=int(np.min(lengths)) if lengths.size else 0,
        max_ep_len=int(np.max(lengths)) if lengths.size else 0,
    )


def write_csv(rows: List[FileSummary], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "project",
        "experiment",
        "model",
        "safety_layer",
        "gamma",
        "sampling_prob",
        "seed",
        "n_episodes",
        "finished_count",
        "finished_rate",
        "avg_return",
        "std_return",
        "missing_return_count",
        "avg_ep_len",
        "min_ep_len",
        "max_ep_len",
        "file",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fieldnames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs_root",
        default="/home/siwei/Documents/repos/rl-x-NCBF-q/experiments/runs/safe_fall",
        type=str,
        help="Path to experiments/runs/ (or any folder containing rollout/eval_safety_*.pkl files).",
    )
    ap.add_argument(
        "--threshold",
        type=int,
        default=250,
        help="Episode length >= threshold counts as finished.",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="**/rollout/eval_safety_*_gamma_*_sampling_prob_*_seed_*.pkl",
        help="Glob pattern (relative to runs_root) to find pkl files.",
    )
    ap.add_argument(
        "--project_filter",
        type=str,
        default=None,
        help="Only include pkls whose inferred project folder name contains this substring (e.g., 'no_safety').",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional explicit output CSV path. If omitted, writes one CSV per project folder.",
    )
    args = ap.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    if not runs_root.exists():
        raise SystemExit(f"runs_root does not exist: {runs_root}")

    pkls = sorted(runs_root.glob(args.pattern))
    pkls = [p for p in pkls if p.is_file()]

    if not pkls:
        raise SystemExit(f"No PKLs found under {runs_root} with pattern: {args.pattern}")

    summaries: List[FileSummary] = []
    for p in pkls:
        try:
            s = summarize_one_pkl(p, args.threshold)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
            continue

        if args.project_filter and args.project_filter not in Path(s.project).name:
            continue

        summaries.append(s)

    if not summaries:
        raise SystemExit("No summaries produced (all files skipped or filtered).")

    # If user requests one output file, write everything there.
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        write_csv(summaries, out_path)
        print(f"Wrote CSV: {out_path}")
        return

    # Otherwise write one CSV per inferred project folder (recommended).
    by_project: Dict[str, List[FileSummary]] = defaultdict(list)
    for s in summaries:
        by_project[s.project].append(s)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for proj, rows in sorted(by_project.items(), key=lambda kv: kv[0]):
        proj_path = Path(proj)
        out_path = proj_path / f"eval_summary_{ts}.csv"
        write_csv(rows, out_path)
        print(f"Wrote CSV: {out_path}  (n_files={len(rows)})")


if __name__ == "__main__":
    main()
