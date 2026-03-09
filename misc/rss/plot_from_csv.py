#!/usr/bin/env python3
"""
Plot eval CSV with:
  - one figure per (experiment, run_root) where run_root is the folder under .../experiments/runs/
  - x: sampling_prob
  - left y: finished rate (mean ± std over entry_name)
  - right y: avg return (mean ± std over entry_name)
  - one line per gamma (same color; return is dashed)

Requires CSV to have a 'file' column with the original pkl path.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _ensure_finished_rate(df: pd.DataFrame) -> pd.DataFrame:
    if "finished_rate" in df.columns and df["finished_rate"].notna().any():
        return df

    if "finished_count" in df.columns and "n_episodes" in df.columns:
        df["finished_rate"] = df["finished_count"] / df["n_episodes"].replace(0, np.nan)
        return df

    if "finished_episodes(>=thr)" in df.columns and "n_episodes" in df.columns:
        df["finished_rate"] = df["finished_episodes(>=thr)"] / df["n_episodes"].replace(0, np.nan)
        return df

    raise ValueError(
        "Could not compute finished_rate. Need one of:\n"
        "  - finished_rate\n"
        "  - finished_count + n_episodes\n"
        "  - finished_episodes(>=thr) + n_episodes"
    )


def infer_runroot_experiment_entry_from_file(p: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Expected:
      .../experiments/runs/<run_root>/<experiment>/<entry_name>/rollout/eval_....

    Returns (run_root, experiment, entry_name). If parsing fails, returns (None,None,None).
    """
    try:
        parts = Path(p).parts
        # find ".../experiments/runs/"
        for i in range(len(parts) - 3):
            if parts[i] == "experiments" and parts[i + 1] == "runs":
                run_root = parts[i + 2]
                experiment = parts[i + 3]
                entry = parts[i + 4] if i + 4 < len(parts) else None
                return run_root, experiment, entry
        return None, None, None
    except Exception:
        return None, None, None


def aggregate_over_entries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate over entry_name within each (run_root, experiment, gamma, sampling_prob).
    """
    group_cols = ["run_root", "experiment", "gamma", "sampling_prob"]

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_entries=("entry_name", "nunique"),
            finished_mean=("finished_rate", "mean"),
            finished_std=("finished_rate", "std"),
            ret_mean=("avg_return", "mean"),
            ret_std=("avg_return", "std"),
        )
        .reset_index()
    )
    agg["finished_std"] = agg["finished_std"].fillna(0.0)
    agg["ret_std"] = agg["ret_std"].fillna(0.0)
    return agg


def _sorted_unique_nonan(xs: pd.Series) -> List[float]:
    vals = [float(x) for x in xs.dropna().unique().tolist()]
    return sorted(vals)


def plot_one(agg: pd.DataFrame, run_root: str, experiment: str, out_dir: Path, show: bool) -> None:
    sub = agg[(agg["run_root"] == run_root) & (agg["experiment"] == experiment)].copy()
    if sub.empty:
        return

    gammas = _sorted_unique_nonan(sub["gamma"])
    if not gammas:
        gammas = [np.nan]

    fig, axL = plt.subplots(figsize=(10, 5))
    axR = axL.twinx()

    axL.set_title(f"{run_root} | {experiment}  (γ lines; mean±std over entry folders)")
    axL.set_xlabel("sampling_prob")
    axL.set_ylabel("finished_rate (mean ± 1 std)")
    axR.set_ylabel("avg_return (mean ± 1 std)")

    for g in gammas:
        grp = sub[sub["gamma"].isna()] if np.isnan(g) else sub[sub["gamma"] == g]
        if grp.empty:
            continue
        grp = grp.sort_values("sampling_prob")

        x = grp["sampling_prob"].to_numpy(dtype=float)

        fm = grp["finished_mean"].to_numpy(dtype=float)
        fs = grp["finished_std"].to_numpy(dtype=float)
        rm = grp["ret_mean"].to_numpy(dtype=float)
        rs = grp["ret_std"].to_numpy(dtype=float)

        label_g = "nan" if np.isnan(g) else f"{g:g}"

        (lineL,) = axL.plot(x, fm, marker="o", linewidth=2, label=f"γ={label_g} finished")
        col = lineL.get_color()
        axL.fill_between(x, fm - fs, fm + fs, alpha=0.18, color=col)

        axR.plot(x, rm, marker="x", linestyle="--", linewidth=2, color=col, label=f"γ={label_g} return")
        axR.fill_between(x, rm - rs, rm + rs, alpha=0.12, color=col)

    axL.grid(True, linestyle="--", alpha=0.3)

    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axR.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, loc="best", fontsize=9, ncol=2)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_root}__{experiment}__gamma_dualaxis.png"
    fig.savefig(out_path, dpi=200)
    print(f"Wrote: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--csv_path", default="/home/siwei/Documents/repos/rl-x-NCBF-q/experiments/runs/ours/eval_summary_20260130_082633.csv", help="Path to summary CSV (must include 'file' column)")
    ap.add_argument("--csv_path", default="/home/siwei/Documents/repos/rl-x-NCBF/experiments/runs/ncbf-state/eval_summary_20260206_074513.csv", help="Path to summary CSV (must include 'file' column)")
    ap.add_argument("--out_dir", type=str, default="./plots_eval", help="Directory to write plots")
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--runroot_filter", type=str, default=None, help="Only include run_root containing this substring (e.g. ours)")
    ap.add_argument("--experiment_filter", type=str, default=None, help="Only include experiments containing this substring")
    args = ap.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Must have file to infer the grouping you want
    if "file" not in df.columns:
        raise SystemExit("CSV must contain a 'file' column (full pkl path) to infer run_root/experiment/entry_name.")

    # Required
    for c in ["avg_return", "sampling_prob", "gamma"]:
        if c not in df.columns:
            raise SystemExit(f"CSV missing required column: {c}")

    df = _to_numeric(df, ["sampling_prob", "avg_return", "gamma", "n_episodes",
                          "finished_count", "finished_episodes(>=thr)", "finished_rate"])

    df = _ensure_finished_rate(df)
    df = df.dropna(subset=["sampling_prob"]).copy()

    # Infer run_root / experiment / entry_name from file path
    triples = df["file"].astype(str).map(infer_runroot_experiment_entry_from_file)
    df["run_root"] = [t[0] for t in triples]
    df["experiment_inferred"] = [t[1] for t in triples]
    df["entry_name"] = [t[2] for t in triples]

    # Use inferred experiment (this matches your directory layout)
    df["experiment"] = df["experiment_inferred"]

    df = df.dropna(subset=["run_root", "experiment", "entry_name"]).copy()

    # Only plot eval_safety_False rows if that's what you want (optional)
    # df = df[df["file"].astype(str).str.contains("eval_safety_False", na=False)]

    if args.runroot_filter:
        df = df[df["run_root"].astype(str).str.contains(args.runroot_filter, na=False)]
    if args.experiment_filter:
        df = df[df["experiment"].astype(str).str.contains(args.experiment_filter, na=False)]

    if df.empty:
        raise SystemExit("No data left after filtering.")

    agg = aggregate_over_entries(df)

    out_dir = Path(args.out_dir).expanduser().resolve()
    run_roots = sorted(agg["run_root"].dropna().unique().tolist())

    for rr in run_roots:
        exps = sorted(agg.loc[agg["run_root"] == rr, "experiment"].dropna().unique().tolist())
        for exp in exps:
            plot_one(agg, rr, exp, out_dir / rr, show=(not args.no_show))

    print(f"Done. Plots in: {out_dir}")


if __name__ == "__main__":
    main()