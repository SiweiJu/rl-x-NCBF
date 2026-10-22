#!/usr/bin/env python3
"""
Paper-style line plots (mean ± std as shaded area) from MULTIPLE CSVs, split by experiment.

For each experiment (inferred from 'file' path as:
  .../experiments/runs/<run_root>/<experiment>/<entry_name>/rollout/eval_....
), we create TWO figures:
  1) finished_rate vs ratio (mean ± std over entry folders / models)
  2) avg_return   vs ratio (mean ± std over entry folders / models)

- Inputs: multiple CSV paths (each CSV is one method / batch)
- X axis: ratio (default: sampling_prob; override via --xcol)
- One colored line per CSV
- Shaded band: ±1 std (computed over entry_name)
- Separate plots per experiment (your 4 experiment settings become 4 folders)

Important detail:
  We compute per-entry statistics first:
    for each (entry_name, ratio): take mean of all rows in that entry at that ratio
  Then aggregate across entries to get mean/std per ratio.
  This avoids overweighting entries that have more rows.

Usage:
  python plot_eval_csv_lines_multi_by_experiment_paper.py \
    a.csv b.csv c.csv \
    --labels Ours SAC PPO \
    --out_dir ./plots_lines --no_show

Optional filters:
  --runroot_filter ours
  --experiment_filter action_noise
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------- style (RSS / paper-like) -----------------------
TABLEAU10 = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.7,
    })


# ----------------------- data utils -----------------------
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
    Returns (run_root, experiment, entry_name). If parsing fails -> (None,None,None).
    """
    try:
        parts = Path(p).parts
        for i in range(len(parts) - 3):
            if parts[i] == "experiments" and parts[i + 1] == "runs":
                run_root = parts[i + 2] if i + 2 < len(parts) else None
                experiment = parts[i + 3] if i + 3 < len(parts) else None
                entry = parts[i + 4] if i + 4 < len(parts) else None
                return run_root, experiment, entry
        return None, None, None
    except Exception:
        return None, None, None


def _read_one_csv(path: Path, xcol: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "file" not in df.columns:
        raise ValueError(f"{path} missing required column: file (needed to infer experiment)")

    for c in [xcol, "avg_return"]:
        if c not in df.columns:
            raise ValueError(f"{path} missing required column: {c}")

    df = _to_numeric(
        df,
        [xcol, "avg_return", "gamma", "n_episodes", "finished_count", "finished_episodes(>=thr)", "finished_rate"],
    )
    df = _ensure_finished_rate(df)
    df = df.dropna(subset=[xcol]).copy()

    triples = df["file"].astype(str).map(infer_runroot_experiment_entry_from_file)
    df["run_root"] = [t[0] for t in triples]
    df["experiment"] = [t[1] for t in triples]
    df["entry_name"] = [t[2] for t in triples]

    df = df.dropna(subset=["run_root", "experiment", "entry_name"]).copy()
    return df


def _sorted_unique(vals: List[float]) -> List[float]:
    xs = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return sorted(set(xs))


def aggregate_mean_std_over_entries(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    """
    Returns a table with columns:
      ratio, mean, std, n_entries

    Steps:
      1) per entry: mean(y) for each (entry_name, ratio)
      2) across entries: mean/std of those per-entry means for each ratio
    """
    tmp = df[[xcol, ycol, "entry_name"]].dropna(subset=[xcol, ycol, "entry_name"]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=[xcol, "mean", "std", "n_entries"])

    # Per-entry mean at each ratio (equal weight per entry)
    per_entry = (
        tmp.groupby(["entry_name", xcol], dropna=False)[ycol]
        .mean()
        .reset_index()
        .rename(columns={ycol: "entry_mean"})
    )

    # Aggregate over entries
    out = (
        per_entry.groupby(xcol, dropna=False)["entry_mean"]
        .agg(mean="mean", std="std", n_entries="count")
        .reset_index()
    )
    out["std"] = out["std"].fillna(0.0)
    return out


# ----------------------- plotting -----------------------
def plot_lines_mean_std(
    series_list: List[pd.DataFrame],
    labels: List[str],
    colors: List[str],
    xcol: str,
    ylabel: str,
    title: str,
    out_path: Path,
    show: bool,
) -> None:
    # collect union of ratios for x ticks
    all_ratios: List[float] = []
    for s in series_list:
        if xcol in s.columns:
            all_ratios.extend(s[xcol].dropna().astype(float).tolist())
    ratios = _sorted_unique(all_ratios)
    if not ratios:
        print(f"[WARN] Nothing to plot: {title}")
        return

    fig_w = max(9.0, 1.2 * len(ratios))
    fig, ax = plt.subplots(figsize=(fig_w, 4.6))

    ax.set_title(title)
    ax.set_xlabel("ratio")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.grid(False, axis="x")

    # fixed x locations for clean spacing (ratios can be floats)
    x_vals = np.array(ratios, dtype=float)

    for s, lab, col in zip(series_list, labels, colors):
        if s.empty:
            continue

        s2 = s.copy()
        s2[xcol] = pd.to_numeric(s2[xcol], errors="coerce")
        s2 = s2.dropna(subset=[xcol]).sort_values(xcol)

        # align to union ratios
        mean_map = {float(r): float(m) for r, m in zip(s2[xcol].tolist(), s2["mean"].tolist())}
        std_map = {float(r): float(sd) for r, sd in zip(s2[xcol].tolist(), s2["std"].tolist())}
        n_map = {float(r): int(n) for r, n in zip(s2[xcol].tolist(), s2["n_entries"].tolist())}

        y_mean = np.array([mean_map.get(float(r), np.nan) for r in x_vals], dtype=float)
        y_std = np.array([std_map.get(float(r), np.nan) for r in x_vals], dtype=float)

        ax.plot(
            x_vals, y_mean,
            marker="o", linewidth=2.2, color=col,
            label=f"{lab}",
        )
        ax.fill_between(
            x_vals, y_mean - y_std / 2.236, y_mean + y_std/2.236,
            color=col, alpha=0.18, linewidth=0
        )

    ax.set_xticks(x_vals)
    ax.set_xticklabels([f"{r:g}" for r in ratios])

    if ylabel.lower().startswith("finished"):
        ax.set_ylim(-0.02, 1.02)

    ax.legend(loc="best", frameon=False, ncol=1 if len(labels) <= 4 else 2)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Wrote: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    apply_paper_style()

    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="Input CSV paths (each becomes one colored mean±std line).")
    ap.add_argument("--labels", nargs="*", default=None, help="Legend labels for CSVs (same length as csvs).")
    ap.add_argument("--xcol", type=str, default="sampling_prob", help="X column name (ratio), default sampling_prob.")
    ap.add_argument("--out_dir", type=str, default="./plots_lines_by_experiment", help="Directory to write plots.")
    ap.add_argument("--no_show", action="store_true", help="Do not open plot windows.")
    ap.add_argument("--runroot_filter", type=str, default=None, help="Only include run_root containing this substring (e.g. ours)")
    ap.add_argument("--experiment_filter", type=str, default=None, help="Only include experiments containing this substring")
    args = ap.parse_args()

    csv_paths = [Path(p).expanduser().resolve() for p in args.csvs]
    for p in csv_paths:
        if not p.is_file():
            raise SystemExit(f"CSV not found: {p}")

    labels = args.labels
    if not labels or len(labels) != len(csv_paths):
        labels = [p.stem for p in csv_paths]

    colors = [TABLEAU10[i % len(TABLEAU10)] for i in range(len(labels))]

    # read all csvs
    dfs_all: List[pd.DataFrame] = []
    for p in csv_paths:
        dfs_all.append(_read_one_csv(p, args.xcol))

    # union of experiments across all CSVs
    experiments = sorted(set(pd.concat(dfs_all, ignore_index=True)["experiment"].astype(str).unique().tolist()))

    out_dir = Path(args.out_dir).expanduser().resolve()
    show = not args.no_show

    for exp in experiments:
        if args.experiment_filter and args.experiment_filter not in str(exp):
            continue

        # Build per-CSV aggregated series for this experiment
        series_finish: List[pd.DataFrame] = []
        series_return: List[pd.DataFrame] = []

        for df in dfs_all:
            sub = df[df["experiment"].astype(str) == str(exp)].copy()

            if args.runroot_filter:
                sub = sub[sub["run_root"].astype(str).str.contains(args.runroot_filter, na=False)]

            if sub.empty:
                series_finish.append(pd.DataFrame(columns=[args.xcol, "mean", "std", "n_entries"]))
                series_return.append(pd.DataFrame(columns=[args.xcol, "mean", "std", "n_entries"]))
                continue

            series_finish.append(aggregate_mean_std_over_entries(sub, args.xcol, "finished_rate"))
            series_return.append(aggregate_mean_std_over_entries(sub, args.xcol, "avg_return"))

        exp_dir = out_dir / str(exp)

        plot_lines_mean_std(
            series_list=series_finish,
            labels=labels,
            colors=colors,
            xcol=args.xcol,
            ylabel="Success rate (mean ± 1 std over models)",
            title=f"{exp}: Success rate rate (mean±std over model entries)",
            out_path=exp_dir / "lines_finished_rate_meanstd.png",
            show=show,
        )

        plot_lines_mean_std(
            series_list=series_return,
            labels=labels,
            colors=colors,
            xcol=args.xcol,
            ylabel="avg_return (mean ± 1 std over models)",
            title=f"{exp}: Average return (mean±std over model entries)",
            out_path=exp_dir / "lines_avg_return_meanstd.png",
            show=show,
        )

    print(f"Done. Plots in: {out_dir}")


if __name__ == "__main__":
    main()