#!/usr/bin/env python3
"""
IEEEtran-style ONE figure with 16 subplots (4x4), each row for a different metric.

For each experiment (inferred from:
  .../experiments/runs/<run_root>/<experiment>/<entry_name>/rollout/eval_....
), we pick an x-axis column *per experiment* from a candidate list.

Each row corresponds to one of the four metrics:
- Row 1: accuracy
- Row 2: false_positives
- Row 3: false_negatives
- Row 4: prediction_steps_true_pos

Each CSV = one method (colored line + shaded band).

MODIFICATION IN THIS VERSION:
  - Each experiment is plotted in its own column.
  - Each row corresponds to one of the four metrics.

Usage:
  python plot_ieee_4x4_lines_multi_x.py ours.csv base.csv \
    --labels Ours Baseline \
    --x_candidates sampling_prob action_noise obs_noise perturb_scale \
    --out ieee_4x4.png --no_show

If you want a fixed experiment order:
  --experiments action_noise_2 action_noise_hfield perturbation_2 uneven_terrain

Notes:
- CSV must contain 'file' column.
- If an experiment has >1 candidate x-column with variation, the first in
  --x_candidates is used.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TABLEAU10 = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]

exp_name_table = {
    "action_noise_2": "Action Noise - Flat Terrain",
    "action_noise_hfield": "Action Noise - Heightfield Terrain",
    "perturbation": "Perturbation - Flat Terrain",
    "perturbation_hfield": "Perturbation - Heightfield Terrain",
}

LEGEND_NAME_MAP = {
    "no_safety": "PPO-RL",
    "NCBF": "H=1",
    "ours": "ours",
    "ncbf": "state safety critic",
    "H1": "H=1",
    "H50": "H=50",
    "H10": "H=10",
}

def map_legend_label(name: str) -> str:
    # try exact match, then case-insensitive match
    if name in LEGEND_NAME_MAP:
        return LEGEND_NAME_MAP[name]
    low = name.lower()
    for k, v in LEGEND_NAME_MAP.items():
        if low == k.lower():
            return v
    return name

def apply_ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.2,
        "grid.linewidth": 0.4,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    })


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _ensure_finished_rate(df: pd.DataFrame) -> pd.DataFrame:
    if "accuracy" in df.columns and df["accuracy"].notna().any():
        return df
    raise ValueError(
        "Could not compute accuracy. Need 'accuracy' column."
    )


def infer_runroot_experiment_entry_from_file(p: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
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


def _read_one_csv(path: Path, x_candidates: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "file" not in df.columns:
        raise ValueError(f"{path} missing required column: file")

    # numeric conversion for common cols + candidates
    numeric_cols = ["accuracy", "false_positives", "false_negatives", "prediction_steps_true_pos"]
    numeric_cols += [c for c in x_candidates if c in df.columns]
    df = _to_numeric(df, numeric_cols)

    if "accuracy" not in df.columns:
        raise ValueError(f"{path} missing required column: accuracy")

    df = _ensure_finished_rate(df)

    triples = df["file"].astype(str).map(infer_runroot_experiment_entry_from_file)
    df["run_root"] = [t[0] for t in triples]
    df["experiment"] = [t[1] for t in triples]
    df["entry_name"] = [t[2] for t in triples]
    df = df.dropna(subset=["run_root", "experiment", "entry_name"]).copy()
    return df


def pick_xcol_for_experiment(df_union: pd.DataFrame, exp: str, x_candidates: List[str]) -> Optional[str]:
    sub = df_union[df_union["experiment"].astype(str) == str(exp)]
    for c in x_candidates:
        if c in sub.columns:
            vals = sub[c].dropna().unique()
            if len(vals) >= 2:
                return c
    # fallback: if candidate exists but only 1 value, still usable (flat line)
    for c in x_candidates:
        if c in sub.columns and sub[c].dropna().shape[0] > 0:
            return c
    return None


def aggregate_mean_std_over_entries(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    tmp = df[[xcol, ycol, "entry_name"]].dropna(subset=[xcol, ycol, "entry_name"]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=[xcol, "mean", "std", "n_entries"])

    per_entry = (
        tmp.groupby(["entry_name", xcol], dropna=False)[ycol]
        .mean()
        .reset_index()
        .rename(columns={ycol: "entry_mean"})
    )

    out = (
        per_entry.groupby(xcol, dropna=False)["entry_mean"]
        .agg(mean="mean", std="std", n_entries="count")
        .reset_index()
    )
    out["std"] = out["std"].fillna(0.0)
    return out


def _sorted_unique_numeric(s: pd.Series) -> List[float]:
    xs = pd.to_numeric(s, errors="coerce").dropna().astype(float).unique().tolist()
    return sorted(set(float(x) for x in xs))


def draw_subplot(ax: plt.Axes,
                 series_list: List[pd.DataFrame],
                 labels: List[str],
                 colors: List[str],
                 xcol: str,
                 ylabel: str,
                 title: str,
                 y_is_rate: bool,
                 show_ylabel: bool) -> None:
    ax.set_title(title, pad=2.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.grid(False, axis="x")

    # union x for this subplot only
    all_x: List[float] = []
    for s in series_list:
        if xcol in s.columns and not s.empty:
            all_x.extend(_sorted_unique_numeric(s[xcol]))
    x_vals = np.array(sorted(set(all_x)), dtype=float)
    if x_vals.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    for s, lab, col in zip(series_list, labels, colors):
        if s.empty or xcol not in s.columns:
            continue
        s2 = s.copy()
        s2[xcol] = pd.to_numeric(s2[xcol], errors="coerce")
        s2 = s2.dropna(subset=[xcol]).sort_values(xcol)

        mean_map = {float(r): float(m) for r, m in zip(s2[xcol].tolist(), s2["mean"].tolist())}
        std_map = {float(r): float(sd) for r, sd in zip(s2[xcol].tolist(), s2["std"].tolist())}

        y_mean = np.array([mean_map.get(float(r), np.nan) for r in x_vals], dtype=float)
        y_std = np.array([std_map.get(float(r), np.nan) for r in x_vals], dtype=float)

        ax.plot(x_vals, y_mean, marker="o", color=col, label=lab)
        ax.fill_between(
            x_vals,
            y_mean - y_std / 2.236,
            y_mean + y_std / 2.236,
            color=col,
            alpha=0.18,
            linewidth=0,
        )

    # x ticks: only bottom row (identified by ylabel)
    if ylabel == "Accuracy":
        ax.set_xlabel("sampling probability")
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{r:g}" for r in x_vals])
    else:
        ax.set_xticklabels([])

    # y label: only left column
    if show_ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    if y_is_rate:
        ax.set_ylim(-0.02, 1.02)


def main():
    apply_ieee_style()

    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--x_candidates", nargs="*", default=["sampling_prob"],
                    help="Ordered list of possible x columns. First with variation is used per experiment.")
    ap.add_argument("--out", type=str, default="/home/siwei/Pictures/RSS2026/exps_safe_fall.pdf")
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--runroot_filter", type=str, default=None)
    ap.add_argument("--experiment_filter", type=str, default=None)
    ap.add_argument("--experiments", nargs="*", default=None,
                    help="Optional explicit list of exactly 4 experiments in desired order.")
    ap.add_argument("--fig_width_in", type=float, default=7.16)
    args = ap.parse_args()

    csv_paths = [Path(p).expanduser().resolve() for p in args.csvs]
    for p in csv_paths:
        if not p.is_file():
            raise SystemExit(f"CSV not found: {p}")

    labels = args.labels
    if not labels or len(labels) != len(csv_paths):
        labels = [p.stem for p in csv_paths]

    # apply legend mapping
    labels = [map_legend_label(lab) for lab in labels]

    colors = [TABLEAU10[i % len(TABLEAU10)] for i in range(len(labels))]

    dfs_all: List[pd.DataFrame] = []
    for p in csv_paths:
        df = _read_one_csv(p, args.x_candidates)

        if args.runroot_filter:
            df = df[df["run_root"].astype(str).str.contains(args.runroot_filter, na=False)]
        if args.experiment_filter:
            df = df[df["experiment"].astype(str).str.contains(args.experiment_filter, na=False)]

        dfs_all.append(df)

    df_union = pd.concat(dfs_all, ignore_index=True)
    if df_union.empty:
        raise SystemExit("No data after loading/filtering.")

    # choose 4 experiments
    if args.experiments:
        if len(args.experiments) != 4:
            raise SystemExit("--experiments must list exactly 4 experiment names.")
        experiments = [str(e) for e in args.experiments]
    else:
        exp_counts = df_union["experiment"].astype(str).value_counts()
        experiments = exp_counts.index.tolist()[:4]

    if len(experiments) < 4:
        raise SystemExit(f"Need 4 experiments, found {len(experiments)}: {experiments}")

    # pick xcol per experiment
    exp_to_xcol: Dict[str, str] = {}
    for exp in experiments:
        xcol = pick_xcol_for_experiment(df_union, exp, args.x_candidates)
        if xcol is None:
            raise SystemExit(
                f"Could not pick x column for experiment '{exp}'. "
                f"None of {args.x_candidates} present with data."
            )
        exp_to_xcol[exp] = xcol

    # build aggregated series per (exp, csv)
    series_accuracy_by_exp: Dict[str, List[pd.DataFrame]] = {}
    series_false_positives_by_exp: Dict[str, List[pd.DataFrame]] = {}
    series_false_negatives_by_exp: Dict[str, List[pd.DataFrame]] = {}
    series_true_pos_by_exp: Dict[str, List[pd.DataFrame]] = {}

    for exp in experiments:
        xcol = exp_to_xcol[exp]
        s_acc: List[pd.DataFrame] = []
        s_fp: List[pd.DataFrame] = []
        s_fn: List[pd.DataFrame] = []
        s_tp: List[pd.DataFrame] = []
        for df in dfs_all:
            sub = df[df["experiment"].astype(str) == str(exp)].copy()
            if sub.empty or xcol not in sub.columns:
                s_acc.append(pd.DataFrame(columns=[xcol, "mean", "std", "n_entries"]))
                s_fp.append(pd.DataFrame(columns=[xcol, "mean", "std", "n_entries"]))
                s_fn.append(pd.DataFrame(columns=[xcol, "mean", "std", "n_entries"]))
                s_tp.append(pd.DataFrame(columns=[xcol, "mean", "std", "n_entries"]))
            else:
                s_acc.append(aggregate_mean_std_over_entries(sub, xcol, "accuracy"))
                s_fp.append(aggregate_mean_std_over_entries(sub, xcol, "false_positives"))
                s_fn.append(aggregate_mean_std_over_entries(sub, xcol, "false_negatives"))
                s_tp.append(aggregate_mean_std_over_entries(sub, xcol, "prediction_steps_true_pos"))
        series_accuracy_by_exp[exp] = s_acc
        series_false_positives_by_exp[exp] = s_fp
        series_false_negatives_by_exp[exp] = s_fn
        series_true_pos_by_exp[exp] = s_tp

    # plot
    fig_w = float(args.fig_width_in)
    fig_h = 4.4  # Adjusted height to fit 4 rows
    fig, axes = plt.subplots(4, 4, figsize=(fig_w, fig_h), sharex=False)

    # top row: accuracy
    for j, exp in enumerate(experiments):
        xcol = exp_to_xcol[exp]
        draw_subplot(
            ax=axes[0, j],
            series_list=series_accuracy_by_exp[exp],
            labels=labels,
            colors=colors,
            xcol=xcol,
            ylabel="Accuracy",
            title=f"{exp_name_table[exp]}",
            y_is_rate=True,
            show_ylabel=(j == 0),
        )

    # second row: false positives
    for j, exp in enumerate(experiments):
        xcol = exp_to_xcol[exp]
        draw_subplot(
            ax=axes[1, j],
            series_list=series_false_positives_by_exp[exp],
            labels=labels,
            colors=colors,
            xcol=xcol,
            ylabel="False Positives",
            title="",
            y_is_rate=False,
            show_ylabel=(j == 0),
        )

    # third row: false negatives
    for j, exp in enumerate(experiments):
        xcol = exp_to_xcol[exp]
        draw_subplot(
            ax=axes[2, j],
            series_list=series_false_negatives_by_exp[exp],
            labels=labels,
            colors=colors,
            xcol=xcol,
            ylabel="False Negatives",
            title="",
            y_is_rate=False,
            show_ylabel=(j == 0),
        )

    # fourth row: prediction_steps_true_pos
    for j, exp in enumerate(experiments):
        xcol = exp_to_xcol[exp]
        draw_subplot(
            ax=axes[3, j],
            series_list=series_true_pos_by_exp[exp],
            labels=labels,
            colors=colors,
            xcol=xcol,
            ylabel="Prediction Steps True Pos",
            title="",
            y_is_rate=False,
            show_ylabel=(j == 0),
        )

    # shared legend
    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            leg_labels,
            loc="upper center",
            ncol=min(len(labels), 5),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )

    plt.subplots_adjust(left=0.06, right=0.995, top=0.92, bottom=0.2, wspace=0.1, hspace=0.1)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Wrote: {out_path}")
    print("Per-experiment x-axis:", {k: v for k, v in exp_to_xcol.items()})

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    main()