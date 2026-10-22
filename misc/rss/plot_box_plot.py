#!/usr/bin/env python3
"""
Paper-style grouped boxplots from MULTIPLE CSVs, split by experiment setting.

For each experiment (inferred from the 'file' path as:
  .../experiments/runs/<run_root>/<experiment>/<entry_name>/rollout/eval_....
), we create TWO figures:
  1) finished_rate boxplots (grouped by ratio; one box per CSV)
  2) avg_return boxplots    (grouped by ratio; one box per CSV)

Improvements vs previous:
  - Distinct, colorblind-friendly colors per CSV (Tableau palette).
  - Filled boxes with consistent styling, clean grid, compact legend.
  - Higher DPI, tight layout, minimal chartjunk (RSS-ready).

Usage:
  python plot_eval_csv_box_multi_by_experiment_paper.py \
    a.csv b.csv c.csv \
    --labels Ours SAC PPO \
    --out_dir ./plots_box --no_show

Notes:
  - Each CSV must have: file, sampling_prob (or --xcol), avg_return
  - finished_rate computed from finished_count/n_episodes or finished_episodes(>=thr)/n_episodes if absent.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------- style (RSS / paper-like) -----------------------
# Tableau 10 (colorblind-friendly, common in papers)
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


def build_samples_per_ratio(
    dfs: List[pd.DataFrame],
    xcol: str,
    ycol: str
) -> Tuple[List[float], List[List[np.ndarray]]]:
    all_ratios: List[float] = []
    for df in dfs:
        all_ratios.extend(df[xcol].dropna().astype(float).tolist())
    ratios = _sorted_unique(all_ratios)

    data_per_ratio: List[List[np.ndarray]] = []
    for r in ratios:
        per_csv: List[np.ndarray] = []
        for df in dfs:
            sub = df[np.isclose(df[xcol].astype(float), r)]
            arr = sub[ycol].dropna().to_numpy(dtype=float)
            per_csv.append(arr)
        data_per_ratio.append(per_csv)

    return ratios, data_per_ratio


# ----------------------- plotting -----------------------
def _lighten_color(hex_color: str, factor: float = 0.25) -> str:
    """Blend hex color with white by 'factor' (0..1)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02X}{g:02X}{b:02X}"


def plot_grouped_box_paper(
    ratios: List[float],
    data_per_ratio: List[List[np.ndarray]],
    labels: List[str],
    colors: List[str],
    title: str,
    ylabel: str,
    out_path: Path,
    show: bool,
) -> None:
    n_csv = len(labels)
    if n_csv == 0 or len(ratios) == 0:
        print(f"[WARN] Nothing to plot for {title}")
        return

    centers = np.arange(len(ratios), dtype=float) + 1.0
    group_width = 0.82
    box_width = group_width / max(1, n_csv)

    fig_w = max(9.5, 1.35 * len(ratios))
    fig, ax = plt.subplots(figsize=(fig_w, 4.6))

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("ratio")

    # Subtle grid like in papers
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.grid(False, axis="x")

    legend_handles = []
    for j in range(n_csv):
        positions = centers - group_width / 2.0 + (j + 0.5) * box_width

        series = []
        for i in range(len(ratios)):
            arr = data_per_ratio[i][j]
            # Empty -> NaN placeholder so spacing stays consistent
            series.append(arr if arr.size > 0 else np.array([np.nan], dtype=float))

        # Filled boxes
        bp = ax.boxplot(
            series,
            positions=positions,
            widths=box_width * 0.92,
            patch_artist=True,
            showfliers=True,
            whis=1.5,
            manage_ticks=False,
        )

        col = colors[j % len(colors)]
        face = _lighten_color(col, 0.35)

        for box in bp["boxes"]:
            box.set(facecolor=face, edgecolor=col, linewidth=1.2)

        for whisk in bp["whiskers"]:
            whisk.set(color=col, linewidth=1.1)
        for cap in bp["caps"]:
            cap.set(color=col, linewidth=1.1)
        for med in bp["medians"]:
            med.set(color=col, linewidth=1.8)

        # Fliers: small, same color, partially transparent
        for fl in bp["fliers"]:
            fl.set(marker="o", markersize=3, markerfacecolor=col, markeredgecolor=col, alpha=0.35)

        # Use a median line as legend handle
        legend_handles.append(bp["medians"][0])

    ax.set_xticks(centers)
    ax.set_xticklabels([f"{r:g}" for r in ratios])

    # Tight y-lims for finished_rate
    if ylabel.lower().startswith("finished"):
        ax.set_ylim(-0.02, 1.02)

    # Legend: compact, no frame (paper-like)
    ax.legend(
        legend_handles,
        labels,
        loc="upper right",
        frameon=False,
        ncol=1 if len(labels) <= 4 else 2,
        handlelength=1.8,
        columnspacing=1.0,
    )

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
    ap.add_argument("csvs", nargs="+", help="Input CSV paths (each becomes one method color / one box per ratio).")
    ap.add_argument("--labels", nargs="*", default=None, help="Legend labels for CSVs (same length as csvs).")
    ap.add_argument("--xcol", type=str, default="sampling_prob", help="X column name (ratio), default sampling_prob.")
    ap.add_argument("--out_dir", type=str, default="./plots_box_by_experiment", help="Directory to write plots.")
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

    dfs_all: List[pd.DataFrame] = []
    for p, lab in zip(csv_paths, labels):
        df = _read_one_csv(p, args.xcol)
        df["__source_label__"] = lab
        dfs_all.append(df)

    experiments = sorted(set(pd.concat(dfs_all, ignore_index=True)["experiment"].astype(str).unique().tolist()))

    out_dir = Path(args.out_dir).expanduser().resolve()
    show = not args.no_show

    for exp in experiments:
        dfs_exp: List[pd.DataFrame] = []
        for df in dfs_all:
            sub = df[df["experiment"].astype(str) == str(exp)].copy()

            if args.runroot_filter:
                sub = sub[sub["run_root"].astype(str).str.contains(args.runroot_filter, na=False)]
            if args.experiment_filter:
                if args.experiment_filter not in str(exp):
                    sub = sub.iloc[0:0]
                else:
                    sub = sub[sub["experiment"].astype(str).str.contains(args.experiment_filter, na=False)]

            dfs_exp.append(sub)

        ratios, finish_data = build_samples_per_ratio(dfs_exp, args.xcol, "finished_rate")
        _, return_data = build_samples_per_ratio(dfs_exp, args.xcol, "avg_return")

        if len(ratios) == 0:
            continue

        exp_dir = out_dir / str(exp)

        plot_grouped_box_paper(
            ratios=ratios,
            data_per_ratio=finish_data,
            labels=labels,
            colors=colors,
            title=f"{exp}: Finished rate",
            ylabel="finished_rate",
            out_path=exp_dir / "box_finished_rate.png",
            show=show,
        )

        plot_grouped_box_paper(
            ratios=ratios,
            data_per_ratio=return_data,
            labels=labels,
            colors=colors,
            title=f"{exp}: Average return",
            ylabel="avg_return",
            out_path=exp_dir / "box_avg_return.png",
            show=show,
        )

    print(f"Done. Plots in: {out_dir}")


if __name__ == "__main__":
    main()