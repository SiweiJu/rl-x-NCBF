#!/usr/bin/env python3
"""Paper Fig. 2 layout from eval_sweep summaries: 2 x 4 subplots (IEEE double column).

Row 1: success rate vs sampling probability, row 2: average return, columns = the 4 test settings
(action noise / push x flat / heightfield). One line per method (and deployment mode), mean over training seeds
(checkpoints) with a +-1 standard error band (the paper used std / sqrt(5) for 5 seeds, i.e. the same thing).

Usage:
  python misc/rss/plot_fig2_from_eval_summaries.py <sweep_root> [<root> ...] \
      --methods ppo ncbf_h1 ours_h25 --labels PPO "H = 1" "H = 25" --mode A --out fig2.pdf --no_show
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_summary_reader import PAPER_EXPERIMENT_ORDER, aggregate_over_checkpoints, read_eval_summaries  # noqa: E402
from plot_line_plot_one_figure import TABLEAU10, apply_ieee_style, exp_name_table, map_legend_label  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--methods", nargs="*", default=None, help="method tags to plot (default: all, sorted)")
    ap.add_argument("--labels", nargs="*", default=None, help="legend labels for --methods")
    ap.add_argument("--mode", nargs="*", default=["A"], help="deployment mode(s) to plot: A, B or both")
    ap.add_argument("--experiments", nargs="*", default=PAPER_EXPERIMENT_ORDER)
    ap.add_argument("--out", default="fig2.pdf")
    ap.add_argument("--fig_width_in", type=float, default=7.16)
    ap.add_argument("--fig_height_in", type=float, default=2.2)
    ap.add_argument("--no_show", action="store_true")
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_ieee_style()
    df = read_eval_summaries(args.roots)
    df = df[df["deployment_mode"].isin(args.mode)]
    methods = args.methods or sorted(df["method"].unique())
    labels = args.labels if args.labels and len(args.labels) == len(methods) else [map_legend_label(m) for m in methods]
    success = aggregate_over_checkpoints(df, "success_rate")
    returns = aggregate_over_checkpoints(df, "mean_episode_return")

    series = [(m, mode) for m in methods for mode in args.mode]
    fig, axes = plt.subplots(2, len(args.experiments), figsize=(args.fig_width_in, args.fig_height_in), sharex=False, squeeze=False)
    for j, exp in enumerate(args.experiments):
        for row, (table, ylabel) in enumerate(((success, "Success rate"), (returns, "Avg. return"))):
            ax = axes[row, j]
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            sub = table[table["experiment"] == exp]
            x_all = np.array(sorted(sub["sampling_prob"].astype(float).unique()))
            for k, (method, mode) in enumerate(series):
                s = sub[(sub["method"] == method) & (sub["deployment_mode"] == mode)].sort_values("sampling_prob")
                if s.empty:
                    continue
                label = labels[methods.index(method)] + (f" ({mode})" if len(args.mode) > 1 else "")
                color = TABLEAU10[k % len(TABLEAU10)]
                ax.plot(s["sampling_prob"], s["mean"], marker="o", color=color, label=label,
                        linestyle="-" if mode == "A" else "--")
                ax.fill_between(s["sampling_prob"], s["mean"] - s["se"], s["mean"] + s["se"], color=color, alpha=0.18, linewidth=0)
            if row == 0:
                ax.set_title(exp_name_table.get(exp, exp), pad=2.0)
                ax.set_ylim(-0.02, 1.02)
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("sampling probability")
                if x_all.size:
                    ax.set_xticks(x_all)
                    ax.set_xticklabels([f"{x:g}" for x in x_all], rotation=45)
            if j == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(axis="y", labelleft=False)
            if sub.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        if len(h) > len(handles):
            handles, leg_labels = h, l
    if handles:
        fig.legend(handles, leg_labels, loc="upper center", ncol=min(len(handles), 5), frameon=False, bbox_to_anchor=(0.5, 1.02))
    plt.subplots_adjust(left=0.06, right=0.995, top=0.85, bottom=0.25, wspace=0.1, hspace=0.08)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Wrote: {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
