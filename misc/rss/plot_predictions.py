#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    x = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-x))


def apply_ieee_style():
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
        "lines.linewidth": 1.2,
        "lines.markersize": 2.8,
        "grid.linewidth": 0.4,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    })


# Professional, paper-friendly palette + styles
COL = {
    "pred": "#4E79A7",        # blue
    "safe": "#000000",        # black
    "std":  "#E15759",        # red
    "raw":  "#F28E2B",        # orange
    "delta":"#59A14F",        # green
    "safe_a":"#4E79A7",       # blue
    "active":"#B07AA1",       # purple
}
LS = {
    "pred": "-",
    "safe": "-",
    "std":  "--",
    "raw":  "--",
    "delta":"-",
    "safe_a":"-",
    "active":"-",
}


def load_rollouts(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def ensure_plot_dir(rollout_path: Path) -> Path:
    plot_dir = rollout_path.parent / rollout_path.stem
    plot_dir.mkdir(exist_ok=True)
    return plot_dir


def to_time_series(sequence):
    arr = np.asarray(sequence)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.squeeze()


def plot_episode_first_three(
    episode,
    episode_idx: int,
    plot_dir: Path,
    fig_width_in: float = 3.5,   # ~IEEE single-column width (inches)
    out_ext: str = "pdf",        # "pdf" or "png"
):
    apply_ieee_style()

    predictions = np.asarray(episode["predictions"])
    T = predictions.shape[0]
    x = np.arange(T)

    pred_series = predictions.reshape(T, -1)
    safe_pred = to_time_series(episode["safe_prediction"])

    # figure layout: 3 stacked panels, compact IEEE-like
    fig_h = 2.35  # tuned for 3 panels at column width
    fig, (ax_pred, ax_act) = plt.subplots(
        2, 1, figsize=(fig_width_in, fig_h), sharex=True
    )

    _plot_pred_panel(ax_pred, x, pred_series, safe_pred, predictions)
    _plot_action_norm_panel(ax_act, x, episode)
    # _plot_constraint_panel(ax_con, x, episode)

    ax_act.set_xlabel("timestep")

    # compact spacing
    fig.tight_layout(pad=0.25, h_pad=0.15)
    out = plot_dir / f"episode_{episode_idx + 1:03d}_ieee.{out_ext}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _plot_pred_panel(ax_pred, x, pred_series, safe_pred, predictions_raw):
    # predictions (sigmoid)
    for col in range(pred_series.shape[1]):
        ax_pred.plot(
            x, sigmoid(pred_series[:, col]),
            color=COL["pred"], linestyle=LS["pred"], linewidth=0.9, alpha=0.35,
            label="single prediction" if col == 0 else None,

        )

    ax_pred.plot(
        x, safe_pred,
        color=COL["safe"], linestyle=LS["safe"], linewidth=1.4,
        label="CVaR safety prediction"
    )

    ax_pred.set_ylabel("safety prediction")
    ax_pred.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax_pred.grid(False, axis="x")

    # std on second y-axis (line plot)
    ax_std = ax_pred.twinx()
    std_series = np.std(sigmoid(predictions_raw.reshape(predictions_raw.shape[0], -1)), axis=-1)
    ax_std.plot(
        x, std_series,
        color=COL["std"], linestyle=LS["std"], linewidth=1.1,
        label="standard deviation"
    )
    ax_std.set_ylabel("std $\sigma_t$", color=COL["std"])
    ax_std.tick_params(axis="y", colors=COL["std"])

    # legend (combine both axes)
    h1, l1 = ax_pred.get_legend_handles_labels()
    h2, l2 = ax_std.get_legend_handles_labels()
    if h1 or h2:
        ax_pred.legend(
            h1 + h2, l1 + l2,
            bbox_to_anchor=(1.12, 1.0),
            loc="upper right",
            frameon=False,
            handlelength=1.6,
            borderpad=0.2,
            labelspacing=0.2,
        )


def _plot_action_norm_panel(ax, x, episode):
    delta_u = np.asarray(episode["delta_u"]).squeeze()
    raw_a = np.asarray(episode["raw_action"]).squeeze()
    safe_a = np.asarray(episode["safe_action"]).squeeze()

    # norms
    dn = delta_u
    rn = np.linalg.norm(raw_a, axis=-1)
    sn = rn - dn # np.linalg.norm(safe_a, axis=-1)

    ax.plot(x, rn, color=COL["raw"], linestyle=LS["raw"], linewidth=1.2,
            label=r"$\|u_t^{0}\|$")
    ax.plot(x, sn, color=COL["safe_a"], linestyle=LS["safe_a"], linewidth=0.5,
            label=r"$\|\tilde{u}_t\|$")
    ax.plot(x, dn, color=COL["delta"], linestyle=LS["delta"], linewidth=1.1,
            label=r"$\|\tilde{u}_t - u_t^{0}\|$")
    ax.set_ylabel("action")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.grid(False, axis="x")
    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.2,
        ncol=1,
    )


def _plot_constraint_panel(ax, x, episode):
    active = np.asarray(episode["constraint_active"]).squeeze().astype(float)
    active = active > 0
    ax.plot(
        x, active,
        color=COL["active"], linestyle=LS["active"], linewidth=1.2,
        label="constraint active"
    )
    ax.set_ylabel("safety shield active")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.grid(False, axis="x")
    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.2,
    )


def plot_one_exp(rollout_file: Path, fig_width_in: float, out_ext: str, max_episodes: int | None):
    rollouts = load_rollouts(rollout_file)
    plot_dir = ensure_plot_dir(rollout_file)

    n = len(rollouts) if max_episodes is None else min(len(rollouts), max_episodes)
    for idx in range(n):
        plot_episode_first_three(
            rollouts[idx],
            idx,
            plot_dir,
            fig_width_in=fig_width_in,
            out_ext=out_ext,
        )

    print(f"Saved {n} IEEE-style episode plots to {plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot rollout diagnostics (IEEE column width, first three panels only).")
    parser.add_argument("rollout_files", nargs="+", help="Paths to rollout pickle files.")
    parser.add_argument("--fig_width_in", type=float, default=3.5,
                        help="Figure width in inches (IEEE single-column ~3.5in).")
    parser.add_argument("--out_ext", type=str, default="pdf", choices=["pdf", "png"],
                        help="Output format.")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Optional cap on number of episodes per rollout file.")
    args = parser.parse_args()

    for rf in args.rollout_files:
        rf = Path(rf)
        print(f"Processing: {rf}")
        plot_one_exp(rf, fig_width_in=args.fig_width_in, out_ext=args.out_ext, max_episodes=args.max_episodes)