#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rollouts(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def ensure_plot_dir(rollout_path: Path) -> Path:
    plot_dir = rollout_path.parent / rollout_path.stem
    plot_dir.mkdir(exist_ok=True)
    return plot_dir


def to_time_series(sequence):
    arr = np.array(sequence)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.squeeze()


def plot_episode(episode, episode_idx: int, plot_dir: Path):
    predictions = np.array(episode["predictions"])
    timesteps = predictions.shape[0]

    prediction_series = predictions.reshape(timesteps, -1)
    safe_prediction = to_time_series(episode["safe_prediction"])
    done_flags = to_time_series(episode["dones"]).astype(int)

    fig, ax_pred = plt.subplots(figsize=(10, 5))
    x = np.arange(timesteps)

    for col in range(prediction_series.shape[1]):
        ax_pred.plot(x, prediction_series[:, col], label=f"prediction_{col}", linewidth=1)

    ax_pred.plot(x, safe_prediction, label="safe_prediction", color="black", linewidth=2)
    ax_pred.set_xlabel("Timestep")
    ax_pred.set_ylabel("Prediction value")
    ax_pred.set_title(f"Episode {episode_idx + 1}")
    ax_pred.grid(True, linestyle="--", alpha=0.3)

    ax_done = ax_pred.twinx()
    ax_done.step(x, done_flags, where="post", color="red", label="done", linewidth=1.5)
    ax_done.set_ylabel("Done flag")
    ax_done.set_ylim(-0.1, 1.1)

    lines, labels = ax_pred.get_legend_handles_labels()
    lines2, labels2 = ax_done.get_legend_handles_labels()
    ax_pred.legend(lines + lines2, labels + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(plot_dir / f"episode_{episode_idx + 1:03d}.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot predictions, safe predictions, and done flags per episode.")
    parser.add_argument("rollout_file", type=Path, help="Path to the saved rollout .pkl file")
    args = parser.parse_args()

    rollouts = load_rollouts(args.rollout_file)
    plot_dir = ensure_plot_dir(args.rollout_file)

    for idx, episode in enumerate(rollouts):
        plot_episode(episode, idx, plot_dir)

    print(f"Saved {len(rollouts)} episode plots to {plot_dir}")


if __name__ == "__main__":
    main()
