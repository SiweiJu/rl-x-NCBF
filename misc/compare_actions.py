#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_rollouts(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)  # expected: list[episode_dict]


def ensure_2d_actions(arr: np.ndarray) -> np.ndarray:
    """
    Convert action array to shape [T, A].
    Handles cases like [T, 1, A], [T, A], [T, A, 1] (best-effort).
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        # [A] -> [1, A]
        return arr[None, :]
    if arr.ndim == 2:
        # [T, A]
        return arr
    # common logging shapes: [T, 1, A] or [T, A, 1]
    if arr.ndim == 3:
        if arr.shape[1] == 1:
            return arr[:, 0, :]
        if arr.shape[2] == 1:
            return arr[:, :, 0]
    # fallback: squeeze then re-check
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Cannot coerce actions to [T,A]. Got shape {np.asarray(arr).shape}")


def get_action_dim(episodes, action_key: str) -> int:
    # Find first episode with that key and infer action_dim
    for ep in episodes:
        if action_key in ep:
            acts = ensure_2d_actions(ep[action_key])
            return acts.shape[-1]
    raise KeyError(f"No episode contains action key '{action_key}'")


def plot_episode_actions_across_files(
    all_rollouts,  # list of (label, episodes)
    ep_idx: int,
    out_dir: Path,
    plot_raw: bool,
    plot_safe: bool,
    raw_key: str = "raw_action",
    safe_key: str = "safe_action",
):
    # Determine action_dim from first available
    action_dim = None
    for _, eps in all_rollouts:
        if len(eps) > ep_idx:
            if raw_key in eps[ep_idx]:
                action_dim = ensure_2d_actions(eps[ep_idx][raw_key]).shape[-1]
                break
            if safe_key in eps[ep_idx]:
                action_dim = ensure_2d_actions(eps[ep_idx][safe_key]).shape[-1]
                break
    if action_dim is None:
        print(f"[WARN] episode {ep_idx}: no actions found in any file, skipping.")
        return

    ep_dir = out_dir / f"episode_{ep_idx+1:03d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    action_dir = out_dir / "actions"
    action_dir.mkdir(parents=True, exist_ok=True)


    for dim in range(action_dim):
        fig, ax = plt.subplots(figsize=(10, 4))
        any_plotted = False

        for label, episodes in all_rollouts:
            if ep_idx >= len(episodes):
                continue
            ep = episodes[ep_idx]

            # Plot raw
            if plot_raw and raw_key in ep:
                raw = ensure_2d_actions(ep[raw_key])  # [T,A]
                T = raw.shape[0]
                x = np.arange(T)
                ax.plot(x, raw[:, dim], linewidth=1.0, label=f"{label}::{raw_key}")

            # Plot safe
            if plot_safe and safe_key in ep:
                safe = ensure_2d_actions(ep[safe_key])  # [T,A]
                T = safe.shape[0]
                x = np.arange(T)
                ax.plot(x, safe[:, dim], linewidth=1.0, linestyle="--", label=f"{label}::{safe_key}")

            # plot safe - raw
            raw = ensure_2d_actions(ep[raw_key])  # [T,A]
            safe = ensure_2d_actions(ep[safe_key])  # [T,A]
            T = raw.shape[0]
            x = np.arange(T)
            ax.plot(x, safe[:, dim] - raw[:, dim], linewidth=1.0,  label=f"{label}::safe-raw")

        ax.set_title(f"Episode {ep_idx+1} — action dim {dim}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Action value")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        fig.savefig(ep_dir / "actions" / f"action_dim_{dim:02d}.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        norm_plotted = False

        for label, episodes in all_rollouts:
            if ep_idx >= len(episodes):
                continue
            ep = episodes[ep_idx]

            if raw_key not in ep or safe_key not in ep:
                continue

            raw = ensure_2d_actions(ep[raw_key])
            safe = ensure_2d_actions(ep[safe_key])
            T = min(raw.shape[0], safe.shape[0])
            diff_norm = np.linalg.norm(safe[:T] - raw[:T], axis=1)
            x = np.arange(T)
            ax.plot(x, diff_norm, linewidth=1.2, label=f"{label}::||safe-raw||")
            norm_plotted = True

        if norm_plotted:
            ax.set_title(f"Episode {ep_idx + 1} — ||safe - raw||")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Norm value")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(ep_dir / "safe_raw_norm.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare per-dimension actions across multiple rollout pickle files, episode by episode."
    )
    parser.add_argument("pkl_files", nargs="+", help="Paths to rollout .pkl files (each contains a list of episodes).")
    parser.add_argument("--out", type=str, default="compare_actions", help="Output directory for plots.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episode indices to plot (default: min across files).")
    parser.add_argument("--plot_raw", action="store_true", help="Plot raw_action curves.")
    parser.add_argument("--plot_safe", action="store_true", help="Plot safe_action curves.")
    parser.add_argument("--raw_key", type=str, default="raw_action", help="Key for raw actions in episode dict.")
    parser.add_argument("--safe_key", type=str, default="safe_action", help="Key for safe actions in episode dict.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rollouts = []
    for p in args.pkl_files:
        path = Path(p)
        episodes = load_rollouts(path)
        label = path.stem
        all_rollouts.append((label, episodes))
        print(f"Loaded {len(episodes)} episodes from {path}")

    # default episodes: min length across files
    if args.episodes is None:
        max_eps = min(len(eps) for _, eps in all_rollouts)
    else:
        max_eps = args.episodes

    for ep_idx in range(max_eps):
        plot_episode_actions_across_files(
            all_rollouts=all_rollouts,
            ep_idx=ep_idx,
            out_dir=out_dir,
            plot_raw=args.plot_raw,
            plot_safe=args.plot_safe,
            raw_key=args.raw_key,
            safe_key=args.safe_key,
        )

    print(f"Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()