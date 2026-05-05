import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    # return x
    x = np.array(x)
    return 1 / (1 + np.exp(-x))

JOINT_LIMITS = np.array(
    [
        [-1.0472, 1.0472],
        [-1.5708, 3.4907],
        [-2.7227, -0.83776],
        [-1.0472, 1.0472],
        [-1.5708, 3.4907],
        [-2.7227, -0.83776],
        [-1.0472, 1.0472],
        [-0.5236, 4.5379],
        [-2.7227, -0.83776],
        [-1.0472, 1.0472],
        [-0.5236, 4.5379],
        [-2.7227, -0.83776],
    ],
    dtype=float,
)

nominal_positions = np.array([-0.1,  0.8, -1.5,  0.1,  0.8, -1.5, -0.1,  0.8, -1.5,  0.1,  0.8,
       -1.5])

JOINT_LIMITS = (JOINT_LIMITS - nominal_positions[:, None]) / 0.3

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


def as_2d_time_series(sequence):
    arr = np.asarray(sequence, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr.reshape(arr.shape[0], -1)


def get_ball_plate_prediction_names(episode, dim):
    metadata = episode.get("prediction_metadata", {})
    names = metadata.get("ball_plate_prediction_names", [])
    names = [name.replace("ball_plate/", "") for name in names]
    if len(names) == dim:
        return names

    fallback_names = [
        "plate_support_relative_position_x",
        "plate_support_relative_position_y",
        "plate_support_relative_position_z",
        "plate_velocity_x",
        "plate_velocity_y",
        "plate_velocity_z",
        "ball_relative_position_x",
        "ball_relative_position_y",
        "ball_relative_position_z",
        "ball_relative_velocity_x",
        "ball_relative_velocity_y",
        "ball_relative_velocity_z",
    ]
    fallback_names.extend(f"ball_plate_{idx}" for idx in range(len(fallback_names), dim))
    return fallback_names[:dim]


def plot_ball_plate_next_step_prediction(episode, episode_idx: int, plot_dir: Path):
    if len(episode.get("ball_plate_next_step_pred", [])) == 0:
        return

    pred_key = "ball_plate_next_step_pred_physical"
    true_key = "ball_plate_next_step_true_physical"
    if pred_key not in episode or true_key not in episode or len(episode[pred_key]) == 0:
        pred_key = "ball_plate_next_step_pred"
        true_key = "ball_plate_next_step_true"

    predictions = as_2d_time_series(episode[pred_key])
    targets = as_2d_time_series(episode[true_key])
    timesteps = min(predictions.shape[0], targets.shape[0])
    predictions = predictions[:timesteps]
    targets = targets[:timesteps]
    if timesteps == 0:
        return

    names = get_ball_plate_prediction_names(episode, predictions.shape[1])
    x = np.arange(timesteps)

    ball_plate_dir = plot_dir / "ball_plate_next_step"
    ball_plate_dir.mkdir(parents=True, exist_ok=True)

    cols = 3
    rows = int(np.ceil(predictions.shape[1] / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.8 * rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for dim_idx in range(predictions.shape[1]):
        ax = axes[dim_idx]
        ax.plot(x, targets[:, dim_idx], label="true", linewidth=1.2)
        ax.plot(x, predictions[:, dim_idx], label="pred", linewidth=1.0)
        ax.set_title(names[dim_idx])
        ax.grid(True, linestyle="--", alpha=0.3)
        if dim_idx == 0:
            ax.legend(loc="upper left")

    for ax in axes[predictions.shape[1]:]:
        ax.axis("off")
    for ax in axes[-cols:]:
        ax.set_xlabel("Timestep")

    fig.suptitle(f"Episode {episode_idx + 1} ball-plate next-step prediction", y=0.995)
    fig.tight_layout()
    fig.savefig(ball_plate_dir / f"episode_{episode_idx + 1:03d}_ball_plate_next_step.png", dpi=200)
    plt.close(fig)

    if predictions.shape[1] >= 12:
        error = predictions - targets
        fig_error, ax_error = plt.subplots(figsize=(10, 4))
        ax_error.plot(x, np.linalg.norm(error[:, :3], axis=-1), label="plate support rel pos")
        ax_error.plot(x, np.linalg.norm(error[:, 3:6], axis=-1), label="plate velocity")
        ax_error.plot(x, np.linalg.norm(error[:, 6:9], axis=-1), label="ball rel pos")
        ax_error.plot(x, np.linalg.norm(error[:, 9:12], axis=-1), label="ball rel velocity")
        ax_error.set_xlabel("Timestep")
        ax_error.set_ylabel("L2 error")
        ax_error.grid(True, linestyle="--", alpha=0.3)
        ax_error.legend(loc="upper left")
        fig_error.tight_layout()
        fig_error.savefig(ball_plate_dir / f"episode_{episode_idx + 1:03d}_ball_plate_error.png", dpi=200)
        plt.close(fig_error)


def plot_episode(episode, episode_idx: int, plot_dir: Path):
    predictions = np.array(episode["predictions"])
    timesteps = predictions.shape[0]

    prediction_series = predictions.reshape(timesteps, -1)
    safe_prediction = to_time_series(episode["safe_prediction"])
    done_flags = to_time_series(episode["dones"]).astype(int)

    # Check if action fields exist
    has_actions = all(key in episode for key in ["delta_u", "raw_action", "safe_action"])
    has_joint_positions = "joint_position_obs" in episode

    has_h_u0 = "h_u0" in episode

    if has_actions:
        if not has_h_u0:
            fig, (ax_pred, ax_action, ax_constraint) = plt.subplots(3, 1, figsize=(10, 8))
        else:
            fig, (ax_pred, ax_action, ax_constraint, ax_h_u0) = plt.subplots(4, 1, figsize=(10, 8))
    else:
        fig, ax_pred = plt.subplots(figsize=(10, 5))

    x = np.arange(timesteps)

    # Plot predictions
    for col in range(prediction_series.shape[1]):
        ax_pred.plot(x, sigmoid(prediction_series[:, col]), label=f"prediction_{col}", linewidth=1)

    ax_pred.plot(x, safe_prediction, label="safe_prediction", color="black", linewidth=2)
    # ax_pred.set_ylim(0, 1)

    ax_pred.set_xlabel("Timestep")
    ax_pred.set_ylabel("Prediction value")
    ax_pred.set_title(f"Episode {episode_idx + 1}")
    ax_pred.grid(True, linestyle="--", alpha=0.3)

    ax_done = ax_pred.twinx()
    ax_done.step(x, np.std(prediction_series, axis=-1), where="post", color="red", label="std", linewidth=1)
    ax_done.set_ylabel("Done flag")
    # ax_done.set_ylim(0.0, 1.0)

    lines, labels = ax_pred.get_legend_handles_labels()
    lines2, labels2 = ax_done.get_legend_handles_labels()
    ax_pred.legend(lines + lines2, labels + labels2, loc="upper left")

    # Plot action norms if available
    if has_actions:
        delta_u = np.array(episode["delta_u"])
        raw_action = np.array(episode["raw_action"])
        safe_action = np.array(episode["safe_action"])
        constraint_active = np.array(episode["constraint_active"])

        # Ensure we have shape [timesteps, action_dim]
        if delta_u.ndim == 3:
            delta_u = delta_u.squeeze()
        if raw_action.ndim == 3:
            raw_action = raw_action.squeeze()
        if safe_action.ndim == 3:
            safe_action = safe_action.squeeze()
        if constraint_active.ndim == 3:
            constraint_active = constraint_active.squeeze()

        delta_u_norm = np.linalg.norm(delta_u, axis=-1)
        raw_action_norm = np.linalg.norm(raw_action, axis=-1)
        safe_action_norm = np.linalg.norm(safe_action, axis=-1)

        ax_action.plot(x, delta_u_norm, label="||delta_u||", linewidth=1.5)
        ax_action.plot(x, raw_action_norm, label="||raw_action||", linewidth=1.5)
        ax_action.plot(x, safe_action_norm, label="||safe_action||", linewidth=1.5)

        ax_action.set_xlabel("Timestep")
        ax_action.set_ylabel("Action norm")
        ax_action.grid(True, linestyle="--", alpha=0.3)
        ax_action.legend(loc="upper left")

        ax_constraint.plot(x, constraint_active, label="constraint_active", color="purple", linewidth=1.5)
        ax_constraint.set_xlabel("Timestep")
        ax_constraint.set_ylabel("Constraint Active")
        ax_constraint.grid(True, linestyle="--", alpha=0.3)
        ax_constraint.legend(loc="upper left")

        if has_h_u0:
            ax_h_u0.plot(x, sigmoid(episode["h_u0"]), label="h_u0", color="orange", linewidth=1.5)
            ax_h_u0.plot(x, np.roll(safe_prediction, -1), label="safe_prediction", linewidth=1.5)
            ax_h_u0.set_xlabel("Timestep")
            ax_h_u0.set_ylabel("h_u0")
            ax_h_u0.grid(True, linestyle="--", alpha=0.3)
            ax_h_u0.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(plot_dir / f"episode_{episode_idx + 1:03d}.png", dpi=200)
    plt.close(fig)

    action_plot_dir = plot_dir / "actions"
    action_plot_dir.mkdir(parents=True, exist_ok=True)

    action_dim = raw_action.shape[-1]
    rows, cols = 3, 4
    fig_actions, axes = plt.subplots(rows, cols, figsize=(12, 8), sharex=True)
    axes = axes.ravel()

    lower_limits = JOINT_LIMITS[:, 0]
    upper_limits = JOINT_LIMITS[:, 1]

    joint_obs = np.array(episode["joint_position_obs"])

    for dim_idx in range(min(action_dim, rows * cols)):
        ax_dim = axes[dim_idx]
        ax_dim.plot(x, raw_action[:, dim_idx], label="raw_action", linewidth=0.5)
        ax_dim.plot(x, safe_action[:, dim_idx], label="safe_action", linewidth=0.5)
        ax_dim.plot(x, safe_action[:, dim_idx] - raw_action[:, dim_idx], label="action_correction", linewidth=1.2)
        # ax_dim.plot(x, joint_obs[:, dim_idx], label="joint_position_obs", linewidth=1.2)
        if lower_limits is not None:
            ax_dim.axhline(lower_limits[dim_idx], color="tab:red", linestyle="--", linewidth=0.8, label="lower_limit")
        if upper_limits is not None:
            ax_dim.axhline(upper_limits[dim_idx], color="tab:green", linestyle="--", linewidth=0.8, label="upper_limit")
        ax_dim.set_ylabel(f"Dim {dim_idx}")
        ax_dim.grid(True, linestyle="--", alpha=0.3)
        if dim_idx == 0:
            ax_dim.legend(loc="upper left")

    for ax in axes[action_dim:]:
        ax.axis("off")
    for ax in axes[-cols:]:
        ax.set_xlabel("Timestep")

    fig_actions.suptitle(f"Episode {episode_idx + 1} actions", y=0.98)
    fig_actions.tight_layout()
    fig_actions.savefig(action_plot_dir / f"episode_{episode_idx + 1:03d}_actions.png", dpi=200)
    plt.close(fig_actions)

    if has_joint_positions:
        joint_obs = np.array(episode["joint_position_obs"])
        # print("min of joints positions",  joint_obs.min(axis=0), "max of joints positions",  joint_obs.max(axis=0))

        joint_min = joint_obs.min(axis=0)
        joint_max = joint_obs.max(axis=0)
        # print("min of joints positions", joint_min, "max of joints positions", joint_max)

        lower_bounds = JOINT_LIMITS[:, 0]
        upper_bounds = JOINT_LIMITS[:, 1]
        violations = np.logical_or(joint_obs < lower_bounds, joint_obs > upper_bounds)

        #
        # if np.any(violations):
        #     violating_indices = np.argwhere(violations)
        #     extended_lower = np.minimum(lower_bounds, joint_min)
        #     extended_upper = np.maximum(upper_bounds, joint_max)
        #     extended_limits = np.column_stack((extended_lower, extended_upper))
        #
        #     print(
        #         "joint_position_obs out of bounds at indices",
        #         violating_indices.tolist(),
        #     )
        #     print("original joint limits:", JOINT_LIMITS.tolist())
        #     print("observed joint range:", np.column_stack((joint_min, joint_max)).tolist())
        #     print("extended joint limits:", extended_limits.tolist())
        # else:
        #     print("joint_position_obs within specified limits")

    plot_ball_plate_next_step_prediction(episode, episode_idx, plot_dir)


def plot_one_exp(rollout_file):
    # print summary of episode
    rollouts = load_rollouts(rollout_file)
    plot_dir = ensure_plot_dir(rollout_file)

    episode_lengths = []
    episode_rets = []
    for idx, episode in enumerate(rollouts):
        plot_episode(episode, idx, plot_dir)
        episode_lengths.append(np.array(episode["predictions"]).shape[0])

        if "ret" in episode:
            episode_rets.append(episode["ret"])

    less_than_1000 = sum(length < 250 for length in episode_lengths)
    avg_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    avg_ret = float(np.mean(episode_rets)) if episode_rets else 0.0

    print(f"Episodes < 250 steps: {less_than_1000}")
    print(f"Average episode length: {avg_length:.2f}")
    print(f"Average episode return: {avg_ret:.2f}")

    print(f"Saved {len(rollouts)} episode plots to {plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot rollout predictions from pickle files.")
    parser.add_argument("rollout_files", nargs="+", help="Paths to rollout pickle files.")
    args = parser.parse_args()

    rollout_files = args.rollout_files

    for rollout_file in rollout_files:
        print(f"Processing rollout file: {rollout_file}")
        plot_one_exp(Path(rollout_file))
