"""Offline check: can an H-step collision critic infer obstacle velocity from a history of range scans?

The robot does not move (zero action; it settles on the ground) while obstacles move past it in the MJX environment. We record
range scans, the privileged obstacle state and the collision indicator, build the receding-horizon
labels y_t = 1 if there is no collision in steps t+1..t+H, and train small MLP critics by supervised
learning on different inputs:

    scan_k      the last k scans, flattened into an MLP (k = 1 is a single frame, which has no
                velocity information); this is what the FFNN history encoder does
    scanconv_k  the last k scans through a circular convolution over the rays (frames as channels)
                and pooling over rays; the pooling makes it rotation invariant, which fits a static
                round robot but drops the direction relative to the heading
    state_pos   privileged relative position and radius of each obstacle (no velocity)
    state_full  privileged relative position, velocity and radius (oracle input)

Two scores need no training: the nearest scan return (what a one-step distance barrier uses) and a
constant-velocity oracle from the true obstacle state. We report AUROC / average precision / log
loss on held-out environments, a linear probe of obstacle velocity from the critic's last hidden
layer, and a counterfactual where the current scan is fixed and only the past differs. Every critic
is trained with early stopping on validation environments.

This isolates the perception question with a static robot (no ego-motion). It is not the closed-loop method: the
policy, the shield and ego-motion are absent.

Run from the repository root (CPU is fine, ~30-40 min):
    JAX_PLATFORMS=cpu PYTHONPATH=. python experiments/ncbf_tools/obstacle_scan_critic_probe.py --out_dir experiments/runs/scan_probe
"""
import argparse
import json
import os
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ml_collections import config_dict
from scipy.stats import rankdata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default="experiments/runs/scan_probe")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nr_envs", type=int, default=128)
    parser.add_argument("--nr_steps", type=int, default=2500, help="control steps per env (50 Hz)")
    parser.add_argument("--episode_seconds", type=int, default=20)
    parser.add_argument("--nr_obstacles", type=int, default=2)
    parser.add_argument("--aim_offset", type=float, default=1.0, help="m; larger than robot + obstacle radius so some obstacles miss")
    parser.add_argument("--speed_range", type=float, nargs=2, default=(0.5, 2.5))
    parser.add_argument("--nr_rays", type=int, default=64)
    parser.add_argument("--max_range", type=float, default=5.0)
    parser.add_argument("--scan_noise", type=float, default=0.02)
    parser.add_argument("--history", type=int, nargs="+", default=(1, 2, 5, 10, 25), help="history lengths for the MLP critics")
    parser.add_argument("--conv_history", type=int, nargs="+", default=(1, 10, 25), help="history lengths for the convolutional critics")
    parser.add_argument("--horizons", type=int, nargs="+", default=(5, 25, 50))
    parser.add_argument("--train_steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--test_fraction", type=float, default=0.2, help="fraction of envs held out for evaluation")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="fraction of envs held out for early stopping")
    parser.add_argument("--eval_every", type=int, default=250, help="training steps between validation checks")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def make_env(args):
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.default_config import get_config
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.create_env import create_env

    env_config = get_config("ncbf_mujoco.robot_locomotion.mjx")
    env_config.nr_envs = args.nr_envs
    env_config.device = "cpu"
    env_config.train_robot = "unitree_go2"
    env_config.terrain.type = "plane"
    env_config.episode_length_in_seconds = args.episode_seconds
    # Without a trained policy the zero action does not hold the robot up. The scan only depends on the
    # base position, so let the robot fall and lie on the ground instead of terminating on height or tilt.
    env_config.termination.height_percentage_threshold = -100.0  # the IMU can end up below the ground reference when the robot rolls over
    env_config.termination.body_tilt_threshold = 100.0
    obstacles = env_config.obstacles
    obstacles.enabled = True
    obstacles.nr_obstacles = args.nr_obstacles
    obstacles.policy_observation = "scan"
    obstacles.ncbf_observation = "policy"
    obstacles.value_observation = "state_and_scan"  # puts the privileged block into the observation as well
    obstacles.terminate_on_collision = False  # obstacles pass through, so episodes do not end at a collision
    obstacles.aim_offset = args.aim_offset
    obstacles.speed_range = tuple(args.speed_range)
    obstacles.scan.nr_rays = args.nr_rays
    obstacles.scan.max_range = args.max_range
    obstacles.scan.noise = args.scan_noise
    config = config_dict.ConfigDict({"environment": env_config, "runner": {"mode": "train"}})
    return create_env(config)


def collect(env, args):
    keys = jax.random.split(jax.random.PRNGKey(args.seed), env.nr_envs)
    state = env.reset(keys, True)  # eval mode: full speed range and full sensor noise from the start
    action = jnp.zeros((env.nr_envs,) + env.single_action_space.shape)
    scan_idx, state_idx = env.obstacle_scan_obs_idx, env.obstacle_obs_idx

    def body(state, _):
        state = env.step(state, action)
        record = {
            "scan": state.next_observation[:, scan_idx],
            "obstacle_state": state.next_observation[:, state_idx],
            "collision": state.internal_state["obstacle_collision"],
            "done": jnp.logical_or(state.terminated, state.truncated),
        }
        return state, record

    rollout = jax.jit(lambda s: jax.lax.scan(body, s, None, length=args.nr_steps))
    _, records = rollout(state)
    return {name: np.asarray(value) for name, value in records.items()}  # [T, E, ...]


def build_samples(data, history_max, horizon_max, test_fraction, val_fraction):
    """Indices (t, e) usable by every input and horizon, split by environment."""
    collision = data["collision"].astype(bool)
    episode = np.cumsum(data["done"].astype(np.int32), axis=0)  # obs at t belongs to episode[t]
    nr_steps, nr_envs = collision.shape
    t = np.arange(history_max - 1, nr_steps - horizon_max)
    same_episode = episode[t - history_max + 1] == episode[t + horizon_max]  # [len(t), E]
    valid = same_episode & ~collision[t]  # decide only in states that are not colliding yet
    t_idx, e_idx = np.nonzero(valid)
    t_idx = t[t_idx]
    nr_test_envs = max(1, int(round(test_fraction * nr_envs)))
    nr_val_envs = max(1, int(round(val_fraction * nr_envs)))
    split = np.where(e_idx >= nr_envs - nr_test_envs, 2, np.where(e_idx >= nr_envs - nr_test_envs - nr_val_envs, 1, 0))
    return t_idx, e_idx, split  # 0 train, 1 validation, 2 test


def labels_for(data, t_idx, e_idx, horizon):
    collision = data["collision"].astype(bool)
    future = np.stack([collision[t_idx + k, e_idx] for k in range(1, horizon + 1)], axis=-1)
    return (~future.any(axis=-1)).astype(np.float32)  # 1 = safe for the next H steps


# ---------------------------------------------------------------------------
# Critics
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    hidden: int

    @nn.compact
    def __call__(self, x):
        x = nn.elu(nn.Dense(self.hidden)(x))
        features = nn.elu(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(features)[..., 0], features


class ScanConvCritic(nn.Module):
    hidden: int
    history: int
    nr_rays: int
    channels: int = 32

    @nn.compact
    def __call__(self, x):
        frames = x.reshape(x.shape[0], self.history, self.nr_rays)
        h = jnp.swapaxes(frames, 1, 2)  # [B, rays, frames]: rays are a circular axis, frames are channels
        h = nn.elu(nn.Conv(self.channels, (5,), padding="CIRCULAR")(h))
        h = nn.elu(nn.Conv(self.channels, (5,), padding="CIRCULAR")(h))
        h = jnp.concatenate([h.max(axis=1), h.mean(axis=1)], axis=-1)
        features = nn.elu(nn.Dense(self.hidden)(h))
        return nn.Dense(1)(features)[..., 0], features


def input_fn(kind, history, scans, obstacle_state, nr_obstacles):
    """Returns f(t_idx, e_idx) -> [B, D] built on device from the stored rollouts."""
    if kind in ("scan", "scanconv"):
        offsets = jnp.arange(-history + 1, 1)

        def f(t_idx, e_idx):
            frames = scans[t_idx[:, None] + offsets[None, :], e_idx[:, None]]  # [B, k, rays], oldest first
            return frames.reshape(frames.shape[0], -1)
        return f
    if kind == "state_pos":
        keep = jnp.asarray([5 * i + j for i in range(nr_obstacles) for j in (0, 1, 4)])
        return lambda t_idx, e_idx: obstacle_state[t_idx, e_idx][:, keep]
    if kind == "state_full":
        return lambda t_idx, e_idx: obstacle_state[t_idx, e_idx]
    raise ValueError(kind)


def train_critic(model, make_input, train, val, args, key):
    """Adam on minibatches; keeps the parameters with the lowest validation log loss."""
    (t_train, e_train, y_train), (t_val, e_val, y_val) = train, val
    dummy = make_input(jnp.asarray(t_train[:2]), jnp.asarray(e_train[:2]))
    params = model.init(key, dummy)
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(params)
    t_train, e_train, y_train = jnp.asarray(t_train), jnp.asarray(e_train), jnp.asarray(y_train)

    def loss_fn(params, t_b, e_b, y_b):
        logits, _ = model.apply(params, make_input(t_b, e_b))
        return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y_b))

    # A Python loop over a jitted step: on the XLA CPU backend, convolutions inside lax.scan run ~30x slower.
    @jax.jit
    def step(carry):
        params, opt_state, key = carry
        key, sample_key = jax.random.split(key)
        batch = jax.random.randint(sample_key, (args.batch_size,), 0, t_train.shape[0])
        loss, grads = jax.value_and_grad(loss_fn)(params, t_train[batch], e_train[batch], y_train[batch])
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state, key), loss

    def run_chunk(carry):
        losses = []
        for _ in range(args.eval_every):
            carry, loss = step(carry)
            losses.append(loss)
        return carry, jnp.stack(losses)
    val_subset = np.random.default_rng(0).permutation(len(t_val))[:30000]
    t_val, e_val, y_val = jnp.asarray(t_val[val_subset]), jnp.asarray(e_val[val_subset]), jnp.asarray(y_val[val_subset])
    val_loss_fn = jax.jit(loss_fn)

    carry = (params, opt_state, key)
    best = (np.inf, params, 0, np.nan)
    for chunk in range(max(1, args.train_steps // args.eval_every)):
        carry, losses = run_chunk(carry)
        val_loss = float(val_loss_fn(carry[0], t_val, e_val, y_val))
        if val_loss < best[0]:
            best = (val_loss, carry[0], (chunk + 1) * args.eval_every, float(np.mean(np.asarray(losses))))
    val_loss, params, best_step, train_loss = best
    return params, {"val_log_loss": val_loss, "best_step": best_step, "train_log_loss": train_loss}


def predict(model, params, make_input, t_idx, e_idx, chunk=16384):
    probabilities, features = [], []
    apply = jax.jit(lambda t, e: model.apply(params, make_input(t, e)))
    for start in range(0, len(t_idx), chunk):
        logits, feats = apply(jnp.asarray(t_idx[start:start + chunk]), jnp.asarray(e_idx[start:start + chunk]))
        probabilities.append(np.asarray(jax.nn.sigmoid(logits)))
        features.append(np.asarray(feats))
    return np.concatenate(probabilities), np.concatenate(features)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auroc(unsafe_score, is_unsafe):
    positives = is_unsafe.astype(bool)
    nr_pos, nr_neg = positives.sum(), (~positives).sum()
    if nr_pos == 0 or nr_neg == 0:
        return float("nan")
    ranks = rankdata(unsafe_score)
    return float((ranks[positives].sum() - nr_pos * (nr_pos + 1) / 2) / (nr_pos * nr_neg))


def average_precision(unsafe_score, is_unsafe):
    order = np.argsort(-unsafe_score, kind="stable")
    hits = is_unsafe[order].astype(np.float64)
    if hits.sum() == 0:
        return float("nan")
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(np.sum(precision * hits) / hits.sum())


def classification_metrics(p_safe, y_safe):
    p = np.clip(p_safe, 1e-6, 1 - 1e-6)
    is_unsafe = 1.0 - y_safe
    return {
        "auroc": auroc(1.0 - p, is_unsafe),
        "average_precision": average_precision(1.0 - p, is_unsafe),
        "log_loss": float(-np.mean(y_safe * np.log(p) + (1 - y_safe) * np.log(1 - p))),
        "brier": float(np.mean(np.square(p - y_safe))),
    }


def ridge_r2(features, targets, split, l2=1e-2):
    """Fit a linear map on features[:split], report R^2 per target column on features[split:]."""
    x = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1)
    x_fit, y_fit, x_eval, y_eval = x[:split], targets[:split], x[split:], targets[split:]
    weights = np.linalg.solve(x_fit.T @ x_fit + l2 * np.eye(x.shape[1]), x_fit.T @ y_fit)
    residual = y_eval - x_eval @ weights
    total = y_eval - y_eval.mean(axis=0)
    return (1.0 - np.sum(np.square(residual), axis=0) / np.sum(np.square(total), axis=0)).tolist()


def constant_velocity_min_clearance(obstacle_state, nr_obstacles, horizon_seconds, robot_radius=0.3, nr_samples=51):
    """Smallest clearance over [0, horizon] if every obstacle keeps its current relative velocity."""
    per_obstacle = obstacle_state.reshape(obstacle_state.shape[0], nr_obstacles, 5)
    pos, vel, radius = per_obstacle[..., 0:2] * 5.0, per_obstacle[..., 2:4] * 5.0, per_obstacle[..., 4] * 0.5
    times = np.linspace(0.0, horizon_seconds, nr_samples)
    future = pos[:, :, None, :] + vel[:, :, None, :] * times[None, None, :, None]
    clearance = np.linalg.norm(future, axis=-1) - radius[..., None] - robot_radius
    return clearance.min(axis=(1, 2))


# ---------------------------------------------------------------------------
# Counterfactual: same current scan, different past
# ---------------------------------------------------------------------------

def counterfactual_inputs(env, args, history, distance_now, speeds, radius=0.25):
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.obstacle_scan import obstacle_range_scan

    angles = env.obstacle_scan_angles
    far_obstacles = jnp.full((args.nr_obstacles - 1, 2), 50.0)
    inputs = []
    for speed in speeds:  # positive = approaching along the forward axis
        frames = []
        for lag in range(history - 1, -1, -1):  # oldest first
            x = distance_now + speed * lag * env.dt
            positions = jnp.concatenate([jnp.asarray([[x, 0.0]]), far_obstacles], axis=0)
            radii = jnp.full((args.nr_obstacles,), radius)
            ranges = obstacle_range_scan(jnp.zeros(2), 0.0, angles, positions, radii, args.max_range)
            frames.append(1.0 - ranges / args.max_range)
        inputs.append(jnp.concatenate(frames))
    return jnp.stack(inputs)


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()

    env = make_env(args)
    print(f"Collecting {args.nr_steps} steps x {args.nr_envs} envs ...", flush=True)
    data = collect(env, args)
    print(f"  done in {time.time() - started:.0f}s; collision rate per step {data['collision'].mean():.4f}, "
          f"episode ends {int(data['done'].sum())}", flush=True)

    history_max, horizon_max = max(list(args.history) + list(args.conv_history)), max(args.horizons)
    t_idx, e_idx, split = build_samples(data, history_max, horizon_max, args.test_fraction, args.val_fraction)
    t_train, e_train = t_idx[split == 0], e_idx[split == 0]
    t_val, e_val = t_idx[split == 1], e_idx[split == 1]
    t_test, e_test = t_idx[split == 2], e_idx[split == 2]
    print(f"  samples: {len(t_train)} train, {len(t_val)} validation, {len(t_test)} test", flush=True)

    scans = jnp.asarray(data["scan"])
    obstacle_state = jnp.asarray(data["obstacle_state"])
    obstacle_state_test = np.asarray(data["obstacle_state"][t_test, e_test])

    # Ground truth for the probe: relative position and velocity of the nearest obstacle (yaw frame, m and m/s).
    per_obstacle = obstacle_state_test.reshape(-1, args.nr_obstacles, 5)
    nearest = np.argmin(np.linalg.norm(per_obstacle[..., 0:2], axis=-1), axis=-1)
    nearest_state = per_obstacle[np.arange(len(nearest)), nearest]
    nearest_pos, nearest_vel = nearest_state[:, 0:2] * 5.0, nearest_state[:, 2:4] * 5.0
    nearest_speed = np.linalg.norm(nearest_vel, axis=-1)
    nearest_distance = np.linalg.norm(nearest_pos, axis=-1)
    closing_speed = -np.sum(nearest_pos * nearest_vel, axis=-1) / np.maximum(nearest_distance, 1e-6)  # > 0 = approaching
    visible = np.linalg.norm(nearest_pos, axis=-1) < args.max_range - 0.5

    variants = [("scan", k) for k in args.history] + [("scanconv", k) for k in args.conv_history] + [("state_pos", 0), ("state_full", 0)]
    results = {"args": vars(args), "nr_train": int(len(t_train)), "nr_val": int(len(t_val)), "nr_test": int(len(t_test)), "horizons": {}}
    speed_bins = np.array([0.5, 1.0, 1.5, 2.0, 2.5 + 1e-6])
    counterfactual = {}
    key = jax.random.PRNGKey(args.seed + 1)

    for horizon in args.horizons:
        y_train = labels_for(data, t_train, e_train, horizon)
        y_val = labels_for(data, t_val, e_val, horizon)
        y_test = labels_for(data, t_test, e_test, horizon)
        entry = {"unsafe_rate_test": float(1.0 - y_test.mean()), "variants": {}, "baselines": {}}

        # Scores that need no training.
        nearest_return = np.asarray(scans[t_test, e_test]).max(axis=-1)  # proximity of the closest return
        entry["baselines"]["nearest_scan_return"] = {
            "auroc": auroc(nearest_return, 1 - y_test), "average_precision": average_precision(nearest_return, 1 - y_test)}
        cv_clearance = constant_velocity_min_clearance(obstacle_state_test, args.nr_obstacles, horizon * env.dt)
        entry["baselines"]["constant_velocity_oracle"] = {
            "auroc": auroc(-cv_clearance, 1 - y_test), "average_precision": average_precision(-cv_clearance, 1 - y_test)}

        for kind, history in variants:
            name = f"{kind}_{history}" if kind in ("scan", "scanconv") else kind
            key, train_key = jax.random.split(key)
            make_input = input_fn(kind, history, scans, obstacle_state, args.nr_obstacles)
            model = ScanConvCritic(args.hidden, history, args.nr_rays) if kind == "scanconv" else Critic(args.hidden)
            tic = time.time()
            params, training = train_critic(model, make_input, (t_train, e_train, y_train), (t_val, e_val, y_val), args, train_key)
            p_test, features = predict(model, params, make_input, t_test, e_test)
            metrics = classification_metrics(p_test, y_test)
            metrics.update(training)
            # AUROC per speed of the nearest obstacle.
            metrics["auroc_by_speed"] = []
            for low, high in zip(speed_bins[:-1], speed_bins[1:]):
                in_bin = (nearest_speed >= low) & (nearest_speed < high)
                metrics["auroc_by_speed"].append(auroc(1.0 - p_test[in_bin], 1 - y_test[in_bin]))
            if horizon == 25:
                split = len(features[visible]) // 2
                metrics["probe_r2_velocity"] = ridge_r2(features[visible], nearest_vel[visible], split)
                metrics["probe_r2_position"] = ridge_r2(features[visible], nearest_pos[visible], split)
                metrics["probe_r2_closing_speed"] = ridge_r2(features[visible], closing_speed[visible][:, None], split)[0]
                if kind in ("scan", "scanconv"):
                    speeds = np.linspace(-2.5, 2.5, 51)
                    for distance_now in (1.2, 1.6):
                        cf_inputs = counterfactual_inputs(env, args, history, distance_now, speeds)
                        cf_logits, _ = model.apply(params, cf_inputs)
                        counterfactual.setdefault(str(distance_now), {})[name] = np.asarray(jax.nn.sigmoid(cf_logits)).tolist()
            entry["variants"][name] = metrics
            print(f"H={horizon:2d} {name:12s} AUROC {metrics['auroc']:.3f} AP {metrics['average_precision']:.3f} "
                  f"logloss {metrics['log_loss']:.4f} (train {metrics['train_log_loss']:.4f}, best step {metrics['best_step']}, "
                  f"{time.time() - tic:.0f}s)", flush=True)
        results["horizons"][str(horizon)] = entry

    # Raw-input probe for reference: how linearly decodable is velocity from the stacked scans themselves?
    for history in (1, 10):
        if history in args.history:
            raw = np.asarray(input_fn("scan", history, scans, obstacle_state, args.nr_obstacles)(jnp.asarray(t_test), jnp.asarray(e_test)))
            split = int(visible.sum()) // 2
            results.setdefault("raw_input_probe_r2_velocity", {})[f"scan_{history}"] = ridge_r2(raw[visible], nearest_vel[visible], split)
            results.setdefault("raw_input_probe_r2_closing_speed", {})[f"scan_{history}"] = ridge_r2(raw[visible], closing_speed[visible][:, None], split)[0]

    results["counterfactual"] = {"speeds": np.linspace(-2.5, 2.5, 51).tolist(), "horizon": 25, "by_distance": counterfactual}
    results["robot_radius"] = float(env.obstacle_config["robot_radius"])
    results["dt"] = float(env.dt)
    results["runtime_seconds"] = time.time() - started
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_report(results, args)
    print(f"Wrote {args.out_dir}/summary.json and results.md in {time.time() - started:.0f}s", flush=True)


def write_report(results, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = [str(h) for h in args.horizons]
    families = [("scan", "MLP on stacked scans", list(args.history)), ("scanconv", "circular conv over rays", list(args.conv_history))]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (family, title, histories) in zip(axes, families):
        for horizon in horizons:
            variants = results["horizons"][horizon]["variants"]
            ax.plot(histories, [variants[f"{family}_{k}"]["average_precision"] for k in histories],
                    marker="o", label=f"H={horizon} ({int(horizon) * results['dt']:.1f} s)")
            color = ax.lines[-1].get_color()
            ax.axhline(variants["state_full"]["average_precision"], linestyle=":", color=color)
            ax.axhline(variants["state_pos"]["average_precision"], linestyle="--", color=color, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks(histories, [str(k) for k in histories])
        ax.set_xlabel("scan history length k (frames)")
        ax.set_title(title)
    axes[0].set_ylabel("average precision (unsafe)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Collision critic from scans; dotted = privileged position + velocity, dashed = position only", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "ap_vs_history.png"), dpi=150)
    plt.close(fig)

    speeds = np.array(results["counterfactual"]["speeds"])
    by_distance = results["counterfactual"]["by_distance"]
    fig, axes = plt.subplots(len(families), len(by_distance), figsize=(5 * len(by_distance), 3.3 * len(families)), squeeze=False)
    for row, (family, title, histories) in enumerate(families):
        for column, (distance, curves) in enumerate(by_distance.items()):
            ax = axes[row][column]
            for k in histories:
                name = f"{family}_{k}"
                if name in curves:
                    ax.plot(speeds, curves[name], label=name)
            threshold = (float(distance) - 0.25 - results["robot_radius"]) / (results["counterfactual"]["horizon"] * results["dt"])
            ax.axvline(threshold, color="k", linestyle="--", linewidth=0.8)
            ax.set_title(f"{title}: obstacle {distance} m ahead", fontsize=9)
            ax.set_xlabel("approach speed in the past frames (m/s, <0 = receding)")
            ax.set_ylabel("P(no collision in 0.5 s)")
            ax.set_ylim(-0.02, 1.02)
            ax.legend(fontsize=7)
    fig.suptitle("Same current scan, different past; dashed line = true threshold", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "counterfactual.png"), dpi=150)
    plt.close(fig)

    lines = ["# Offline scan-critic probe", "",
             f"Static Go2 (zero action, lying on the ground), {args.nr_obstacles} moving obstacles, speeds "
             f"{args.speed_range[0]}-{args.speed_range[1]} m/s, aim offset ±{args.aim_offset} m, {args.nr_rays} rays over 360°, "
             f"range noise ±{args.scan_noise} m. {results['nr_train']} train / {results['nr_val']} validation / "
             f"{results['nr_test']} test samples, split by environment. Early stopping on validation log loss.", ""]
    for horizon in horizons:
        entry = results["horizons"][horizon]
        lines += [f"## H = {horizon} steps ({int(horizon) * results['dt']:.2f} s), unsafe rate {entry['unsafe_rate_test']:.3f}", "",
                  "| input | AUROC | AP | test log loss | train log loss |", "|---|---|---|---|---|"]
        for name, m in entry["variants"].items():
            lines.append(f"| {name} | {m['auroc']:.3f} | {m['average_precision']:.3f} | {m['log_loss']:.4f} | {m['train_log_loss']:.4f} |")
        for name, m in entry["baselines"].items():
            lines.append(f"| {name} (no training) | {m['auroc']:.3f} | {m['average_precision']:.3f} | – | – |")
        lines.append("")
    entry = results["horizons"].get("25")
    if entry is not None:
        lines += ["## Linear probe of the last hidden layer (H = 25 critics, nearest visible obstacle)", "",
                  "| input | R² closing speed | R² velocity (x, y) | R² position (x, y) |", "|---|---|---|---|"]
        for name, m in entry["variants"].items():
            if "probe_r2_velocity" in m:
                v, p = m["probe_r2_velocity"], m["probe_r2_position"]
                lines.append(f"| {name} | {m['probe_r2_closing_speed']:.2f} | {v[0]:.2f}, {v[1]:.2f} | {p[0]:.2f}, {p[1]:.2f} |")
        for name, v in results.get("raw_input_probe_r2_velocity", {}).items():
            closing = results.get("raw_input_probe_r2_closing_speed", {}).get(name, float("nan"))
            lines.append(f"| raw {name} input (no network) | {closing:.2f} | {v[0]:.2f}, {v[1]:.2f} | – |")
        lines += ["", "## AUROC by speed of the nearest obstacle (H = 25)", "",
                  "| input | 0.5–1 | 1–1.5 | 1.5–2 | 2–2.5 m/s |", "|---|---|---|---|---|"]
        for name, m in entry["variants"].items():
            lines.append("| " + name + " | " + " | ".join(f"{a:.3f}" for a in m["auroc_by_speed"]) + " |")
    lines += ["", "Figures: `ap_vs_history.png`, `counterfactual.png`."]
    with open(os.path.join(args.out_dir, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
