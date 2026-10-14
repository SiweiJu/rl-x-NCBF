"""Validation of a trained tray-carrying (ball-on-plate) locomotion policy under fixed velocity commands.

The policy is meant to become the frozen low-level controller of the command-level obstacle task, so this
script checks the three things that task relies on:

1. Command grid: for a set of fixed (vx, vy, wz) commands, the ball/plate drop rate, the fall rate and
   the velocity tracking error over one full episode.
2. Step response: the command jumps from zero to a forward speed; the time the mean forward velocity
   needs to reach 63 % / 90 % of the step is the low-level response lag a command-level shield has to
   look ahead over.
3. Braking: the command drops from a forward speed to zero; how often the ball falls when the robot
   stops hard (an obstacle-avoidance manoeuvre).

Commands are given as fractions of the per-axis maximum the environment trains on
(max_command_velocity * velocity_ratio), so the same profiles fit any robot.

The environment's own command sampling is overridden every control step: before each policy call the
command is written into internal_state["goal_velocities"] and into the command slots of the observation.
After a ball/plate drop the environment switches to a zero command (stand_after_drop); this script stops
counting an env from the step its first episode ends or its ball drops, so that switch does not matter.

Run from experiments/ with the training flags of the run, for example:

    python ncbf_tools/eval_tray_baseline.py \
        --algorithm.name=ncbf_ppo.flax_full_jit \
        --environment.name=ncbf_mujoco.robot_locomotion.mjx \
        --environment.train_robot=unitree_g1 \
        --environment.ball_plate.enabled=True \
        --environment.ball_plate.include_observations=True \
        --environment.ball_plate.stand_after_drop=True \
        --environment.nr_envs=4096 \
        --runner.mode=test \
        --runner.load_model=runs/.../models/latest.model \
        --tray_eval_output=tray_eval.json

The result is a JSON file with per-profile metrics and pass/fail against the thresholds below.
"""

import json
import os
import subprocess
import sys

import numpy as np
import jax
import jax.numpy as jnp
from absl import flags

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rl_x.runner.runner import Runner  # noqa: E402


flags.DEFINE_string("tray_eval_output", "tray_eval.json", "Path of the JSON result file.")
flags.DEFINE_float("tray_eval_transient_seconds", 1.0, "Seconds after a command change excluded from tracking error.")
flags.DEFINE_float("tray_eval_switch_seconds", 4.0, "Time of the command change in the step and braking profiles.")
flags.DEFINE_float("tray_eval_seconds", 0.0, "Evaluated seconds per env; <= 0 uses the full episode length.")
# Pass/fail thresholds (proposed; see experiments/TRAY_BASELINE_PLAN.md).
flags.DEFINE_float("tray_eval_max_drop_rate", 0.05, "Max ball/plate drop rate per episode on in-range grid cells.")
flags.DEFINE_float("tray_eval_max_fall_rate", 0.01, "Max fall (termination) rate per episode on in-range grid cells.")
flags.DEFINE_float("tray_eval_max_lin_rmse", 0.15, "Max linear velocity tracking RMSE (m/s) on in-range grid cells.")
flags.DEFINE_float("tray_eval_max_yaw_rmse", 0.2, "Max yaw-rate tracking RMSE (rad/s) on in-range grid cells.")
flags.DEFINE_float("tray_eval_in_range_fraction", 0.6, "Grid cells whose largest axis fraction is <= this are 'in range'.")
FLAGS = flags.FLAGS


# Each profile: (name, kind, command before the switch, command after the switch), commands as fractions
# of the per-axis maximum. "grid" profiles hold one command for the whole episode.
GRID = [
    (0.0, 0.0, 0.0),
    (0.3, 0.0, 0.0), (0.6, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.5, 0.0, 0.0),
    (0.0, 0.5, 0.0), (0.0, -0.5, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 0.5), (0.0, 0.0, -0.5), (0.0, 0.0, 1.0),
    (0.6, 0.5, 0.0), (0.6, 0.0, 0.5), (0.6, -0.5, -0.5),
]
PROFILES = (
    [(f"grid_{vx:+.1f}_{vy:+.1f}_{wz:+.1f}", "grid", (vx, vy, wz), (vx, vy, wz)) for vx, vy, wz in GRID]
    + [(f"step_vx_{f:.1f}", "step", (0.0, 0.0, 0.0), (f, 0.0, 0.0)) for f in (0.3, 0.6, 1.0)]
    + [(f"brake_vx_{f:.1f}", "brake", (f, 0.0, 0.0), (0.0, 0.0, 0.0)) for f in (0.6, 1.0)]
)


def git_commit():
    try:
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def first_crossing_time(trace, level, dt):
    """Seconds until trace first reaches level (trace starts at the switch), or None."""
    hits = np.nonzero(trace >= level)[0]
    return None if len(hits) == 0 else float(hits[0] * dt)


def run_tray_eval(model, env, config):
    nr_envs = env.nr_envs
    dt = float(env.dt)
    nr_profiles = len(PROFILES)
    if nr_envs < nr_profiles:
        raise ValueError(f"nr_envs={nr_envs} is smaller than the number of profiles ({nr_profiles}).")

    horizon = int(env.horizon)
    nr_steps = horizon if FLAGS.tray_eval_seconds <= 0 else min(horizon, int(round(FLAGS.tray_eval_seconds / dt)))
    switch_step = int(round(FLAGS.tray_eval_switch_seconds / dt))
    transient_steps = int(round(FLAGS.tray_eval_transient_seconds / dt))

    model.key, reset_key = jax.random.split(model.key)
    initial_state = env.reset(jax.random.split(reset_key, nr_envs), True)  # eval mode: full curriculum
    max_command = float(jax.device_get(initial_state.internal_state["max_command_velocity"])[0])
    axis_max = max_command * np.asarray(env.command_function.velocity_ratio, dtype=np.float32)

    # Envs are assigned to profiles round-robin.
    profile_id = np.arange(nr_envs) % nr_profiles
    before = np.array([PROFILES[p][2] for p in profile_id], dtype=np.float32) * axis_max
    after = np.array([PROFILES[p][3] for p in profile_id], dtype=np.float32) * axis_max
    switches = np.array([PROFILES[p][1] != "grid" for p in profile_id])
    before, after, switches, profile_id_j = map(jnp.asarray, (before, after, switches, profile_id))

    goal_idx = jnp.asarray(env.goal_velocities_obs_idx)
    keep_nominal_default = env.command_function.default_actuator_joint_keep_nominal

    def command_at(t):
        return jnp.where((switches & (t >= switch_step))[:, None], after, before)

    def set_command(env_state, command):
        internal_state = dict(env_state.internal_state)
        internal_state["goal_velocities"] = command
        all_zero = jnp.all(command == 0.0, axis=-1, keepdims=True)
        internal_state["actuator_joint_keep_nominal"] = jnp.where(all_zero, jnp.ones_like(keep_nominal_default, dtype=bool), keep_nominal_default[None, :])
        next_observation = env_state.next_observation.at[:, goal_idx].set(command)
        return env_state.replace(internal_state=internal_state, next_observation=next_observation)

    def local_velocity(env_state):
        yaw = env_state.internal_state["imu_orientation_euler"][:, 2]
        vx_w, vy_w = env_state.data.qvel[:, 0], env_state.data.qvel[:, 1]
        vx = jnp.cos(yaw) * vx_w + jnp.sin(yaw) * vy_w
        vy = -jnp.sin(yaw) * vx_w + jnp.cos(yaw) * vy_w
        wz = env_state.data.qvel[:, 5]
        return jnp.stack([vx, vy, wz], axis=-1)

    def scan_step(carry, t):
        env_state, acc = carry
        command = command_at(t)
        env_state = set_command(env_state, command)
        action, _, _ = model._shielded_action(env_state)
        env_state = env.step(env_state, action)

        done = env_state.terminated | env_state.truncated
        dropped = env_state.info.get("env_info/ball_plate_drop_latched", jnp.zeros(nr_envs)) > 0.5
        active = acc["active"]
        # Velocity of the state the step ended in; for envs that just finished this is the reset state,
        # so they are excluded through `done`.
        velocity = local_velocity(env_state)
        since_change = jnp.where(switches & (t >= switch_step), t - switch_step, t)
        track = active & ~done & ~dropped & (since_change >= transient_steps)
        error = jnp.square(velocity - command)

        new_acc = {
            "active": active & ~done & ~dropped,
            "dropped": acc["dropped"] | (active & dropped),
            "fell": acc["fell"] | (active & env_state.terminated),
            "ended": acc["ended"] | (active & done),
            "sq_error": acc["sq_error"] + jnp.where(track[:, None], error, 0.0),
            "nr_tracked": acc["nr_tracked"] + track.astype(jnp.float32),
            "steps_alive": acc["steps_alive"] + active.astype(jnp.float32),
        }
        # Per-profile mean forward velocity of envs that are still active, for the step/brake traces.
        alive = (active & ~done).astype(jnp.float32)
        vx_sum = jax.ops.segment_sum(alive * velocity[:, 0], profile_id_j, num_segments=nr_profiles)
        count = jax.ops.segment_sum(alive, profile_id_j, num_segments=nr_profiles)
        return (env_state, new_acc), (vx_sum, count)

    @jax.jit
    def evaluate(env_state):
        false = jnp.zeros(nr_envs, dtype=bool)
        acc = {
            "active": jnp.ones(nr_envs, dtype=bool), "dropped": false, "fell": false, "ended": false,
            "sq_error": jnp.zeros((nr_envs, 3)), "nr_tracked": jnp.zeros(nr_envs), "steps_alive": jnp.zeros(nr_envs),
        }
        (_, acc), traces = jax.lax.scan(scan_step, (env_state, acc), jnp.arange(nr_steps))
        return acc, traces

    acc, (vx_sum, count) = jax.device_get(evaluate(initial_state))
    vx_mean = np.where(count > 0, vx_sum / np.maximum(count, 1.0), np.nan)  # [T, profiles]

    results = {}
    for p, (name, kind, cmd_before, cmd_after) in enumerate(PROFILES):
        m = profile_id == p
        n = int(m.sum())
        nr_tracked = float(acc["nr_tracked"][m].sum())
        # No tracked step (every env fell or dropped within the transient) gives no RMSE, not zero.
        rmse = np.sqrt(acc["sq_error"][m].sum(0) / nr_tracked) if nr_tracked > 0 else np.full(3, np.nan)
        entry = {
            "kind": kind,
            "command_fraction_before": list(cmd_before),
            "command_fraction_after": list(cmd_after),
            "command_before": [float(x) for x in np.asarray(cmd_before) * axis_max],
            "command_after": [float(x) for x in np.asarray(cmd_after) * axis_max],
            "nr_envs": n,
            "drop_rate": float(acc["dropped"][m].mean()),
            "fall_rate": float(acc["fell"][m].mean()),
            # Reaching the episode time limit is not a failure; only falls and drops are.
            "no_fall_no_drop_rate": float((~acc["fell"][m] & ~acc["dropped"][m]).mean()),
            "mean_seconds_alive": float(acc["steps_alive"][m].mean() * dt),
            "tracked_steps": int(nr_tracked),
            "tracking_rmse_vx": None if np.isnan(rmse[0]) else float(rmse[0]),
            "tracking_rmse_vy": None if np.isnan(rmse[1]) else float(rmse[1]),
            "tracking_rmse_wz": None if np.isnan(rmse[2]) else float(rmse[2]),
        }
        if kind in ("step", "brake"):
            trace = vx_mean[switch_step:, p]
            window = vx_mean[max(0, switch_step - 25):switch_step, p]
            start = float(np.nanmean(window)) if np.isfinite(window).any() else None  # None: no env alive at the switch
            target = float(np.asarray(cmd_after)[0] * axis_max[0])
            entry["vx_before_switch"] = start
            delta = None if start is None else target - start
            if delta is not None and abs(delta) > 1e-6:
                normalized = (trace - start) / delta  # 0 at the old command, 1 at the new one
                entry["t63_seconds"] = first_crossing_time(np.nan_to_num(normalized, nan=-np.inf), 0.63, dt)
                entry["t90_seconds"] = first_crossing_time(np.nan_to_num(normalized, nan=-np.inf), 0.90, dt)
                entry["overshoot_fraction"] = float(max(0.0, np.nanmax(normalized) - 1.0)) if np.isfinite(normalized).any() else None
            entry["mean_vx_trace_after_switch"] = [None if np.isnan(v) else round(float(v), 4) for v in trace[: int(round(3.0 / dt))]]
        results[name] = entry

    in_range = [
        name for name, kind, _, cmd in PROFILES
        if kind == "grid" and max(abs(c) for c in cmd) <= FLAGS.tray_eval_in_range_fraction + 1e-9
    ]
    def worst_rmse(keys):
        # A cell without any tracked step counts as failing (infinite error).
        values = [results[n][k] for n in in_range for k in keys]
        return float("inf") if any(v is None for v in values) else max(values)

    worst = {
        "drop_rate": max(results[n]["drop_rate"] for n in in_range),
        "fall_rate": max(results[n]["fall_rate"] for n in in_range),
        "lin_rmse": worst_rmse(("tracking_rmse_vx", "tracking_rmse_vy")),
        "yaw_rmse": worst_rmse(("tracking_rmse_wz",)),
    }
    checks = {
        "drop_rate": worst["drop_rate"] <= FLAGS.tray_eval_max_drop_rate,
        "fall_rate": worst["fall_rate"] <= FLAGS.tray_eval_max_fall_rate,
        "lin_rmse": worst["lin_rmse"] <= FLAGS.tray_eval_max_lin_rmse,
        "yaw_rmse": worst["yaw_rmse"] <= FLAGS.tray_eval_max_yaw_rmse,
    }
    summary = {
        "checkpoint": config.runner.load_model,
        "git_commit": git_commit(),
        "train_robot": config.environment.train_robot,
        "nr_envs": nr_envs,
        "dt": dt,
        "evaluated_seconds": nr_steps * dt,
        "switch_seconds": switch_step * dt,
        "max_command_velocity": max_command,
        "axis_max_command": [float(x) for x in axis_max],
        "in_range_grid_cells": in_range,
        "worst_in_range": {k: (None if v == float("inf") else v) for k, v in worst.items()},  # None: a cell had no tracked step
        "thresholds": {
            "max_drop_rate": FLAGS.tray_eval_max_drop_rate,
            "max_fall_rate": FLAGS.tray_eval_max_fall_rate,
            "max_lin_rmse": FLAGS.tray_eval_max_lin_rmse,
            "max_yaw_rmse": FLAGS.tray_eval_max_yaw_rmse,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "profiles": results,
    }
    with open(FLAGS.tray_eval_output, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("checkpoint", "max_command_velocity", "worst_in_range", "checks", "passed")}, indent=2))
    for name in results:
        if "t63_seconds" in results[name]:
            r = results[name]
            print(f"{name}: t63={r['t63_seconds']} s, t90={r['t90_seconds']} s, drop_rate={r['drop_rate']:.3f}")
    print(f"Wrote {FLAGS.tray_eval_output}")
    return summary


class TrayEvalRunner(Runner):
    def _test(self, _):
        self.init_config()
        run_path = os.path.abspath(f"runs/{self._config.runner.project_name}/{self._config.runner.exp_name}/{self._config.runner.run_name}")
        env = self._create_env(self._config)
        if self._config.runner.load_model:
            explicitly_set = [p for p in self._algorithm_config_flag._flagvalues if p.startswith("algorithm.")]
            model = self._model_class.load(self._config, env, run_path, None, explicitly_set)
        else:
            print("WARNING: no --runner.load_model given; evaluating an untrained policy (smoke test only).")
            model = self._model_class(self._config, env, run_path, None)
        try:
            run_tray_eval(model, env, self._config)
        finally:
            env.close()


if __name__ == "__main__":
    if not any(arg.startswith("--runner.mode=") for arg in sys.argv):
        sys.argv.append("--runner.mode=test")
    TrayEvalRunner().run()
