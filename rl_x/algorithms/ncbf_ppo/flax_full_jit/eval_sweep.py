"""Batched perturbation-sweep evaluation (paper Fig. 2 protocol, rewritten for the full-jit path).

Called from `PPO.test()` when `algorithm.eval_sweep.enabled=True` (runner.mode=test). One process evaluates one
checkpoint on one terrain in one deployment mode, over

    perturbation in {action_noise, push}  x  sampling probability  x  eval seed,

with `environment.nr_envs` parallel envs x exactly one episode each per sweep point, and writes one
`eval_summary.json` (+ per-env `eval_episodes.npz`) per sweep point to

    <out_dir>/<checkpoint_label>/<perturbation>_<terrain_label>/mode_<A|B>/sp_<p>/seed_<s>/

Protocol (paper code: `plots` branch, `ncbf_ppo/flax/ppo.py:test()` and the CPU `mujoco/` env):
  * action noise: with probability `p` per env and control step, add N(0, noise_std^2) to every joint of the
    processed policy action (before the shield in mode B, like the paper). noise_std is absolute (paper: 1.0)
    or a fraction of each joint's action range (`noise_std_mode=action_range_fraction`).
  * push: with probability `p` per env and control step, apply the env's own perturbation function (random trunk
    velocity, joint velocity/position kick), its trunk velocity bound scaled by `push_scale`.
  * policy: deterministic (action mean). Mode A = policy only (shield never called), mode B = policy + shield.
  * env in eval mode (curriculum coefficient 1, domain randomization at every episode start).
  * success = episode not terminated within the horizon (`environment.episode_length_in_seconds`).
  * the same eval seed gives the same initial states, domain randomization, noise and push draws for every
    checkpoint (paired comparison); draws are nested across sampling probabilities (u < p).

Everything here only uses the env API (reset/step/State) and an agent adapter, so this file also works on the
paper-code branch (`ncbf-q`, see `PaperAgent`).
"""
import copy
import json
import logging
import os
import subprocess
import time
from contextlib import contextmanager

import numpy as np
import jax
import jax.numpy as jnp

from rl_x.algorithms.ncbf_ppo.flax_full_jit.ncbf_targets import episode_first_step


rlx_logger = logging.getLogger("rl_x")

SCHEMA_VERSION = "pncbf-eval/v1"
PERTURBATIONS = ("action_noise", "push")
TERRAIN_LABELS = {"plane": "flat", "hfield_diverse": "hfield"}


def _float_list(text):
    if isinstance(text, (list, tuple)):
        return [float(v) for v in text]
    return [float(v) for v in str(text).replace(";", ",").split(",") if v.strip() != ""]


def _int_list(text):
    return [int(v) for v in _float_list(text)]


def prob_token(p):
    """0.002 -> '0p002' (same token style as the paper's rollout file names)."""
    return f"{float(p):g}".replace("-", "m").replace(".", "p")


RL_X_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _git_commit():
    try:
        repo = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


@contextmanager
def _patched_attributes(obj, **values):
    old = {name: getattr(obj, name) for name in values}
    for name, value in values.items():
        setattr(obj, name, value)
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(obj, name, value)


# ----------------------------------------------------------------------------------------------------------------
# Agent adapters: policy(env_state) -> (raw processed action [B, A], aux); shield(env_state, action, aux) ->
# (action, constraint_active [B], delta_u [B]).
# ----------------------------------------------------------------------------------------------------------------
class MultistepAgent:
    """pncbf-v2 / multistep PPO (history encoder, decoder, ensemble critic). Mirrors `PPO._policy_and_shield`."""

    def __init__(self, model):
        self.model = model

    @property
    def use_safety_layer(self):
        return bool(self.model.ncbf_use_safety_layer)

    def policy(self, env_state):
        m = self.model
        latent_z = m.encoder_state.apply_fn(m.encoder_state.params, env_state.history_stack)
        decoder_output = m._get_policy_decoder_output(
            m.decoder_state.params, latent_z, env_state.last_state, env_state.last_action, env_state.next_observation)
        observation_input = m._mask_policy_ball_plate_inputs(env_state.next_observation)
        action_mean, _ = m.policy.apply(m.policy_state.params, observation_input, latent_z, decoder_output)
        return m.get_processed_action(action_mean), latent_z

    def shield(self, env_state, action, latent_z):
        m = self.model
        safe_action, constraint_active, delta_u, _ = m.batched_ncbf_safety_layer(
            action, env_state.next_observation, env_state.last_action, env_state.last_state, latent_z,
            jnp.asarray(1.0, dtype=action.dtype), m._ncbf_params_stack(m.ncbf_state.params),
            episode_first_step(env_state))
        safe_action, constraint_active, delta_u, _ = m._bypass_post_drop_safety_layer(
            action, safe_action, constraint_active, delta_u, {}, env_state.next_observation)
        return safe_action, constraint_active, delta_u


class PaperAgent:
    """Paper code (`ncbf-q` branch): policy(obs) and one of two shields.

    shield="train": the full-jit shield used during training (ppo.py:189-192 there):
        `batched_ncbf_safety_layer(u, obs, last_action, last_state, stacked ensemble params)`, last_state = previous obs.
    shield="cpu_test": the shield of the paper's evaluation, i.e. the non-jit `ncbf_ppo/flax` test() on the CPU env:
        `flax/ncbf.py:make_get_safe_action_q` (CVaR over k=2 of 5 members instead of 3, slack scaled by the std of h(x)
        instead of clip(std h(u), 1, 10), shield off while last_action == 0), called with last_state = the *current*
        observation (`flax/ppo.py:1001-1002`: last_state = state after env.step).
    """

    def __init__(self, model, shield="train"):
        self.model = model
        self.shield_mode = shield
        if shield == "cpu_test":
            from ml_collections import config_dict
            from rl_x.algorithms.ncbf_ppo.flax.ncbf import get_ncbf as get_flax_ncbf
            config = config_dict.ConfigDict(model.config.to_dict())
            config.algorithm.ncbf.n_enssemble = config.algorithm.ncbf.n_ensemble
            config.algorithm.ncbf.use_robust_safety_layer = False
            _, _, self._cpu_test_shield, _ = get_flax_ncbf(config, model.env)
        elif shield != "train":
            raise ValueError("eval_sweep.paper_shield must be train or cpu_test")

    @property
    def use_safety_layer(self):
        return bool(self.model.config.algorithm.ncbf.use_safety_layer)

    def policy(self, env_state):
        m = self.model
        action_mean, _ = m.policy.apply(m.policy_state.params, env_state.next_observation)
        return m.get_processed_action(action_mean), None

    def shield(self, env_state, action, aux):
        m = self.model
        states = m.ncbf_state if isinstance(m.ncbf_state, (list, tuple)) else [m.ncbf_state]
        params_stack = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[s.params for s in states])
        if self.shield_mode == "cpu_test":
            obs = env_state.next_observation
            safe_action, delta, delta_u, _, _ = self._cpu_test_shield(action, obs, env_state.last_action, obs, params_stack)
            last_action_zero = jnp.all(env_state.last_action == 0.0, axis=-1)
            return safe_action, (delta > 0.0) & ~last_action_zero, delta_u
        safe_action, constraint_active, delta_u = m.batched_ncbf_safety_layer(
            action, env_state.next_observation, env_state.last_action, env_state.last_state, params_stack)
        return safe_action, constraint_active, delta_u


def make_agent(model, paper_shield="train"):
    return MultistepAgent(model) if hasattr(model, "encoder_state") else PaperAgent(model, paper_shield)


# ----------------------------------------------------------------------------------------------------------------
# Sweep configuration
# ----------------------------------------------------------------------------------------------------------------
def read_sweep_config(config):
    """Plain dict of the sweep protocol from `config.algorithm.eval_sweep` (defaults for missing keys)."""
    cfg = config.algorithm.get("eval_sweep", None)
    get = (lambda name, default: cfg.get(name, default)) if cfg is not None else (lambda name, default: default)
    sampling_probs = get("sampling_probs", "0.0,0.002,0.004,0.008,0.016")
    noise_probs = get("action_noise_sampling_probs", "") or sampling_probs
    push_probs = get("push_sampling_probs", "") or sampling_probs
    terrain_type = config.environment.terrain.type if "terrain" in config.environment else "unknown"
    sweep = {
        "out_dir": get("out_dir", ""),
        "checkpoint_label": get("checkpoint_label", ""),
        "method": get("method", ""),
        "train_seed": int(get("train_seed", -1)),
        "terrain_label": get("terrain_label", "") or TERRAIN_LABELS.get(terrain_type, terrain_type),
        "terrain_type": terrain_type,
        "perturbations": [p.strip() for p in str(get("perturbations", "action_noise,push")).split(",") if p.strip()],
        "sampling_probs": {"action_noise": _float_list(noise_probs), "push": _float_list(push_probs)},
        "eval_seeds": _int_list(get("eval_seeds", "0")),
        "deployment_mode": str(get("deployment_mode", "A")).upper(),
        "noise_std": float(get("noise_std", 1.0)),
        "noise_std_mode": get("noise_std_mode", "absolute"),
        "noise_point": get("noise_point", "pre_shield"),
        "push_scale": float(get("push_scale", 1.0)),
        "push_scale_joints": bool(get("push_scale_joints", False)),
        "push_impl": get("push_impl", "env"),
        "background_push_prob": float(get("background_push_prob", -1.0)),
        "success_threshold_steps": int(get("success_threshold_steps", 250)),
        "initial_command_zero_prob": float(get("initial_command_zero_prob", 1.0)),
        "paper_shield": get("paper_shield", "train"),
        "checkpoint_source": get("checkpoint_source", ""),
        "save_episodes": bool(get("save_episodes", True)),
    }
    for name in sweep["perturbations"]:
        if name not in PERTURBATIONS:
            raise ValueError(f"eval_sweep.perturbations: unknown '{name}', expected {PERTURBATIONS}")
    if sweep["deployment_mode"] not in ("A", "B"):
        raise ValueError("eval_sweep.deployment_mode must be A (policy only) or B (policy + shield)")
    if sweep["noise_std_mode"] not in ("absolute", "action_range_fraction"):
        raise ValueError("eval_sweep.noise_std_mode must be absolute or action_range_fraction")
    if sweep["noise_point"] not in ("pre_shield", "post_shield"):
        raise ValueError("eval_sweep.noise_point must be pre_shield or post_shield")
    if sweep["push_impl"] not in ("env", "eval_loop"):
        raise ValueError("eval_sweep.push_impl must be env or eval_loop")
    return sweep


def noise_std_vector(env, sweep):
    action_dim = int(np.prod(env.single_action_space.shape))
    if sweep["noise_std_mode"] == "absolute":
        return np.full(action_dim, sweep["noise_std"], dtype=np.float32)
    action_range = np.asarray(env.single_action_space.high, dtype=np.float32) - np.asarray(env.single_action_space.low, dtype=np.float32)
    return (sweep["noise_std"] * action_range).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------------
# One batched episode per env
# ----------------------------------------------------------------------------------------------------------------
def _scaled_perturbation_function(perturbation_function, scale, scale_joints):
    """Copy of the env's perturbation function whose trunk velocity bound (and optionally joint kicks) is scaled.
    max_trunk_velocity = min(mass * f, limit) -> scale * min(mass * f, limit)."""
    scaled = copy.copy(perturbation_function)
    scaled.trunk_velocity_clip_mass_factor = perturbation_function.trunk_velocity_clip_mass_factor * scale
    scaled.trunk_velocity_clip_limit = perturbation_function.trunk_velocity_clip_limit * scale
    if scale_joints:
        scaled.max_joint_velocity = perturbation_function.max_joint_velocity * scale
        scaled.max_joint_position = perturbation_function.max_joint_position * scale
    return scaled


def apply_initial_commands(env, env_state, key, zero_prob):
    """The env starts every episode with the command of the previous one (0 after the first reset) and resamples it
    with a small per-step probability. The paper's CPU eval ran its episodes in sequence, so most of them started with
    a random command, whereas a batched reset starts every env at 0. Keep 0 with probability `zero_prob`, otherwise
    draw the command from the env's own command distribution (1.0 = env default, nothing changes)."""
    if zero_prob >= 1.0:
        return env_state
    nr_envs = env_state.next_observation.shape[0]
    keep_key, command_key = jax.random.split(key)
    resample = jax.random.uniform(keep_key, (nr_envs,)) >= zero_prob

    def one(state, flag, k):
        internal_state = dict(state.internal_state)
        env.command_function.get_next_command(internal_state, flag, k)
        return state.replace(internal_state=internal_state)

    return jax.jit(jax.vmap(one))(env_state, resample, jax.random.split(command_key, nr_envs))


def make_episode_runner(env, agent, deployment_mode, noise_point="pre_shield", loop_pushes=False,
                        push_scale_joints=False, success_threshold_steps=250, horizon=None):
    """jitted run(env_state, noise_key, push_key, noise_prob, noise_std[A], push_prob, push_scale) -> per-env dict.

    Every env runs exactly one episode of `horizon` steps (later auto-reset episodes are ignored).
    `loop_pushes=True` applies pushes here (traced probability/scale, before the env step); otherwise the env applies
    its own pushes (probability/magnitude fixed when the env step is traced).
    """
    horizon = int(horizon or env.horizon)
    use_shield = deployment_mode == "B"
    target_indices = getattr(env, "ncbf_target_indices", None)
    target_indices = None if target_indices is None or len(target_indices) == 0 else jnp.asarray(target_indices)
    perturbation_function = env.domain_randomization_perturbation_function

    def act(env_state, noise):
        raw_action, aux = agent.policy(env_state)
        if not use_shield:
            action = raw_action + noise
            zeros = jnp.zeros(action.shape[0], dtype=action.dtype)
            return action, jnp.zeros(action.shape[0], dtype=bool), zeros
        if noise_point == "pre_shield":
            return agent.shield(env_state, raw_action + noise, aux)
        action, active, delta_u = agent.shield(env_state, raw_action, aux)
        return action + noise, active, delta_u

    def step(carry, step_index):
        env_state, stats = carry
        nr_envs = stats["alive"].shape[0]
        step_noise_key = jax.random.fold_in(stats["noise_key"], step_index)
        bernoulli_key, normal_key = jax.random.split(step_noise_key)
        noise_on = jax.random.uniform(bernoulli_key, (nr_envs,)) < stats["noise_prob"]
        noise = jax.random.normal(normal_key, (nr_envs, stats["noise_std"].shape[0])) * stats["noise_std"]
        noise = noise * noise_on[:, None]

        if loop_pushes:
            step_push_key = jax.random.fold_in(stats["push_key"], step_index)
            push_bernoulli_key, push_sample_key = jax.random.split(step_push_key)
            push_on = jax.random.uniform(push_bernoulli_key, (nr_envs,)) < stats["push_prob"]
            scaled = _scaled_perturbation_function(perturbation_function, stats["push_scale"], push_scale_joints)
            data = jax.vmap(lambda s, flag, key: scaled.sample(s.internal_state, s.mjx_model, s.data, flag, key))(
                env_state, push_on, jax.random.split(push_sample_key, nr_envs))
            env_state = env_state.replace(data=data)
        else:
            push_on = jnp.zeros(nr_envs, dtype=bool)

        action, constraint_active, delta_u = act(env_state, noise)
        env_state = env.step(env_state, action)

        alive = stats["alive"]
        alive_f = alive.astype(jnp.float32)
        terminated = env_state.terminated.astype(bool)
        truncated = env_state.truncated.astype(bool)
        constraint_terminated = env_state.info.get("constraint/terminated", terminated).astype(bool)
        collision = env_state.info.get("env_info/obstacle_collision", jnp.zeros(nr_envs)).astype(bool)

        new = dict(stats)
        new["return"] = stats["return"] + alive_f * env_state.reward
        new["length"] = stats["length"] + alive.astype(jnp.int32)
        new["terminated"] = stats["terminated"] | (alive & terminated)
        new["truncated"] = stats["truncated"] | (alive & truncated & ~terminated)
        new["constraint_terminated"] = stats["constraint_terminated"] | (alive & constraint_terminated)
        new["qvel_limit_terminated"] = stats["qvel_limit_terminated"] | (alive & terminated & ~constraint_terminated)
        new["collision"] = stats["collision"] | (alive & collision)
        new["noise_events"] = stats["noise_events"] + (alive & noise_on).astype(jnp.int32)
        new["push_events"] = stats["push_events"] + (alive & push_on).astype(jnp.int32)
        new["interventions"] = stats["interventions"] + (alive & constraint_active.astype(bool)).astype(jnp.int32)
        new["delta_u_sum"] = stats["delta_u_sum"] + alive_f * delta_u
        if target_indices is not None:
            unsafe = env_state.actual_next_observation[:, target_indices] < 0.5
            new["head_unsafe"] = stats["head_unsafe"] | (alive[:, None] & unsafe)
        new["alive"] = alive & ~(terminated | truncated)
        return (env_state, new), None

    @jax.jit
    def run(env_state, noise_key, push_key, noise_prob, noise_std, push_prob, push_scale):
        nr_envs = env_state.next_observation.shape[0]
        false = jnp.zeros(nr_envs, dtype=bool)
        zero_i = jnp.zeros(nr_envs, dtype=jnp.int32)
        stats = {
            "alive": jnp.ones(nr_envs, dtype=bool),
            "return": jnp.zeros(nr_envs, dtype=jnp.float32), "length": zero_i,
            "terminated": false, "truncated": false, "constraint_terminated": false,
            "qvel_limit_terminated": false, "collision": false,
            "noise_events": zero_i, "push_events": zero_i, "interventions": zero_i,
            "delta_u_sum": jnp.zeros(nr_envs, dtype=jnp.float32),
            "head_unsafe": jnp.zeros((nr_envs, 0 if target_indices is None else target_indices.shape[0]), dtype=bool),
            "noise_key": noise_key, "push_key": push_key,
            "noise_prob": jnp.asarray(noise_prob, jnp.float32), "noise_std": jnp.asarray(noise_std, jnp.float32),
            "push_prob": jnp.asarray(push_prob, jnp.float32), "push_scale": jnp.asarray(push_scale, jnp.float32),
        }
        (_, stats), _ = jax.lax.scan(step, (env_state, stats), jnp.arange(horizon))
        for name in ("alive", "noise_key", "push_key", "noise_prob", "noise_std", "push_prob", "push_scale"):
            stats.pop(name)
        return stats

    return run


# ----------------------------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------------------------
def _se(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


def summarize_episodes(episodes, horizon, success_threshold_steps, head_names=None, shield_used=False):
    """eval_summary.json fields from per-env episode outcomes (one episode per env)."""
    n = int(episodes["length"].shape[0])
    terminated = episodes["terminated"].astype(bool)
    success = ~terminated
    success_rate = float(np.mean(success))
    returns = episodes["return"].astype(np.float64)
    lengths = episodes["length"].astype(np.float64)
    steps = float(np.sum(lengths))
    survived = lengths >= min(success_threshold_steps, horizon)
    survival_rate = float(np.mean(survived | success))

    event_rates = {
        "termination": float(np.mean(terminated)),
        "fall": float(np.mean(episodes["constraint_terminated"] & ~episodes["collision"])),
        "qvel_limit": float(np.mean(episodes["qvel_limit_terminated"])),
        "obstacle_collision": float(np.mean(episodes["collision"])),
    }
    head_unsafe = episodes.get("head_unsafe")
    if head_unsafe is not None and head_unsafe.shape[-1] > 0:
        names = list(head_names) if head_names is not None else [f"head_{i}" for i in range(head_unsafe.shape[-1])]
        for i, name in enumerate(names):
            event_rates[f"unsafe_{name}"] = float(np.mean(head_unsafe[:, i]))

    per_env_intervention = episodes["interventions"] / np.maximum(lengths, 1.0)
    interventions = float(np.sum(episodes["interventions"]))
    summary = {
        "schema": SCHEMA_VERSION,
        # fields of the multistep test() summary (same meaning; one episode per env here)
        "nr_envs": n,
        "nr_steps": int(horizon),
        "nr_finished_episodes": n,
        "failure_rate": float(np.mean(terminated)),
        "terminations_per_env": float(np.mean(terminated)),
        "collisions_per_env": float(np.mean(episodes["collision"])),
        "mean_episode_return": float(np.mean(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "intervention_rate": (interventions / max(steps, 1.0)) if shield_used else None,
        "mean_delta_u": (float(np.sum(episodes["delta_u_sum"])) / max(steps, 1.0)) if shield_used else None,
        # new fields
        "nr_episodes": n,
        "horizon": int(horizon),
        "success_rate": success_rate,
        "success_rate_se": float(np.sqrt(success_rate * (1.0 - success_rate) / max(n, 1))),
        "mean_episode_return_se": _se(returns),
        "episode_return_std": float(np.std(returns, ddof=1)) if n > 1 else 0.0,
        "mean_episode_length_se": _se(lengths),
        "success_threshold_steps": int(success_threshold_steps),
        "survival_rate_at_threshold": survival_rate,
        "event_rates": event_rates,
        "intervention_rate_se": _se(per_env_intervention) if shield_used else None,
        "mean_delta_u_when_active": (float(np.sum(episodes["delta_u_sum"])) / interventions if interventions > 0 else 0.0) if shield_used else None,
        "noise_events_per_episode": float(np.mean(episodes["noise_events"])),
        "push_events_per_episode": float(np.mean(episodes["push_events"])),
    }
    return summary


# ----------------------------------------------------------------------------------------------------------------
# Driver (inside one process)
# ----------------------------------------------------------------------------------------------------------------
def sweep_points(sweep):
    for perturbation in sweep["perturbations"]:
        for p in sweep["sampling_probs"][perturbation]:
            for seed in sweep["eval_seeds"]:
                yield perturbation, float(p), int(seed)


def point_dir(sweep, perturbation, p, seed):
    return os.path.join(sweep["out_dir"], sweep["checkpoint_label"], f"{perturbation}_{sweep['terrain_label']}",
                        f"mode_{sweep['deployment_mode']}", f"sp_{prob_token(p)}", f"seed_{seed}")


def eval_keys(seed):
    """Fixed per eval seed and independent of checkpoint/method: reset (initial state + DR), noise, push."""
    base = jax.random.PRNGKey(int(seed))
    return jax.random.fold_in(base, 0), jax.random.fold_in(base, 1), jax.random.fold_in(base, 2)


def command_key(seed):
    return jax.random.fold_in(jax.random.PRNGKey(int(seed)), 3)


def run_eval_sweep(model, sweep=None):
    """Run the sweep for a loaded model (PPO of this branch or of the paper code). Returns the list of summaries."""
    env, config = model.env, model.config
    sweep = sweep or read_sweep_config(config)
    agent = make_agent(model, sweep["paper_shield"])
    if sweep["deployment_mode"] == "B" and not agent.use_safety_layer:
        raise ValueError("deployment_mode=B needs algorithm.ncbf.use_safety_layer=True (the shield is a no-op otherwise).")
    if not sweep["out_dir"]:
        sweep["out_dir"] = os.path.join(os.path.dirname(model.save_path), "eval_sweep")
    if not sweep["checkpoint_label"]:
        load_path = config.runner.get("load_model", "") or "untrained"
        parts = os.path.abspath(load_path).split(os.sep)
        sweep["checkpoint_label"] = parts[-3] if len(parts) >= 3 and parts[-2] == "models" else os.path.basename(load_path)

    sampling_function = env.domain_randomization_perturbation_sampling_function
    perturbation_function = env.domain_randomization_perturbation_function
    env_push_prob = float(getattr(sampling_function, "probability", 0.0))
    background_push_prob = env_push_prob if sweep["background_push_prob"] < 0 else sweep["background_push_prob"]
    noise_std = jnp.asarray(noise_std_vector(env, sweep))
    zero_std = jnp.zeros_like(noise_std)
    loop_pushes = sweep["push_impl"] == "eval_loop"
    head_names = getattr(env, "ncbf_target_names", None)
    commit = _git_commit()
    rlx_logger.info(f"Eval sweep code: rl_x from {RL_X_PATH}, commit {commit}")

    def env_push_settings(perturbation, p):
        """(env-internal push probability, scale) for push_impl=env; the env step is retraced for each value."""
        if loop_pushes:
            return 0.0, 1.0
        if perturbation == "push":
            return p, sweep["push_scale"]
        return background_push_prob, 1.0

    runners = {}

    def get_runner(env_setting):
        """Returns (runner, is_new); a new runner is traced and compiled on its first call."""
        if env_setting in runners:
            return runners[env_setting], False
        if not loop_pushes:
            jax.clear_caches()  # env.step is jitted with the env as a static argument: force a retrace
            runners.clear()
        runners[env_setting] = make_episode_runner(
            env, agent, sweep["deployment_mode"], sweep["noise_point"], loop_pushes,
            sweep["push_scale_joints"], sweep["success_threshold_steps"])
        return runners[env_setting], True

    summaries = []
    rlx_logger.info(f"Eval sweep: {env.nr_envs} envs x 1 episode ({env.horizon} steps), mode {sweep['deployment_mode']}, "
                    f"terrain {sweep['terrain_label']}, push_impl {sweep['push_impl']}, out {sweep['out_dir']}")
    for perturbation, p, seed in sweep_points(sweep):
        env_prob, env_scale = env_push_settings(perturbation, p)
        scaled = _scaled_perturbation_function(perturbation_function, env_scale, sweep["push_scale_joints"])
        attrs = {name: getattr(scaled, name) for name in ("trunk_velocity_clip_mass_factor", "trunk_velocity_clip_limit",
                                                         "max_joint_velocity", "max_joint_position")}
        with _patched_attributes(sampling_function, probability=env_prob), _patched_attributes(perturbation_function, **attrs):
            run, includes_compile = get_runner((env_prob, env_scale))
            if perturbation == "action_noise":
                noise_prob, push_prob, push_scale = p, (background_push_prob if loop_pushes else 0.0), 1.0
            else:
                noise_prob, push_prob, push_scale = 0.0, (p if loop_pushes else 0.0), sweep["push_scale"]
            reset_key, noise_key, push_key = eval_keys(seed)
            start = time.time()
            env_state = env.reset(jax.random.split(reset_key, env.nr_envs), True)
            env_state = apply_initial_commands(env, env_state, command_key(seed), sweep["initial_command_zero_prob"])
            episodes = run(env_state, noise_key, push_key, noise_prob, noise_std if noise_prob > 0 else zero_std, push_prob, push_scale)
            episodes = {name: np.asarray(value) for name, value in jax.device_get(episodes).items()}
            wall_time = time.time() - start

        summary = summarize_episodes(episodes, env.horizon, sweep["success_threshold_steps"], head_names,
                                     shield_used=sweep["deployment_mode"] == "B")
        if not loop_pushes:
            summary["push_events_per_episode"] = None  # env-internal pushes are not counted
        ncbf_config = config.algorithm.ncbf
        summary.update({
            "checkpoint": sweep["checkpoint_source"] or config.runner.get("load_model", None),
            "checkpoint_label": sweep["checkpoint_label"],
            "method": sweep["method"],
            "train_seed": sweep["train_seed"],
            "setting": f"{perturbation}_{sweep['terrain_label']}",
            "perturbation": perturbation,
            "terrain": sweep["terrain_label"],
            "terrain_type": sweep["terrain_type"],
            "sampling_prob": p,
            "eval_seed": seed,
            "deployment_mode": sweep["deployment_mode"],
            "use_safety_layer": sweep["deployment_mode"] == "B",
            "gamma_c": float(ncbf_config.gamma_c),
            "eta_cbf": float(ncbf_config.eta_cbf),
            "H": int(ncbf_config.get("H", -1)),
            "protocol": {
                "noise_prob": noise_prob,
                "noise_std": sweep["noise_std"], "noise_std_mode": sweep["noise_std_mode"],
                "noise_std_per_joint": [float(v) for v in np.asarray(noise_std)],
                "noise_point": sweep["noise_point"],
                "push_impl": sweep["push_impl"],
                "push_prob": p if perturbation == "push" else (background_push_prob if loop_pushes else env_prob),
                "push_scale": sweep["push_scale"] if perturbation == "push" else 1.0,
                "push_scale_joints": sweep["push_scale_joints"],
                "background_push_prob": background_push_prob,
                "env_push_prob_config": env_push_prob,
                "episode_length_in_seconds": float(config.environment.get("episode_length_in_seconds", -1)),
                "eval_mode": True,
                "deterministic_policy": True,
                "episodes_per_env": 1,
                "initial_command_zero_prob": sweep["initial_command_zero_prob"],
                "paper_shield": sweep["paper_shield"] if isinstance(agent, PaperAgent) else None,
            },
            "git_commit": commit,
            "rl_x_path": RL_X_PATH,
            "wall_time_s": wall_time,
            "wall_time_includes_compile": includes_compile,
        })
        out = point_dir(sweep, perturbation, p, seed)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "eval_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        if sweep["save_episodes"]:
            np.savez_compressed(os.path.join(out, "eval_episodes.npz"), **episodes)
        rlx_logger.info(f"[{summary['setting']} mode {sweep['deployment_mode']} p={p:g} seed={seed}] success "
                        f"{summary['success_rate']:.3f} +- {summary['success_rate_se']:.3f}, return "
                        f"{summary['mean_episode_return']:.2f}, {wall_time:.1f} s -> {out}")
        summaries.append(summary)
    return summaries
