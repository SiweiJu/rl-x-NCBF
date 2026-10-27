"""CaT: Constraints as Terminations (Chane-Sane et al. 2024, arXiv 2403.18765). Pure, jittable.

A constraint violation c+ = max(0, c) becomes a termination probability (paper eq. 6, one constraint)

    delta = p_max * clip(c+ / c_max, 0, 1)
    c_max <- tau * c_max + (1 - tau) * max_batch c+          (EMA of the largest violation)

and PPO is run on the modified data (paper Alg. 1): rewards <- rewards * (1 - delta), dones <- delta. The episode
is NOT reset: delta only terminates the *future reward* (the bootstrap and the GAE trace are multiplied by
1 - delta). Hard constraints use p_max = 1 (paper: contacts), so a binary collision with p_max = 1 cuts the return.

Details follow the reference code (github.com/gepetto/constraints-as-terminations,
exts/cat_envs/cat_envs/tasks/utils/cat/{constraint_manager.py, cat_env.py}, read 2026-10-27):
  * the running max is updated at every env step with the max over envs, tau = 0.95, initialized to the first
    step's max, clamped to >= 1e-6;
  * delta = min_p + clip(c / c_max, 0, 1) * (p_max - min_p) on violating samples only (min_p = 0 default), 0 else;
  * reward = clip(reward * (1 - delta), 0, inf);
  * real terminations stay terminations: effective done = max(terminated, delta). (The reference also sets
    done = 1 at time-outs; the repo's PPO GAE does not cut at truncations, so neither do we, for parity.)

Constraint sources (``cat.constraint_source``):
  * ``collision``    env_info/obstacle_collision (binary: base inside robot_radius of an obstacle)
  * ``penetration``  max(0, -env_info/obstacle_min_clearance) in m (graded collision depth)
  * ``cost``         constraint/cost (the PPO-Lag / P3O cost: falls, plus collisions when they terminate)
For collision/penetration run the env with ``environment.obstacles.terminate_on_collision=False`` (existing env
switch), otherwise every collision is already a hard termination and CaT changes nothing.
"""

import jax
import jax.numpy as jnp


CONSTRAINT_SOURCES = ("collision", "penetration", "cost")


def init_cat_state():
    return {
        "running_max": jnp.zeros((), dtype=jnp.float32),
        "initialized": jnp.zeros((), dtype=jnp.float32),  # float, not bool: orbax-friendly
    }


def constraint_values(source, infos, costs):
    """Per-step constraint values c (nr_steps, nr_envs) from the stacked rollout infos."""
    if source == "collision":
        return infos["env_info/obstacle_collision"].astype(jnp.float32)
    if source == "penetration":
        return jnp.maximum(-infos["env_info/obstacle_min_clearance"].astype(jnp.float32), 0.0)
    if source == "cost":
        return costs.astype(jnp.float32)
    raise ValueError(f"Unknown cat.constraint_source {source!r}, expected one of {CONSTRAINT_SOURCES}")


def step_probabilities(cat_state, constraint, tau, p_max, min_p=0.0):
    """One env step. constraint: (nr_envs,). Returns (new_state, delta (nr_envs,))."""
    violation = jnp.maximum(constraint.astype(jnp.float32), 0.0)
    batch_max = jnp.maximum(jnp.max(violation), 1e-6)
    running_max = jnp.where(cat_state["initialized"] > 0.5, tau * cat_state["running_max"] + (1.0 - tau) * batch_max, batch_max)
    normalized = jnp.clip(violation / running_max, 0.0, 1.0)
    delta = jnp.where(violation > 0.0, min_p + normalized * (p_max - min_p), 0.0)
    return {"running_max": running_max, "initialized": jnp.ones((), dtype=jnp.float32)}, delta


def rollout_probabilities(cat_state, constraints, tau, p_max, min_p=0.0):
    """constraints: (nr_steps, nr_envs). Same as calling step_probabilities at every env step."""
    def body(state, c):
        return step_probabilities(state, c, tau, p_max, min_p)
    return jax.lax.scan(body, cat_state, constraints)


def apply(rewards, terminations, delta, clip_reward_at_zero=True):
    """Returns (CaT rewards, effective terminations as float) for the reward GAE."""
    rewards = rewards * (1.0 - delta)
    if clip_reward_at_zero:
        rewards = jnp.maximum(rewards, 0.0)
    effective_terminations = jnp.maximum(terminations.astype(jnp.float32), delta)
    return rewards, effective_terminations
