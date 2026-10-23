"""Lagrange-multiplier updates and constraint-cost estimates for the constrained baselines.

Pure, jittable functions shared by PPO-Lagrangian and CPO.

Cost measures (``cost_measure``):
  * ``"rollout"``: mean over envs of the summed cost of one rollout (``nr_steps`` steps).
    This was the only measure before the T07 rewrite and stays the PPO-Lagrangian default
    so old runs are reproducible.
  * ``"episode"``: expected cost per episode, estimated from one rollout as
    ``sum(costs) / #finished episodes`` (renewal-reward estimator). With the default
    Go2 cost (1 on the failure step, 0 otherwise) this is the failure probability per episode,
    so ``cost_limit = 0.01`` means "at most 1 failure per 100 episodes". If no episode finished
    in a rollout the previous estimate is kept.

Lagrange update rules (``lagrangian.update_rule``):
  * ``"gradient"``: projected gradient ascent ``lambda <- clip(lambda + lr * (J_c - d), 0, lambda_max)``
    (old behavior).
  * ``"pid"``: PID Lagrangian, Stooke, Achiam, Abbeel, ICML 2020 (arXiv 2007.03964), Algorithm 2,
    with the EMA smoothing of the P and D terms and the derivative delay from the paper's code
    (rlpyt ``CppoPID``; the same code is used in OmniSafe ``common/pid_lagrange.py``):
        delta   = J_c - d
        I       = max(0, I + K_I * delta)                      (clipped to [0, 1] with diff_norm)
        delta_p = a_p * delta_p + (1 - a_p) * delta            (EMA of the error)
        cost_d  = a_d * cost_d  + (1 - a_d) * J_c              (EMA of the cost)
        D       = max(0, cost_d - cost_d[k - d_delay])
        lambda  = clip(K_P * delta_p + I + K_D * D, 0, lambda_max)  (lambda_max = 1 with diff_norm)
"""

import jax.numpy as jnp


UPDATE_RULES = ("gradient", "pid")
COST_MEASURES = ("rollout", "episode")


def init_lagrange_state(lambda_init, pid_d_delay):
    lambda_init = jnp.asarray(lambda_init, dtype=jnp.float32)
    return {
        "lambda": lambda_init,
        "pid_i": lambda_init,
        "pid_delta_p": jnp.zeros((), dtype=jnp.float32),
        "pid_cost_d": jnp.zeros((), dtype=jnp.float32),
        # cost_ds[0] is the cost_d value from d_delay updates ago (zeros before that, like the
        # deque([0.0], maxlen=d_delay) of the reference implementation).
        "pid_cost_ds": jnp.zeros((max(int(pid_d_delay), 1),), dtype=jnp.float32),
        "cost_estimate": jnp.zeros((), dtype=jnp.float32),
    }


def measure_cost(cost_measure, costs, dones, previous_estimate):
    """Constraint value J_c of one rollout. costs, dones: (nr_steps, nr_envs)."""
    costs = costs.astype(jnp.float32)
    dones = dones.astype(jnp.float32)
    nr_finished_episodes = jnp.sum(dones)
    if cost_measure == "rollout":
        return jnp.mean(jnp.sum(costs, axis=0)), nr_finished_episodes
    if cost_measure == "episode":
        estimate = jnp.sum(costs) / jnp.maximum(nr_finished_episodes, 1.0)
        return jnp.where(nr_finished_episodes > 0, estimate, previous_estimate), nr_finished_episodes
    raise ValueError(f"Unknown cost_measure {cost_measure!r}, expected one of {COST_MEASURES}")


def gradient_update(state, cost, cost_limit, lr, lambda_max):
    new_state = dict(state)
    new_state["lambda"] = jnp.clip(state["lambda"] + lr * (cost - cost_limit), 0.0, lambda_max)
    new_state["cost_estimate"] = jnp.asarray(cost, dtype=jnp.float32)
    return new_state


def pid_update(state, cost, cost_limit, kp, ki, kd, delta_p_ema_alpha, delta_d_ema_alpha, diff_norm, lambda_max):
    cost = jnp.asarray(cost, dtype=jnp.float32)
    delta = cost - cost_limit
    pid_i = jnp.maximum(0.0, state["pid_i"] + ki * delta)
    if diff_norm:
        pid_i = jnp.clip(pid_i, 0.0, 1.0)
    delta_p = delta_p_ema_alpha * state["pid_delta_p"] + (1.0 - delta_p_ema_alpha) * delta
    cost_d = delta_d_ema_alpha * state["pid_cost_d"] + (1.0 - delta_d_ema_alpha) * cost
    pid_d = jnp.maximum(0.0, cost_d - state["pid_cost_ds"][0])
    pid_output = kp * delta_p + pid_i + kd * pid_d
    upper = 1.0 if diff_norm else lambda_max
    new_state = dict(state)
    new_state["lambda"] = jnp.clip(pid_output, 0.0, upper)
    new_state["pid_i"] = pid_i
    new_state["pid_delta_p"] = delta_p
    new_state["pid_cost_d"] = cost_d
    new_state["pid_cost_ds"] = jnp.concatenate([state["pid_cost_ds"][1:], cost_d[None]])
    new_state["cost_estimate"] = cost
    return new_state


def make_lagrange_update(algorithm_config):
    """Returns update(state, cost) -> new_state for the configured rule."""
    lag = algorithm_config.lagrangian
    rule = lag.update_rule
    cost_limit = float(algorithm_config.cost_limit)
    if rule == "gradient":
        lr = float(algorithm_config.cost_lambda_lr)
        lambda_max = float(algorithm_config.cost_lambda_max)
        return lambda state, cost: gradient_update(state, cost, cost_limit, lr, lambda_max)
    if rule == "pid":
        return lambda state, cost: pid_update(
            state, cost, cost_limit,
            kp=float(lag.pid_kp), ki=float(lag.pid_ki), kd=float(lag.pid_kd),
            delta_p_ema_alpha=float(lag.pid_delta_p_ema_alpha),
            delta_d_ema_alpha=float(lag.pid_delta_d_ema_alpha),
            diff_norm=bool(lag.diff_norm),
            lambda_max=float(algorithm_config.cost_lambda_max),
        )
    raise ValueError(f"Unknown lagrangian.update_rule {rule!r}, expected one of {UPDATE_RULES}")


def lagrangian_advantage(reward_advantage, cost_advantage, cost_lambda, scale_objective, diff_norm):
    """Advantage of the penalized objective.

    sum_norm (scale_lagrangian_objective, old default): (A_r - lambda A_c) / (1 + lambda)
    diff_norm (lambda in [0, 1]):                        (1 - lambda) A_r - lambda A_c
    neither:                                             A_r - lambda A_c
    """
    if diff_norm:
        return (1.0 - cost_lambda) * reward_advantage - cost_lambda * cost_advantage
    advantage = reward_advantage - cost_lambda * cost_advantage
    if scale_objective:
        advantage = advantage / (1.0 + cost_lambda)
    return advantage
