"""CPU tests for the constrained baselines (T07): PPO-Lagrangian (gradient / PID) and CPO.

Run: JAX_PLATFORMS=cpu python -m pytest tests/test_constrained_baselines.py -q
"""

import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.linen.initializers import constant
from jax.flatten_util import ravel_pytree
from ml_collections import config_dict

from rl_x.algorithms.ppo_lagrangian.flax_full_jit import lagrange
from rl_x.algorithms.cpo.flax_full_jit import cpo_core


# ----------------------------------------------------------------------------- Lagrange multiplier

def _lagrangian_config(update_rule, cost_limit, **pid):
    from rl_x.algorithms.ppo_lagrangian.flax_full_jit.default_config import get_config

    config = get_config("ppo_lagrangian.flax_full_jit")
    config.cost_limit = cost_limit
    config.lagrangian.update_rule = update_rule
    for key, value in pid.items():
        config.lagrangian[key] = value
    return config


def _run_lagrange(config, costs):
    update = jax.jit(lagrange.make_lagrange_update(config))
    state = lagrange.init_lagrange_state(config.cost_lambda_init, config.lagrangian.pid_d_delay)
    lambdas = []
    for cost in costs:
        state = update(state, jnp.float32(cost))
        lambdas.append(float(state["lambda"]))
    return np.array(lambdas)


@pytest.mark.parametrize(
    "update_rule, pid",
    [
        ("gradient", {}),
        ("pid", {}),  # OmniSafe / Stooke et al. default gains
        ("pid", {"pid_kp": 1.0, "pid_ki": 0.1, "pid_kd": 0.5}),
    ],
)
def test_lambda_increases_when_violated_and_decays_to_zero_otherwise(update_rule, pid):
    cost_limit = 0.05
    config = _lagrangian_config(update_rule, cost_limit, **pid)
    violated = _run_lagrange(config, [0.5] * 100)
    assert violated[-1] > violated[0] > 0.0
    assert np.all(np.diff(violated) >= -1e-7), "lambda must not decrease while the constraint is violated"

    # Continue from the violated state with a satisfied constraint (cost 0 < limit).
    satisfied = _run_lagrange(config, [0.5] * 100 + [0.0] * 3000)[100:]
    assert satisfied[-1] < 0.05 * violated[-1] + 1e-6
    assert satisfied[-1] == pytest.approx(0.0, abs=1e-3)
    # Decreasing trend (the PID D term only acts on cost increases, so no overshoot upwards).
    assert np.all(np.diff(satisfied[20:]) <= 1e-7)


def test_pid_terms_match_the_reference_recursion():
    config = _lagrangian_config("pid", 0.1, pid_kp=0.3, pid_ki=0.2, pid_kd=0.4, pid_d_delay=3,
                                pid_delta_p_ema_alpha=0.5, pid_delta_d_ema_alpha=0.6)
    costs = [0.5, 0.0, 0.3, 0.9, 0.2, 0.0, 0.7]
    lambdas = _run_lagrange(config, costs)

    # Plain-python port of the rlpyt / OmniSafe PIDLagrangian.
    from collections import deque
    pid_i, delta_p, cost_d = 0.0, 0.0, 0.0
    cost_ds = deque([0.0], maxlen=3)
    expected = []
    for cost in costs:
        delta = cost - 0.1
        pid_i = max(0.0, pid_i + delta * 0.2)
        delta_p = 0.5 * delta_p + 0.5 * delta
        cost_d = 0.6 * cost_d + 0.4 * cost
        pid_d = max(0.0, cost_d - cost_ds[0])
        expected.append(min(max(0.0, 0.3 * delta_p + pid_i + 0.4 * pid_d), 100.0))
        cost_ds.append(cost_d)
    np.testing.assert_allclose(lambdas, expected, rtol=1e-5, atol=1e-6)


def test_episode_cost_measure():
    costs = jnp.zeros((8, 4)).at[3, 0].set(1.0).at[5, 2].set(1.0)
    dones = jnp.zeros((8, 4)).at[3, 0].set(1.0).at[5, 2].set(1.0).at[7, 1].set(1.0).at[6, 3].set(1.0)
    estimate, nr_finished = lagrange.measure_cost("episode", costs, dones, jnp.float32(0.3))
    assert float(nr_finished) == 4.0
    assert float(estimate) == pytest.approx(0.5)  # 2 failures in 4 episodes
    estimate, _ = lagrange.measure_cost("episode", costs * 0, dones * 0, jnp.float32(0.3))
    assert float(estimate) == pytest.approx(0.3)  # no finished episode -> keep the last estimate
    rollout_cost, _ = lagrange.measure_cost("rollout", costs, dones, jnp.float32(0.0))
    assert float(rollout_cost) == pytest.approx(0.5)  # mean over envs of the summed cost


# ----------------------------------------------------------------------------- CPO math

def test_conjugate_gradient_solves_random_spd_system():
    rng = np.random.default_rng(0)
    n = 30
    m = rng.normal(size=(n, n))
    a = jnp.asarray(m @ m.T / n + 0.1 * np.eye(n), dtype=jnp.float32)
    b = jnp.asarray(rng.normal(size=n), dtype=jnp.float32)
    x = cpo_core.conjugate_gradient(lambda v: a @ v, b, nr_iterations=n * 3)
    np.testing.assert_allclose(np.asarray(a @ x), np.asarray(b), atol=1e-3)
    np.testing.assert_allclose(np.asarray(x), np.linalg.solve(np.asarray(a, np.float64), np.asarray(b, np.float64)), rtol=1e-2, atol=1e-3)


class TinyPolicy(nn.Module):
    action_dim: int = 2

    @nn.compact
    def __call__(self, x):
        h = jnp.tanh(nn.Dense(5)(x))
        mean = nn.Dense(self.action_dim)(h)
        logstd = self.param("policy_logstd", constant(-0.3), (1, self.action_dim))
        return mean, logstd


def _tiny_policy_and_batch(seed=0, nr_samples=64, obs_dim=3):
    policy = TinyPolicy()
    key_init, key_obs, key_act, key_adv, key_cadv = jax.random.split(jax.random.PRNGKey(seed), 5)
    observations = jax.random.normal(key_obs, (nr_samples, obs_dim))
    params = policy.init(key_init, observations)
    mean, logstd = policy.apply(params, observations)
    actions = mean + jnp.exp(logstd) * jax.random.normal(key_act, mean.shape)
    log_probs = cpo_core.gaussian_log_prob(mean, logstd, actions)
    reward_advantages = jax.random.normal(key_adv, (nr_samples,))
    cost_advantages = jax.random.normal(key_cadv, (nr_samples,))
    return policy, params, observations, actions, log_probs, reward_advantages, cost_advantages


def test_fisher_vector_product_matches_explicit_fisher():
    policy, params, observations, *_ = _tiny_policy_and_batch()
    theta, unravel = ravel_pytree(params)
    damping = 0.1
    fvp = cpo_core.make_fisher_vector_product(policy.apply, unravel, theta, observations, damping)
    fvp_matrix = jax.vmap(fvp)(jnp.eye(theta.size)).T - damping * jnp.eye(theta.size)

    # Explicit Fisher of a diagonal Gaussian with state-independent log-std:
    # F = E_s[J_mu^T diag(1/sigma^2) J_mu + 2 J_logstd^T J_logstd].
    def mean_fn(t, obs):
        return policy.apply(unravel(t), obs[None])[0][0]

    def logstd_fn(t):
        return policy.apply(unravel(t), observations[:1])[1][0]

    jac_mean = jax.vmap(jax.jacobian(mean_fn), in_axes=(None, 0))(theta, observations)  # (N, A, P)
    _, logstd = policy.apply(params, observations[:1])
    inv_var = jnp.exp(-2.0 * logstd[0])
    fisher_mean = jnp.mean(jnp.einsum("nap,a,naq->npq", jac_mean, inv_var, jac_mean), axis=0)
    jac_logstd = jax.jacobian(logstd_fn)(theta)  # (A, P)
    fisher = fisher_mean + 2.0 * jac_logstd.T @ jac_logstd

    np.testing.assert_allclose(np.asarray(fvp_matrix), np.asarray(fisher), atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("cost_violation", [-1.0, -1e-3, 0.0, 1e-3])
def test_cpo_step_respects_the_trust_region(cost_violation):
    policy, params, observations, actions, log_probs, adv_r, adv_c = _tiny_policy_and_batch(seed=1)
    target_kl = 0.01
    new_params, metrics = cpo_core.cpo_policy_update(
        policy.apply, params, observations, actions, log_probs, adv_r, adv_c - adv_c.mean(),
        jnp.float32(cost_violation), target_kl=target_kl, cg_iterations=15, cg_damping=0.1,
        line_search_steps=15, line_search_decay=0.8,
    )
    mean_old, logstd_old = policy.apply(params, observations)
    mean_new, logstd_new = policy.apply(new_params, observations)
    kl = float(jnp.mean(cpo_core.gaussian_kl(mean_old, logstd_old, mean_new, logstd_new)))
    assert float(metrics["cpo/accepted"]) == 1.0
    assert kl <= target_kl * 1.001
    assert kl == pytest.approx(float(metrics["cpo/kl"]), rel=1e-3, abs=1e-7)
    assert kl > 0.0
    assert np.all(np.isfinite(np.asarray(ravel_pytree(new_params)[0])))


def test_cpo_enters_recovery_when_infeasible():
    policy, params, observations, actions, log_probs, adv_r, adv_c = _tiny_policy_and_batch(seed=2)
    adv_c = adv_c - adv_c.mean()
    target_kl = 0.01
    # Far outside the constraint: the linearized feasible set does not intersect the trust region (B < 0).
    new_params, metrics = cpo_core.cpo_policy_update(
        policy.apply, params, observations, actions, log_probs, adv_r, adv_c,
        jnp.float32(10.0), target_kl=target_kl, cg_iterations=15, cg_damping=0.1,
        line_search_steps=15, line_search_decay=0.8,
    )
    assert int(metrics["cpo/optim_case"]) == 0
    assert float(metrics["cpo/B"]) < 0.0
    assert float(metrics["cpo/lambda_star"]) == 0.0

    # The recovery step only reduces the cost surrogate and is the direction -F^-1 b.
    theta, unravel = ravel_pytree(params)
    def surrogates(p):
        mean, logstd = policy.apply(p, observations)
        ratio = jnp.exp(cpo_core.gaussian_log_prob(mean, logstd, actions) - log_probs)
        return jnp.mean(ratio * adv_r), jnp.mean(ratio * adv_c)
    _, cost_old = surrogates(params)
    _, cost_new = surrogates(new_params)
    assert float(cost_new) < float(cost_old)

    b = jax.grad(lambda t: surrogates(unravel(t))[1])(theta)
    fvp = cpo_core.make_fisher_vector_product(policy.apply, unravel, theta, observations, 0.1)
    p = cpo_core.conjugate_gradient(fvp, b, 15)
    step = ravel_pytree(new_params)[0] - theta
    cosine = float(jnp.dot(step, -p) / (jnp.linalg.norm(step) * jnp.linalg.norm(p)))
    assert cosine > 0.999


def test_cpo_dual_cases():
    rng = np.random.default_rng(3)
    n = 6
    g = jnp.asarray(rng.normal(size=n), dtype=jnp.float32)
    b = jnp.asarray(rng.normal(size=n), dtype=jnp.float32)
    # Identity Fisher: x = g, p = b.
    def case(c):
        _, optim_case, info = cpo_core.cpo_step_direction(g, b, g, b, jnp.dot(g, g), jnp.float32(c), 0.01)
        return int(optim_case), info
    assert case(-10.0)[0] == 3  # whole trust region feasible -> TRPO
    assert case(-1e-3)[0] == 2  # feasible, boundary intersects the trust region
    assert case(1e-3)[0] == 1  # infeasible but recoverable
    assert case(10.0)[0] == 0  # infeasible -> recovery
    _, optim_case, _ = cpo_core.cpo_step_direction(g, b * 1e-5, g, b * 1e-5, jnp.dot(g, g), jnp.float32(-1e-3), 0.01)
    assert int(optim_case) == 4  # no cost gradient and feasible -> TRPO

    # Case 1/2 step satisfies the linearized constraint c + b^T step <= 0 and the quadratic trust region.
    for c in (-1e-3, 1e-3):
        step, _, _ = cpo_core.cpo_step_direction(g, b, g, b, jnp.dot(g, g), jnp.float32(c), 0.01)
        assert float(c + jnp.dot(b, step)) <= 1e-5
        assert float(0.5 * jnp.dot(step, step)) <= 0.01 * 1.01


def test_cpo_dual_matches_numerical_qp():
    """Cases 1/2: the analytic dual step solves max g^T x s.t. c + b^T x <= 0, 0.5 x^T H x <= delta."""
    optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(4)
    n, delta, checked = 6, 0.01, 0
    for _ in range(60):
        m = rng.normal(size=(n, n))
        h = m @ m.T / n + 0.5 * np.eye(n)
        g = rng.normal(size=n)
        b = rng.normal(size=n) + rng.uniform(-2, 2) * g
        c = rng.uniform(-0.5, 0.5)
        h_inv = np.linalg.inv(h)
        x, p = h_inv @ g, h_inv @ b
        step, optim_case, _ = cpo_core.cpo_step_direction(
            jnp.asarray(g, jnp.float32), jnp.asarray(b, jnp.float32), jnp.asarray(x, jnp.float32),
            jnp.asarray(p, jnp.float32), jnp.float32(g @ x), jnp.float32(c), delta)
        if int(optim_case) not in (1, 2):
            continue
        result = optimize.minimize(
            lambda z: -g @ z, np.zeros(n), method="SLSQP",
            constraints=[{"type": "ineq", "fun": lambda z: -(c + b @ z)},
                         {"type": "ineq", "fun": lambda z: delta - 0.5 * z @ h @ z}])
        if not result.success:
            continue
        checked += 1
        np.testing.assert_allclose(np.asarray(step), result.x, atol=2e-3 * max(1.0, np.linalg.norm(result.x)))
    assert checked >= 20


# ----------------------------------------------------------------------------- end-to-end on Go2

@pytest.fixture(scope="module")
def go2_env():
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.default_config import get_config
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.create_env import create_env

    env_config = get_config("ncbf_mujoco.robot_locomotion.mjx")
    env_config.nr_envs = 16
    env_config.device = "cpu"
    env_config.train_robot = "unitree_go2"
    env_config.terrain.type = "plane"
    env_config.episode_length_in_seconds = 1
    config = config_dict.ConfigDict({"environment": env_config, "runner": {"mode": "train"}})
    return create_env(config), env_config


def _algorithm_config(name, **overrides):
    if name == "cpo":
        from rl_x.algorithms.cpo.flax_full_jit.default_config import get_config
    else:
        from rl_x.algorithms.ppo_lagrangian.flax_full_jit.default_config import get_config
    config = get_config(f"{name}.flax_full_jit")
    config.device = "cpu"
    config.nr_steps = 8
    config.minibatch_size = 32
    config.nr_epochs = 2
    config.total_timesteps = 2 * 8 * 16  # 2 updates with 16 envs
    config.evaluation_and_save_frequency = -1
    config.evaluation_active = False
    for key, value in overrides.items():
        if "." in key:
            outer, inner = key.split(".")
            config[outer][inner] = value
        else:
            config[key] = value
    return config


@pytest.mark.parametrize(
    "name, overrides",
    [
        ("ppo_lagrangian", {}),
        ("ppo_lagrangian", {"lagrangian.update_rule": "pid", "lagrangian.cost_measure": "episode", "cost_limit": 0.01}),
        ("cpo", {"cost_limit": 0.01}),
    ],
    ids=["ppo_lag_gradient", "ppo_lag_pid", "cpo"],
)
def test_end_to_end_training_has_no_nan(go2_env, name, overrides):
    if name == "cpo":
        from rl_x.algorithms.cpo.flax_full_jit.cpo import CPO as Algorithm
    else:
        from rl_x.algorithms.ppo_lagrangian.flax_full_jit.ppo_lagrangian import PPOLagrangian as Algorithm

    env, env_config = go2_env
    config = config_dict.ConfigDict({
        "algorithm": _algorithm_config(name, **overrides),
        "environment": env_config,
        "runner": {"mode": "train", "save_model": False, "track_console": False, "track_tb": False, "track_wandb": False},
    })
    model = Algorithm(config, env, "/tmp/unused", None)
    logged = {}
    model.log = lambda key, value, step: logged.setdefault(key, []).append(float(value))
    model.train()

    assert len(logged["steps/nr_env_steps"]) == 2
    bad = {key: values for key, values in logged.items() if not all(math.isfinite(v) for v in values)}
    assert not bad, f"non-finite metrics: {bad}"
    assert "constraint/lambda" in logged or "cpo/optim_case" in logged


@pytest.mark.parametrize("name", ["ppo_lagrangian", "cpo"])
def test_checkpoint_loads_into_the_eval_sweep(go2_env, tmp_path, name):
    """Save after 1 update, load with Algorithm.load and run a 1-point mode-A eval sweep (policy only)."""
    import json as _json
    if name == "cpo":
        from rl_x.algorithms.cpo.flax_full_jit.cpo import CPO as Algorithm
    else:
        from rl_x.algorithms.ppo_lagrangian.flax_full_jit.ppo_lagrangian import PPOLagrangian as Algorithm

    env, env_config = go2_env
    algorithm_config = _algorithm_config(name, total_timesteps=8 * 16)
    runner = {"mode": "train", "save_model": True, "track_console": False, "track_tb": False, "track_wandb": False, "load_model": ""}
    config = config_dict.ConfigDict({"algorithm": algorithm_config, "environment": env_config, "runner": runner})
    run_path = tmp_path / "run"
    model = Algorithm(config, env, str(run_path), None)
    model.log = lambda *a: None
    model.train()
    checkpoint = run_path / "models" / "latest.model"
    assert checkpoint.exists()

    load_config = config_dict.ConfigDict({"algorithm": _algorithm_config(name), "environment": env_config,
                                          "runner": dict(runner, mode="test", save_model=False, load_model=str(checkpoint))})
    sweep = load_config.algorithm.eval_sweep
    sweep.enabled = True
    sweep.out_dir = str(tmp_path / "sweep")
    sweep.perturbations = "action_noise"
    sweep.action_noise_sampling_probs = "0.5"
    sweep.save_episodes = False
    loaded = Algorithm.load(load_config, env, str(tmp_path / "eval_run"), None, set())
    fresh = Algorithm(config_dict.ConfigDict(dict(load_config.to_dict(), runner=dict(runner, save_model=False))), env, str(tmp_path / "fresh"), None)
    assert not np.allclose(np.asarray(ravel_pytree(loaded.policy_state.params)[0]),
                           np.asarray(ravel_pytree(fresh.policy_state.params)[0])), "checkpoint params were not restored"
    summaries = loaded.test(1)
    assert len(summaries) == 1
    summary_files = list((tmp_path / "sweep").rglob("eval_summary.json"))
    assert len(summary_files) == 1
    summary = _json.loads(summary_files[0].read_text())
    assert summary["deployment_mode"] == "A" and summary["algorithm"] == f"{name}.flax_full_jit"
    assert 0.0 <= summary["success_rate"] <= 1.0
