"""T09 (P3O) and T10 (CaT) tests. CPU: JAX_PLATFORMS=cpu python -m pytest tests/test_p3o.py -q"""

import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from ml_collections import config_dict

from rl_x.algorithms.p3o.flax_full_jit import cat
from rl_x.algorithms.p3o.flax_full_jit import penalty


# ----------------------------------------------------------------------------- P3O penalty

def _standardized(n, seed):
    a = np.random.default_rng(seed).normal(size=n).astype(np.float32)
    return jnp.asarray((a - a.mean()) / a.std())


@pytest.mark.parametrize("clip", [True, False])
def test_penalty_is_zero_when_feasible_and_grows_when_violated(clip):
    kappa, eps = 20.0, 0.1
    adv = _standardized(512, 0)
    ratio = jnp.ones_like(adv)  # pi = pi_k: surrogate = mean(A) = 0, so the penalty is kappa * relu(offset)
    offsets = jnp.linspace(-1.0, 1.0, 41)
    values = np.array([float(penalty.penalty(ratio, adv, o, kappa, eps, clip)[0]) for o in offsets])
    np.testing.assert_allclose(values, kappa * np.maximum(np.asarray(offsets), 0.0), atol=1e-4)
    assert np.all(values[offsets < 0] == 0.0)
    assert np.all(np.diff(values[offsets > 0]) > 0)

    # Gradient w.r.t. the policy (here: the log ratio) is exactly zero while feasible and kappa * d surrogate when not.
    log_ratio = jnp.asarray(np.random.default_rng(1).normal(scale=0.02, size=512).astype(np.float32))
    def penalty_of(lr, offset):
        return penalty.penalty(jnp.exp(lr), adv, offset, kappa, eps, clip)[0]
    surrogate = float(jnp.mean(penalty.clipped_cost_surrogate(jnp.exp(log_ratio), adv, eps, clip)))
    feasible = surrogate - 0.5
    violated = -surrogate + 0.5
    assert float(penalty_of(log_ratio, feasible)) == 0.0
    assert float(jnp.abs(jax.grad(penalty_of)(log_ratio, feasible)).max()) == 0.0
    grad_violated = jax.grad(penalty_of)(log_ratio, violated)
    grad_surrogate = jax.grad(lambda lr: jnp.mean(penalty.clipped_cost_surrogate(jnp.exp(lr), adv, eps, clip)))(log_ratio)
    np.testing.assert_allclose(np.asarray(grad_violated), kappa * np.asarray(grad_surrogate), rtol=1e-5, atol=1e-8)
    assert float(penalty_of(log_ratio, violated)) == pytest.approx(kappa * 0.5, rel=1e-5)


def test_cost_surrogate_clipping_matches_the_paper():
    """max(r A, clip(r) A) (eq. 7): cost decreases beyond the trust region earn nothing, increases always count."""
    eps = 0.2
    ratios = jnp.array([0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.6])
    for a in (1.0, -1.0, 0.3, -2.5):
        adv = jnp.full_like(ratios, a)
        got = penalty.clipped_cost_surrogate(ratios, adv, eps)
        expected = np.maximum(np.asarray(ratios) * a, np.clip(np.asarray(ratios), 1 - eps, 1 + eps) * a)
        np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-6)
        # Mirror of PPO's reward clip: max(r A, clip(r) A) = -min(r (-A), clip(r) (-A)).
        ppo = jnp.minimum(ratios * -a, jnp.clip(ratios, 1 - eps, 1 + eps) * -a)
        np.testing.assert_allclose(np.asarray(got), -np.asarray(ppo), rtol=1e-6)
        grads = jax.vmap(jax.grad(lambda r: penalty.clipped_cost_surrogate(r, jnp.asarray(a), eps)))(ratios)
        for r, g in zip(np.asarray(ratios), np.asarray(grads)):
            inside = 1 - eps < r < 1 + eps
            # A > 0 (cost-increasing action): only lowering r below 1 - eps is clipped (no extra credit).
            # A < 0 (cost-decreasing action): only raising r above 1 + eps is clipped.
            clipped = (a > 0 and r < 1 - eps) or (a < 0 and r > 1 + eps)
            assert g == pytest.approx(0.0 if clipped else a), (a, r, g, inside)
    # clip=False is the OmniSafe surrogate r A.
    np.testing.assert_allclose(np.asarray(penalty.clipped_cost_surrogate(ratios, jnp.ones_like(ratios), eps, clip=False)),
                               np.asarray(ratios))


def test_constraint_offset_units():
    j, d, T, gamma = 0.06, 0.01, 800.0, 0.99
    assert float(penalty.constraint_offset(j, d, T, gamma, "per_step")) == pytest.approx((j - d) / T)
    assert float(penalty.constraint_offset(j, d, T, gamma, "one_minus_gamma")) == pytest.approx((1 - gamma) * (j - d))
    assert float(penalty.constraint_offset(j, d, T, gamma, "none")) == pytest.approx(j - d)
    with pytest.raises(ValueError):
        penalty.constraint_offset(j, d, T, gamma, "bogus")


def test_consistent_normalization_keeps_the_activation_decision():
    """N-P3O: relu argument after normalization = unnormalized argument / sigma (unclipped, mean ratio 1)."""
    rng = np.random.default_rng(3)
    raw = jnp.asarray((rng.exponential(0.02, size=1024) - 0.015).astype(np.float32))
    r = np.exp(rng.normal(scale=0.05, size=1024)).astype(np.float32)
    ratio = jnp.asarray(r / r.mean())
    sigma = float(jnp.std(raw)) + 1e-8
    for offset in (-3e-3, -1e-4, 0.0, 1e-4, 3e-3):
        unnormalized = float(jnp.mean(ratio * raw)) + offset
        adv, off = penalty.normalize_cost_advantages(raw, offset, "consistent")
        normalized = float(penalty.penalty(ratio, adv, off, 1.0, 0.1, clip=False)[1])
        assert normalized == pytest.approx(unnormalized / sigma, rel=1e-3, abs=1e-5)
    adv, off = penalty.normalize_cost_advantages(raw, 1e-3, "standardize")
    assert off == 1e-3 and float(jnp.std(adv)) == pytest.approx(1.0, rel=1e-4)
    adv, off = penalty.normalize_cost_advantages(raw, 1e-3, "center")
    assert float(jnp.mean(adv)) == pytest.approx(0.0, abs=1e-7) and float(jnp.std(adv)) == pytest.approx(sigma, rel=1e-4)


# ----------------------------------------------------------------------------- CaT

def test_cat_binary_collision_gives_p_max_and_scales_the_reward():
    state = cat.init_cat_state()
    collision = jnp.array([0.0, 1.0, 0.0, 1.0])
    state, delta = cat.step_probabilities(state, collision, tau=0.95, p_max=1.0)
    np.testing.assert_allclose(np.asarray(delta), [0.0, 1.0, 0.0, 1.0])
    state, delta = cat.step_probabilities(state, collision, tau=0.95, p_max=0.3)
    np.testing.assert_allclose(np.asarray(delta), [0.0, 0.3, 0.0, 0.3], rtol=1e-6)
    rewards = jnp.array([1.0, 2.0, 0.5, 0.0])
    terminated = jnp.array([True, False, False, False])
    new_rewards, terms = cat.apply(rewards, terminated, delta)
    np.testing.assert_allclose(np.asarray(new_rewards), [1.0, 1.4, 0.5, 0.0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(terms), [1.0, 0.3, 0.0, 0.3], rtol=1e-6)
    # Negative rewards are clipped at 0 (reference code), unless switched off.
    np.testing.assert_allclose(np.asarray(cat.apply(jnp.array([-1.0]), jnp.array([False]), jnp.array([0.5]))[0]), [0.0])
    np.testing.assert_allclose(np.asarray(cat.apply(jnp.array([-1.0]), jnp.array([False]), jnp.array([0.5]), False)[0]), [-0.5])


def test_cat_graded_violation_and_running_max():
    tau = 0.9
    state = cat.init_cat_state()
    state, delta = cat.step_probabilities(state, jnp.array([0.0, 0.1, 0.2]), tau, 1.0)
    assert float(state["running_max"]) == pytest.approx(0.2)  # first step: initialized to the batch max
    np.testing.assert_allclose(np.asarray(delta), [0.0, 0.5, 1.0], rtol=1e-5)
    state, delta = cat.step_probabilities(state, jnp.array([0.0, 0.05, 0.4]), tau, 0.5)
    expected_max = tau * 0.2 + (1 - tau) * 0.4
    assert float(state["running_max"]) == pytest.approx(expected_max)
    np.testing.assert_allclose(np.asarray(delta), [0.0, 0.5 * 0.05 / expected_max, 0.5], rtol=1e-5)
    # rollout_probabilities == step-by-step
    seq = jnp.asarray(np.random.default_rng(0).uniform(-0.1, 0.3, size=(6, 5)).astype(np.float32))
    s, deltas = cat.rollout_probabilities(cat.init_cat_state(), seq, tau, 1.0)
    s2, rows = cat.init_cat_state(), []
    for row in seq:
        s2, d = cat.step_probabilities(s2, row, tau, 1.0)
        rows.append(np.asarray(d))
    np.testing.assert_allclose(np.asarray(deltas), np.stack(rows), rtol=1e-6)
    assert float(s["running_max"]) == pytest.approx(float(s2["running_max"]))
    assert np.all(np.asarray(deltas)[np.asarray(seq) <= 0] == 0.0)


# ----------------------------------------------------------------------------- end-to-end on Go2

def _env(nr_envs=16, obstacles=None):
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.default_config import get_config
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.create_env import create_env

    env_config = get_config("ncbf_mujoco.robot_locomotion.mjx")
    env_config.nr_envs = nr_envs
    env_config.device = "cpu"
    env_config.train_robot = "unitree_go2"
    env_config.terrain.type = "plane"
    env_config.episode_length_in_seconds = 1
    if obstacles is not None:
        env_config.obstacles.enabled = True
        for key, value in obstacles.items():
            env_config.obstacles[key] = value
    config = config_dict.ConfigDict({"environment": env_config, "runner": {"mode": "train"}})
    return create_env(config), env_config


@pytest.fixture(scope="module")
def go2_env():
    return _env()


@pytest.fixture(scope="module")
def go2_obstacle_env():
    # Obstacles spawn 0.2 m from the base, so the base is inside (radius + robot_radius) and collides at once.
    return _env(obstacles={"terminate_on_collision": False, "spawn_distance_range": (0.2, 0.2), "speed_range": (0.5, 0.5)})


def _algorithm_config(**overrides):
    from rl_x.algorithms.p3o.flax_full_jit.default_config import get_config
    config = get_config("p3o.flax_full_jit")
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


def _train(env, env_config, **overrides):
    from rl_x.algorithms.p3o.flax_full_jit.p3o import P3O
    config = config_dict.ConfigDict({
        "algorithm": _algorithm_config(**overrides),
        "environment": env_config,
        "runner": {"mode": "train", "save_model": False, "track_console": False, "track_tb": False, "track_wandb": False},
    })
    model = P3O(config, env, "/tmp/unused", None)
    logged = {}
    model.log = lambda key, value, step: logged.setdefault(key, []).append(float(value))
    model.train()
    return model, logged


@pytest.mark.parametrize(
    "overrides",
    [
        {"cost_limit": 0.01},
        {"cost_limit": 0.05, "cost_measure": "rollout", "p3o.cost_advantage_normalization": "center",
         "p3o.constraint_normalization": "none", "p3o.clip_cost_surrogate": False},
    ],
    ids=["p3o_default", "p3o_omnisafe_style"],
)
def test_end_to_end_training_has_no_nan(go2_env, overrides):
    env, env_config = go2_env
    _, logged = _train(env, env_config, **overrides)
    assert len(logged["steps/nr_env_steps"]) == 2
    bad = {key: values for key, values in logged.items() if not all(math.isfinite(v) for v in values)}
    assert not bad, f"non-finite metrics: {bad}"
    for key in ("loss/p3o_penalty_loss", "p3o/surrogate_violation", "p3o/penalty_active", "p3o/constraint_offset",
                "constraint/measured_cost"):
        assert key in logged
    assert "cat/termination_prob" not in logged


def test_p3o_disabled_is_plain_ppo_loss(go2_env):
    env, env_config = go2_env
    _, logged = _train(env, env_config, **{"p3o.enabled": False, "cost_limit": 0.0})
    assert all(v == 0.0 for v in logged["loss/p3o_penalty_loss"])
    assert all(v == 0.0 for v in logged["p3o/kappa"])


def test_checkpoint_loads_into_the_eval_sweep(go2_env, tmp_path):
    """Save after 1 update, load with P3O.load and run a 1-point mode-A eval sweep (policy only), as T08."""
    from rl_x.algorithms.p3o.flax_full_jit.p3o import P3O

    env, env_config = go2_env
    algorithm_config = _algorithm_config(total_timesteps=8 * 16, cost_limit=0.05, **{"p3o.kappa": 7.0})
    runner = {"mode": "train", "save_model": True, "track_console": False, "track_tb": False, "track_wandb": False, "load_model": ""}
    config = config_dict.ConfigDict({"algorithm": algorithm_config, "environment": env_config, "runner": runner})
    run_path = tmp_path / "run"
    model = P3O(config, env, str(run_path), None)
    model.log = lambda *a: None
    model.train()
    checkpoint = run_path / "models" / "latest.model"
    assert checkpoint.exists()

    load_config = config_dict.ConfigDict({"algorithm": _algorithm_config(), "environment": env_config,
                                          "runner": dict(runner, mode="test", save_model=False, load_model=str(checkpoint))})
    sweep = load_config.algorithm.eval_sweep
    sweep.enabled = True
    sweep.out_dir = str(tmp_path / "sweep")
    sweep.perturbations = "action_noise"
    sweep.action_noise_sampling_probs = "0.5"
    sweep.save_episodes = False
    loaded = P3O.load(load_config, env, str(tmp_path / "eval_run"), None, {"algorithm.cost_limit"})
    # Nested config restored from the checkpoint; explicitly set keys are kept.
    assert loaded.config.algorithm.p3o.kappa == 7.0 and loaded.kappa == 7.0
    assert loaded.config.algorithm.cost_limit == 0.0
    fresh = P3O(config_dict.ConfigDict(dict(load_config.to_dict(), runner=dict(runner, save_model=False))), env, str(tmp_path / "fresh"), None)
    assert not np.allclose(np.asarray(ravel_pytree(loaded.policy_state.params)[0]),
                           np.asarray(ravel_pytree(fresh.policy_state.params)[0])), "checkpoint params were not restored"
    assert float(loaded.p3o_state["episode_length_estimate"]) > 0
    summaries = loaded.test(1)
    assert len(summaries) == 1
    summary_files = list((tmp_path / "sweep").rglob("eval_summary.json"))
    assert len(summary_files) == 1
    summary = json.loads(summary_files[0].read_text())
    assert summary["deployment_mode"] == "A" and summary["algorithm"] == "p3o.flax_full_jit"
    assert 0.0 <= summary["success_rate"] <= 1.0


# ----------------------------------------------------------------------------- CaT on the obstacle env

def test_cat_on_the_obstacle_env(go2_obstacle_env):
    """Real env: a collision does not terminate (terminate_on_collision=False), CaT turns it into delta = p_max."""
    env, _ = go2_obstacle_env
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), env.nr_envs), False)
    action = jnp.zeros((env.nr_envs,) + env.single_action_space.shape)
    infos, rewards, terminated = [], [], []
    for _ in range(3):
        state = env.step(state, action)
        infos.append({k: np.asarray(state.info[k]) for k in ("env_info/obstacle_collision", "env_info/obstacle_min_clearance", "constraint/cost")})
        rewards.append(np.asarray(state.reward))
        terminated.append(np.asarray(state.terminated))
    stacked = {k: jnp.asarray(np.stack([i[k] for i in infos])) for k in infos[0]}
    collision = np.asarray(stacked["env_info/obstacle_collision"])
    assert collision.mean() > 0.5, "obstacles spawned inside the collision radius should collide"
    np.testing.assert_array_equal(collision > 0, np.asarray(stacked["env_info/obstacle_min_clearance"]) < 0)
    # Collisions are no longer terminations or costs; falls would still be.
    assert np.all(np.asarray(stacked["constraint/cost"])[collision > 0] == 0.0)

    c = cat.constraint_values("collision", stacked, None)
    _, delta = cat.rollout_probabilities(cat.init_cat_state(), c, 0.95, 1.0)
    np.testing.assert_allclose(np.asarray(delta), collision)
    new_rewards, terms = cat.apply(jnp.asarray(np.stack(rewards)), jnp.asarray(np.stack(terminated)), delta)
    assert np.all(np.asarray(new_rewards)[collision > 0] == 0.0)
    np.testing.assert_allclose(np.asarray(new_rewards)[collision == 0], np.maximum(np.stack(rewards)[collision == 0], 0.0), rtol=1e-6)
    assert np.all(np.asarray(terms)[collision > 0] == 1.0)

    pen = cat.constraint_values("penetration", stacked, None)
    _, delta_pen = cat.rollout_probabilities(cat.init_cat_state(), pen, 0.95, 1.0)
    delta_pen = np.asarray(delta_pen)
    assert np.all(delta_pen[collision == 0] == 0.0) and np.all((delta_pen[collision > 0] > 0) & (delta_pen[collision > 0] <= 1))


def test_ppo_cat_training_on_the_obstacle_env(go2_obstacle_env):
    env, env_config = go2_obstacle_env
    _, logged = _train(env, env_config, **{"p3o.enabled": False, "cat.enabled": True})
    bad = {key: values for key, values in logged.items() if not all(math.isfinite(v) for v in values)}
    assert not bad, f"non-finite metrics: {bad}"
    assert min(logged["cat/violation_rate"]) > 0.0
    assert logged["cat/termination_prob_when_violated"][0] == pytest.approx(1.0)
    assert max(logged["cat/reward_scale"]) < 1.0


def test_cat_refuses_an_env_that_already_terminates_on_collision():
    from rl_x.algorithms.p3o.flax_full_jit.p3o import P3O
    env, env_config = _env(nr_envs=2, obstacles={"terminate_on_collision": True})
    config = config_dict.ConfigDict({
        "algorithm": _algorithm_config(**{"cat.enabled": True}),
        "environment": env_config,
        "runner": {"mode": "train", "save_model": False, "track_console": False, "track_tb": False, "track_wandb": False},
    })
    with pytest.raises(ValueError, match="terminate_on_collision"):
        P3O(config, env, "/tmp/unused", None)
