"""CPU tests for the safety-critic fixes and the moving-obstacle environment.

Run from the repository root:
    JAX_PLATFORMS=cpu python -m pytest tests/test_safety_critic.py -q
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import config_dict

from rl_x.algorithms.ncbf_ppo.flax_full_jit.ncbf import make_get_safe_action


ACTION_DIM = 3
OBS_DIM = 4


def _make_stub_ncbf_apply(member_offsets, std_logit=0.0):
    """Critic stub: logit_e(x) = w . x + offset_e for each ensemble member e (one safety head)."""
    member_offsets = jnp.asarray(member_offsets, dtype=jnp.float32)
    weights = jnp.linspace(0.2, 0.8, OBS_DIM + ACTION_DIM)

    def ncbf_apply(params, x):
        mean_logits = (jnp.dot(x, weights) + member_offsets)[:, None]  # [E, 1]
        std_logits = jnp.full_like(mean_logits, std_logit)
        eps_shape = (8,) + mean_logits.shape
        details = {
            "mean_logits": mean_logits,
            "std_logits": std_logits,
            "residual_eps_current": jax.random.normal(jax.random.PRNGKey(0), eps_shape),
            "residual_eps_next": jax.random.normal(jax.random.PRNGKey(1), eps_shape),
        }
        return jnp.mean(jax.nn.sigmoid(mean_logits)), jnp.asarray(0.0), details

    return ncbf_apply, weights


def _make_shield(ncbf_apply, **overrides):
    kwargs = dict(
        ncbf_apply=ncbf_apply,
        ncbf_obs_from_obs_idx=jnp.arange(OBS_DIM),
        use_safety_layer=True,
        act_low=-jnp.ones(ACTION_DIM),
        act_high=jnp.ones(ACTION_DIM),
        gamma_c=0.6,
        eta_cbf=0.5,
        lambda_s=1000.0,
        max_delta_u=0.0,
        action_clipping=False,
        safety_layer_projection="soft_slack",
        safety_layer_min_grad_norm=0.0,
        post_check_actual_residual=False,
    )
    kwargs.update(overrides)
    return make_get_safe_action(**kwargs)


def _shield_inputs():
    obs_t = jnp.array([0.1, -0.2, 0.3, 0.0])
    obs_last = jnp.array([0.0, 0.1, 0.2, -0.1])
    u_last = jnp.array([0.1, 0.0, -0.1])
    u0 = jnp.array([-0.5, 0.2, 0.1])
    latent = jnp.zeros(0)
    return u0, obs_t, u_last, obs_last, latent


def test_gamma_c_is_scaled_by_eta_in_the_residual():
    ncbf_apply, weights = _make_stub_ncbf_apply([0.0])
    u0, obs_t, u_last, obs_last, latent = _shield_inputs()
    h_next = jax.nn.sigmoid(jnp.dot(jnp.concatenate([obs_t, u0]), weights))
    h_cur = jax.nn.sigmoid(jnp.dot(jnp.concatenate([obs_last, u_last]), weights))

    for scaled, gamma_term in ((True, 0.5 * 0.6), (False, 0.6)):
        shield = _make_shield(ncbf_apply, gamma_c_scaled_by_eta=scaled)
        _, _, _, diag = shield(u0, obs_t, u_last, obs_last, latent, 0.0, None)
        expected = h_next - 0.5 * h_cur - gamma_term
        np.testing.assert_allclose(diag["residual_mean"], expected, rtol=1e-5, atol=1e-6)


def test_correction_restores_the_linearized_constraint():
    ncbf_apply, _ = _make_stub_ncbf_apply([0.0])
    u0, obs_t, u_last, obs_last, latent = _shield_inputs()
    shield = _make_shield(ncbf_apply, safety_layer_projection="hard_projection", gamma_c=0.95, eta_cbf=1.0)
    u_safe, active, _, diag = shield(u0, obs_t, u_last, obs_last, latent, 0.0, None)
    assert bool(active)
    assert float(diag["post_linearized_margin"]) > -1e-5
    assert float(jnp.linalg.norm(u_safe - u0)) > 0.0


def test_epistemic_std_separates_identical_and_distinct_members():
    u0, obs_t, u_last, obs_last, latent = _shield_inputs()
    same, _ = _make_stub_ncbf_apply([0.0, 0.0, 0.0])
    distinct, _ = _make_stub_ncbf_apply([-1.0, 0.0, 1.0])
    _, _, _, diag_same = _make_shield(same)(u0, obs_t, u_last, obs_last, latent, 1.0, None)
    _, _, _, diag_distinct = _make_shield(distinct)(u0, obs_t, u_last, obs_last, latent, 1.0, None)
    assert float(diag_same["residual_std_epistemic"]) < 1e-3
    assert float(diag_distinct["residual_std_epistemic"]) > 0.05
    # Deterministic heads have no aleatoric spread.
    assert float(diag_distinct["residual_std_aleatoric"]) < 1e-3


def test_uncertainty_scaled_slack_matches_the_closed_form():
    ncbf_apply, _ = _make_stub_ncbf_apply([-1.0, 0.0, 1.0])
    u0, obs_t, u_last, obs_last, latent = _shield_inputs()
    lambda_s = 2.0
    shield = _make_shield(ncbf_apply, gamma_c=0.95, eta_cbf=1.0, lambda_s=lambda_s, slack_uncertainty="epistemic", tighten_uncertainty="none")
    u_safe, active, _, diag = shield(u0, obs_t, u_last, obs_last, latent, 1.0, None)
    assert bool(active)
    sigma = float(diag["slack_sigma"])
    grad_norm = float(diag["constraint_grad_norm"])
    delta = float(diag["constraint_delta"])
    expected_step = delta / (grad_norm ** 2 + sigma ** 2 / lambda_s) * grad_norm
    np.testing.assert_allclose(float(jnp.linalg.norm(u_safe - u0)), expected_step, rtol=1e-4)
    # More uncertainty means a smaller correction (the constraint is relaxed).
    calm, _ = _make_stub_ncbf_apply([0.0, 0.0, 0.0])
    u_calm, _, _, _ = _make_shield(calm, gamma_c=0.95, eta_cbf=1.0, lambda_s=lambda_s, slack_uncertainty="epistemic", tighten_uncertainty="none")(u0, obs_t, u_last, obs_last, latent, 1.0, None)
    assert float(jnp.linalg.norm(u_safe - u0)) < float(jnp.linalg.norm(u_calm - u0)) + 1e-6


def test_tightening_with_beta_increases_the_correction():
    ncbf_apply, _ = _make_stub_ncbf_apply([-1.0, 0.0, 1.0])
    u0, obs_t, u_last, obs_last, latent = _shield_inputs()
    shield = _make_shield(ncbf_apply, gamma_c=0.7, eta_cbf=1.0, tighten_uncertainty="epistemic")
    _, _, delta_relaxed, _ = shield(u0, obs_t, u_last, obs_last, latent, -2.0, None)
    _, _, delta_tight, _ = shield(u0, obs_t, u_last, obs_last, latent, 1.0, None)
    assert float(delta_tight) > float(delta_relaxed)


# ---------------------------------------------------------------------------
# Moving-obstacle environment
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def obstacle_env():
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.default_config import get_config
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.create_env import create_env

    env_config = get_config("ncbf_mujoco.robot_locomotion.mjx")
    env_config.nr_envs = 2
    env_config.device = "cpu"
    env_config.terrain.type = "plane"
    env_config.episode_length_in_seconds = 2
    env_config.obstacles.enabled = True
    env_config.obstacles.nr_obstacles = 2
    config = config_dict.ConfigDict({"environment": env_config, "runner": {"mode": "train"}})
    return create_env(config)


def test_obstacle_observation_layout(obstacle_env):
    env = obstacle_env
    assert env.nr_obstacles == 2
    assert len(env.obstacle_obs_idx) == 10
    assert len(env.obstacle_no_collision_obs_idx) == 1
    assert env.ncbf_target_names[-1] == "collision"
    assert int(env.ncbf_target_indices[-1]) == int(env.obstacle_no_collision_obs_idx[0])
    assert set(np.asarray(env.obstacle_obs_idx)).issubset(set(np.asarray(env.policy_observation_indices)))
    assert set(np.asarray(env.obstacle_obs_idx)).issubset(set(np.asarray(env.ncbf_observation_indices)))


def test_obstacles_move_and_collisions_terminate(obstacle_env):
    env = obstacle_env
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), env.nr_envs), False)
    obs = np.asarray(state.next_observation)
    assert obs.shape[-1] == env.single_observation_space.shape[0]
    # Freshly spawned obstacles are outside the collision radius.
    assert np.all(obs[:, int(env.obstacle_no_collision_obs_idx[0])] == 1.0)
    clearance = np.asarray(state.internal_state["obstacle_min_clearance"])
    assert np.all(clearance > 0.0)

    action = jnp.zeros((env.nr_envs,) + env.single_action_space.shape)
    pos_before = np.asarray(state.internal_state["obstacle_pos"])
    state = env.step(state, action)
    pos_after = np.asarray(state.internal_state["obstacle_pos"])
    assert np.all(np.linalg.norm(pos_after - pos_before, axis=-1) > 0.0)

    # Place obstacle 0 of every env on top of the robot: the next step must report a collision and terminate.
    robot_xy = np.asarray(state.data.qpos[:, :2])
    internal_state = dict(state.internal_state)
    internal_state["obstacle_pos"] = internal_state["obstacle_pos"].at[:, 0, :].set(jnp.asarray(robot_xy))
    internal_state["obstacle_vel"] = internal_state["obstacle_vel"].at[:, 0, :].set(0.0)
    state = state.replace(internal_state=internal_state)
    state = env.step(state, action)
    assert np.all(np.asarray(state.terminated))
    assert np.all(np.asarray(state.info["env_info/obstacle_collision"]) == 1.0)
    final_obs = np.asarray(state.actual_next_observation)
    assert np.all(final_obs[:, int(env.obstacle_no_collision_obs_idx[0])] == 0.0)


# ---------------------------------------------------------------------------
# Range-scan obstacle observation
# ---------------------------------------------------------------------------

def test_range_scan_geometry():
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.obstacle_scan import obstacle_range_scan

    angles = jnp.array([0.0, jnp.pi / 2, jnp.pi])
    origin = jnp.zeros(2)

    def scan(obstacle_pos, obstacle_radius, yaw=0.0, origin=origin):
        return np.asarray(obstacle_range_scan(origin, yaw, angles, jnp.asarray(obstacle_pos, dtype=jnp.float32), jnp.asarray(obstacle_radius, dtype=jnp.float32), 5.0))

    # Straight ahead: hit at distance 2 - 0.5; sideways and behind see nothing.
    np.testing.assert_allclose(scan([[2.0, 0.0]], [0.5]), [1.5, 5.0, 5.0], atol=1e-5)
    # Rays rotate with the yaw.
    np.testing.assert_allclose(scan([[0.0, 2.0]], [0.5], yaw=jnp.pi / 2), [1.5, 5.0, 5.0], atol=1e-5)
    # The nearer obstacle occludes the farther one on the same ray.
    np.testing.assert_allclose(scan([[4.0, 0.0], [2.0, 0.0]], [0.5, 0.5])[0], 1.5, atol=1e-5)
    # A ray passing beside the obstacle misses it; an obstacle beyond max_range is not seen.
    np.testing.assert_allclose(scan([[2.0, 0.6]], [0.5])[0], 5.0, atol=1e-5)
    np.testing.assert_allclose(scan([[7.0, 0.0]], [0.5])[0], 5.0, atol=1e-5)
    # An origin inside an obstacle gives 0 on every ray.
    np.testing.assert_allclose(scan([[0.1, 0.0]], [0.5]), [0.0, 0.0, 0.0], atol=1e-5)


def _make_obstacle_env(**obstacle_overrides):
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.default_config import get_config
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.create_env import create_env

    env_config = get_config("ncbf_mujoco.robot_locomotion.mjx")
    env_config.nr_envs = 2
    env_config.device = "cpu"
    env_config.terrain.type = "plane"
    env_config.episode_length_in_seconds = 2
    env_config.obstacles.enabled = True
    env_config.obstacles.nr_obstacles = 2
    for key, value in obstacle_overrides.items():
        if key.startswith("scan_"):
            env_config.obstacles.scan[key[len("scan_"):]] = value
        else:
            env_config.obstacles[key] = value
    config = config_dict.ConfigDict({"environment": env_config, "runner": {"mode": "train"}})
    return create_env(config)


def _as_set(indices):
    return set(np.asarray(indices).tolist())


@pytest.fixture(scope="module")
def scan_env():
    return _make_obstacle_env(policy_observation="scan", value_observation="state_and_scan", scan_nr_rays=16, scan_noise=0.0)


def test_scan_observation_routing(scan_env):
    env = scan_env
    scan_idx, state_idx = _as_set(env.obstacle_scan_obs_idx), _as_set(env.obstacle_obs_idx)
    assert len(scan_idx) == 16 and len(state_idx) == 10
    policy, ncbf, value = _as_set(env.policy_observation_indices), _as_set(env.ncbf_observation_indices), _as_set(env.critic_observation_indices)
    assert scan_idx <= policy and not (state_idx & policy)
    assert scan_idx <= ncbf and not (state_idx & ncbf)
    assert scan_idx <= value and state_idx <= value
    np.testing.assert_array_equal(np.asarray(env.history_observation_indices), np.asarray(env.ncbf_observation_indices))

    # Policy blind to obstacles, safety critic (and so the history encoder) sees the scan.
    blind = _make_obstacle_env(policy_observation="none", ncbf_observation="scan", scan_nr_rays=16)
    blind_obstacle_idx = _as_set(blind.obstacle_scan_obs_idx) | _as_set(blind.obstacle_obs_idx)
    assert not (blind_obstacle_idx & _as_set(blind.policy_observation_indices))
    assert _as_set(blind.obstacle_scan_obs_idx) <= _as_set(blind.ncbf_observation_indices)
    assert _as_set(blind.obstacle_scan_obs_idx) <= _as_set(blind.history_observation_indices)
    assert len(blind.obstacle_obs_idx) == 0


def test_scan_observation_in_env_and_lifetime(scan_env):
    env = scan_env
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), env.nr_envs), False)
    internal_state = state.internal_state
    obs = np.asarray(state.next_observation)
    assert obs.shape[-1] == env.single_observation_space.shape[0]
    scan = obs[:, np.asarray(env.obstacle_scan_obs_idx)]
    assert np.all((scan >= 0.0) & (scan <= 1.0))

    # Lifetime covers the expected arrival time (distance / speed) and respects the cap.
    distance = np.linalg.norm(np.asarray(internal_state["obstacle_pos"]) - np.asarray(state.data.qpos[:, None, :2]), axis=-1)
    speed = np.linalg.norm(np.asarray(internal_state["obstacle_vel"]), axis=-1)
    lifetime = np.asarray(internal_state["obstacle_lifetime"])
    assert np.all(lifetime <= env.obstacle_config["max_lifetime_seconds"] + 1e-5)
    assert np.all(lifetime >= np.minimum(distance / speed, env.obstacle_config["max_lifetime_seconds"]) - 1e-4)

    # Put obstacle 0 two metres away along the forward-most ray: that ray reads 1 - (2 - r) / max_range.
    env_index = 0
    angles = np.asarray(env.obstacle_scan_angles)
    forward = int(np.argmin(np.abs(angles)))
    behind = int(np.argmin(np.abs(np.abs(angles) - np.pi)))
    ray_heading = float(internal_state["imu_orientation_euler"][env_index, 2]) + angles[forward]
    robot_xy = np.asarray(state.data.qpos[env_index, :2])
    single_internal = jax.tree_util.tree_map(lambda x: x[env_index], internal_state)
    single_internal["obstacle_pos"] = jnp.asarray([robot_xy + 2.0 * np.array([np.cos(ray_heading), np.sin(ray_heading)]), robot_xy + 50.0])
    single_internal["obstacle_radius"] = jnp.asarray([0.25, 0.25])
    single_data = jax.tree_util.tree_map(lambda x: x[env_index], state.data)
    proximity = np.asarray(env.get_obstacle_scan_observation(single_data, single_internal, jax.random.PRNGKey(1)))
    np.testing.assert_allclose(proximity[forward], 1.0 - 1.75 / env.obstacle_scan_max_range, atol=1e-3)
    assert proximity[behind] == 0.0


def test_gru_history_encoder_only_reads_selected_observations():
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.history_encoder import GRU_encoder

    encoder = GRU_encoder(hidden_size=8, observation_indices=jnp.array([0, 2]))
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 4, 5))
    params = encoder.init(jax.random.PRNGKey(1), x)
    z = encoder.apply(params, x)
    assert z.shape == (3, 8)
    # Changing an unselected dimension (e.g. a safety label) must not change the latent.
    np.testing.assert_allclose(encoder.apply(params, x.at[..., 1].set(100.0)), z, atol=1e-6)
    assert not np.allclose(encoder.apply(params, x.at[..., 2].add(1.0)), z)


def test_scan_conv_history_encoder():
    from types import SimpleNamespace
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.history_encoder import get_history_encoder

    # Observation layout: 3 proprioceptive dims, 16 scan rays, 2 dims the encoder must ignore.
    env = SimpleNamespace(history_observation_indices=jnp.arange(19), obstacle_scan_obs_idx=jnp.arange(3, 19), nr_history_steps=4)
    config = config_dict.ConfigDict({"algorithm": {"next_step_predictor": {
        "history_encoder_type": "scan_conv", "history_encoder_hidden_size": 8, "scan_encoder_sectors": 4}}})
    encoder = get_history_encoder(config, env)
    x = jax.random.uniform(jax.random.PRNGKey(0), (5, 4, 21))
    params = encoder.init(jax.random.PRNGKey(1), x)
    z = encoder.apply(params, x)
    assert z.shape == (5, 8)
    np.testing.assert_allclose(encoder.apply(params, x.at[..., 19:].set(7.0)), z, atol=1e-6)
    assert not np.allclose(encoder.apply(params, x.at[:, -1, 3].add(1.0)), z)
    # Extra leading batch dimensions are kept.
    assert encoder.apply(params, x[None]).shape == (1, 5, 8)

    env.obstacle_scan_obs_idx = jnp.zeros(0, dtype=int)
    with pytest.raises(ValueError):
        get_history_encoder(config, env)
