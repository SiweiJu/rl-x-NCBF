"""F13: environment.forward_after_state_writes runs mjx.forward after the reset writes and after the post-physics
DR / shock writes, so the IMU part of the observation matches the physics state.

    JAX_PLATFORMS=cpu python -m pytest tests/test_f13_forward.py -q
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import config_dict
from mujoco import mjx

NR_ENVS = 4


def _make_env(forward_after_state_writes):
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.create_env import create_env
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.default_config import get_config

    env_config = get_config("ncbf_mujoco.robot_locomotion.mjx")
    env_config.nr_envs = NR_ENVS
    env_config.device = "cpu"
    env_config.terrain.type = "plane"
    env_config.forward_after_state_writes = forward_after_state_writes
    env_config.domain_randomization.observation_noise.type = "none"  # compare observations exactly
    env_config.domain_randomization.perturbation.sampling_probability = 1.0  # a shock on every step
    config = config_dict.ConfigDict({"environment": env_config, "runner": {"mode": "train"}})
    return create_env(config)


def _imu_blocks_from_data(env, data):
    """(lin vel obs, ang vel obs, gravity obs) as get_observation builds them, from the given (batched) data."""
    lin = data.sensordata[:, env.imu_linear_velocity_sensor_adr:env.imu_linear_velocity_sensor_adr + env.imu_linear_velocity_sensor_dim]
    ang = data.sensordata[:, env.imu_angular_velocity_sensor_adr:env.imu_angular_velocity_sensor_adr + env.imu_angular_velocity_sensor_dim]
    rot = data.site_xmat[:, env.imu_site_id].reshape(-1, 3, 3)
    gravity = jnp.einsum("nji,j->ni", rot, jnp.array([0.0, 0.0, -1.0]))  # R^T g
    return np.clip(np.asarray(lin) / 10.0, -1, 1), np.clip(np.asarray(ang) / 50.0, -1, 1), np.asarray(gravity)


def _imu_blocks_from_obs(env, obs):
    obs = np.asarray(obs)
    return obs[:, np.asarray(env.imu_linear_vel_obs_idx)], obs[:, np.asarray(env.imu_angular_vel_obs_idx)], obs[:, np.asarray(env.gravity_vector_obs_idx)]


def _max_mismatch(env, state):
    fresh = jax.vmap(mjx.forward)(state.mjx_model, state.data)
    return max(float(np.max(np.abs(a - b))) for a, b in zip(_imu_blocks_from_obs(env, state.next_observation), _imu_blocks_from_data(env, fresh)))


@pytest.fixture(scope="module", params=[False, True], ids=["off", "on"])
def env_states(request):
    env = _make_env(request.param)
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), NR_ENVS), True)  # eval mode: curriculum 1, shocks active
    reset_mismatch = _max_mismatch(env, state)
    step_mismatches = []
    for _ in range(3):
        state = env.step(state, jnp.zeros((NR_ENVS, env.nr_actuator_joints)))
        step_mismatches.append(_max_mismatch(env, state))
    return request.param, reset_mismatch, step_mismatches


def test_first_observation_and_post_shock_observation_match_a_fresh_forward(env_states):
    enabled, reset_mismatch, step_mismatches = env_states
    if enabled:
        assert reset_mismatch < 1e-4
        assert max(step_mismatches) < 1e-4
    else:
        # Legacy (default): stale IMU values in the first observation (random initial roll/pitch) and after shocks.
        assert reset_mismatch > 1e-3
        assert max(step_mismatches) > 1e-3
