"""T11 tests: world-model state x^-/x^+, xi, action delay, o(x) and analytic margins against the MJX env.

    JAX_PLATFORMS=cpu python -m pytest tests/test_wm_state.py -q
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import config_dict

from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_state

NR_ENVS = 8
NR_STEPS = 120
FLIP_STEP = 40  # envs 0 and 1 are flipped upside down before this step -> they fall (imu height < 0)


def _make_env(obs_noise="none"):
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.default_config import get_config
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.create_env import create_env

    c = get_config("ncbf_mujoco.robot_locomotion.mjx")
    c.nr_envs = NR_ENVS
    c.device = "cpu"
    c.terrain.type = "plane"
    c.episode_length_in_seconds = 1  # 50 steps: truncations inside the rollout too
    c.record_wm_state = True
    c.termination.height_curriculum_scaled = True  # T06b
    c.termination.body_tilt_threshold = 1e6  # T06b
    c.command.velocity_ratio = (1.0, 1.0, 1.0)
    c.command.clip_max_velocity = 2.0
    c.domain_randomization.observation_noise.type = obs_noise
    c.domain_randomization.sampling_probability = 0.05  # mid-episode DR resampling
    c.domain_randomization.perturbation.sampling_probability = 0.1  # shocks
    c.domain_randomization.action_delay.mixed_chance = 0.5  # exercise the random delay
    config = config_dict.ConfigDict({"environment": c, "runner": {"mode": "train"}})
    return create_env(config)


def _rollout(env, eval_mode, seed):
    lay = wm_state.state_layout(env)
    ext_x = jax.jit(jax.vmap(lambda d, i: wm_state.extract_state(env, d, i, i["last_action"])))
    ext_xi = jax.jit(jax.vmap(lambda m, i: wm_state.extract_xi(env, m, i)))
    state = env.reset(jax.random.split(jax.random.PRNGKey(seed), env.nr_envs), eval_mode)
    rec = {k: [] for k in ("x", "xi", "obs", "command", "buffer", "u", "x_minus_next", "x_plus_next", "xi_next",
                           "u_applied", "delay_steps", "buffer_after", "terminated", "truncated", "perturbed",
                           "dr_resampled", "episode_step", "constraint_terminated")}
    key = jax.random.PRNGKey(seed + 100)
    for t in range(NR_STEPS):
        if t == FLIP_STEP:
            qpos = state.data.qpos.at[:2, 3:7].set(jnp.array([0.0, 1.0, 0.0, 0.0]))
            state = state.replace(data=state.data.replace(qpos=qpos))
        key, k = jax.random.split(key)
        u = 0.8 * jax.random.normal(k, (env.nr_envs, env.nr_actuator_joints))
        rec["x"].append(np.asarray(ext_x(state.data, state.internal_state)))
        rec["xi"].append(np.asarray(ext_xi(state.mjx_model, state.internal_state)))
        rec["obs"].append(np.asarray(state.next_observation[:, env.policy_observation_indices]))
        rec["command"].append(np.asarray(state.internal_state["goal_velocities"]))
        rec["buffer"].append(np.asarray(state.internal_state["action_history"]))
        rec["episode_step"].append(np.asarray(state.info_episode_store["episode_step"]))
        rec["u"].append(np.asarray(u))
        state = env.step(state, u)
        for name, info_key in (("x_minus_next", "wm/x_minus"), ("x_plus_next", "wm/x_plus"), ("xi_next", "wm/xi_next"),
                               ("u_applied", "wm/u_applied"), ("delay_steps", "wm/delay_steps"),
                               ("perturbed", "shock/perturbed"), ("dr_resampled", "shock/dr_resampled"),
                               ("constraint_terminated", "constraint/terminated")):
            rec[name].append(np.asarray(state.info[info_key]))
        rec["buffer_after"].append(np.asarray(state.internal_state["action_history"]))
        rec["terminated"].append(np.asarray(state.terminated))
        rec["truncated"].append(np.asarray(state.truncated))
    out = {k: np.stack(v) for k, v in rec.items()}
    out["done"] = out["terminated"] | out["truncated"]
    out["lay"] = lay
    return out


@pytest.fixture(scope="module")
def env():
    return _make_env()


@pytest.fixture(scope="module")
def roll(env):
    return _rollout(env, True, 0)


@pytest.fixture(scope="module")
def roll_train(env):
    return _rollout(env, False, 1)  # curriculum coefficient 0: height threshold 0.8 * nominal


def test_layouts_and_schema(env):
    lay, xl = wm_state.state_layout(env), wm_state.xi_layout(env)
    assert lay.dim == 51
    assert list(lay.slices)[:5] == ["joint_pos", "joint_vel", "imu_ang_vel", "projected_gravity", "last_action"]
    schema = wm_state.schema_dict(env)
    assert schema["state"]["groups"]["base_yaw"] == [50, 51]
    assert schema["xi"]["dim"] == xl.dim
    assert schema["policy_obs"]["dim"] == 45
    assert schema["margin"]["height_curriculum_scaled"] and not schema["margin"]["use_tilt"]


def test_rollout_has_falls_shocks_dr_and_delays(roll):
    assert roll["terminated"].sum() >= 2 and roll["truncated"].any()
    assert (roll["perturbed"] > 0.5).any() and (roll["dr_resampled"] > 0.5).any()
    assert (roll["delay_steps"] == 1).any() and (roll["delay_steps"] == 0).any()


def test_policy_obs_from_state_equals_noise_free_env_obs(env, roll):
    obs_fn, full_obs_fn = wm_state.make_policy_obs_fn(env)
    o = np.asarray(obs_fn(jnp.asarray(roll["x"]), jnp.asarray(roll["command"]), jnp.asarray(roll["xi"])))
    np.testing.assert_allclose(o, roll["obs"], atol=2e-6, rtol=0)
    full = np.asarray(full_obs_fn(jnp.asarray(roll["x"][:2]), jnp.asarray(roll["command"][:2]), jnp.asarray(roll["xi"][:2])))
    np.testing.assert_array_equal(full[..., np.asarray(env.policy_observation_indices)], o[:2])


def test_x_plus_next_is_the_next_state_and_x_minus_differs_only_by_the_shock(roll):
    lay = roll["lay"]
    cont = ~roll["done"][:-1]
    np.testing.assert_allclose(roll["x_plus_next"][:-1][cont], roll["x"][1:][cont], atol=1e-6)
    d = roll["x_plus_next"] - roll["x_minus_next"]
    quiet = (roll["perturbed"] < 0.5) & (roll["dr_resampled"] < 0.5)
    np.testing.assert_allclose(d[quiet], 0.0, atol=1e-6)
    shocked = roll["perturbed"] > 0.5
    moved = np.abs(d[shocked][:, lay.slices["joint_vel"]]).sum(-1) + np.abs(d[shocked][:, lay.slices["base_lin_vel"]]).sum(-1)
    assert np.all(moved > 0)
    # IMU-derived groups are never changed by the shock / DR step (the env does not recompute kinematics there)
    for name in ("imu_ang_vel", "projected_gravity", "imu_height", "roll_pitch", "base_yaw", "last_action"):
        np.testing.assert_array_equal(d[..., lay.slices[name]], 0.0)


def test_projected_gravity_matches_roll_pitch(roll):
    lay = roll["lay"]
    x = roll["x"][roll["episode_step"] > 0]
    r, p = x[:, lay.slices["roll_pitch"]].T
    g = np.stack([np.sin(p), -np.cos(p) * np.sin(r), -np.cos(p) * np.cos(r)], -1)
    np.testing.assert_allclose(x[:, lay.slices["projected_gravity"]], g, atol=1e-5)


def test_delay_step_reproduces_the_env(env, roll):
    u_tilde, buffer_after = wm_state.delay_step(jnp.asarray(roll["buffer"]), jnp.asarray(roll["u"]), jnp.asarray(roll["delay_steps"]))
    np.testing.assert_array_equal(np.asarray(u_tilde), roll["u_applied"])
    cont = ~roll["done"]
    np.testing.assert_array_equal(np.asarray(buffer_after)[cont], roll["buffer_after"][cont])
    # the realized delay is 0 unless the env's delay is mixed; and delay_steps_from_xi agrees for non-mixed envs
    xl = wm_state.xi_layout(env)
    mixed = roll["xi"][..., xl.slices["delay_mixed"]][..., 0] > 0.5
    assert np.all(roll["delay_steps"][~mixed] == 0)
    d_xi = np.asarray(wm_state.delay_steps_from_xi(jnp.asarray(roll["xi"]), xl))
    np.testing.assert_array_equal(d_xi[~mixed], roll["delay_steps"][~mixed])
    assert np.all(d_xi[mixed] == 1)  # worst case without a key
    # gradient flows to u through the delay buffer when the delay is 0
    g = jax.grad(lambda u: wm_state.delay_step(jnp.zeros((2, 12)), u, jnp.asarray(0))[0].sum())(jnp.ones(12))
    np.testing.assert_array_equal(np.asarray(g), 1.0)


def test_xi_is_constant_between_dr_events(env, roll):
    cont = ~roll["done"][:-1]
    np.testing.assert_array_equal(roll["xi_next"][:-1][cont], roll["xi"][1:][cont])
    same = np.all(roll["xi_next"] == roll["xi"], axis=-1)
    assert np.all(same[roll["dr_resampled"] < 0.5])
    assert not np.all(same[roll["dr_resampled"] > 0.5])
    xl = wm_state.xi_layout(env)
    kp = roll["xi"][..., xl.slices["kp"]]
    assert np.all(kp > 0) or np.any(roll["xi"][..., xl.slices["dropout_open_mask"]] < 0.5)
    np.testing.assert_allclose(roll["xi"][..., xl.slices["curriculum_coeff"]], 1.0)


@pytest.mark.parametrize("which", ["roll", "roll_train"])
def test_margin_negative_exactly_where_env_terminates(env, which, request):
    r = request.getfixturevalue(which)
    lay, xl = wm_state.state_layout(env), wm_state.xi_layout(env)
    cfg = wm_state.margin_config_from_env(env)
    m = np.asarray(wm_state.margin_fall(jnp.asarray(r["x_plus_next"]), jnp.asarray(r["xi_next"]), cfg, lay, xl))
    np.testing.assert_array_equal(m < 0, r["terminated"])
    assert r["terminated"].sum() >= 2
    # the qvel term never fires on ordinary steps; with constraint_terminated only, the height term decides
    h = np.asarray(wm_state.margins(jnp.asarray(r["x_plus_next"]), jnp.asarray(r["xi_next"]), cfg, lay, xl)["height"])
    np.testing.assert_array_equal(h < 0, r["constraint_terminated"])
    soft = np.asarray(wm_state.margin_fall(jnp.asarray(r["x_plus_next"]), jnp.asarray(r["xi_next"]), cfg, lay, xl, 0.05))
    assert np.all(soft <= m + 1e-6)


def test_tilt_switch_and_window_frame(env, roll):
    lay, xl = wm_state.state_layout(env), wm_state.xi_layout(env)
    x = jnp.asarray(roll["x"][5])
    xi = jnp.asarray(roll["xi"][5])
    cfg = wm_state.margin_config_from_env(env, use_tilt=True)
    cfg = wm_state.MarginConfig(**{**cfg.to_json(), "tilt_threshold": 0.0})
    assert np.all(np.asarray(wm_state.margin_fall(x, xi, cfg, lay, xl)) <= 0)
    x0 = jnp.asarray(roll["x"][10])
    oxy, oyaw = wm_state.window_origin(x0, lay)
    xw = wm_state.to_window_frame(jnp.asarray(roll["x"][11]), oxy, oyaw, lay)
    np.testing.assert_allclose(np.asarray(wm_state.to_window_frame(x0, oxy, oyaw, lay))[:, lay.slices["base_xy"]], 0.0, atol=1e-6)
    back = wm_state.from_window_frame(xw, oxy, oyaw, lay)
    np.testing.assert_allclose(np.asarray(back), roll["x"][11], atol=1e-5)
    # invariance: rotating + translating the whole world leaves the window-frame state unchanged
    s = lay.slices
    shift = lambda x: x.at[:, s["base_xy"]].add(jnp.array([3.0, -1.0])).at[:, s["base_yaw"]].add(0.0)
    xw2 = wm_state.to_window_frame(shift(jnp.asarray(roll["x"][11])), *wm_state.window_origin(shift(x0), lay), lay)
    np.testing.assert_allclose(np.asarray(xw2), np.asarray(xw), atol=1e-5)


def test_physical_fall_margin(env, roll):
    lay, xl = wm_state.state_layout(env), wm_state.xi_layout(env)
    cfg = wm_state.PhysicalFallConfig()
    m = np.asarray(wm_state.margin_fall_physical(jnp.asarray(roll["x_plus_next"]), jnp.asarray(roll["xi_next"]), cfg, lay, xl))
    assert np.all(m[roll["terminated"]] < 0)  # IMU below ground -> also physically fallen
    # flipped envs are physically fallen right after the flip, before the termination
    assert np.all(m[FLIP_STEP + 1, :2] < 0)
    upright = roll["x"][3:10, 2:]
    assert np.all(np.asarray(wm_state.margin_fall_physical(jnp.asarray(upright), jnp.asarray(roll["xi"][3:10, 2:]), cfg, lay, xl)) > 0)
