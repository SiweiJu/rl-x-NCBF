"""T06a (branch ncbf-q-fixed): tests of the F1 / F2 / replay-permutation fixes on the paper code (cacffe7).

Run from the repository root:
    JAX_PLATFORMS=cpu python -m pytest tests/test_t06a_fixes.py -q
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import config_dict

from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import adjacent_temporal_loss, critic_inputs_and_pairs
from rl_x.algorithms.ncbf_ppo.flax_full_jit.ncbf import make_get_safe_action


def test_critic_inputs_and_adjacent_pairs():
    T, N, D, A = 5, 2, 3, 2
    states = jnp.arange(T * N * D, dtype=jnp.float32).reshape(T, N, D) + 1.0
    exec_actions = jnp.arange(T * N * A, dtype=jnp.float32).reshape(T, N, A) + 1.0
    post_s = jnp.zeros_like(states)
    post_a = jnp.zeros_like(exec_actions)
    y = jnp.ones((T, N), dtype=bool).at[3, 1].set(False)
    masks = jnp.ones((T, N), dtype=bool)
    dones = jnp.zeros((T, N), dtype=bool).at[1, 0].set(True)
    cs, ca, na, ny, valid = critic_inputs_and_pairs(states, exec_actions, post_s, post_a, y, masks, dones, "pre_step")
    np.testing.assert_array_equal(cs, states)
    np.testing.assert_array_equal(ca, exec_actions)
    np.testing.assert_array_equal(na[:-1], exec_actions[1:])
    np.testing.assert_array_equal(ny[:-1], y[1:])
    valid = np.asarray(valid)
    assert not valid[1, 0] and not valid[-1].any() and valid[0, 0] and valid[1, 1]
    cs, ca, *_ = critic_inputs_and_pairs(states, exec_actions, post_s, post_a, y, masks, dones, "post_step")
    np.testing.assert_array_equal(cs, post_s)


@pytest.mark.parametrize("space", ["prob", "logit"])
def test_adjacent_temporal_loss_is_paper_eq_4_on_safe_pairs(space):
    rng = np.random.default_rng(0)
    B = 64
    h, hn = (jnp.asarray(rng.normal(size=B), dtype=jnp.float32) for _ in range(2))
    y, ny, masks, valid = (jnp.asarray(rng.uniform(size=B) > 0.3) for _ in range(4))
    gamma_c, eta = 0.405, 0.5
    loss = adjacent_temporal_loss(h, hn, y, ny, masks, valid, gamma_c, eta, space)
    f = (lambda x: 1 / (1 + np.exp(-x))) if space == "prob" else (lambda x: x)
    h_, hn_, g_ = f(np.asarray(h)), f(np.asarray(hn)), f(gamma_c)
    paper = np.maximum(0.0, -(hn_ - h_ + eta * (h_ - g_)))
    sel = np.asarray(valid & masks & y & ny)
    np.testing.assert_allclose(loss, paper[sel].mean(), rtol=1e-5)


def _stub_apply():
    w = jnp.linspace(0.2, 0.8, 7)

    def apply(phi, x):
        v = jnp.dot(x, w) + 0.3 * jnp.sin(3 * x[4])
        return v, jnp.asarray(0.5), v[None]
    return apply


def test_shield_first_step_uses_gamma_c():
    gamma_c, eta, lam = 0.405, 0.5, 1000.0
    kw = dict(ncbf_apply=_stub_apply(), ncbf_obs_from_obs_idx=jnp.arange(4), use_safety_layer=True,
              act_low=-jnp.ones(3), act_high=jnp.ones(3), gamma_c=gamma_c, eta_cbf=eta, lambda_s=lam,
              action_clipping=False)
    obs, obs_last = jnp.array([0.1, -0.2, 0.3, 0.0]), jnp.zeros(4)
    u0, u_last = jnp.array([-2.0, 0.2, 0.1]), jnp.zeros(3)
    apply = _stub_apply()
    q = lambda u: apply(None, jnp.concatenate([obs, u]))[0]
    a = jax.grad(q)(u0)

    def expected(h_x):
        c = h_x - eta * (h_x - gamma_c) + a @ u0 - q(u0)
        delta = max(0.0, float(c - a @ u0))
        return u0 + delta / (a @ a + 1.0 / lam) * a  # sigma = clip(std, 1, 10) = 1

    fixed = make_get_safe_action(**kw, first_step_prev_value="gamma_c")
    legacy = make_get_safe_action(**kw, first_step_prev_value="input")
    h_zero = apply(None, jnp.concatenate([obs_last, u_last]))[0]  # h(0, 0)
    np.testing.assert_allclose(fixed(u0, obs, u_last, obs_last, None, True)[0], expected(gamma_c), rtol=1e-5)
    np.testing.assert_allclose(fixed(u0, obs, u_last, obs_last, None, False)[0], expected(h_zero), rtol=1e-5)
    np.testing.assert_allclose(legacy(u0, obs, u_last, obs_last, None, True)[0], expected(h_zero), rtol=1e-5)


def _make_env(reset_last_state):
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.default_config import get_config
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.create_env import create_env

    env_config = get_config("ncbf_mujoco.robot_locomotion.mjx")
    env_config.nr_envs = 2
    env_config.device = "cpu"
    env_config.terrain.type = "plane"
    env_config.episode_length_in_seconds = 2
    env_config.reset_last_state = reset_last_state
    return create_env(config_dict.ConfigDict({"environment": env_config, "runner": {"mode": "train"}}))


@pytest.mark.parametrize("reset_last_state", ["observation", "zeros"])
def test_env_first_step_reports_s0_and_terminal_step_reports_zero(reset_last_state):
    env = _make_env(reset_last_state)
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), env.nr_envs), False)
    s0 = np.asarray(state.next_observation)
    action = jnp.zeros((env.nr_envs,) + env.single_action_space.shape)
    state = env.step(state, action)  # first step of the episode
    post = np.asarray(state.last_state)
    if reset_last_state == "observation":
        np.testing.assert_allclose(post, s0)
    else:
        assert np.all(post == 0.0)
    # Terminal step: flip env 0 upside down; the post-step env state is the reset one (0, 0) in both modes.
    qpos = state.data.qpos.at[0, 3:7].set(jnp.array([0.0, 1.0, 0.0, 0.0]))
    state = state.replace(data=state.data.replace(qpos=qpos))
    for _ in range(5):
        state = env.step(state, action)
        if bool(state.terminated[0]):
            break
    assert bool(state.terminated[0])
    assert np.all(np.asarray(state.last_state[0]) == 0.0) and np.all(np.asarray(state.last_action[0]) == 0.0)
