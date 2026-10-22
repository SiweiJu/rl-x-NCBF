"""P0 tests (project plan 8, T02): label alignment, F1, F2, B1, gamma_c / eta form, closed-form projection,
shock recording in the env, and the requeue-safe resume checkpoint.

Run from the repository root:
    JAX_PLATFORMS=cpu python -m pytest tests/test_p0_fixes.py -q
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ml_collections import config_dict

from rl_x.algorithms.ncbf_ppo.flax_full_jit.ncbf import make_get_safe_action
from rl_x.algorithms.ncbf_ppo.flax_full_jit.ncbf_targets import (
    cbf_residual,
    critic_training_inputs,
    legacy_raw_vs_executed_temporal_loss,
    receding_min_target,
    successor_pairs,
    temporal_consistency_loss,
)


# ---------------------------------------------------------------------------
# Environment rollout with a forced fall (shared by the label / F1 / shock tests)
# ---------------------------------------------------------------------------
NR_ENVS = 4
H = 5


def _make_env():
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.default_config import get_config
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.create_env import create_env

    env_config = get_config("ncbf_mujoco.robot_locomotion.mjx")
    env_config.nr_envs = NR_ENVS
    env_config.device = "cpu"
    env_config.terrain.type = "plane"
    env_config.episode_length_in_seconds = 2
    env_config.record_transition_physics_state = True
    env_config.domain_randomization.sampling_probability = 0.0  # no model resampling inside episodes
    env_config.domain_randomization.perturbation.sampling_probability = 0.5  # shocks on about half the steps
    config = config_dict.ConfigDict({"environment": env_config, "runner": {"mode": "train"}})
    return config, create_env(config)


@pytest.fixture(scope="module")
def env_and_config():
    return _make_env()


@pytest.fixture(scope="module")
def rollout(env_and_config):
    """Rollout like ppo.single_rollout records it; env 0 is flipped upside down before step FALL_STEP."""
    _, env = env_and_config
    # eval_mode=True sets the env curriculum to 1, so the curriculum-scaled shocks are active.
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), env.nr_envs), True)
    nr_steps, fall_step = 12, 6
    records = {k: [] for k in ("states", "actions", "next_states", "dones", "terms", "post_last_state",
                               "post_last_action", "x_minus_qpos", "x_minus_qvel", "x_plus_qpos", "x_plus_qvel",
                               "data_qpos", "data_qvel", "perturbed", "dr_resampled", "first_step")}
    key = jax.random.PRNGKey(1)
    for t in range(nr_steps):
        if t == fall_step:
            qpos = state.data.qpos.at[0, 3:7].set(jnp.array([0.0, 1.0, 0.0, 0.0]))
            state = state.replace(data=state.data.replace(qpos=qpos))
        key, subkey = jax.random.split(key)
        action = 0.3 * jax.random.normal(subkey, (env.nr_envs,) + env.single_action_space.shape)
        records["states"].append(np.asarray(state.next_observation))
        records["first_step"].append(np.asarray(state.info_episode_store["episode_step"] == 0))
        records["actions"].append(np.asarray(action))
        state = env.step(state, action)
        records["next_states"].append(np.asarray(state.actual_next_observation))
        records["dones"].append(np.asarray(state.terminated | state.truncated))
        records["terms"].append(np.asarray(state.terminated))
        records["post_last_state"].append(np.asarray(state.last_state))
        records["post_last_action"].append(np.asarray(state.last_action))
        for name in ("x_minus_qpos", "x_minus_qvel", "x_plus_qpos", "x_plus_qvel"):
            records[name].append(np.asarray(state.info[f"transition/{name}"]))
        records["data_qpos"].append(np.asarray(state.data.qpos))
        records["data_qvel"].append(np.asarray(state.data.qvel))
        records["perturbed"].append(np.asarray(state.info["shock/perturbed"]))
        records["dr_resampled"].append(np.asarray(state.info["shock/dr_resampled"]))
    out = {k: np.stack(v) for k, v in records.items()}
    out["fall_step"] = fall_step
    return out


def test_forced_fall_terminates_exactly_at_the_flip(rollout):
    k = rollout["fall_step"]
    assert rollout["terms"][k, 0] and rollout["dones"][k, 0]
    assert not rollout["dones"][:k, 0].any()
    # The next step starts a new episode.
    assert rollout["first_step"][k + 1, 0] and not rollout["first_step"][k, 0]


def test_label_and_critic_input_are_the_same_time_step(env_and_config, rollout):
    """y_t is the outcome of executing a_t at s_t: the fall at step k gives y_k = 0 and the critic input at k is
    (s_k, a_k), the observation and action of that same step."""
    _, env = env_and_config
    k = rollout["fall_step"]
    target, y, mask, dists = receding_min_target(
        jnp.asarray(rollout["next_states"]), jnp.asarray(rollout["dones"]), jnp.asarray(rollout["terms"]),
        H, env.ncbf_target_indices)
    target, y, dists = np.asarray(target), np.asarray(y), np.asarray(dists)
    assert dists[k, 0] == 0
    assert not y[k, 0] and np.any(target[k, 0] < 0.5)  # unsafe label at the failing step
    assert not y[k - 1, 0]  # inside the horizon before the fall
    assert y[k + 1, 0]  # new episode after the reset: safe again
    states, actions = critic_training_inputs(rollout["states"], rollout["actions"], None, rollout["dones"])
    np.testing.assert_array_equal(states[k, 0], rollout["states"][k, 0])
    np.testing.assert_array_equal(actions[k, 0], rollout["actions"][k, 0])


def test_f1_terminal_critic_input_is_the_pre_reset_state(rollout):
    k = rollout["fall_step"]
    states, actions = critic_training_inputs(rollout["states"], rollout["actions"], None, rollout["dones"],
                                             "current", "pre_reset")
    # Non-zero, equal to the observation before the failing step ...
    assert np.abs(states[k, 0]).sum() > 0 and np.abs(actions[k, 0]).sum() > 0
    np.testing.assert_array_equal(states[k, 0], rollout["states"][k, 0])
    # ... and not what the env state holds after the auto-reset (what the paper code read): zero action and
    # the reset observation.
    assert np.all(rollout["post_last_action"][k, 0] == 0.0)
    assert not np.allclose(rollout["post_last_state"][k, 0], states[k, 0])
    # On steps that do not end the episode the post-step env state is the same (s_t, a_t).
    np.testing.assert_allclose(rollout["post_last_state"][k - 1, 0], rollout["states"][k - 1, 0])
    np.testing.assert_allclose(rollout["post_last_action"][k - 1, 0], rollout["actions"][k - 1, 0])
    # The legacy switch reproduces the zero input.
    legacy_states, legacy_actions = critic_training_inputs(rollout["states"], rollout["actions"], None,
                                                           rollout["dones"], "current", "zeros")
    assert np.all(legacy_states[k, 0] == 0.0) and np.all(legacy_actions[k, 0] == 0.0)
    np.testing.assert_array_equal(legacy_states[k - 1, 0], rollout["states"][k - 1, 0])


def test_env_records_shock_flags_and_pre_post_physics_state(rollout):
    perturbed = rollout["perturbed"] > 0.5
    dr = rollout["dr_resampled"] > 0.5
    not_done = ~rollout["dones"]
    assert perturbed.any() and (~perturbed).any()
    assert not dr.any()
    d_qvel = rollout["x_plus_qvel"] - rollout["x_minus_qvel"]
    d_qpos = rollout["x_plus_qpos"] - rollout["x_minus_qpos"]
    # No shock: x^+ == x^-. Shock (default Go2 perturbation): trunk and joint velocities and joint positions
    # jump, the base pose does not.
    np.testing.assert_allclose(d_qvel[~perturbed], 0.0, atol=1e-6)
    np.testing.assert_allclose(d_qpos[~perturbed], 0.0, atol=1e-6)
    assert np.all(np.abs(d_qvel[perturbed][:, :6]).sum(-1) > 0.0)
    assert np.all(np.abs(d_qpos[perturbed][:, 7:]).sum(-1) > 0.0)
    np.testing.assert_allclose(d_qpos[perturbed][:, :7], 0.0, atol=1e-6)
    # x^+ is the physics state the next step starts from (when the episode continues).
    np.testing.assert_allclose(rollout["x_plus_qpos"][not_done], rollout["data_qpos"][not_done], atol=1e-6)
    np.testing.assert_allclose(rollout["x_plus_qvel"][not_done], rollout["data_qvel"][not_done], atol=1e-6)


# ---------------------------------------------------------------------------
# F2: temporal term on adjacent pairs
# ---------------------------------------------------------------------------
def _synthetic_batch():
    # T = 6 steps, N = 2 envs, 1 head. Env 0 has an episode end at t = 2.
    T, N = 6, 2
    env_actions = jnp.arange(T * N * 3, dtype=jnp.float32).reshape(T, N, 3)
    y = jnp.ones((T, N, 1))
    y = y.at[4, 1, 0].set(0.0)
    masks = jnp.ones((T, N), dtype=bool)
    dones = jnp.zeros((T, N), dtype=bool).at[2, 0].set(True)
    return env_actions, y, masks, dones


def test_f2_successor_pairs_are_adjacent_and_within_one_episode():
    env_actions, y, masks, dones = _synthetic_batch()
    next_actions, next_y, valid = successor_pairs(env_actions, y, masks, dones)
    # a_{t+1} paired with a_t, from the same (executed) action array -> same scale.
    np.testing.assert_array_equal(next_actions[:-1], env_actions[1:])
    np.testing.assert_array_equal(next_y[:-1], y[1:])
    valid = np.asarray(valid)
    assert not valid[2, 0]  # episode ends at t = 2: (s_2, s_3) crosses a reset
    assert not valid[-1].any()  # last step of the rollout has no successor
    assert valid[1, 0] and valid[3, 0] and valid[2, 1]


def test_f2_temporal_loss_matches_paper_eq_4_on_safe_pairs():
    rng = np.random.default_rng(0)
    E, B, K = 3, 8, 2
    h_cur = jnp.asarray(rng.uniform(0, 1, (E, B, K)), dtype=jnp.float32)
    h_next = jnp.asarray(rng.uniform(0, 1, (E, B, K)), dtype=jnp.float32)
    y_now = jnp.asarray(rng.uniform(0, 1, (B, K)) > 0.3)
    y_next = jnp.asarray(rng.uniform(0, 1, (B, K)) > 0.3)
    valid = jnp.asarray(rng.uniform(0, 1, B) > 0.2)
    gamma_c, eta = 0.6, 0.3
    loss = temporal_consistency_loss(h_cur, h_next, y_now, y_next, valid, gamma_c, eta)
    # Paper eq. (4): ReLU(-[h(x_{t+1}) - h(x_t) + alpha (h(x_t) - gamma_c)]) on safe -> safe pairs.
    paper = np.maximum(0.0, -(np.asarray(h_next) - np.asarray(h_cur) + eta * (np.asarray(h_cur) - gamma_c)))
    sel = (np.asarray(valid)[:, None] & np.asarray(y_now) & np.asarray(y_next))[None].repeat(E, 0)
    # Summed over ensemble members (each member's mean over the pairs), like the classification loss.
    np.testing.assert_allclose(loss, E * paper[sel].mean(), rtol=1e-5)
    # Pairs with an unsafe label or crossing an episode boundary do not contribute.
    loss_unsafe_changed = temporal_consistency_loss(h_cur, jnp.where(y_now[None], h_next, 0.0), y_now, y_next,
                                                    valid, gamma_c, eta)
    np.testing.assert_allclose(loss, loss_unsafe_changed, rtol=1e-6)


def test_f2_legacy_mode_compares_two_actions_at_the_same_state():
    # The paper code's term is zero whenever h(s_t, a_exec) == h(s_t, a_raw), whatever h does along the
    # trajectory; the adjacent form is not.
    h = jnp.asarray([[[0.9], [0.2], [0.8]]])  # [E=1, B=3, K=1] along time
    y = jnp.asarray([[1.0], [1.0], [1.0]])
    legacy = legacy_raw_vs_executed_temporal_loss(h, h, y, jnp.ones(3, dtype=bool), jnp.full(3, 100), H)
    assert float(legacy) == 0.0
    adjacent = temporal_consistency_loss(h[:, :2], h[:, 1:], y[:2], y[1:], jnp.ones(2, dtype=bool), 0.5, 0.5)
    assert float(adjacent) > 0.0


# ---------------------------------------------------------------------------
# Shield: gamma_c / eta form, first step, closed-form projection
# ---------------------------------------------------------------------------
ACTION_DIM, OBS_DIM = 3, 4


def _stub_ncbf_apply():
    weights = jnp.linspace(0.2, 0.8, OBS_DIM + ACTION_DIM)

    def ncbf_apply(params, x):
        # Nonlinear in the action so the linearization is not exact.
        mean_logits = (jnp.dot(x, weights) + 0.5 * jnp.sin(3.0 * x[OBS_DIM]))[None, None]  # [E=1, 1]
        details = {
            "mean_logits": mean_logits,
            "std_logits": jnp.zeros_like(mean_logits),
            "residual_eps_current": jnp.zeros((4,) + mean_logits.shape),
            "residual_eps_next": jnp.zeros((4,) + mean_logits.shape),
        }
        return jnp.mean(jax.nn.sigmoid(mean_logits)), jnp.asarray(0.0), details

    return ncbf_apply


def _shield(**overrides):
    kwargs = dict(ncbf_apply=_stub_ncbf_apply(), ncbf_obs_from_obs_idx=jnp.arange(OBS_DIM), use_safety_layer=True,
                  act_low=-jnp.ones(ACTION_DIM), act_high=jnp.ones(ACTION_DIM), gamma_c=0.9, eta_cbf=0.5,
                  lambda_s=1000.0, max_delta_u=0.0, action_clipping=False, safety_layer_projection="soft_slack",
                  safety_layer_min_grad_norm=0.0, post_check_actual_residual=False)
    kwargs.update(overrides)
    return make_get_safe_action(**kwargs)


OBS_T = jnp.array([0.1, -0.2, 0.3, 0.0])
OBS_LAST = jnp.array([0.0, 0.1, 0.2, -0.1])
U_LAST = jnp.array([0.1, 0.0, -0.1])
U0 = jnp.array([-0.5, 0.2, 0.1])


def _h(obs, u):
    _, _, d = _stub_ncbf_apply()(None, jnp.concatenate([obs, u]))
    return jax.nn.sigmoid(d["mean_logits"][0, 0])


def test_residual_is_paper_form_and_first_step_uses_gamma_c():
    gamma_c, eta = 0.9, 0.5
    h_next, h_cur = _h(OBS_T, U0), _h(OBS_LAST, U_LAST)
    np.testing.assert_allclose(cbf_residual(h_next, h_cur, gamma_c, eta),
                               h_next - h_cur + eta * (h_cur - gamma_c), rtol=1e-6)
    shield = _shield(gamma_c=gamma_c, eta_cbf=eta)
    _, _, _, diag = shield(U0, OBS_T, U_LAST, OBS_LAST, jnp.zeros(0), 0.0, None, False)
    np.testing.assert_allclose(diag["residual_mean"], h_next - (1 - eta) * h_cur - eta * gamma_c, rtol=1e-5)
    # First step of an episode: h_prev = gamma_c, i.e. the constraint is h_next >= gamma_c.
    _, _, _, diag_first = shield(U0, OBS_T, U_LAST, OBS_LAST, jnp.zeros(0), 0.0, None, True)
    np.testing.assert_allclose(diag_first["residual_mean"], h_next - gamma_c, rtol=1e-5, atol=1e-6)
    # Legacy switch: the (reset) previous input is used even on the first step.
    legacy = _shield(gamma_c=gamma_c, eta_cbf=eta, first_step_prev_value="input")
    _, _, _, diag_legacy = legacy(U0, OBS_T, U_LAST, OBS_LAST, jnp.zeros(0), 0.0, None, True)
    np.testing.assert_allclose(diag_legacy["residual_mean"], diag["residual_mean"], rtol=1e-6)


@pytest.mark.parametrize("projection,lambda_s", [("soft_slack", 5.0), ("soft_slack", 1000.0), ("hard_projection", 1.0)])
def test_closed_form_projection_equals_the_single_constraint_qp(projection, lambda_s):
    from scipy.optimize import minimize

    gamma_c, eta = 0.9, 0.5
    shield = _shield(gamma_c=gamma_c, eta_cbf=eta, lambda_s=lambda_s, safety_layer_projection=projection)
    u_safe, active, _, _ = shield(U0, OBS_T, U_LAST, OBS_LAST, jnp.zeros(0), 0.0, None, False)
    assert bool(active)

    # Independent linearization of the residual and a numerical QP solve (paper eq. 12).
    h_cur = _h(OBS_LAST, U_LAST)
    residual = lambda u: _h(OBS_T, u) - (1 - eta) * h_cur - eta * gamma_c
    a = np.asarray(jax.grad(residual)(U0), dtype=np.float64)
    c = float(a @ np.asarray(U0) - residual(U0))
    u0 = np.asarray(U0, dtype=np.float64)
    if projection == "hard_projection":
        objective = lambda z: np.sum((z[:3] - u0) ** 2)
        constraints = [{"type": "ineq", "fun": lambda z: a @ z[:3] - c}]
        z0 = np.concatenate([u0, [0.0]])
    else:
        objective = lambda z: np.sum((z[:3] - u0) ** 2) + lambda_s * z[3] ** 2
        constraints = [{"type": "ineq", "fun": lambda z: a @ z[:3] + z[3] - c}, {"type": "ineq", "fun": lambda z: z[3]}]
        z0 = np.concatenate([u0, [0.0]])
    result = minimize(objective, z0, constraints=constraints, method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    assert result.success
    np.testing.assert_allclose(np.asarray(u_safe), result.x[:3], atol=1e-5)


# ---------------------------------------------------------------------------
# PPO object: B1 (independent ensemble members) and the resume checkpoint round trip
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ppo_model(env_and_config, tmp_path_factory):
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.default_config import get_config as get_algorithm_config
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import PPO
    from rl_x.runner.default_config import get_config as get_runner_config

    env_cfg, env = env_and_config
    algorithm = get_algorithm_config("ncbf_ppo.flax_full_jit")
    algorithm.device = "cpu"
    algorithm.nr_steps = 16
    algorithm.minibatch_size = 64
    algorithm.total_timesteps = 64 * 8
    algorithm.evaluation_and_save_frequency = 64 * 4
    algorithm.ncbf.n_ensemble = 3
    algorithm.ncbf.nr_hidden_units = 16
    runner = get_runner_config("train")
    runner.resume = True
    config = config_dict.ConfigDict({"algorithm": algorithm, "environment": env_cfg.environment, "runner": runner})
    run_path = str(tmp_path_factory.mktemp("run"))
    return PPO, config, env, run_path


def test_b1_ensemble_members_have_different_parameters(ppo_model):
    PPO, config, env, run_path = ppo_model
    model = PPO(config, env, run_path, None)
    leaves = jax.tree_util.tree_leaves(model.ncbf_state.params)
    assert all(leaf.shape[0] == 3 for leaf in leaves)
    kernels = [leaf for leaf in leaves if leaf.ndim == 3]
    assert kernels
    for kernel in kernels:
        assert not np.allclose(kernel[0], kernel[1]) and not np.allclose(kernel[1], kernel[2])


def test_resume_checkpoint_restores_params_optimizer_and_counters(ppo_model):
    import os

    PPO, config, env, run_path = ppo_model
    model = PPO(config, env, run_path, None)
    bump = lambda state: state.replace(
        step=state.step + 7,
        params=jax.tree_util.tree_map(lambda x: x + 0.25, state.params),
        opt_state=jax.tree_util.tree_map(lambda x: x + 1 if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating) else x, state.opt_state),
    )
    states = [bump(s) for s in (model.policy_state, model.critic_state, model.ncbf_state, model.encoder_state, model.decoder_state)]
    curriculum = {"env_curriculum_coeff": np.linspace(0.0, 1.0, NR_ENVS), "env_curriculum_levels_in_a_row": np.arange(NR_ENVS, dtype=np.float32)}
    model._save_resume_checkpoint(*states, curriculum, 5)
    # An interrupted later save (no pointer switch) must not be picked up.
    os.makedirs(os.path.join(run_path, "resume", "step_000000006.partial"))

    restored = PPO(config, env, run_path, None)
    restored._restore_resume_checkpoint()
    assert restored.resume_start_update == 5
    np.testing.assert_allclose(restored.resume_env_curriculum["env_curriculum_coeff"], curriculum["env_curriculum_coeff"])
    for original, new in zip(states, (restored.policy_state, restored.critic_state, restored.ncbf_state, restored.encoder_state, restored.decoder_state)):
        assert int(new.step) == int(original.step)
        for x, y in zip(jax.tree_util.tree_leaves((original.params, original.opt_state)), jax.tree_util.tree_leaves((new.params, new.opt_state))):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y))
