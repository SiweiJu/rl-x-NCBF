"""T06c: switches for the PPO-side differences between the paper code (ncbf-q @ cacffe7) and pncbf-v2.

    JAX_PLATFORMS=cpu python -m pytest tests/test_t06c_switches.py -q
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from ml_collections import config_dict

from rl_x.algorithms.ncbf_ppo.flax_full_jit.critic import Critic
from rl_x.algorithms.ncbf_ppo.flax_full_jit.paper_compat import (
    PAPER_PPO_SWITCHES,
    anticipation_squared_error,
    gae_advantages,
    maybe_clip_logstd,
)
from rl_x.algorithms.ncbf_ppo.flax_full_jit.policy import Policy


def _reference_gae(delta, cut, gamma, lam):
    """The loop both codes implement: A_T-1 = delta_T-1, A_t = delta_t + gamma lam (1 - cut_t) A_t+1."""
    adv = np.zeros_like(delta)
    adv[-1] = delta[-1]
    for t in range(len(delta) - 2, -1, -1):
        adv[t] = delta[t] + gamma * lam * (1.0 - cut[t]) * adv[t + 1]
    return adv


def test_gae_switch_cuts_at_dones_or_only_at_terminations():
    rng = np.random.default_rng(0)
    T, N = 9, 3
    delta = rng.normal(size=(T, N)).astype(np.float32)
    terminations = np.zeros((T, N), bool)
    truncations = np.zeros((T, N), bool)
    terminations[2, 0] = True  # env 0: a fall
    truncations[4, 1] = True  # env 1: a time-out
    dones = terminations | truncations
    v2 = np.asarray(gae_advantages(jnp.asarray(delta), jnp.asarray(terminations), jnp.asarray(dones), 0.99, 0.9, True))
    paper = np.asarray(gae_advantages(jnp.asarray(delta), jnp.asarray(terminations), jnp.asarray(dones), 0.99, 0.9, False))
    np.testing.assert_allclose(v2, _reference_gae(delta, dones.astype(np.float32), 0.99, 0.9), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(paper, _reference_gae(delta, terminations.astype(np.float32), 0.99, 0.9), rtol=1e-5, atol=1e-6)
    # They differ only before the time-out of env 1 (the paper leaks the next episode's advantage into it).
    differs = ~np.isclose(v2, paper, atol=1e-6)
    assert differs[:5, 1].all() and not differs[5:, 1].any()
    assert not differs[:, 0].any() and not differs[:, 2].any()


class PaperPolicy(nn.Module):
    """Policy of the paper code, copied from cacffe7 rl_x/algorithms/ncbf_ppo/flax_full_jit/policy.py."""
    as_shape: tuple
    std_dev: float
    policy_observation_indices: tuple

    @nn.compact
    def __call__(self, x):
        x = x[..., jnp.asarray(self.policy_observation_indices)]
        policy_mean = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        policy_mean = nn.LayerNorm()(policy_mean)
        policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(policy_mean)
        policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(policy_mean)
        policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(np.prod(self.as_shape).item(), kernel_init=orthogonal(0.01), bias_init=constant(0.0))(policy_mean)
        policy_logstd = self.param("policy_logstd", constant(jnp.log(self.std_dev)), (1, np.prod(self.as_shape).item()))
        return policy_mean, policy_logstd


class PaperCritic(nn.Module):
    """Value network of the paper code (cacffe7 critic.py)."""
    critic_observation_indices: tuple

    @nn.compact
    def __call__(self, x):
        x = x[..., jnp.asarray(self.critic_observation_indices)]
        critic = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = nn.LayerNorm()(critic)
        critic = nn.elu(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = nn.elu(critic)
        critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = nn.elu(critic)
        return nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)


def _tree_paths(params):
    return sorted(jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_flatten_with_path(params)[0])


def test_layer_norm_switch_gives_the_paper_networks():
    obs = jax.random.normal(jax.random.PRNGKey(1), (5, 20))
    idx = tuple(range(0, 20, 2))
    paper = PaperPolicy((12,), 1.0, idx)
    v2_paper_mode = Policy((12,), 1.0, jnp.asarray(idx), (512, 256, 128), layer_norm=True)
    v2_default = Policy((12,), 1.0, jnp.asarray(idx), (512, 256, 128))
    params = paper.init(jax.random.PRNGKey(2), obs)
    # Same parameter tree -> a paper policy's parameters load into the switched v2 policy, and the outputs agree.
    assert _tree_paths(params) == _tree_paths(v2_paper_mode.init(jax.random.PRNGKey(3), obs))
    for a, b in zip(paper.apply(params, obs), v2_paper_mode.apply(params, obs)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)
    # Default: no LayerNorm parameters, so existing pncbf-v2 checkpoints keep loading.
    assert not any("LayerNorm" in p for p in _tree_paths(v2_default.init(jax.random.PRNGKey(3), obs)))

    paper_critic = PaperCritic(idx)
    v2_critic = Critic(jnp.asarray(idx), (512, 256, 128), layer_norm=True)
    critic_params = paper_critic.init(jax.random.PRNGKey(4), obs)
    assert _tree_paths(critic_params) == _tree_paths(v2_critic.init(jax.random.PRNGKey(5), obs))
    np.testing.assert_allclose(np.asarray(paper_critic.apply(critic_params, obs)), np.asarray(v2_critic.apply(critic_params, obs)), rtol=1e-6, atol=1e-6)
    assert not any("LayerNorm" in p for p in _tree_paths(Critic(jnp.asarray(idx), (512, 256, 128)).init(jax.random.PRNGKey(5), obs)))


def test_logstd_clip_and_anticipation_reduction():
    logstd = jnp.array([-7.0, 0.0, 3.0])
    np.testing.assert_allclose(maybe_clip_logstd(logstd, True), [-5.0, 0.0, 2.0])
    np.testing.assert_allclose(maybe_clip_logstd(logstd, False), logstd)
    mu, target = jnp.zeros((2, 4)), jnp.array([[1.0, 1.0, 1.0, 1.0], [2.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(anticipation_squared_error(mu, target, "mean_action"), [4.0, 4.0])
    np.testing.assert_allclose(anticipation_squared_error(mu, target, "sampled_action"), [1.0, 1.0])


# ---------------------------------------------------------------------------
# End to end: a tiny shield-off PPO run with all paper switches trains, saves and reloads
# ---------------------------------------------------------------------------
def _set(config, dotted, value):
    node = config
    *parents, leaf = dotted.split(".")[1:]  # drop "algorithm"
    for p in parents:
        node = node[p]
    node[leaf] = value


@pytest.mark.parametrize("paper_switches", [False, True])
def test_tiny_ppo_run_with_switches_trains_and_reloads(paper_switches, tmp_path):
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.default_config import get_config as get_algorithm_config
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import PPO
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.create_env import create_env
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.default_config import get_config as get_env_config
    from rl_x.runner.default_config import get_config as get_runner_config

    env_config = get_env_config("ncbf_mujoco.robot_locomotion.mjx")
    env_config.nr_envs = 4
    env_config.device = "cpu"
    env_config.terrain.type = "plane"
    env_config.episode_length_in_seconds = 1  # 50 control steps: a time-out inside the 2 x 32-step rollouts
    algorithm = get_algorithm_config("ncbf_ppo.flax_full_jit")
    algorithm.device = "cpu"
    algorithm.nr_steps = 32
    algorithm.minibatch_size = 32
    algorithm.total_timesteps = 128 * 2
    algorithm.evaluation_and_save_frequency = 128 * 2
    algorithm.evaluation_active = False
    algorithm.ncbf.use_safety_layer = False
    algorithm.ncbf.n_ensemble = 2
    algorithm.ncbf.nr_hidden_units = 16
    algorithm.hidden_layers = (32, 16)
    if paper_switches:
        for key, value in PAPER_PPO_SWITCHES.items():
            _set(algorithm, key, value)
    runner = get_runner_config("train")
    runner.save_model = True
    config = config_dict.ConfigDict({"algorithm": algorithm, "environment": env_config, "runner": runner})
    env = create_env(config)
    model = PPO(config, env, str(tmp_path), None)
    has_ln = any("LayerNorm" in p for p in _tree_paths(model.policy_state.params))
    assert has_ln == paper_switches
    before = jax.tree_util.tree_leaves(model.policy_state.params)
    model.train()

    load_config = config_dict.ConfigDict(config.to_dict())
    load_config.algorithm = get_algorithm_config("ncbf_ppo.flax_full_jit")  # defaults: switches come from the checkpoint
    load_config.algorithm.device = "cpu"
    load_config.runner.load_model = f"{model.save_path}/latest.model"
    loaded = PPO.load(load_config, env, str(tmp_path / "load"), None, set())
    assert loaded.config.algorithm.policy_layer_norm == paper_switches
    assert loaded.config.algorithm.gae_stop_at_truncation == (not paper_switches)
    after = jax.tree_util.tree_leaves(loaded.policy_state.params)
    assert len(after) == len(before)
    assert all(np.all(np.isfinite(np.asarray(x))) for x in after)
    assert any(not np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(before, after))  # it trained
