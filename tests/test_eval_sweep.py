"""CPU tests for the perturbation-sweep evaluation (T04).

Run from the repository root:
    JAX_PLATFORMS=cpu python -m pytest tests/test_eval_sweep.py -q
"""
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from ml_collections import config_dict

from rl_x.algorithms.ncbf_ppo.flax_full_jit import eval_sweep
from rl_x.algorithms.ncbf_ppo.flax_full_jit.eval_sweep import make_episode_runner, summarize_episodes

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------------------------
# Fake env + stub agent: fast checks of the noise model and of the shield calls
# ---------------------------------------------------------------------------------------------------------------
@struct.dataclass
class FakeState:
    next_observation: jax.Array
    actual_next_observation: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    info: dict
    step_count: jax.Array


class FakeEnv:
    """reward = sum(action^2); never terminates; truncates after `horizon` steps."""

    def __init__(self, nr_envs=4096, action_dim=6, horizon=40):
        self.nr_envs, self.horizon, self.action_dim = nr_envs, horizon, action_dim
        self.domain_randomization_perturbation_function = None

    def reset(self, nr_envs):
        z = jnp.zeros(nr_envs)
        return FakeState(jnp.zeros((nr_envs, 3)), jnp.zeros((nr_envs, 3)), z, z.astype(bool), z.astype(bool), {}, z.astype(jnp.int32))

    def step(self, state, action):
        count = state.step_count + 1
        return state.replace(reward=jnp.sum(action ** 2, axis=-1), truncated=count >= self.horizon, step_count=count)


class StubAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.shield_traces = 0
        self.use_safety_layer = True

    def policy(self, env_state):
        return jnp.zeros((env_state.next_observation.shape[0], self.action_dim)), None

    def shield(self, env_state, action, aux):
        self.shield_traces += 1
        n = action.shape[0]
        return action, jnp.ones(n, dtype=bool), jnp.full(n, 0.5)


def _run_fake(mode, noise_prob, noise_std, nr_envs=4096, action_dim=6, horizon=40, seed=0):
    env, agent = FakeEnv(nr_envs, action_dim, horizon), StubAgent(action_dim)
    run = make_episode_runner(env, agent, mode, loop_pushes=False, horizon=horizon)
    _, noise_key, push_key = eval_sweep.eval_keys(seed)
    out = run(env.reset(nr_envs), noise_key, push_key, noise_prob, jnp.full(action_dim, noise_std), 0.0, 1.0)
    return {k: np.asarray(v) for k, v in out.items()}, agent


def test_noise_has_the_right_probability_and_std():
    p, std, action_dim, horizon, n = 0.1, 0.7, 6, 40, 4096
    out, _ = _run_fake("A", p, std, n, action_dim, horizon)
    nr_steps = n * horizon
    event_rate = out["noise_events"].sum() / nr_steps
    assert abs(event_rate - p) < 4 * np.sqrt(p * (1 - p) / nr_steps)
    # E[sum_t |noise_t|^2] = horizon * p * A * std^2 (zero policy action)
    per_component_var = out["return"].sum() / (out["noise_events"].sum() * action_dim)
    assert abs(per_component_var - std ** 2) < 0.02 * std ** 2
    assert np.all(out["length"] == horizon) and not out["terminated"].any()


def test_noise_off_gives_exactly_the_unperturbed_actions():
    out, _ = _run_fake("A", 0.0, 1.0, nr_envs=64)
    assert out["noise_events"].sum() == 0
    assert np.all(out["return"] == 0.0)


def test_noise_draws_are_nested_and_seeded():
    low, _ = _run_fake("A", 0.05, 1.0, nr_envs=256)
    high, _ = _run_fake("A", 0.2, 1.0, nr_envs=256)
    again, _ = _run_fake("A", 0.05, 1.0, nr_envs=256)
    np.testing.assert_array_equal(low["return"], again["return"])
    assert np.all(low["noise_events"] <= high["noise_events"])
    other_seed, _ = _run_fake("A", 0.05, 1.0, nr_envs=256, seed=1)
    assert not np.array_equal(low["return"], other_seed["return"])


def test_mode_a_never_calls_the_shield_and_mode_b_does():
    out_a, agent_a = _run_fake("A", 0.0, 1.0, nr_envs=8)
    assert agent_a.shield_traces == 0
    assert out_a["interventions"].sum() == 0 and np.all(out_a["delta_u_sum"] == 0.0)
    out_b, agent_b = _run_fake("B", 0.0, 1.0, nr_envs=8, horizon=10)
    assert agent_b.shield_traces > 0
    assert np.all(out_b["interventions"] == 10) and np.allclose(out_b["delta_u_sum"], 5.0)


def test_summary_statistics():
    episodes = {
        "return": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), "length": np.array([10, 10, 4, 10]),
        "terminated": np.array([False, False, True, False]), "truncated": np.array([True, True, False, True]),
        "constraint_terminated": np.array([False, False, True, False]), "qvel_limit_terminated": np.zeros(4, bool),
        "collision": np.zeros(4, bool), "noise_events": np.array([1, 0, 2, 1]), "push_events": np.zeros(4, int),
        "interventions": np.array([5, 0, 2, 0]), "delta_u_sum": np.array([1.0, 0.0, 1.0, 0.0]),
        "head_unsafe": np.array([[False, False], [False, True], [True, False], [False, False]]),
    }
    s = summarize_episodes(episodes, horizon=10, success_threshold_steps=4, head_names=["height", "tilt"], shield_used=True)
    assert s["success_rate"] == 0.75 and s["failure_rate"] == 0.25
    assert np.isclose(s["success_rate_se"], np.sqrt(0.75 * 0.25 / 4))
    assert np.isclose(s["mean_episode_return_se"], np.std([1, 2, 3, 4], ddof=1) / 2)
    assert s["survival_rate_at_threshold"] == 1.0
    assert s["event_rates"]["fall"] == 0.25 and s["event_rates"]["unsafe_tilt"] == 0.25
    assert np.isclose(s["intervention_rate"], 7 / 34) and np.isclose(s["mean_delta_u_when_active"], 2 / 7)
    s_a = summarize_episodes(episodes, 10, 4, shield_used=False)
    assert s_a["intervention_rate"] is None and s_a["mean_delta_u"] is None


# ---------------------------------------------------------------------------------------------------------------
# Real (untrained) PPO model + MJX env on CPU, tiny
# ---------------------------------------------------------------------------------------------------------------
NR_ENVS = 4


def _make_model(tmp_path, use_safety_layer=True, **sweep_overrides):
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.default_config import get_config as get_algorithm_config
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import PPO
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.create_env import create_env
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.default_config import get_config as get_env_config
    from rl_x.runner.default_config import get_config as get_runner_config

    env_config = get_env_config("ncbf_mujoco.robot_locomotion.mjx")
    env_config.nr_envs = NR_ENVS
    env_config.device = "cpu"
    env_config.terrain.type = "plane"
    env_config.episode_length_in_seconds = 1  # 50 control steps
    algorithm = get_algorithm_config("ncbf_ppo.flax_full_jit")
    algorithm.device = "cpu"
    algorithm.nr_steps = 8
    algorithm.minibatch_size = 32
    algorithm.ncbf.use_safety_layer = use_safety_layer
    algorithm.eval_sweep.out_dir = str(tmp_path / "sweep")
    algorithm.eval_sweep.checkpoint_label = "untrained"
    for key, value in sweep_overrides.items():
        algorithm.eval_sweep[key] = value
    runner = get_runner_config("test")
    config = config_dict.ConfigDict({"environment": env_config, "algorithm": algorithm, "runner": runner})
    env = create_env(config)
    return PPO(config, env, str(tmp_path / "run"), None)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return _make_model(tmp_path_factory.mktemp("eval_sweep"))


def _reference_episode(model, env_state, shielded):
    """Plain loop: deterministic policy (optionally shielded) through PPO's own methods, one episode per env."""

    @jax.jit
    def step(env_state):
        raw, safe, active, _, _ = model._policy_and_shield(env_state)
        return model.env.step(env_state, safe if shielded else raw), active

    alive = np.ones(NR_ENVS, bool)
    ret, length, term, interventions = np.zeros(NR_ENVS), np.zeros(NR_ENVS, int), np.zeros(NR_ENVS, bool), np.zeros(NR_ENVS, int)
    for _ in range(model.env.horizon):
        env_state, active = step(env_state)
        ret += alive * np.asarray(env_state.reward)
        length += alive
        interventions += alive & np.asarray(active)
        term |= alive & np.asarray(env_state.terminated)
        alive &= ~(np.asarray(env_state.terminated) | np.asarray(env_state.truncated))
    return ret, length, term, interventions


@pytest.mark.parametrize("mode", ["A", "B"])
def test_noise_off_matches_the_unperturbed_evaluation(model, mode):
    agent = eval_sweep.make_agent(model)
    assert isinstance(agent, eval_sweep.MultistepAgent)
    reset_key, noise_key, push_key = eval_sweep.eval_keys(3)
    reset = lambda: model.env.reset(jax.random.split(reset_key, NR_ENVS), True)
    ret, length, term, interventions = _reference_episode(model, reset(), shielded=(mode == "B"))
    run = make_episode_runner(model.env, agent, mode)
    out = run(reset(), noise_key, push_key, 0.0, jnp.ones(12), 0.0, 1.0)
    np.testing.assert_allclose(np.asarray(out["return"]), ret, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(out["length"]), length)
    np.testing.assert_array_equal(np.asarray(out["terminated"]), term)
    if mode == "B":
        np.testing.assert_array_equal(np.asarray(out["interventions"]), interventions)


def test_sweep_writes_summaries_in_the_collected_schema(model, tmp_path):
    sweep = eval_sweep.read_sweep_config(model.config)
    sweep.update({"out_dir": str(tmp_path), "perturbations": ["action_noise", "push"],
                  "sampling_probs": {"action_noise": [0.0, 0.5], "push": [0.5]}, "eval_seeds": [0],
                  "push_impl": "eval_loop", "push_scale": 2.0})
    summaries = eval_sweep.run_eval_sweep(model, sweep)
    assert len(summaries) == 3
    files = sorted(tmp_path.rglob("eval_summary.json"))
    assert len(files) == 3
    expected = {"untrained/action_noise_flat/mode_A/sp_0/seed_0", "untrained/action_noise_flat/mode_A/sp_0p5/seed_0",
                "untrained/push_flat/mode_A/sp_0p5/seed_0"}
    assert {str(f.parent.relative_to(tmp_path)) for f in files} == expected
    required = {"schema", "nr_envs", "nr_steps", "nr_finished_episodes", "failure_rate", "mean_episode_return",
                "mean_episode_length", "intervention_rate", "mean_delta_u", "success_rate", "success_rate_se",
                "mean_episode_return_se", "event_rates", "checkpoint_label", "setting", "perturbation", "terrain",
                "sampling_prob", "eval_seed", "deployment_mode", "protocol", "git_commit", "wall_time_s"}
    for f in files:
        s = json.loads(f.read_text())
        assert required <= set(s), required - set(s)
        assert s["nr_envs"] == NR_ENVS and s["deployment_mode"] == "A" and s["intervention_rate"] is None
        assert 0.0 <= s["success_rate"] <= 1.0
        assert (f.parent / "eval_episodes.npz").exists()
    by_point = {(s["perturbation"], s["sampling_prob"]): s for s in summaries}
    assert by_point[("action_noise", 0.0)]["noise_events_per_episode"] == 0
    assert by_point[("action_noise", 0.5)]["noise_events_per_episode"] > 0
    assert by_point[("push", 0.5)]["push_events_per_episode"] > 0
    assert by_point[("push", 0.5)]["protocol"]["push_scale"] == 2.0

    # collector + Fig. 2 reader
    out_csv = tmp_path / "all.csv"
    subprocess.run([sys.executable, str(REPO / "experiments/ncbf_tools/collect_eval_summaries.py"), str(tmp_path),
                    "--out", str(out_csv)], check=True, cwd=REPO)
    assert out_csv.exists()
    sys.path.insert(0, str(REPO / "misc/rss"))
    from eval_summary_reader import load_eval_summary_rows
    rows = load_eval_summary_rows([tmp_path])
    assert len(rows) == 3 and all({"success_rate", "sampling_prob", "setting", "experiment"} <= set(r) for r in rows)
    assert {r["experiment"] for r in rows} == {"action_noise_2", "perturbation"}


def test_mode_b_requires_the_shield(model):
    sweep = eval_sweep.read_sweep_config(model.config)
    sweep["deployment_mode"] = "B"
    model.ncbf_use_safety_layer = False
    try:
        with pytest.raises(ValueError, match="use_safety_layer"):
            eval_sweep.run_eval_sweep(model, sweep)
    finally:
        model.ncbf_use_safety_layer = True


def test_initial_commands(model):
    env = model.env
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), NR_ENVS), True)
    zero = np.asarray(state.internal_state["goal_velocities"])
    same = eval_sweep.apply_initial_commands(env, state, jax.random.PRNGKey(1), 1.0)
    np.testing.assert_array_equal(np.asarray(same.internal_state["goal_velocities"]), zero)
    sampled = eval_sweep.apply_initial_commands(env, state, jax.random.PRNGKey(1), 0.0)
    assert np.any(np.asarray(sampled.internal_state["goal_velocities"]) != zero)
