"""T12 tests: label rules and a CPU smoke run of the world-model data collector (untrained model).

    JAX_PLATFORMS=cpu python -m pytest tests/test_wm_collect.py -q
"""
import json
import os

import numpy as np
import pytest
from ml_collections import config_dict

from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_collect, wm_state


def test_actual_labels_rules():
    # one env, T = 8: termination at t=2 (episode 0), truncation at t=5 (episode 1), window ends at t=7
    term = np.zeros((8, 1), bool)
    trunc = np.zeros((8, 1), bool)
    term[2, 0] = True
    trunc[5, 0] = True
    fail, mask = wm_collect.actual_labels(term, trunc, horizons=(1, 2, 3))
    f, m = fail[:, 0], mask[:, 0]
    # t=2: fails within 1; t=1: within 2; t=0: within 3
    assert f[2].tolist() == [True, True, True] and f[1].tolist() == [False, True, True] and f[0].tolist() == [False, False, True]
    assert m[:3].all()
    # episode 1 (t=3..5) is truncated: t=3 has 3 transitions observed -> H<=3 known, no failure
    assert not f[3:6].any()
    assert m[3].tolist() == [True, True, True] and m[4].tolist() == [True, True, False] and m[5].tolist() == [True, False, False]
    # episode 2 (t=6, 7) is cut by the window end
    assert m[6].tolist() == [True, True, False] and m[7].tolist() == [True, False, False]


def test_oracle_labels_rules():
    H = 5
    term = np.zeros((3, H), bool)
    trunc = np.zeros((3, H), bool)
    term[0, 2] = True  # fails at step 2
    trunc[1, 1] = True  # truncated at step 1 -> 2 transitions observed
    term[2, 4] = True  # fails at the last step
    first, fail, mask = wm_collect.oracle_labels(term, trunc, horizons=(1, 3, 5))
    assert first.tolist() == [2, H, 4]
    assert fail[0].tolist() == [False, True, True] and fail[2].tolist() == [False, False, True]
    assert mask[1].tolist() == [True, False, False] and mask[0].all() and mask[2].all()


def test_xi_table_dedup():
    tab = wm_collect.XiTable(2, 3)
    a = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    i0 = tab.index(a)
    i1 = tab.index(a)
    b = a.copy()
    b[1, 0] = 9
    i2 = tab.index(b)
    assert i0.tolist() == [0, 1] and i1.tolist() == [0, 1] and i2.tolist() == [0, 2]
    np.testing.assert_array_equal(tab.table()[i2], b)


NR_ENVS = 8


@pytest.fixture(scope="module")
def shard(tmp_path_factory):
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.default_config import get_config as get_algorithm_config
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import PPO
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.create_env import create_env
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.default_config import get_config as get_env_config
    from rl_x.runner.default_config import get_config as get_runner_config

    tmp = tmp_path_factory.mktemp("wm_collect")
    env_config = get_env_config("ncbf_mujoco.robot_locomotion.mjx")
    env_config.nr_envs = NR_ENVS
    env_config.device = "cpu"
    env_config.terrain.type = "plane"
    env_config.episode_length_in_seconds = 1  # 50 steps: truncations inside the 60-step window
    env_config.record_wm_state = True
    env_config.termination.height_curriculum_scaled = True
    env_config.termination.body_tilt_threshold = 1e6
    algorithm = get_algorithm_config("ncbf_ppo.flax_full_jit")
    algorithm.device = "cpu"
    algorithm.nr_steps = 8
    algorithm.minibatch_size = 32
    algorithm.ncbf.use_safety_layer = False
    w = algorithm.wm_collect
    w.enabled = True
    w.out_dir = str(tmp / "out")
    w.checkpoint_label = "untrained"
    w.settings = "push:0.3"
    w.nr_steps = 60
    w.oracle_interval = 10
    w.oracle_nr_envs = 3
    w.oracle_horizon = 12
    w.chunk_blocks = 3
    w.noise_std = 3.0
    config = config_dict.ConfigDict({"environment": env_config, "algorithm": algorithm, "runner": get_runner_config("test")})
    env = create_env(config)
    model = PPO(config, env, str(tmp / "run"), None)
    summaries = model.test(1)
    return env, os.path.join(w.out_dir, "train", "untrained", "push_p0p3"), summaries


def test_collect_smoke_shard_contents(shard):
    env, path, summaries = shard
    assert len(summaries) == 1
    load = lambda name: np.load(os.path.join(path, f"{name}.npy"))
    T, N = 60, NR_ENVS
    lay, xl = wm_state.state_layout(env), wm_state.xi_layout(env)
    assert load("x_plus").shape == (T, N, lay.dim) and load("xi_table").shape[1] == xl.dim
    assert load("oracle_x_plus_next").shape == (6, 3, 12, lay.dim)
    schema = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(path))), "schema.json")))
    assert set(schema["fields"]) >= {"x_plus", "u_applied", "oracle_fail_within"}
    summary = json.load(open(os.path.join(path, "summary.json")))
    assert summary["margin_consistent"] and summary["oracle_margin_consistent"]
    assert summary["shock_rate"] > 0.1
    term, done = load("terminated"), load("done")
    # x_plus_next[t] == x_plus[t+1] unless the env reset
    cont = ~done[:-1]
    np.testing.assert_allclose(load("x_plus_next")[:-1][cont], load("x_plus")[1:][cont], atol=1e-6)
    # the delay function reproduces the applied action
    u_tilde, _ = wm_state.delay_step(load("delay_buffer"), load("u_cmd"), load("delay_steps").astype(np.int32))
    np.testing.assert_array_equal(np.asarray(u_tilde), load("u_applied"))
    # xi indices: xi_next differs from xi only at DR resamples
    table = load("xi_table")
    same = np.all(table[load("xi_idx")] == table[load("xi_next_idx")], -1)
    assert np.all(same[~load("dr_resampled_next")])
    # margin < 0 exactly at terminations
    np.testing.assert_array_equal(load("margin_next") < 0, term)
    # oracle: no shocks -> x^+ == x^- (no recorded jump) is implied by perturbed flags being off; the first action is
    # the policy on o(x) without noise; the oracle labels are consistent with its margins
    first = load("oracle_first_failure")
    margin = load("oracle_margin")
    for p in range(first.shape[0]):
        for k in range(first.shape[1]):
            f = first[p, k]
            if f < 12:
                assert margin[p, k, f] < 0 and np.all(margin[p, k, :f] >= 0)
    meta = json.load(open(os.path.join(path, "meta.json")))
    assert meta["push_prob"] == 0.3 and meta["oracle"]["horizon"] == 12
