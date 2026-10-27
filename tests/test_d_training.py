"""T16 / T17 (method D): replay windows and masks, WM update, warm start, fade-out, D2 gradient flow, save/resume,
and a tiny end-to-end D run (CPU)."""
import os

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from ml_collections import config_dict

from rl_x.algorithms.ncbf_ppo.flax_full_jit import d_training as D
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model as W


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------
def _tiny_config(tmp_path, nr_envs=16, nr_steps=32, nr_updates=4, **d_overrides):
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.default_config import get_config as get_algorithm_config
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.default_config import get_config as get_env_config
    from rl_x.runner.default_config import get_config as get_runner_config

    env_config = get_env_config("ncbf_mujoco.robot_locomotion.mjx")
    env_config.nr_envs = nr_envs
    env_config.device = "cpu"
    env_config.terrain.type = "plane"
    env_config.episode_length_in_seconds = 1   # 50 control steps: time-outs inside the rollouts
    env_config.record_wm_state = True
    algorithm = get_algorithm_config("ncbf_ppo.flax_full_jit")
    algorithm.device = "cpu"
    algorithm.method = "d"
    algorithm.nr_steps = nr_steps
    algorithm.minibatch_size = nr_envs * nr_steps // 2
    algorithm.nr_epochs = 2
    algorithm.total_timesteps = nr_envs * nr_steps * nr_updates
    algorithm.evaluation_and_save_frequency = nr_envs * nr_steps * 2
    algorithm.evaluation_active = False
    algorithm.ncbf.use_safety_layer = False
    algorithm.ncbf.n_ensemble = 2
    algorithm.ncbf.nr_hidden_units = 16
    algorithm.ncbf.nr_minibatches = 2
    algorithm.hidden_layers = (32, 16)
    d = algorithm.d
    d.wm_width, d.wm_depth, d.k_train, d.wm_batch, d.wm_updates_per_iteration = 32, 2, 4, 64, 3
    d.replay_steps, d.neg_capacity, d.neg_segment_len, d.neg_new_per_iteration = 64, 16, 8, 8
    d.eval_windows, d.auc_horizon, d.H, d.d2_H = 64, 8, 4, 4
    d.anticipation_subset, d.d2_subset, d.xi_probe_envs = 16, 8, 8
    d.warm_start_max_frac = 0.3       # shield on from the 2nd update
    for k, v in d_overrides.items():
        setattr(d, k, v)
    runner = get_runner_config("train")
    runner.save_model = True
    return config_dict.ConfigDict({"algorithm": algorithm, "environment": env_config, "runner": runner})


def _model(tmp_path, **kw):
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import PPO
    from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.create_env import create_env
    config = _tiny_config(tmp_path, **kw)
    env = create_env(config)
    return PPO(config, env, str(tmp_path), None), config, env


def _toy_rows(T=40, N=3, S=5, A=2, P=2, seed=0):
    rng = np.random.default_rng(seed)
    recs = {"x_plus": rng.normal(size=(T, N, S)), "u_applied": rng.normal(size=(T, N, A)),
            "u_chosen": rng.normal(size=(T, N, A)), "xi": rng.normal(size=(T, N, P)),
            "x_minus_next": rng.normal(size=(T, N, S)), "x_plus_next": rng.normal(size=(T, N, S)),
            "shock_next": np.zeros((T, N)), "done": np.zeros((T, N)), "terminated": np.zeros((T, N)),
            "start_ok": np.ones((T, N))}
    return {k: jnp.asarray(v, jnp.float32) for k, v in recs.items()}


# ---------------------------------------------------------------------------------------------------------------
# replay / windows / masks
# ---------------------------------------------------------------------------------------------------------------
def test_ring_write_wraps_and_windows_respect_episode_and_stale_masks():
    T, N, k = 40, 3, 4
    recs = _toy_rows(T, N)
    recs["x_plus"] = recs["x_plus"].at[:, :, 0].set(jnp.arange(T)[:, None] + 100.0 * jnp.arange(N)[None, :])
    recs["done"] = recs["done"].at[9, 1].set(1.0)          # env 1: an episode ends at step 9
    recs["start_ok"] = recs["start_ok"].at[20, 2].set(0.0)  # env 2: stale IMU at step 20
    ring = D.init_ring(50, N, (5, 2, 2))
    ring = D.ring_write(ring, recs)
    ring = D.ring_write(ring, jax.tree_util.tree_map(lambda a: a + 0.0, recs))   # wraps: size 50, pos 30
    assert int(ring["size"]) == 50 and int(ring["pos"]) == 30
    win = D.sample_ring_windows(ring, jax.random.PRNGKey(0), 4000, k)
    x = np.asarray(win["x_plus"][..., 0])
    w = np.asarray(win["weight"]) > 0
    env = (x[:, 0] // 100).astype(int)
    t = x % 100
    # logical order: t 30..39 of the first write, then t 0..39 of the second; consecutive rollouts continue each
    # other (39 -> 0 is a legal step), but no window may wrap around the write head (39 -> 30)
    steps = np.diff(t, axis=1)
    assert np.all(np.isin(steps[w], [1, -39]))
    assert not np.any(w & (env == 1) & (t[:, 0] <= 9) & (t[:, -2] >= 9))  # no done inside a window (last step ok)
    assert np.any(w & (env == 1) & (t[:, -1] == 9))                # ... but a window may END at the done step
    assert not np.any(w & (env == 2) & (t[:, 0] == 20))            # no window starts at a stale row
    assert w.mean() > 0.6


def test_neg_buffer_keeps_segments_ending_at_terminations():
    T, N, L, Lseg = 32, 4, 64, 8
    recs = _toy_rows(T, N)
    recs["x_plus"] = recs["x_plus"].at[:, :, 0].set(jnp.arange(T)[:, None] + 100.0 * jnp.arange(N)[None, :])
    recs["terminated"] = recs["terminated"].at[20, 1].set(1.0).at[3, 2].set(1.0)
    recs["done"] = recs["done"] + recs["terminated"]
    ring0 = D.init_ring(L, N, (5, 2, 2))
    ring = D.ring_write(ring0, recs)
    neg = D.init_neg(16, Lseg, (5, 2, 2))
    neg, n_new = D.neg_write(neg, ring, ring0["pos"], ring0["size"], recs["terminated"] > 0.5, jax.random.PRNGKey(1), 8)
    assert int(n_new) == 2 and int(neg["count"]) == 2
    segs = {int(np.asarray(neg["x_plus"][i, -1, 0])): i for i in range(2)}
    assert set(segs) == {120, 203}
    i = segs[203]   # env 2 terminated at step 3: the rows before the data start must be unusable
    assert np.all(np.asarray(neg["done"][i, :Lseg - 4]) == 1) and np.all(np.asarray(neg["start_ok"][i, :Lseg - 4]) == 0)
    wins = D.sample_neg_windows(neg, jax.random.PRNGKey(2), 500, 4)
    ok = np.asarray(wins["weight"]) > 0
    assert ok.any()
    assert np.all(np.asarray(wins["x_plus"][ok, 0, 0]) % 100 >= 0)
    last = np.asarray(wins["x_plus"][ok, -1, 0])
    assert np.all((last <= 120) & (last >= 116) | (last == 203))


def test_wm_update_decreases_loss_on_a_tiny_env(tmp_path):
    model, _, _ = _model(tmp_path, wm_lr=2e-3, wm_updates_per_iteration=10, wm_batch=256)
    d = model.d
    env = model.env
    keys = jax.random.split(jax.random.PRNGKey(0), d.nr_envs)
    env_state = env.reset(keys, False)
    roll = d.rollout_carry(d.init_state(), 0)
    recs = []
    for t in range(32):
        u0 = 0.3 * jax.random.normal(jax.random.PRNGKey(t), (d.nr_envs, env.nr_actuator_joints))
        u, inputs, vp, ve, diag = d.shield_step(roll, model.policy_state.params, env_state, u0)
        env_state = env.step(env_state, u)
        recs.append(d.record(inputs, u, vp, env_state, diag))
    recs = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *recs)
    # last_action of x-_{t+1} is the chosen action u_t
    la = d.lay.slices["last_action"]
    np.testing.assert_allclose(np.asarray(recs["x_minus_next"][..., la]), np.asarray(recs["u_chosen"]), atol=1e-5)
    st = d.init_state()
    losses = []
    upd = jax.jit(lambda s, k: d.iteration_update(s, model.policy_state.params, recs, 0, k))
    for i in range(12):
        st, m = upd(st, jax.random.PRNGKey(i))
        losses.append(float(m["d/wm_loss"]))
    assert np.isfinite(losses).all()
    assert np.mean(losses[-3:]) < 0.5 * np.mean(losses[:2]), losses
    assert float(m["d/norm_count"]) > 0 and float(m["d/replay_size"]) == 64


# ---------------------------------------------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------------------------------------------
def test_warm_start_switch_logic():
    cfg = D.DConfig(warm_start_auc=0.9, warm_start_consecutive=2, warm_start_min_positives=5, warm_start_max_frac=0.05)
    ready, streak = jnp.asarray(False), jnp.asarray(0)
    seq = [(0.95, 10), (0.80, 10), (0.95, 10), (float("nan"), 0), (0.95, 3), (0.95, 10), (0.92, 10)]
    out = []
    for i, (auc, npos) in enumerate(seq):
        ready, streak, sw = D.warm_start_gate(ready, streak, jnp.asarray(auc), jnp.asarray(npos), i, cfg, 1000)
        out.append((bool(ready), int(streak), bool(sw)))
    assert [o[0] for o in out] == [False] * 6 + [True]
    assert [o[2] for o in out] == [False] * 6 + [True]
    ready2, _, sw2 = D.warm_start_gate(ready, jnp.asarray(0), jnp.asarray(0.1), jnp.asarray(0), 8, cfg, 1000)
    assert bool(ready2) and not bool(sw2)    # latched
    # the cap: on at 5 % of training whatever the AUC
    r, _, s = D.warm_start_gate(jnp.asarray(False), jnp.asarray(0), jnp.asarray(0.1), jnp.asarray(0), 49, cfg, 1000)
    assert bool(r) and bool(s)
    r, _, _ = D.warm_start_gate(jnp.asarray(False), jnp.asarray(0), jnp.asarray(0.1), jnp.asarray(0), 47, cfg, 1000)
    assert not bool(r)


def test_fade_out_schedule():
    n = 1000
    s = np.asarray([float(D.fade_strength(u, n, 0.2)) for u in (0, 500, 799, 800, 900, 1000)])
    np.testing.assert_allclose(s, [1, 1, 1, 1, 0.5, 0], atol=1e-6)
    assert float(D.fade_strength(999, n, 0.0)) == 1.0


def test_masked_auc_matches_numpy():
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import _binary_auroc
    rng = np.random.default_rng(0)
    score = rng.normal(size=300)
    pos = rng.uniform(size=300) < 0.3
    score = score + pos
    valid = rng.uniform(size=300) < 0.8
    auc, npos = D.masked_auc(jnp.asarray(score), jnp.asarray(pos), jnp.asarray(valid))
    assert abs(float(auc) - _binary_auroc(score[valid], pos[valid])) < 1e-5
    assert int(npos) == int((pos & valid).sum())
    assert np.isnan(float(D.masked_auc(jnp.asarray(score), jnp.zeros(300, bool), jnp.asarray(valid))[0]))


# ---------------------------------------------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------------------------------------------
def _batch_from_env(model, n=8):
    d, env = model.d, model.env
    st = env.reset(jax.random.split(jax.random.PRNGKey(3), d.nr_envs), False)
    for t in range(3):
        st = env.step(st, jnp.zeros((d.nr_envs, env.nr_actuator_joints)))
    u, inputs, vp, ve, diag = d.shield_step(d.rollout_carry(d.init_state(), 0), model.policy_state.params, st,
                                            jnp.zeros((d.nr_envs, env.nr_actuator_joints)))
    rec = d.record(inputs, u, vp, env.step(st, u), diag)
    b = d.ppo_batch(jax.tree_util.tree_map(lambda a: a[None], rec))
    obs = st.next_observation
    return jax.tree_util.tree_map(lambda a: a[:n], b), obs[:n]


def test_d2_gradient_reaches_policy_not_wm(tmp_path):
    model, _, _ = _model(tmp_path, d2_coef=0.1)
    d = model.d
    b, obs = _batch_from_env(model)
    st = {**d.init_state(), "wm_ready": jnp.asarray(1, jnp.int32)}

    def loss(policy_params, wm_params):
        mean, _ = model.policy.apply(policy_params, obs, None, None)
        return d.d2_loss({**st, "wm_params": wm_params}, policy_params, b, model.get_processed_action(mean))

    gp, gw = jax.grad(loss, argnums=(0, 1))(model.policy_state.params, st["wm_params"])
    assert sum(float(jnp.sum(jnp.abs(x))) for x in jax.tree_util.tree_leaves(gp)) > 0
    assert all(float(jnp.max(jnp.abs(x))) == 0.0 for x in jax.tree_util.tree_leaves(gw))
    # before the warm start: no loss, no gradient
    st0 = d.init_state()
    l0 = d.d2_loss(st0, model.policy_state.params, b, jnp.zeros((obs.shape[0], 12)))
    assert float(l0) == 0.0


def test_anticipation_through_wm_shield_is_differentiable(tmp_path):
    model, _, _ = _model(tmp_path)
    d = model.d
    b, obs = _batch_from_env(model)
    st = d.init_state()
    # v_prev = gamma_c: most envs need no correction (their zero correction once gave NaN gradients, cap_correction);
    # v_prev = 30: more than any V, the shield must act everywhere
    for vp in (d.gamma_c, 30.0):
        bb = {**b, "v_prev": jnp.full_like(b["v_prev"], vp)}

        def loss(policy_params):
            mean, _ = model.policy.apply(policy_params, obs, None, None)
            return d.anticipation_loss(st, policy_params, bb, model.get_processed_action(mean), jnp.asarray(1.0))[0]

        val, g = jax.value_and_grad(loss)(model.policy_state.params)
        assert np.isfinite(float(val))
        assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(g)), vp
    assert float(val) > 0
    l_off = d.anticipation_loss(st, model.policy_state.params, b, jnp.zeros((obs.shape[0], 12)), jnp.asarray(0.0))[0]
    assert float(l_off) == 0.0


# ---------------------------------------------------------------------------------------------------------------
# save / resume and the end-to-end run
# ---------------------------------------------------------------------------------------------------------------
def test_save_resume_restores_wm_state(tmp_path):
    model, config, env = _model(tmp_path)
    d = model.d
    p = jax.tree_util.tree_map(lambda a: a + 0.5 if jnp.issubdtype(a.dtype, jnp.floating) else a + 3, d.persist0)
    p = {**p, "wm_ready": jnp.asarray(1, jnp.int32), "switch_update": jnp.asarray(7, jnp.int32)}
    model.resume = True
    states = (model.policy_state, model.critic_state, model.ncbf_state, model.encoder_state, model.decoder_state)
    model._save_resume_checkpoint(*states, {}, 5, d_persist=p)
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import PPO
    config.runner.resume = True
    restored = PPO(config, env, str(tmp_path), None)
    restored._restore_resume_checkpoint()
    assert restored.resume_start_update == 5
    for a, b in zip(jax.tree_util.tree_leaves(restored.d.persist), jax.tree_util.tree_leaves(p)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))
    assert int(restored.d.init_state()["wm_ready"]) == 1


def test_tiny_end_to_end_d_run_without_nan(tmp_path):
    model, config, env = _model(tmp_path, nr_updates=4)
    logged = {}
    model.start_logging = lambda step: None
    model.end_logging = lambda: None
    model.log = lambda name, value, step: logged.setdefault(name, []).append(float(value))
    model.train()
    for key in ("d/wm_loss", "d/wm_latest_kstep_loss", "loss/policy_gradient_loss", "loss/anticipation_loss"):
        assert key in logged and len(logged[key]) == 4, key
        assert np.all(np.isfinite(logged[key])), (key, logged[key])
    assert logged["d/wm_ready"][0] == 0.0 and logged["d/wm_ready"][-1] == 1.0
    assert max(logged["d/shield_strength"]) > 0
    assert os.path.exists(os.path.join(model.save_path, "wm_latest.msgpack"))
    from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model_eval as WE
    wm, bundle, meta = WE.load_bundle(os.path.join(model.save_path, "wm_latest.msgpack"))
    assert meta["schema"]["xi_active"] == [int(i) for i in model.d.xi_active]
    # the checkpoint loads as a plain policy (mode A eval)
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import PPO
    load_config = config_dict.ConfigDict(config.to_dict())
    load_config.runner.load_model = f"{model.save_path}/latest.model"
    loaded = PPO.load(load_config, env, str(tmp_path / "load"), None, set())
    assert loaded.d is None
