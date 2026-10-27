"""CPU tests for the P1 world model (T13).

    JAX_PLATFORMS=cpu python -m pytest tests/test_world_model.py -q
"""
import os

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model as W
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model_data as WD
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model_eval as WE


@pytest.fixture(scope="module")
def toy():
    data, schema = WD.synthetic_dataset(n_envs=16, T=300, seed=3)
    return data, schema, W.WMLayout.from_schema(schema)


def _wm(layout, **kw):
    cfg = dict(width=32, depth=2)
    cfg.update(kw)
    return W.WorldModel(layout, W.WMConfig(**cfg))


def _batch(data, starts, k):
    dev = WD.to_device(data)
    return WD.gather_windows(dev, jnp.asarray(starts, jnp.int32), k)


def test_layout_and_shapes(toy):
    data, schema, layout = toy
    assert layout.xy_idx and layout.yaw_idx and len(layout.last_action_idx) == layout.action_dim
    wm = _wm(layout, ensemble_size=3)
    bundle = wm.init(jax.random.PRNGKey(0))
    assert jax.tree_util.tree_leaves(bundle["params"])[0].shape[0] == 3
    N, D, A, P = 5, layout.state_dim, layout.action_dim, layout.xi_dim
    x, u, xi = data["x_plus"][:N], data["u_applied"][:N], data["xi"][:N]
    assert wm.step(bundle, x, u, xi, u).shape == (N, D)
    assert wm.step_members(bundle, x, u, xi, u).shape == (3, N, D)
    # the network never sees base_xy / base_yaw, and last_action is the chosen action
    assert wm.n_in == D - 3 + A + P and wm.n_out == D - A
    xn = wm.step(bundle, x, u, xi, u)
    np.testing.assert_allclose(xn[:, list(layout.last_action_idx)], u, atol=0)
    pos, _ = WD.split_window_starts(data, 8)
    b = _batch(data, pos[:7], 8)
    assert b["x_plus"].shape == (7, 8, D)
    loss, aux = W.k_step_loss(wm, bundle["params"], bundle["norm"], b)
    assert loss.shape == () and np.isfinite(float(loss))
    traj, acts = W.rollout(wm, bundle, x, xi, None, u, lambda s: jnp.tanh(s[:, :A]), H=6, return_actions=True)
    assert traj.shape == (6, N, D) and acts.shape == (6, N, A)
    # translation / yaw invariance: shifting the base pose shifts the prediction the same way
    x2 = np.array(x)
    x2[:, list(layout.xy_idx)] += np.array([3.0, -2.0])
    np.testing.assert_allclose(wm.step(bundle, x2, u, xi, u)[:, list(layout.xy_idx)],
                               xn[:, list(layout.xy_idx)] + np.array([3.0, -2.0]), atol=1e-5)


def test_normalization_round_trip(toy):
    data, schema, layout = toy
    wm = _wm(layout)
    rows = WD.transition_rows_for_norm(data)
    x, u, uc, xi, xn = (data[f][rows] for f in ("x_plus", "u_applied", "u_chosen", "xi", "x_minus_next"))
    norm = W.fit_norm(layout, wm.cfg, x, u, xi, xn)
    d = W.state_delta(layout, wm.cfg, jnp.asarray(x), jnp.asarray(xn))
    dn = W.normalize_delta(norm, d)
    np.testing.assert_allclose(np.mean(dn, 0), 0, atol=1e-3)
    np.testing.assert_allclose(np.std(dn, 0), 1, atol=1e-3)
    np.testing.assert_allclose(W.denormalize_delta(norm, dn), d, atol=1e-5)
    # state_delta / apply_delta are inverse (heading-frame xy, wrapped yaw, last_action from u_chosen)
    back = W.apply_delta(layout, wm.cfg, jnp.asarray(x), d, u_chosen=jnp.asarray(uc))
    np.testing.assert_allclose(W.state_error(layout, back, jnp.asarray(xn)), 0, atol=1e-5)
    # normalized inputs
    z = W.raw_input(layout, wm.cfg, jnp.asarray(x), jnp.asarray(u), jnp.asarray(xi))
    zn = (z - norm["in_mean"]) / norm["in_std"]
    np.testing.assert_allclose(zn * norm["in_std"] + norm["in_mean"], z, atol=1e-5)
    # save / load of the bundle keeps the predictions
    bundle = wm.init(jax.random.PRNGKey(1), norm)
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"wm_test_{os.getpid()}.msgpack")
    WE.save_bundle(path, wm, bundle, schema)
    wm2, bundle2, meta = WE.load_bundle(path)
    os.remove(path)
    np.testing.assert_allclose(wm2.step(bundle2, x[:4], u[:4], xi[:4], uc[:4]), wm.step(bundle, x[:4], u[:4], xi[:4], uc[:4]),
                               atol=1e-6)


def _shock_batch(data, layout, k=6, shock_at=2):
    pos, _ = WD.split_window_starts(data, k)
    b = {f: np.array(v) for f, v in _batch(data, pos[10:14], k).items()}
    b["shock_next"][:] = 0
    b["shock_next"][0, shock_at] = 1.0
    b["x_plus_next"][0, shock_at] = b["x_minus_next"][0, shock_at] + 0.5   # the recorded shocked state
    return {f: jnp.asarray(v) for f, v in b.items()}


def test_teacher_forcing_at_shocks(toy):
    data, schema, layout = toy
    wm = _wm(layout)
    bundle = wm.init(jax.random.PRNGKey(2))
    k, j = 6, 2
    b = _shock_batch(data, layout, k, j)
    step_fn = lambda x, u, xi, uc: wm.step(bundle, x, u, xi, uc)
    preds = W.k_step_rollout_on_window(wm, step_fn, b, teacher_forcing=True)
    # step j+1 starts from the recorded x+ after the shock
    exp = wm.step(bundle, b["x_plus_next"][:, j], b["u_applied"][:, j + 1], b["xi"][:, j + 1], b["u_chosen"][:, j + 1])
    np.testing.assert_allclose(preds[0, j + 1], exp[0], atol=1e-5)
    # windows without shocks: x_plus_next is ignored
    b2 = dict(b)
    b2["x_plus_next"] = b["x_plus_next"].at[1:].add(10.0)
    p2 = W.k_step_rollout_on_window(wm, step_fn, b2, teacher_forcing=True)
    np.testing.assert_allclose(p2[1:], preds[1:], atol=1e-6)
    # changing the shocked x+ changes only the later predictions of that window
    b3 = dict(b)
    b3["x_plus_next"] = b["x_plus_next"].at[0, j].add(1.0)
    p3 = W.k_step_rollout_on_window(wm, step_fn, b3, teacher_forcing=True)
    np.testing.assert_allclose(p3[0, :j + 1], preds[0, :j + 1], atol=1e-6)
    assert float(jnp.abs(p3[0, j + 1:] - preds[0, j + 1:]).max()) > 1e-4
    # without teacher forcing the shock is ignored
    p4 = W.k_step_rollout_on_window(wm, step_fn, b3, teacher_forcing=False)
    p5 = W.k_step_rollout_on_window(wm, step_fn, b, teacher_forcing=False)
    np.testing.assert_allclose(p4, p5, atol=1e-6)


def test_method_b_masking(toy):
    data, schema, layout = toy
    wm = _wm(layout)
    bundle = wm.init(jax.random.PRNGKey(3))
    k = 6
    b = _shock_batch(data, layout, k, shock_at=2)
    m = W.window_mask_method_b(b["shock_next"])
    np.testing.assert_array_equal(np.array(m), [False, True, True, True])
    # a shock after the last target does not matter
    last = b["shock_next"].at[1, k - 1].set(1.0)
    np.testing.assert_array_equal(np.array(W.window_mask_method_b(last)), [False, True, True, True])
    p1 = jax.tree_util.tree_map(lambda a: a[0], bundle["params"])
    loss_b, aux = W.k_step_loss_single(wm, p1, bundle["norm"], b, method="B")
    clean = {f: v[1:] for f, v in b.items()}
    loss_clean, _ = W.k_step_loss_single(wm, p1, bundle["norm"], clean, method="A")
    np.testing.assert_allclose(loss_b, loss_clean, rtol=1e-5)
    assert abs(float(aux["valid_frac"]) - 0.75) < 1e-6
    # the masked window gets no gradient: perturbing its data leaves the loss unchanged
    b_mod = dict(b)
    b_mod["x_minus_next"] = b["x_minus_next"].at[0].add(5.0)
    np.testing.assert_allclose(W.k_step_loss_single(wm, p1, bundle["norm"], b_mod, method="B")[0], loss_b, rtol=1e-6)


def test_window_indexing():
    track = np.array([0] * 6 + [1] * 5)
    done = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1], float)
    term = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], float)   # track 0 ep 0 terminated; track 1 truncated
    np.testing.assert_array_equal(WD.valid_window_starts(track, done, 3), [0, 3, 6, 7, 8])
    st = WD.steps_to_termination(track, done, term)
    assert list(st[:3]) == [2, 1, 0] and all(st[3:] > 1000)
    np.testing.assert_array_equal(WD.near_failure_rows(track, done, term, 2), [0, 1, 1] + [0] * 8)


@pytest.mark.parametrize("delay", [0, 1])
@pytest.mark.parametrize("open_loop", [False, True])
def test_rollout_gradient_vs_finite_differences(toy, delay, open_loop):
    data, schema, layout = toy
    with jax.experimental.enable_x64():
        wm = _wm(layout, width=16, depth=2, ensemble_size=2)
        bundle = wm.init(jax.random.PRNGKey(4))
        bundle = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float64), bundle)
        A = layout.action_dim
        N, H = 3, 7
        x0 = jnp.asarray(data["x_plus"][:N], jnp.float64)
        xi = jnp.asarray(data["xi"][:N], jnp.float64)
        u0 = jnp.asarray(data["u_chosen"][:N], jnp.float64)
        rng = np.random.default_rng(0)
        Kp = jnp.asarray(rng.normal(0, 0.5, (layout.state_dim, A)))
        policy_fn = lambda x: jnp.tanh(x @ Kp)
        buf0 = jnp.asarray(rng.normal(0, 0.3, (N, delay + 1, A)))
        dstep = WD.synthetic_delay_step(delay)
        w = jnp.asarray(rng.normal(0, 1, (H, N, layout.state_dim)))

        def obj(u):
            traj = W.rollout(wm, bundle, x0, xi, buf0, u, policy_fn, H, delay_step_fn=dstep, open_loop=open_loop)
            return jnp.sum(w * jnp.sin(traj))

        g = jax.grad(obj)(u0)
        eps = 1e-6
        fd = np.zeros(u0.shape)
        for i in range(N):
            for a in range(A):
                e = jnp.zeros(u0.shape).at[i, a].set(eps)
                fd[i, a] = (obj(u0 + e) - obj(u0 - e)) / (2 * eps)
        if not open_loop:
            np.testing.assert_allclose(np.array(g), fd, rtol=1e-5, atol=1e-7)
        else:
            # open loop drops the policy's reaction: equal to FD of a rollout whose later actions are frozen
            _, acts = W.rollout(wm, bundle, x0, xi, buf0, u0, policy_fn, H, delay_step_fn=dstep, return_actions=True)

            def frozen(u):   # same rollout with the later actions fixed to their values at u0
                ut, buf = dstep(buf0, u)
                x = wm.step(bundle, x0, ut, xi, u)
                out = [x]
                for j in range(1, H):
                    ut, buf = dstep(buf, acts[j])
                    x = wm.step(bundle, x, ut, xi, acts[j])
                    out.append(x)
                return jnp.sum(w * jnp.sin(jnp.stack(out)))

            fd2 = np.zeros(u0.shape)
            for i in range(N):
                for a in range(A):
                    e = jnp.zeros(u0.shape).at[i, a].set(eps)
                    fd2[i, a] = (frozen(u0 + e) - frozen(u0 - e)) / (2 * eps)
            np.testing.assert_allclose(np.array(g), fd2, rtol=1e-5, atol=1e-7)
            assert np.abs(np.array(g) - fd).max() > 1e-6   # and it differs from the closed-loop gradient
        assert np.abs(fd).max() > 1e-4


def test_training_reduces_loss_and_resumes(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_wm_offline", os.path.join(os.path.dirname(__file__), "..", "experiments", "pncbf", "wm", "train_wm_offline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    common = ["--run_dir", str(tmp_path / "run"), "--data", "synthetic", "--synthetic_envs", "16", "--synthetic_T", "300",
              "--width", "32", "--depth", "2", "--batch", "64", "--updates_per_call", "20", "--log_every", "20",
              "--eval_every", "0", "--eval_ks", "1,8", "--eval_windows", "200", "--k_train", "4"]
    mod.main(common + ["--updates", "40"])
    ck = WE.load_msgpack(str(tmp_path / "run" / "ckpt_latest.msgpack"))
    assert int(ck["step"]) == 40
    mod.main(common + ["--updates", "200"])
    ck2 = WE.load_msgpack(str(tmp_path / "run" / "ckpt_latest.msgpack"))
    assert int(ck2["step"]) == 200
    wm, bundle, meta = WE.load_bundle(str(tmp_path / "run" / "wm_final.msgpack"))
    assert meta["updates"] == 200
    import json
    res = json.load(open(tmp_path / "run" / "eval_heldout.json"))
    g = res["overall"]["1"]
    assert g["q"]["model"] < g["q"]["persist"]


def _write_wmdata_shard(path, data, T, split="train"):
    """Write track-format synthetic data as a wm-data shard directory (time-major .npy files + meta.json)."""
    import json
    os.makedirs(path, exist_ok=True)
    N = len(data["done"]) // T
    tm = lambda a: np.swapaxes(a.reshape((N, T) + a.shape[1:]), 0, 1)
    table, inv = np.unique(data["xi"], axis=0, return_inverse=True)
    ep_step = np.zeros(len(data["done"]), np.int32)
    for i in range(1, len(ep_step)):
        ep_step[i] = 0 if (data["done"][i - 1] > 0 or data["track_id"][i] != data["track_id"][i - 1]) else ep_step[i - 1] + 1
    files = {"x_plus": data["x_plus"], "x_minus_next": data["x_minus_next"], "x_plus_next": data["x_plus_next"],
             "u_applied": data["u_applied"], "u_cmd": data["u_chosen"], "xi_idx": inv.reshape(-1).astype(np.int32),
             "done": data["done"] > 0, "terminated": data["terminated"] > 0, "perturbed_next": data["shock_next"] > 0,
             "dr_resampled_next": np.zeros(len(data["done"]), bool), "episode_step": ep_step}
    for f, v in files.items():
        np.save(os.path.join(path, f + ".npy"), tm(np.asarray(v)))
    np.save(os.path.join(path, "xi_table.npy"), table.astype(np.float32))
    json.dump({"split": split, "checkpoint": "toy"}, open(os.path.join(path, "meta.json"), "w"))
    return ep_step


def test_wmdata_shard_loader_matches_dense(toy, tmp_path):
    data, schema, layout = toy
    T = 300
    ep_step = _write_wmdata_shard(str(tmp_path / "train" / "a"), data, T)
    _write_wmdata_shard(str(tmp_path / "test" / "b"), data, T, split="test")
    L = WD.load_wmdata(str(tmp_path / "*" / "*"), log=lambda *a: None)
    comp, info = L["train"]
    assert L["heldout"][0] is not None and info[0]["x_minus_ne_next_x_plus_outside_kept"] == 0
    assert "x_minus_next" not in comp and "xi_table" in comp
    # window starts: same as dense except starts at episode_step == 0 or right after a shock (stale IMU)
    k = 8
    dense_starts = WD.valid_window_starts(data["track_id"], data["done"], k)
    prev_shock = np.concatenate([[0], data["shock_next"][:-1]]) > 0
    prev_shock[::T] = False
    keep = (ep_step[dense_starts] != 0) & ~prev_shock[dense_starts]
    np.testing.assert_array_equal(WD.data_window_starts(comp, k), dense_starts[keep])
    s = jnp.asarray(dense_starts[::7], jnp.int32)
    bd = WD.gather_windows(WD.to_device(data), s, k)
    bc = WD.gather_windows(WD.to_device(comp), s, k)
    for f in ("x_plus", "u_applied", "u_chosen", "xi", "x_minus_next", "shock_next"):
        np.testing.assert_allclose(bc[f], bd[f], atol=0, err_msg=f)
    # x_plus_next agrees wherever teacher forcing can use it (inside the window, before its last step)
    np.testing.assert_allclose(bc["x_plus_next"][:, :-1], bd["x_plus_next"][:, :-1], atol=0)
    rows = np.arange(0, len(data["done"]), 5)
    for a, b in zip(WD.rows_arrays(comp, rows), WD.rows_arrays(data, rows)):
        np.testing.assert_allclose(a, b, atol=0)
    # concatenation of two shards keeps xi_idx / xm_rows consistent
    both = WD.concat_data([comp, L["heldout"][0]])
    n = WD.n_rows(comp)
    for a, b in zip(WD.rows_arrays(both, rows + n), WD.rows_arrays(data, rows)):
        np.testing.assert_allclose(a, b, atol=0)
    act = WD.active_xi_dims(comp)
    assert len(act) == layout.xi_dim


def test_gated_grad_and_bf16(toy):
    data, schema, layout = toy
    wm = _wm(layout)
    bundle = wm.init(jax.random.PRNGKey(5))
    N, A = 12, layout.action_dim
    x0, xi, u0 = (jnp.asarray(data[f][:N]) for f in ("x_plus", "xi", "u_chosen"))
    pf = lambda x: jnp.tanh(x[:, :A])
    vfn = lambda u, x0, xi: jnp.sum(W.rollout(wm, bundle, x0, xi, None, u, pf, 4) ** 2, axis=(0, 2))
    v, g, idx = W.gated_value_and_grad(vfn, u0, (x0, xi), 5)
    g_full = jax.grad(lambda u: jnp.sum(vfn(u, x0, xi)))(u0)
    order = np.argsort(np.array(v))[:5]
    assert set(np.array(idx)) == set(order)
    np.testing.assert_allclose(g[idx], g_full[idx], rtol=1e-5, atol=1e-6)
    mask = np.ones(N, bool)
    mask[np.array(idx)] = False
    assert float(jnp.abs(g[mask]).max()) == 0.0
    # bf16 matmuls (f32 accumulation) give the same parameters' predictions up to bf16 rounding
    wm16 = _wm(layout, matmul_dtype="bfloat16")
    x16 = wm16.step(bundle, x0, u0, xi, u0)
    x32 = wm.step(bundle, x0, u0, xi, u0)
    assert x16.dtype == jnp.float32
    np.testing.assert_allclose(x16, x32, atol=5e-2)


def test_roll_wrap_and_post_shock_mask(toy):
    data, schema, layout = toy
    sch = dict(schema)
    sch["groups"] = dict(schema["groups"], roll_pitch=list(schema["groups"]["q"][:1]) + [schema["groups"]["q"][0] + 2])
    lay = W.WMLayout.from_schema(sch)          # roll = first dim of q (toy stand-in)
    assert lay.angle_idx == (schema["groups"]["q"][0],)
    cfg = W.WMConfig()
    r = lay.angle_idx[0]
    x = jnp.zeros((1, lay.state_dim)).at[0, r].set(3.1)
    xn = x.at[0, r].set(-3.1)                  # +3.1 -> -3.1 is a +0.083 rad turn, not -6.2
    d = W.state_delta(lay, cfg, x, xn)
    pidx = list(W.pred_idx(lay, cfg))
    np.testing.assert_allclose(d[0, pidx.index(r)], 2 * np.pi - 6.2, atol=1e-5)
    back = W.apply_delta(lay, cfg, x, d, u_chosen=jnp.zeros((1, lay.action_dim)))
    np.testing.assert_allclose(W.state_error(lay, back, xn)[0, r], 0, atol=1e-5)
    # post-shock step mask: corrupting the target of the step after a shock does not change the method-A loss
    wm = _wm(layout)
    bundle = wm.init(jax.random.PRNGKey(6))
    p1 = jax.tree_util.tree_map(lambda a: a[0], bundle["params"])
    b = _shock_batch(data, layout, 6, shock_at=2)
    l0 = W.k_step_loss_single(wm, p1, bundle["norm"], b, "A")[0]
    b2 = dict(b)
    b2["x_minus_next"] = b["x_minus_next"].at[0, 3].add(9.0)
    np.testing.assert_allclose(W.k_step_loss_single(wm, p1, bundle["norm"], b2, "A")[0], l0, rtol=1e-6)
    wm_off = _wm(layout, mask_post_shock_step=False)
    assert float(W.k_step_loss_single(wm_off, p1, bundle["norm"], b2, "A")[0]) > float(l0) + 1e-3
