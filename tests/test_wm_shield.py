"""T15: H-step safety value (safety_value.py) and the WM shield (wm_shield.py) on a tiny world model (CPU)."""
import math

import numpy as np
import pytest
import jax
import jax.numpy as jnp
enable_x64 = lambda: jax.enable_x64(True)

from rl_x.algorithms.ncbf_ppo.flax_full_jit import safety_value as SV
from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_shield as WS
from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_state
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model as W

A = 2
STATE_GROUPS = [("joint_pos", A), ("joint_vel", A), ("imu_ang_vel", 3), ("projected_gravity", 3), ("last_action", A),
                ("base_lin_vel", 3), ("imu_height", 1), ("roll_pitch", 2), ("base_xy", 2), ("base_yaw", 1)]
XI_GROUPS = [("nominal_joint_pos", A), ("delay_mixed", 1), ("delay_nr_steps", 1), ("curriculum_coeff", 1),
             ("nominal_imu_height", 1)]
NOM_H = 0.3


def _setup(delay=0, H=6, open_loop=False, events=("fall",), T_l=(1.0,), dtype=jnp.float32, n=8, seed=0,
           source="wm"):
    lay, xl = wm_state.Layout(STATE_GROUPS), wm_state.Layout(XI_GROUPS)
    schema = {"groups": {k: [s.start, s.stop] for k, s in lay.slices.items()}, "state_dim": lay.dim,
              "action_dim": A, "xi_dim": xl.dim}
    wm = W.WorldModel(W.WMLayout.from_schema(schema), W.WMConfig(width=16, depth=2))
    k = jax.random.split(jax.random.PRNGKey(seed), 8)
    bundle = wm.init(k[0])
    # larger output layer + delta scale so the imagined dynamics are visibly nonlinear in u
    bundle["params"] = jax.tree_util.tree_map(lambda p: p * 30.0 if p.ndim == 3 and p.shape[-1] == wm.n_out else p,
                                              bundle["params"])
    bundle["norm"] = dict(bundle["norm"], d_std=jnp.full(wm.n_out, 0.02))
    bundle = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype), bundle)
    obs_dim = 3 * A + 9
    pol = {"W": 0.5 * jax.random.normal(k[1], (obs_dim, A), dtype), "b": 0.1 * jax.random.normal(k[2], (A,), dtype)}
    policy_fn = lambda p, o: jnp.tanh(o @ p["W"] + p["b"])
    obs_fn = lambda x, cmd, xi: wm_state.policy_obs_from_state(x, cmd, xl.get(xi, "nominal_joint_pos"), lay)
    mcfg = wm_state.MarginConfig(height_curriculum_scaled=False, use_qvel=True)
    spec = SV.ImaginationSpec(lay=lay, xl=xl, obs_fn=obs_fn, margin_cfg=mcfg, max_nr_delay_steps=1)
    x0 = 0.1 * jax.random.normal(k[3], (n, lay.dim), dtype)
    x0 = x0.at[:, lay.slices["imu_height"]].set(NOM_H * 0.8 + 0.03 * jax.random.normal(k[4], (n, 1), dtype))
    xi = jnp.concatenate([0.1 * jnp.ones((n, A)), jnp.zeros((n, 1)), jnp.full((n, 1), float(delay)), jnp.ones((n, 1)),
                          jnp.full((n, 1), NOM_H)], axis=-1).astype(dtype)
    buf = 0.2 * jax.random.normal(k[5], (n, 2, A), dtype)
    inputs = SV.ShieldInputs(x0=x0, xi_full=xi, delay_buffer=buf, delay_steps=jnp.full((n,), delay, jnp.int32),
                             command=0.3 * jnp.ones((n, 3), dtype), first_step=jnp.zeros((n,), bool),
                             stale=jnp.zeros((n,), bool))
    cfg = SV.SafetyValueConfig(source=source, H=H, events=events, T_l=T_l, open_loop=open_loop)
    vf = SV.make_safety_value_fn(cfg, spec, wm=wm, policy_fn=policy_fn)
    u0 = 0.3 * jax.random.normal(k[6], (n, A), dtype)
    params = {"wm": bundle, "policy": pol}
    return dict(wm=wm, bundle=bundle, pol=pol, policy_fn=policy_fn, spec=spec, lay=lay, xl=xl, inputs=inputs,
                cfg=cfg, vf=vf, u0=u0, params=params, delay=delay)


def _numpy_value(s, u0, T_l=1.0):
    """Brute force: python loop over the steps, explicit delay buffer, numpy margins, P = Π sigmoid(ℓ/T)."""
    lay, inp, wm, bundle, cfg = s["lay"], s["inputs"], s["wm"], s["bundle"], s["cfg"]
    sl = lay.slices
    n = u0.shape[0]
    out = np.zeros(n)
    for i in range(n):
        x = np.asarray(inp.x0[i], np.float64)
        buf = [np.asarray(b, np.float64) for b in np.asarray(inp.delay_buffer[i])]
        xi = np.asarray(inp.xi_full[i]); cmd = np.asarray(inp.command[i])
        u = np.asarray(u0[i], np.float64)
        logp = 0.0
        for j in range(cfg.H):
            if j > 0:
                o = np.concatenate([(x[sl["joint_pos"]] - xi[:A]) / 3.14, x[sl["joint_vel"]] / 100.0,
                                    x[sl["last_action"]] / 10.0, np.clip(x[sl["imu_ang_vel"]] / 50.0, -1, 1), cmd,
                                    x[sl["projected_gravity"]]])
                o = np.clip(o, -10, 10)
                u = np.tanh(o @ np.asarray(s["pol"]["W"], np.float64) + np.asarray(s["pol"]["b"], np.float64))
            buf = buf[1:] + [u]
            ut = buf[len(buf) - 1 - s["delay"]]
            x = np.asarray(wm.step(bundle, jnp.asarray(x[None], bundle["norm"]["d_std"].dtype),
                                   jnp.asarray(ut[None], bundle["norm"]["d_std"].dtype),
                                   jnp.asarray(xi[None], bundle["norm"]["d_std"].dtype),
                                   u_chosen=jnp.asarray(u[None], bundle["norm"]["d_std"].dtype))[0], np.float64)
            h = (x[sl["imu_height"]][0] - 0.8 * NOM_H) / 0.05
            r, p, y = x[sl["roll_pitch"]][0], x[sl["roll_pitch"]][1], x[sl["base_yaw"]][0]
            from scipy.spatial.transform import Rotation
            v = Rotation.from_euler("xyz", [r, p, y]).as_matrix() @ x[sl["base_lin_vel"]]
            q = (100.0 - 1e-3 - np.max(np.abs(v))) / 10.0
            ell = min(h, q)
            logp += np.log(1.0 / (1.0 + np.exp(-ell / T_l)))
        P = math.exp(logp)
        out[i] = math.log(P / (1.0 - P))
    return out


@pytest.mark.parametrize("delay", [0, 1])
def test_value_matches_bruteforce(delay):
    with enable_x64():
        s = _setup(delay=delay, dtype=jnp.float64)
        v, aux = s["vf"](s["params"], s["inputs"], s["u0"])
        ref = _numpy_value(s, np.asarray(s["u0"]))
        assert v.shape == (8, 1) and aux["logp"].shape == (8, 1)
        np.testing.assert_allclose(np.asarray(v[:, 0]), ref, rtol=1e-8, atol=1e-8)
        assert np.ptp(ref) > 0.1   # non-trivial values


def test_value_numerics_extremes():
    lp = jnp.array([-1e-20, -1e-6, -0.5, -0.7, -1.0, -50.0, -1e4])
    v = SV.logit_from_logp(lp, logp_max=-1e-30)
    p = jnp.exp(lp)
    assert jnp.all(jnp.isfinite(v))
    np.testing.assert_allclose(np.asarray(v[2:5]), np.asarray(jnp.log(p / (1 - p))[2:5]), rtol=1e-5)
    np.testing.assert_allclose(float(v[-1]), -1e4, rtol=1e-6)
    g = jax.grad(lambda l: jnp.sum(SV.logit_from_logp(l)))(lp)
    assert jnp.all(jnp.isfinite(g))


def _fd(f, u, eps=1e-6):
    g = np.zeros(u.shape)
    for i in range(u.shape[0]):
        for a in range(u.shape[1]):
            e = jnp.zeros_like(u).at[i, a].set(eps)
            g[i, a] = (f(u + e)[i] - f(u - e)[i]) / (2 * eps)
    return g


def _frozen_value_fn(s):
    """Open loop reference: the same rollout with the later actions fixed to their values at u0 (event 0)."""
    wm, b, inp, spec, cfg = s["wm"], s["bundle"], s["inputs"], s["spec"], s["cfg"]
    pol = lambda x: s["policy_fn"](s["pol"], spec.obs_fn(x, inp.command, inp.xi_full))
    ds = jnp.full((inp.x0.shape[0],), s["delay"], jnp.int32)
    dfn = lambda bb, u: wm_state.delay_step(bb, u, ds)
    _, acts = W.rollout(wm, b, inp.x0, inp.xi_full, inp.delay_buffer, s["u0"], pol, cfg.H, delay_step_fn=dfn,
                        return_actions=True)

    def f(u):
        x, buf, xs = inp.x0, inp.delay_buffer, []
        for j in range(cfg.H):
            uj = u if j == 0 else acts[j]
            ut, buf = dfn(buf, uj)
            x = wm.step(b, x, ut, inp.xi_full, u_chosen=uj)
            xs.append(x)
        m = SV.event_margins(cfg, spec, jnp.stack(xs), inp.xi_full)
        return SV.values_from_margins(cfg, m)[0]
    return f


@pytest.mark.parametrize("delay,open_loop", [(0, False), (1, False), (0, True), (1, True)])
def test_grad_V_and_G_vs_finite_differences(delay, open_loop):
    with enable_x64():
        s = _setup(delay=delay, open_loop=open_loop, dtype=jnp.float64, n=4,
                   events=("fall", "fall_physical"), T_l=(1.0, 0.5))
        vf, params, inp = s["vf"], s["params"], s["inputs"]
        f = lambda u: vf(params, inp, u)[0][:, 0]
        g = jax.grad(lambda u: jnp.sum(f(u)))(s["u0"])
        frozen = _frozen_value_fn(s)
        fd = _fd((lambda u: frozen(u)[:, 0]) if open_loop else f, s["u0"])
        if delay == 1:   # u_t is applied one step later; it still matters (via the dynamics) unless open loop+H tiny
            assert np.abs(fd).max() > 1e-6
        np.testing.assert_allclose(np.asarray(g), fd, rtol=1e-5, atol=1e-8)
        cfg = WS.ShieldConfig(tau=0.1)
        shield = WS.make_shield(vf, cfg, nr_events=2)
        v_prev = jnp.array([[0.5, 1.0], [2.0, -1.0], [0.0, 0.0], [3.0, 3.0]])
        fG = ((lambda u: WS.combine_residual(cfg, frozen(u), v_prev)) if open_loop
              else (lambda u: shield.G_and_V(params, inp, u, v_prev)[0]))
        (_, _, _), gG = shield.G_value_and_grad(params, inp, s["u0"], v_prev)
        np.testing.assert_allclose(np.asarray(gG), _fd(fG, s["u0"]), rtol=1e-5, atol=1e-8)


def test_open_loop_differs_from_closed_loop():
    with enable_x64():
        sc = _setup(delay=0, open_loop=False, dtype=jnp.float64, n=4)
        so = _setup(delay=0, open_loop=True, dtype=jnp.float64, n=4)
        gc = jax.grad(lambda u: jnp.sum(sc["vf"](sc["params"], sc["inputs"], u)[0]))(sc["u0"])
        go = jax.grad(lambda u: jnp.sum(so["vf"](so["params"], so["inputs"], u)[0]))(so["u0"])
        vc, vo = sc["vf"](sc["params"], sc["inputs"], sc["u0"])[0], so["vf"](so["params"], so["inputs"], so["u0"])[0]
        np.testing.assert_allclose(np.asarray(vc), np.asarray(vo))     # same values
        assert np.abs(np.asarray(gc - go)).max() > 1e-6                 # different gradients


def test_params_get_no_gradient_unless_policy_grad():
    s = _setup(n=4)
    g = jax.grad(lambda p: jnp.sum(s["vf"](p, s["inputs"], s["u0"])[0]))(s["params"])
    assert all(float(jnp.abs(x).max()) == 0.0 for x in jax.tree_util.tree_leaves(g))
    vf2 = SV.make_safety_value_fn(s["cfg"], s["spec"], wm=s["wm"], policy_fn=s["policy_fn"], policy_grad=True)
    g2 = jax.grad(lambda p: jnp.sum(vf2(p, s["inputs"], s["u0"])[0]))(s["params"])
    assert float(jnp.abs(g2["policy"]["W"]).max()) > 0.0
    assert all(float(jnp.abs(x).max()) == 0.0 for x in jax.tree_util.tree_leaves(g2["wm"]))


def test_projection_solves_single_constraint_qp():
    from scipy.optimize import minimize
    rng = np.random.default_rng(0)
    cfg = WS.ShieldConfig(max_delta_u=0.0)
    for _ in range(10):
        u0, a = rng.normal(size=4), rng.normal(size=4)
        G = -abs(rng.normal())
        c = a @ u0 - G
        v, ok = WS.project_halfspace(jnp.asarray(u0)[None], jnp.asarray(a)[None], jnp.asarray([G]), cfg)
        res = minimize(lambda x: np.sum((x - u0) ** 2), u0, jac=lambda x: 2 * (x - u0),
                       constraints=[{"type": "ineq", "fun": lambda x: a @ x - c, "jac": lambda x: a}], method="SLSQP",
                       options={"ftol": 1e-14})
        np.testing.assert_allclose(np.asarray(v[0]), res.x, atol=1e-6)
        assert a @ np.asarray(v[0]) >= c - 1e-6
    # satisfied constraint: no change
    v, _ = WS.project_halfspace(jnp.ones((1, 3)), jnp.ones((1, 3)), jnp.array([0.2]), cfg)
    np.testing.assert_allclose(np.asarray(v), 1.0)


def test_shield_respects_max_delta_u_and_reduces_G():
    s = _setup(n=64, H=8)
    vf = s["vf"]
    # require a high value so that many envs are violated
    cfg = WS.ShieldConfig(p_c=0.98, gate="full", relinearize=1, max_delta_u=0.5)
    shield = WS.make_shield(vf, cfg)
    st = shield.init_state(64)
    u, st2, out = jax.jit(shield.apply)(s["params"], s["inputs"], s["u0"], st)
    act = np.asarray(out["active"])
    assert act.sum() > 5
    assert float(out["delta_u"].max()) <= 0.5 + 1e-5
    np.testing.assert_allclose(np.asarray(out["delta_u"]), np.linalg.norm(np.asarray(u - s["u0"]), axis=-1), atol=1e-6)
    Gb, Ga = np.asarray(out["G_before"]), np.asarray(out["G_after"])
    assert np.mean(Ga[act] > Gb[act]) > 0.9
    # G_after is the true G at u
    G_true = shield.G_and_V(s["params"], s["inputs"], u, out["v_prev"])[0]
    np.testing.assert_allclose(np.asarray(G_true), Ga, rtol=1e-4, atol=1e-4)
    # inactive envs untouched; V^prev for the next step = V at the executed action
    np.testing.assert_allclose(np.asarray(u)[~act], np.asarray(s["u0"])[~act])
    V_exec = vf(s["params"], s["inputs"], u)[0]
    np.testing.assert_allclose(np.asarray(st2["v_prev"]), np.asarray(V_exec), rtol=1e-4, atol=1e-4)
    # strength 0 = no shield
    u0s, _, _ = shield.apply(s["params"], s["inputs"], s["u0"], st, strength=0.0)
    np.testing.assert_allclose(np.asarray(u0s), np.asarray(s["u0"]))


def _scalar_value_fn(params, inputs, u):
    """V(u) = log(1 + u_0) - 1 (concave in u_0): the linearization undershoots."""
    v = jnp.log1p(jnp.maximum(u[:, 0], -0.99)) - 1.0
    return v[:, None], {"logp": v[:, None], "min_margin": v[:, None]}


def _scalar_inputs(n, first=True):
    z = jnp.zeros((n, 1))
    return SV.ShieldInputs(x0=z, xi_full=z, delay_buffer=z, delay_steps=jnp.zeros((n,), jnp.int32), command=z,
                           first_step=jnp.full((n,), first), stale=jnp.zeros((n,), bool))


def test_relinearization_reduces_violation():
    n = 3
    u0 = jnp.zeros((n, 2))
    res = {}
    for r in (0, 1, 2):
        cfg = WS.ShieldConfig(gate="full", relinearize=r, max_delta_u=5.0, p_c=0.5)
        sh = WS.make_shield(_scalar_value_fn, cfg)
        u, _, out = sh.apply({}, _scalar_inputs(n), u0, sh.init_state(n))
        res[r] = float(out["G_after"][0])
        if r == 0:
            np.testing.assert_allclose(np.asarray(u[:, 0]), 1.0, atol=1e-6)   # one Newton step: u = 1
        assert int(out["nr_projections"][0]) == 1 + r
    np.testing.assert_allclose(res[0], math.log(2) - 1, atol=1e-6)
    assert res[0] < res[1] < res[2] < 0.0 or (res[0] < res[1] and res[2] >= res[1])
    assert abs(res[1]) < 0.2 * abs(res[0])


def test_first_step_and_stale_handling():
    n = 4
    cfg = WS.ShieldConfig(gate="full", relinearize=0, p_c=0.7, alpha=0.5)
    sh = WS.make_shield(_scalar_value_fn, cfg)
    g_c = math.log(0.7 / 0.3)
    st = {"v_prev": jnp.full((n, 1), 5.0)}
    inp = _scalar_inputs(n, first=False)._replace(first_step=jnp.array([True, False, True, False]),
                                                  stale=jnp.array([False, False, True, True]))
    u0 = jnp.full((n, 2), 0.5)
    u, st2, out = sh.apply({}, inp, u0, st)
    np.testing.assert_allclose(np.asarray(out["v_prev"][:, 0]), [g_c, 5.0, g_c, 5.0], rtol=1e-6)
    V = math.log1p(0.5) - 1.0
    np.testing.assert_allclose(float(out["G_before"][0]), V - g_c, rtol=1e-5)                       # first step
    np.testing.assert_allclose(float(out["G_before"][1]), V - 0.5 * 5.0 - 0.5 * g_c, rtol=1e-5)     # residual
    assert not bool(out["active"][2]) and not bool(out["active"][3])                                 # stale
    np.testing.assert_allclose(np.asarray(u[2:]), np.asarray(u0[2:]))
    assert bool(out["active"][0]) and bool(out["active"][1])
    np.testing.assert_allclose(np.asarray(st2["v_prev"][2:, 0]), g_c, rtol=1e-6)                    # stale -> γ_c
    # skip_stale off: stale rows are shielded
    sh2 = WS.make_shield(_scalar_value_fn, WS.ShieldConfig(gate="full", relinearize=0, p_c=0.7, skip_stale=False))
    _, _, out2 = sh2.apply({}, inp, u0, st)
    assert bool(out2["active"][2]) and bool(out2["active"][3])


def test_gated_equals_full_on_gated_envs():
    s = _setup(n=32, H=6)
    base = dict(p_c=0.95, relinearize=1, max_delta_u=0.5)
    full = WS.make_shield(s["vf"], WS.ShieldConfig(gate="full", **base))
    gated = WS.make_shield(s["vf"], WS.ShieldConfig(gate="gated", gate_frac=0.25, **base))
    st = full.init_state(32)
    uf, sf, of = full.apply(s["params"], s["inputs"], s["u0"], st)
    ug, sg, og = gated.apply(s["params"], s["inputs"], s["u0"], st)
    gm = np.asarray(og["gate_mask"])
    assert gm.sum() == 8
    np.testing.assert_allclose(np.asarray(og["G_before"]), np.asarray(of["G_before"]), rtol=1e-5, atol=1e-5)
    # gate = the 8 lowest G(u0)
    assert set(np.argsort(np.asarray(of["G_before"]))[:8]) == set(np.where(gm)[0])
    np.testing.assert_allclose(np.asarray(ug)[gm], np.asarray(uf)[gm], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ug)[~gm], np.asarray(s["u0"])[~gm])
    assert np.asarray(of["active"])[~gm].sum() > 0          # full mode corrects some envs the gate leaves out
    assert not np.asarray(og["active"])[~gm].any()
    # functional wrapper (d-train)
    fn = WS.make_shield_fn(WS.ShieldConfig(gate="gated", gate_frac=0.25, **base), s["vf"])
    u2, v_exec, diag = fn(s["params"], s["inputs"], s["u0"], st["v_prev"])
    np.testing.assert_allclose(np.asarray(u2), np.asarray(ug), atol=1e-6)
    np.testing.assert_allclose(np.asarray(v_exec), np.asarray(sg["v_prev"]), atol=1e-6)
    assert diag["v0"].shape == (32,) and diag["delta_u_norm"].shape == (32,)


def test_shield_differentiable_wrt_u0():
    s = _setup(n=16, H=4)
    fn = WS.make_shield_fn(WS.ShieldConfig(p_c=0.95, gate="gated"), s["vf"])
    v_prev = jnp.zeros((16, 1))
    g = jax.grad(lambda u: jnp.sum(fn(s["params"], s["inputs"], u, v_prev)[0] ** 2))(s["u0"])
    assert jnp.all(jnp.isfinite(g))


def test_softmin_reduces_to_min():
    x = jnp.array([[0.3, -0.2, 1.0], [2.0, 2.0, 5.0]])
    np.testing.assert_allclose(np.asarray(WS.softmin(x, 1e-4)), [-0.2, 2.0 - 1e-4 * math.log(2)], atol=1e-6)
    np.testing.assert_allclose(np.asarray(WS.softmin(x, 0.0)), [-0.2, 2.0])
    assert float(WS.softmin(x, 0.1)[0]) <= -0.2
    cfg = WS.ShieldConfig(tau=1e-5, alpha=0.5, p_c=0.5)
    v, vp = jnp.array([[1.0, 0.0]]), jnp.array([[0.0, 4.0]])
    np.testing.assert_allclose(float(WS.combine_residual(cfg, v, vp)[0]), min(1.0, 0.0 - 2.0), atol=1e-4)
    cfg2 = WS.ShieldConfig(tau=1e-5, combine="value")
    np.testing.assert_allclose(float(WS.combine_residual(cfg2, v, vp)[0]), 0.0 - 0.5 * 0.0, atol=1e-4)


def test_shield_grad_finite_on_uncorrected_rows():
    """d-train repro: rows with G(u0) >= 0 inside the gate must not give NaN gradients w.r.t. u0."""
    def vf(params, inputs, u):
        v = jnp.sum(u, axis=-1) - 1.0
        return v[:, None], {"logp": v[:, None], "min_margin": v[:, None]}
    n = 6
    u0 = jnp.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.1], [2.0, 0.0], [-1.0, 0.5], [0.5, 0.5]])
    for gate, frac in (("gated", 1.0), ("gated", 0.5), ("full", 1.0)):
        for relin in (0, 1):
            fn = WS.make_shield_fn(WS.ShieldConfig(gate=gate, gate_frac=frac, relinearize=relin), vf)
            loss = lambda u: jnp.sum((u - fn({}, _scalar_inputs(n), u, jnp.zeros((n, 1)))[0]) ** 2)
            g = jax.grad(loss)(u0)
            assert jnp.all(jnp.isfinite(g)), (gate, frac, relin, g)
            _, _, d = fn({}, _scalar_inputs(n), u0, jnp.zeros((n, 1)))
            assert bool(jnp.any(d["active"])) and not bool(jnp.all(d["active"]))


def test_speed_switches_equivalent():
    """gate_vjp=residuals and relin_frac=gate_frac give the default shield's result; scan_unroll keeps V."""
    import dataclasses
    s = _setup(n=32, H=6)
    base = WS.ShieldConfig(p_c=0.95, gate="gated", gate_frac=0.25, relinearize=1, max_delta_u=0.5)
    st = {"v_prev": jnp.zeros((32, 1))}
    ref = WS.make_shield(s["vf"], base).apply(s["params"], s["inputs"], s["u0"], st)
    for kw in ({"gate_vjp": "residuals"}, {"relin_frac": 0.25}, {"gate_vjp": "residuals", "relin_frac": 0.25}):
        out = WS.make_shield(s["vf"], dataclasses.replace(base, **kw)).apply(s["params"], s["inputs"], s["u0"], st)
        np.testing.assert_allclose(np.asarray(out[0]), np.asarray(ref[0]), rtol=1e-4, atol=1e-5, err_msg=str(kw))
        for k in ("G_before", "G_after", "gate_mask", "active", "nr_projections"):
            np.testing.assert_allclose(np.asarray(out[2][k]), np.asarray(ref[2][k]), rtol=1e-4, atol=1e-5,
                                       err_msg=f"{kw} {k}")
    assert np.asarray(ref[2]["nr_projections"]).max() == 2
    # relin_frac smaller than the number of still-violated envs: fewer second projections, same first step
    small = WS.make_shield(s["vf"], dataclasses.replace(base, relin_frac=1 / 32)).apply(s["params"], s["inputs"],
                                                                                      s["u0"], st)
    assert int((np.asarray(small[2]["nr_projections"]) == 2).sum()) <= 1
    np.testing.assert_array_equal(np.asarray(small[2]["active"]), np.asarray(ref[2]["active"]))
    cfg5 = dataclasses.replace(s["cfg"], scan_unroll=5)
    vf5 = SV.make_safety_value_fn(cfg5, s["spec"], wm=s["wm"], policy_fn=s["policy_fn"])
    np.testing.assert_allclose(np.asarray(vf5(s["params"], s["inputs"], s["u0"])[0]),
                               np.asarray(s["vf"](s["params"], s["inputs"], s["u0"])[0]), rtol=1e-5, atol=1e-6)


def test_secant_relinearization():
    n = 3
    u0 = jnp.zeros((n, 2))
    sh = WS.make_shield(_scalar_value_fn, WS.ShieldConfig(gate="full", relinearize=1, relin_mode="secant",
                                                          max_delta_u=5.0))
    u, _, out = sh.apply({}, _scalar_inputs(n), u0, sh.init_state(n))
    s_star = -1.0 / (-1.0 - (math.log(2) - 1.0))   # G0 / (G0 - G1), G0 = -1, G1 = G(u=1) = log 2 - 1
    np.testing.assert_allclose(np.asarray(u[:, 0]), s_star, rtol=1e-5)
    np.testing.assert_allclose(float(out["G_after"][0]), math.log1p(s_star) - 1.0, rtol=1e-5)
    assert math.log(2) - 1 < float(out["G_after"][0]) < 0.0
    assert int(out["nr_projections"][0]) == 2
    # cap binding at the first projection: no secant extension
    sh2 = WS.make_shield(_scalar_value_fn, WS.ShieldConfig(gate="full", relinearize=1, relin_mode="secant",
                                                           max_delta_u=0.5))
    u2, _, out2 = sh2.apply({}, _scalar_inputs(n), u0, sh2.init_state(n))
    np.testing.assert_allclose(np.asarray(u2[:, 0]), 0.5, rtol=1e-5)
    assert int(out2["nr_projections"][0]) == 1


def test_speed_switches_grad_finite():
    def vf(params, inputs, u):
        v = jnp.log1p(jnp.maximum(jnp.sum(u, axis=-1), -0.9)) - 0.5
        return v[:, None], {"logp": v[:, None], "min_margin": v[:, None]}
    n = 6
    u0 = jnp.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.1], [2.0, 0.0], [-0.3, 0.1], [0.5, 0.5]])
    for kw in ({"gate_vjp": "residuals"}, {"relin_mode": "secant"}, {"relin_frac": 0.2}, {"post_check": False}):
        fn = WS.make_shield_fn(WS.ShieldConfig(gate="gated", gate_frac=1.0, **kw), vf)
        g = jax.grad(lambda u: jnp.sum((u - fn({}, _scalar_inputs(n), u, jnp.zeros((n, 1)))[0]) ** 2))(u0)
        assert jnp.all(jnp.isfinite(g)), kw
