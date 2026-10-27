"""Method D: online world-model training + WM shield in the PPO loop (plan v3.2 sec. 4.9, P2 / T17) and the
internalization terms (sec. 4.7, T16). Active with `algorithm.method=d`; the default (`"default"`) leaves ppo.py
exactly as before (no D state, no extra work).

Per learning iteration (`PPO.train`):
  1. rollout: policy sample u0 -> WM shield (once the warm start is over) -> env. Per step the record holds
     x+_t, u_t (executed, = last_action of x_{t+1}), u~_t, xi_t (full and the WM's active dims), the delay buffer,
     the command, x-_{t+1}, x+_{t+1}, the shock / DR flag, done / terminated, the stale-IMU flag and V^prev.
     The env must run with `environment.record_wm_state=True` (checked at start-up).
  2. `iteration_update`: (a) evaluate the WM on the new rollout before training on it (k_train-step loss, 25-step
     fall-prediction AUC "replay" = WM along the recorded actions with method-A teacher forcing at shocks, and
     "imagination" = the shield's V(x_t, u_t) closed loop through the deterministic policy; label = the episode
     terminated within 25 steps); (b) update the normalizers (running statistics over the first
     `norm_update_iterations` iterations, frozen afterwards; frozen from the start with an init bundle);
     (c) write the rollout into the replay (last `replay_steps` steps per env, time-major ring) and the windows that
     end in a termination into the separate negative buffer (segments of `neg_segment_len` rows ending at the
     terminating step); (d) `wm_updates_per_iteration` Adam updates of the k_train-step loss (T13: method A, stale-
     IMU masks: no window starts at episode step 0 or right after a shock / DR step, no loss on the step after a
     shock), batch `wm_batch` windows with `neg_ratio` from the negative buffer; (e) warm-start gate.
  3. PPO update (+ anticipation loss through the WM shield, + D2).

Warm start (plan 4.9): the shield stays off until the gate metric (`warm_start_metric`, default the replay AUC)
is >= `warm_start_auc` for `warm_start_consecutive` iterations with >= `warm_start_min_positives` falls, or until
`warm_start_max_frac` of training; then it latches on ("wm_ready"). Shield strength s = wm_ready * fade(update)
with fade = 1 until the last `fade_out_frac` of training and then linear to 0 (u = u0 + s * du).

Gradient flow (plan 4.8): WM loss -> psi only. Anticipation loss -> theta through u0 = mu_theta(s) (and through
the shield with anticipation_through_shield), psi and the imagined policy stop_gradient. D2 -> theta through u0
and the imagined closed-loop policy, psi stop_gradient.

Checkpoint (runner.resume): WM params, Adam state, normalizers + their running statistics, the warm-start state
(wm_ready, switch update, AUC streak) are in the resume checkpoint ("d"). NOT restored (like the NCBF negative
buffer): the replay ring, the negative buffer and V^prev; after a requeue they refill from the next rollouts (the
first iteration after a resume trains the WM on 128 steps per env only). `models/wm_latest.msgpack` holds the WM
bundle in the T13 format (world_model_eval.load_bundle) next to latest.model.

Safety value + shield: agent d-shield's safety_value.py (SV) and wm_shield.py (WS), T15.
"""
import dataclasses
import logging
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import optax

from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_state
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model as W
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model_eval as WE
from rl_x.algorithms.ncbf_ppo.flax_full_jit import safety_value as SV
from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_shield as WS

rlx_logger = logging.getLogger("rl_x")

METHODS = ("default", "d")
ROW_FIELDS = ("x_plus", "u_applied", "u_chosen", "xi", "x_minus_next", "x_plus_next", "shock_next", "done",
              "terminated", "start_ok")


@dataclass(frozen=True)
class DConfig:
    # world model (T13 config)
    wm_width: int = 512
    wm_depth: int = 3
    wm_layer_norm: bool = True
    wm_activation: str = "elu"
    k_train: int = 16
    wm_method: str = "A"
    use_xi: bool = True
    stale_imu_mask: bool = True
    wm_lr: float = 2e-4
    wm_grad_clip: float = 1.0
    wm_batch: int = 4096
    neg_ratio: float = 0.3
    wm_updates_per_iteration: int = 50
    replay_steps: int = 1000
    neg_capacity: int = 4096
    neg_segment_len: int = 32
    neg_new_per_iteration: int = 256
    norm_update_iterations: int = 100
    wm_init_bundle: str = ""
    xi_probe_envs: int = 256
    eval_windows: int = 2048
    auc_horizon: int = 25
    # shield
    shield_enabled: bool = True
    H: int = 25
    T_l: float = 1.0
    p_c: float = 0.5
    alpha: float = 0.5
    max_delta_u: float = 0.5
    relinearize: int = 1
    gate_frac: float = 0.25
    shield_bf16: bool = True
    warm_start_metric: str = "replay"     # replay | imagination
    warm_start_auc: float = 0.9
    warm_start_consecutive: int = 3
    warm_start_min_positives: int = 20
    warm_start_max_frac: float = 0.05
    # internalization (T16)
    anticipation_coef: float = 0.01
    anticipation_through_shield: bool = True
    anticipation_subset: int = 1024        # per PPO minibatch (second order through the 25-step shield is costly)
    fade_out_frac: float = 0.2
    d2_coef: float = 0.0
    d2_subset: int = 1024
    d2_H: int = 25

    @staticmethod
    def from_config(cfg):
        fields = {f.name: f.type for f in dataclasses.fields(DConfig)}
        kw = {}
        for name in fields:
            if name in cfg:
                default = getattr(DConfig, name)
                kw[name] = type(default)(cfg[name])
        out = DConfig(**kw)
        if out.wm_method not in ("A", "B"):
            raise ValueError("d.wm_method must be A or B")
        if out.warm_start_metric not in ("replay", "imagination"):
            raise ValueError("d.warm_start_metric must be replay or imagination")
        if out.replay_steps < out.neg_segment_len or out.neg_segment_len < out.k_train:
            raise ValueError("need replay_steps >= neg_segment_len >= k_train")
        return out


def config_defaults(config_dict_module):
    """The `algorithm.d` ConfigDict with the DConfig defaults (default_config.py)."""
    d = config_dict_module.ConfigDict()
    for f in dataclasses.fields(DConfig):
        setattr(d, f.name, getattr(DConfig, f.name))
    return d


# ---------------------------------------------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------------------------------------------
def fade_strength(update, nr_updates, fade_out_frac):
    """1 until the last fade_out_frac of training, then linear to 0 at the last update (0 = no fade-out)."""
    if fade_out_frac <= 0:
        return jnp.asarray(1.0, jnp.float32)
    frac = jnp.asarray(update, jnp.float32) / jnp.maximum(jnp.asarray(nr_updates, jnp.float32), 1.0)
    return jnp.clip((1.0 - frac) / fade_out_frac, 0.0, 1.0)


def warm_start_gate(wm_ready, streak, auc, nr_pos, update, cfg: DConfig, nr_updates):
    """Returns (wm_ready', streak', switched_now). Latched: once on, stays on."""
    good = jnp.isfinite(auc) & (auc >= cfg.warm_start_auc) & (nr_pos >= cfg.warm_start_min_positives)
    streak = jnp.where(good, streak + 1, 0)
    cap = jnp.asarray(update + 1, jnp.float32) >= cfg.warm_start_max_frac * nr_updates
    new = wm_ready | (streak >= cfg.warm_start_consecutive) | cap
    return new, streak, new & ~wm_ready


# ---------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------
def masked_auc(score, positive, valid):
    """AUROC of `score` for the positive class over the valid entries (ties not averaged); nan if a class is empty."""
    valid = valid.astype(bool)
    big = jnp.finfo(jnp.float32).max
    s = jnp.where(valid, score.astype(jnp.float32), big)
    order = jnp.argsort(s)
    ranks = jnp.zeros(s.shape, jnp.float32).at[order].set(jnp.arange(1, s.shape[0] + 1, dtype=jnp.float32))
    pos = valid & positive.astype(bool)
    n_pos = jnp.sum(pos).astype(jnp.float32)
    n_neg = jnp.sum(valid & ~positive.astype(bool)).astype(jnp.float32)
    auc = (jnp.sum(jnp.where(pos, ranks, 0.0)) - n_pos * (n_pos + 1) / 2) / jnp.maximum(n_pos * n_neg, 1.0)
    return jnp.where((n_pos > 0) & (n_neg > 0), auc, jnp.nan), n_pos


# ---------------------------------------------------------------------------------------------------------------
# Time-major row stores (rollout batch (T, N, ...) and replay ring (L, N, ...))
# ---------------------------------------------------------------------------------------------------------------
def gather(store, rows, envs, fields):
    """rows (B, k) physical time rows, envs (B,) -> {f: (B, k, ...)}."""
    return {f: store[f][rows, envs[:, None]] for f in fields}


def window_valid(win, k):
    ok = win["start_ok"][:, 0] > 0.5
    if k > 1:
        ok &= jnp.all(win["done"][:, :k - 1] < 0.5, axis=1)
    return ok


def init_ring(L, N, dims):
    S, A, P = dims
    shapes = {"x_plus": (S,), "u_applied": (A,), "u_chosen": (A,), "xi": (P,), "x_minus_next": (S,),
              "x_plus_next": (S,), "shock_next": (), "done": (), "terminated": (), "start_ok": ()}
    ring = {f: jnp.zeros((L, N) + s, jnp.float32) for f, s in shapes.items()}
    ring["pos"] = jnp.zeros((), jnp.int32)
    ring["size"] = jnp.zeros((), jnp.int32)
    return ring


def init_neg(C, Lseg, dims):
    S, A, P = dims
    shapes = {"x_plus": (S,), "u_applied": (A,), "u_chosen": (A,), "xi": (P,), "x_minus_next": (S,),
              "x_plus_next": (S,), "shock_next": (), "done": (), "terminated": (), "start_ok": ()}
    neg = {f: jnp.zeros((C, Lseg) + s, jnp.float32) for f, s in shapes.items()}
    neg["pos"] = jnp.zeros((), jnp.int32)
    neg["count"] = jnp.zeros((), jnp.int32)
    return neg


def ring_write(ring, recs):
    """recs {f: (T, N, ...)} -> ring with the T new rows at pos..pos+T-1 (mod L)."""
    L = ring["x_plus"].shape[0]
    T = recs["x_plus"].shape[0]
    rows = (ring["pos"] + jnp.arange(T)) % L
    out = {f: ring[f].at[rows].set(recs[f]) for f in ROW_FIELDS}
    out["pos"] = (ring["pos"] + T) % L
    out["size"] = jnp.minimum(ring["size"] + T, L)
    return out


def neg_write(neg, ring_after, pos_before, size_before, terminated, key, M):
    """Copy the segments (Lseg rows) that end at a termination of the latest rollout (terminated (T, N)) from the
    ring into the negative buffer; at most M per call (random subset). Rows before the start of the data get
    done = 1 / start_ok = 0, so no window uses them."""
    C, Lseg = neg["x_plus"].shape[:2]
    L = ring_after["x_plus"].shape[0]
    T, N = terminated.shape
    M = int(min(M, T * N))
    score = terminated.reshape(-1).astype(jnp.float32) + 0.5 * jax.random.uniform(key, (T * N,))
    top, flat = jax.lax.top_k(score, M)
    is_new = top >= 1.0
    n_new = jnp.minimum(jnp.sum(is_new), C).astype(jnp.int32)
    t, e = flat // N, flat % N
    offs = t[:, None] - Lseg + 1 + jnp.arange(Lseg)[None, :]            # rollout-relative step of each row
    rows = (pos_before + offs) % L
    seg = gather(ring_after, rows, e, ROW_FIELDS)
    exists = (size_before + offs) >= 0
    seg["done"] = jnp.where(exists, seg["done"], 1.0)
    seg["start_ok"] = jnp.where(exists, seg["start_ok"], 0.0)
    slots = (neg["pos"] + jnp.arange(M)) % C
    keep = (jnp.arange(M) < n_new)
    out = {}
    for f in ROW_FIELDS:
        old = neg[f][slots]
        m = keep.reshape((M,) + (1,) * (old.ndim - 1))
        out[f] = neg[f].at[slots].set(jnp.where(m, seg[f], old))
    out["pos"] = (neg["pos"] + n_new) % C
    out["count"] = jnp.minimum(neg["count"] + n_new, C)
    return out, n_new


def sample_ring_windows(ring, key, B, k):
    L, N = ring["x_plus"].shape[:2]
    ke, kt = jax.random.split(key)
    envs = jax.random.randint(ke, (B,), 0, N)
    hi = jnp.maximum(ring["size"] - k + 1, 1)
    tau = jax.random.randint(kt, (B,), 0, hi)
    oldest = (ring["pos"] - ring["size"]) % L
    rows = (oldest + tau[:, None] + jnp.arange(k)[None, :]) % L
    win = gather(ring, rows, envs, ROW_FIELDS)
    win["weight"] = (window_valid(win, k) & (ring["size"] >= k)).astype(jnp.float32)
    return win


def sample_neg_windows(neg, key, B, k):
    C, Lseg = neg["x_plus"].shape[:2]
    kc, ks = jax.random.split(key)
    c = jax.random.randint(kc, (B,), 0, jnp.maximum(neg["count"], 1))
    s = jax.random.randint(ks, (B,), 0, Lseg - k + 1)
    rows = s[:, None] + jnp.arange(k)[None, :]
    win = {f: neg[f][c[:, None], rows] for f in ROW_FIELDS}
    win["weight"] = (window_valid(win, k) & (neg["count"] > 0)).astype(jnp.float32)
    return win


def sample_training_batch(ring, neg, key, B, k, neg_ratio):
    n_neg = int(round(B * neg_ratio))
    kp, kn, kx = jax.random.split(key, 3)
    pos = sample_ring_windows(ring, kp, B - n_neg, k)
    if n_neg == 0:
        return pos
    negw = jax.lax.cond(neg["count"] > 0, lambda: sample_neg_windows(neg, kn, n_neg, k),
                        lambda: sample_ring_windows(ring, kx, n_neg, k))
    return jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b], axis=0), pos, negw)


# ---------------------------------------------------------------------------------------------------------------
# Normalizer running statistics (Chan et al. parallel update)
# ---------------------------------------------------------------------------------------------------------------
def init_norm_stats(n_in, n_out):
    z = lambda n: jnp.zeros((n,), jnp.float32)
    return {"count": jnp.zeros((), jnp.float32), "in_mean": z(n_in), "in_m2": z(n_in), "d_mean": z(n_out),
            "d_m2": z(n_out)}


def _combine(count, mean, m2, x, w):
    nb = jnp.sum(w)
    mb = jnp.sum(x * w[:, None], 0) / jnp.maximum(nb, 1.0)
    m2b = jnp.sum(jnp.square(x - mb) * w[:, None], 0)
    n = count + nb
    delta = mb - mean
    mean_n = mean + delta * nb / jnp.maximum(n, 1.0)
    m2_n = m2 + m2b + jnp.square(delta) * count * nb / jnp.maximum(n, 1.0)
    return n, mean_n, m2_n


def update_norm_stats(stats, layout, cfg, x, u_tilde, xi, x_next, w):
    z = W.raw_input(layout, cfg, x, u_tilde, xi)
    d = W.state_delta(layout, cfg, x, x_next)
    n, im, i2 = _combine(stats["count"], stats["in_mean"], stats["in_m2"], z, w)
    _, dm, d2 = _combine(stats["count"], stats["d_mean"], stats["d_m2"], d, w)
    return {"count": n, "in_mean": im, "in_m2": i2, "d_mean": dm, "d_m2": d2}


def norm_from_stats(stats, cfg: W.WMConfig):
    n = jnp.maximum(stats["count"], 1.0)
    d_sd = jnp.sqrt(stats["d_m2"] / n)
    return {"in_mean": stats["in_mean"], "in_std": jnp.maximum(jnp.sqrt(stats["in_m2"] / n), cfg.std_floor),
            "d_mean": stats["d_mean"], "d_std": jnp.maximum(d_sd, cfg.std_floor),
            "err_scale": jnp.maximum(d_sd, cfg.err_std_floor)}


# ---------------------------------------------------------------------------------------------------------------
# Row-safe gradient through the shield
# ---------------------------------------------------------------------------------------------------------------
def _zero_cotangent(x):
    x = jnp.asarray(x)
    if jnp.issubdtype(x.dtype, jnp.inexact):
        return jnp.zeros_like(x)
    return np.zeros(x.shape, dtype=jax.dtypes.float0)


def row_safe_vjp(f):
    """g(u, aux) = f(u, aux); d/du by f's VJP with every row (leading axis) that has a non-finite entry set to 0;
    no gradient to aux. The shield's VJP w.r.t. u0 is NaN on some uncorrected rows of real WM data (0 * inf inside;
    the forward value is finite) - this keeps one bad row from poisoning the whole policy gradient."""
    @jax.custom_vjp
    def g(u, aux):
        return f(u, aux)

    def fwd(u, aux):
        y, vjp = jax.vjp(lambda uu: f(uu, aux), u)
        return y, (vjp, aux)

    def bwd(res, ct):
        vjp, aux = res
        (gu,) = vjp(ct)
        ok = jnp.all(jnp.isfinite(gu), axis=tuple(range(1, gu.ndim)), keepdims=True)
        return jnp.where(ok, gu, 0.0), jax.tree_util.tree_map(_zero_cotangent, aux)

    g.defvjp(fwd, bwd)
    return g


# ---------------------------------------------------------------------------------------------------------------
# The trainer
# ---------------------------------------------------------------------------------------------------------------
def probe_xi_active(env, nr_envs, seed=0, tol=1e-6):
    """xi dims that vary over an eval-mode reset (full DR ranges) + the env curriculum coefficient (varies in
    training only)."""
    keys = jax.random.split(jax.random.PRNGKey(seed), nr_envs)
    st = env.reset(keys, True)
    xi = np.asarray(jax.vmap(lambda m, i: wm_state.extract_xi(env, m, i))(st.mjx_model, st.internal_state))
    act = set(np.nonzero(xi.std(0) > tol)[0].tolist())
    sl = wm_state.xi_layout(env).slices["curriculum_coeff"]
    act |= set(range(sl.start, sl.stop))
    return np.asarray(sorted(act), np.int32)


class DTrainer:
    def __init__(self, ppo, config, env):
        self.cfg = cfg = DConfig.from_config(config.algorithm.get("d", {}))
        if bool(config.algorithm.ncbf.use_safety_layer):
            raise ValueError("algorithm.method=d replaces the critic shield: set algorithm.ncbf.use_safety_layer=False")
        if not getattr(env, "record_wm_state", False):
            raise ValueError("algorithm.method=d needs environment.record_wm_state=True")
        self.env = env
        self.nr_envs = int(config.environment.nr_envs)
        self.nr_updates = int(ppo.nr_updates)
        self.nr_steps = int(ppo.nr_steps)
        self.lay, self.xl = wm_state.state_layout(env), wm_state.xi_layout(env)
        A = int(env.nr_actuator_joints)

        bundle0 = None
        if cfg.wm_init_bundle:
            wm0, bundle0, meta = WE.load_bundle(cfg.wm_init_bundle)
            xi_active = np.asarray(meta["schema"].get("xi_active", np.arange(self.xl.dim)), np.int32)
            wm_cfg = dataclasses.replace(wm0.cfg, matmul_dtype="float32", mask_post_shock_step=cfg.stale_imu_mask)
            rlx_logger.info(f"D: WM initialized from {cfg.wm_init_bundle} ({len(xi_active)} xi dims)")
        else:
            xi_active = probe_xi_active(env, min(cfg.xi_probe_envs, max(self.nr_envs, 2)), int(config.environment.seed))
            wm_cfg = W.WMConfig(width=cfg.wm_width, depth=cfg.wm_depth, layer_norm=cfg.wm_layer_norm,
                                activation=cfg.wm_activation, use_xi=cfg.use_xi, last_action_source="chosen",
                                mask_post_shock_step=cfg.stale_imu_mask)
            rlx_logger.info(f"D: WM from scratch, {len(xi_active)} of {self.xl.dim} xi dims")
        self.xi_active = xi_active
        self.schema = {"state_dim": self.lay.dim, "action_dim": A, "xi_dim": int(len(xi_active)),
                       "groups": self.lay.to_json()["groups"], "xi_active": [int(i) for i in xi_active],
                       "xi_full_dim": self.xl.dim, "source_schema": wm_state.SCHEMA_VERSION}
        self.layout = W.WMLayout.from_schema(self.schema)
        self.wm = W.WorldModel(self.layout, wm_cfg)
        self.dims = (self.lay.dim, A, int(len(xi_active)))
        self.tx = optax.chain(optax.clip_by_global_norm(cfg.wm_grad_clip), optax.adam(cfg.wm_lr))

        self.spec = SV.ImaginationSpec.from_env(env, xi_active=xi_active)
        self.policy_fn = SV.make_imagined_policy(ppo.policy.apply, ppo.get_processed_action)
        dt = "bfloat16" if cfg.shield_bf16 else "float32"
        sv = lambda H, dtype: SV.SafetyValueConfig(source="wm", H=H, events=("fall",), T_l=(cfg.T_l,),
                                                   margin_offset=(0.0,), open_loop=False, matmul_dtype=dtype)
        self.shield_cfg = WS.ShieldConfig(p_c=cfg.p_c, alpha=cfg.alpha, gate="gated", gate_frac=cfg.gate_frac,
                                          relinearize=cfg.relinearize, projection="hard_projection",
                                          max_delta_u=cfg.max_delta_u, skip_stale=cfg.stale_imu_mask)
        self.gamma_c = self.shield_cfg.gamma_c
        self.value_fn = SV.make_safety_value_fn(sv(cfg.H, dt), self.spec, wm=self.wm, policy_fn=self.policy_fn)
        self.shield_fn = WS.make_shield_fn(self.shield_cfg, self.value_fn, nr_events=1)
        self.auc_value_fn = SV.make_safety_value_fn(sv(cfg.auc_horizon, dt), self.spec, wm=self.wm,
                                                    policy_fn=self.policy_fn)
        self.d2_value_fn = SV.make_safety_value_fn(sv(cfg.d2_H, "float32"), self.spec, wm=self.wm,
                                                   policy_fn=self.policy_fn, policy_grad=True)

        key = jax.random.PRNGKey(int(config.environment.seed) + 7919)
        if bundle0 is not None:
            params, norm, frozen = bundle0["params"], bundle0["norm"], 1
        else:
            b = self.wm.init(key)
            params, norm, frozen = b["params"], b["norm"], 0
        self.persist0 = {
            "wm_params": params, "wm_opt": self.tx.init(params), "wm_norm": norm,
            "norm_stats": init_norm_stats(self.wm.n_in, self.wm.n_out),
            "norm_frozen": jnp.asarray(frozen, jnp.int32), "wm_ready": jnp.asarray(0, jnp.int32),
            "switch_update": jnp.asarray(-1, jnp.int32), "auc_streak": jnp.asarray(0, jnp.int32),
            "nr_wm_updates": jnp.asarray(0, jnp.int32),
        }
        self.persist = self.persist0

    # --- state ---------------------------------------------------------------------------------------------
    def persist_keys(self):
        return tuple(self.persist0.keys())

    def init_state(self, persist=None):
        persist = self.persist if persist is None else persist
        c = self.cfg
        return {**persist,
                "replay": init_ring(c.replay_steps, self.nr_envs, self.dims),
                "neg": init_neg(c.neg_capacity, c.neg_segment_len, self.dims),
                "v_prev": jnp.full((self.nr_envs, 1), self.gamma_c, jnp.float32)}

    def persistent(self, d_state):
        return {k: d_state[k] for k in self.persist_keys()}

    def bundle(self, d_state):
        return {"params": d_state["wm_params"], "norm": d_state["wm_norm"]}

    def strength(self, d_state, update):
        on = (d_state["wm_ready"] > 0) & self.cfg.shield_enabled
        return on.astype(jnp.float32) * fade_strength(update, self.nr_updates, self.cfg.fade_out_frac)

    # --- rollout -------------------------------------------------------------------------------------------
    def rollout_carry(self, d_state, update):
        """The small part of the D state the rollout scan needs (not the replay buffers)."""
        return {"bundle": self.bundle(d_state), "v_prev": d_state["v_prev"], "s": self.strength(d_state, update)}

    def shield_step(self, d_roll, policy_params, env_state, u0):
        """Before env.step: (u, inputs, v_prev used, v_exec (the next V^prev), diag)."""
        inputs = SV.inputs_from_env_state(self.env, env_state, self.spec)
        params = {"wm": d_roll["bundle"], "policy": policy_params}
        v_prev, s = d_roll["v_prev"], d_roll["s"]

        def on(_):
            u, v_exec, diag = self.shield_fn(params, inputs, u0, v_prev, s)
            active = diag["active"]
            cap = self.cfg.max_delta_u
            out = {k: diag[k].astype(jnp.float32) for k in ("active", "delta_u_norm", "v0")}
            out["cap_bind"] = (active & (cap > 0) & (diag["delta_u_norm"] >= 0.999 * cap * s)).astype(jnp.float32)
            out["reach_safe"] = (active & (diag["G_after"] >= 0.0)).astype(jnp.float32)
            return u, v_exec, out

        def off(_):
            n = u0.shape[0]
            z = jnp.zeros((n,), jnp.float32)
            return u0, jnp.full_like(v_prev, self.gamma_c), {"active": z, "delta_u_norm": z, "v0": z, "cap_bind": z,
                                                              "reach_safe": z}

        if self.cfg.shield_enabled:
            u, v_exec, diag = jax.lax.cond(s > 0, on, off, None)
        else:
            u, v_exec, diag = off(None)
        diag = {**diag, "shield_on": (s > 0).astype(jnp.float32) * jnp.ones_like(diag["active"])}
        return u, inputs, v_prev, v_exec, diag

    def record(self, inputs, u, v_prev_used, env_state_after, diag):
        info = env_state_after.info
        stale = inputs.stale if self.cfg.stale_imu_mask else jnp.zeros_like(inputs.stale)
        rec = {
            "x_plus": inputs.x0, "xi": self.spec.wm_xi(inputs.xi_full), "xi_full": inputs.xi_full,
            "delay_buffer": inputs.delay_buffer, "delay_steps": inputs.delay_steps, "command": inputs.command,
            "first_step": inputs.first_step, "stale": inputs.stale,
            "u_chosen": u, "u_applied": info["wm/u_applied"], "x_minus_next": info["wm/x_minus"],
            "x_plus_next": info["wm/x_plus"],
            "shock_next": ((info["shock/perturbed"] > 0.5) | (info["shock/dr_resampled"] > 0.5)).astype(jnp.float32),
            "done": (env_state_after.terminated | env_state_after.truncated).astype(jnp.float32),
            "terminated": env_state_after.terminated.astype(jnp.float32),
            "start_ok": (~stale).astype(jnp.float32),
            "v_prev": v_prev_used, "diag": diag,
        }
        return rec

    # --- PPO batch -----------------------------------------------------------------------------------------
    def ppo_batch(self, recs):
        """Per-sample shield inputs for the PPO update, flattened like the PPO batch ((T, N) -> T * N)."""
        flat = lambda a: a.reshape((-1,) + a.shape[2:])
        keys = ("x_plus", "xi_full", "delay_buffer", "delay_steps", "command", "first_step", "stale", "v_prev")
        return {k: flat(recs[k]) for k in keys}

    @staticmethod
    def _inputs(b):
        return SV.ShieldInputs(x0=b["x_plus"], xi_full=b["xi_full"], delay_buffer=b["delay_buffer"],
                               delay_steps=b["delay_steps"], command=b["command"], first_step=b["first_step"],
                               stale=b["stale"])

    def anticipation_loss(self, d_state, policy_params, b, u_mean, s, style="mean_action"):
        """0.5 * mean ||mu - shield(mu)||^2 on the first anticipation_subset samples (gradient through the shield
        into u0 = mu_theta(s) if anticipation_through_shield). Zero while the shield is off (s = 0)."""
        from rl_x.algorithms.ncbf_ppo.flax_full_jit.paper_compat import anticipation_squared_error
        n = min(self.cfg.anticipation_subset, u_mean.shape[0])
        sub = jax.tree_util.tree_map(lambda a: a[:n], b)
        u = u_mean[:n]
        params = {"wm": self.bundle(d_state), "policy": jax.lax.stop_gradient(policy_params)}

        def shield_u(u, aux):
            prm, inp, vp, strength = aux
            u_safe, _, diag = self.shield_fn(prm, inp, u, vp, strength)
            return u_safe, diag["active"].astype(u.dtype)

        def compute(u):
            aux = (params, self._inputs(sub), sub["v_prev"], s)
            if self.cfg.anticipation_through_shield:
                u_safe, active = row_safe_vjp(shield_u)(u, aux)
            else:
                u_safe, active = jax.lax.stop_gradient(shield_u(u, aux))
            u_safe = jnp.where(jnp.all(jnp.isfinite(u_safe), axis=-1, keepdims=True), u_safe, u)
            return 0.5 * jnp.mean(anticipation_squared_error(u, u_safe, style)), jnp.mean(active)

        # No lax.cond here: the gradient of a cond around the shield came out NaN on uncorrected rows (the plain
        # function's gradient is finite; test_anticipation_through_wm_shield_is_differentiable). With s = 0 the
        # shield returns u0, so the loss is 0 anyway (computed but wasted during the warm start).
        loss, active = compute(u)
        on = (s > 0).astype(u.dtype)
        return on * loss, on * active

    def d2_loss(self, d_state, policy_params, b, u_mean):
        """L_wm-safe = mean -log P(s, mu_theta(s)) closed loop through the imagined policy (theta), psi stopped.
        Zero before the warm start is over (the WM is not trusted yet)."""
        n = min(self.cfg.d2_subset, u_mean.shape[0])
        sub = jax.tree_util.tree_map(lambda a: a[:n], b)
        params = {"wm": self.bundle(d_state), "policy": policy_params}

        def compute(u):
            _, aux = self.d2_value_fn(params, self._inputs(sub), u)
            return -jnp.mean(aux["logp"][:, 0])

        # no lax.cond (see anticipation_loss); computed but zeroed before the warm start is over
        return (d_state["wm_ready"] > 0).astype(u_mean.dtype) * compute(u_mean[:n])

    # --- WM training ---------------------------------------------------------------------------------------
    def wm_loss(self, params, norm, batch):
        return W.k_step_loss(self.wm, params, norm, batch, self.cfg.wm_method)

    def _latest_eval(self, d_state, policy_params, recs, key):
        """WM on the new rollout (not trained on yet): k-step loss, fall-prediction AUC (replay / imagination)."""
        c = self.cfg
        T, N = recs["x_plus"].shape[:2]
        k, Ha = c.k_train, c.auc_horizon
        B = c.eval_windows
        out = {}
        k1, k2, k3, k4 = jax.random.split(key, 4)
        norm = d_state["wm_norm"]
        if T >= k:
            envs = jax.random.randint(k1, (B,), 0, N)
            t0 = jax.random.randint(k2, (B,), 0, T - k + 1)
            win = gather(recs, t0[:, None] + jnp.arange(k)[None, :], envs, ROW_FIELDS)
            win["weight"] = window_valid(win, k).astype(jnp.float32)
            loss, aux = self.wm_loss(d_state["wm_params"], norm, win)
            out["d/wm_latest_kstep_loss"] = loss
            out["d/wm_latest_valid_frac"] = aux["valid_frac"]
        if T >= Ha:
            envs = jax.random.randint(k3, (B,), 0, N)
            t0 = jax.random.randint(k4, (B,), 0, T - Ha + 1)
            rows = t0[:, None] + jnp.arange(Ha)[None, :]
            win = gather(recs, rows, envs, ROW_FIELDS + ("xi_full",))
            done = win["done"] > 0.5
            first_done = jnp.argmax(jnp.concatenate([done, jnp.ones_like(done[:, :1])], axis=1), axis=1)
            step_ok = jnp.arange(Ha)[None, :] <= first_done[:, None]
            label = jnp.any(step_ok & (win["terminated"] > 0.5), axis=1)
            valid = win["start_ok"][:, 0] > 0.5
            step_fn = lambda x, u, xi, uc: self.wm.step(self.bundle(d_state), x, u, xi, uc)
            preds = W.k_step_rollout_on_window(self.wm, step_fn, win, teacher_forcing=(c.wm_method == "A"))
            m = wm_state.margin_fall(preds, win["xi_full"], self.spec.margin_cfg, self.lay, self.xl)
            score = jnp.min(jnp.where(step_ok, m, jnp.inf), axis=1)
            auc_r, npos = masked_auc(-score, label, valid)
            # imagination: the shield's V(x_t, u_t) (closed loop through the deterministic policy)
            r0 = jax.tree_util.tree_map(lambda a: a[t0, envs], {f: recs[f] for f in
                                        ("x_plus", "xi_full", "delay_buffer", "delay_steps", "command",
                                         "first_step", "stale", "u_chosen")})
            v, _ = self.auc_value_fn({"wm": self.bundle(d_state), "policy": policy_params}, self._inputs(r0),
                                     r0["u_chosen"])
            auc_i, _ = masked_auc(-v[:, 0], label, valid)
            out.update({"d/wm_auc_replay": auc_r, "d/wm_auc_imagination": auc_i, "d/wm_auc_nr_falls": npos,
                        "d/wm_auc_nr_valid": jnp.sum(valid).astype(jnp.float32)})
        return out

    def iteration_update(self, d_state, policy_params, recs, update, key):
        """After the rollout of PPO update `update` (0-based): eval on the new data, normalizers, replay, WM
        updates, warm-start gate. Returns (d_state', metrics)."""
        c = self.cfg
        k_eval, k_neg, k_train = jax.random.split(key, 3)
        metrics = self._latest_eval(d_state, policy_params, recs, k_eval)

        # normalizers
        in_phase = (d_state["norm_frozen"] == 0) & (update < c.norm_update_iterations)
        flat = lambda a: a.reshape((-1,) + a.shape[2:])
        w = flat((recs["shock_next"] < 0.5) & (recs["start_ok"] > 0.5)).astype(jnp.float32)
        stats_new = update_norm_stats(d_state["norm_stats"], self.layout, self.wm.cfg, flat(recs["x_plus"]),
                                      flat(recs["u_applied"]), flat(recs["xi"]), flat(recs["x_minus_next"]), w)
        stats = jax.tree_util.tree_map(lambda a, b: jnp.where(in_phase, a, b), stats_new, d_state["norm_stats"])
        norm = jax.tree_util.tree_map(lambda a, b: jnp.where(in_phase, a, b), norm_from_stats(stats, self.wm.cfg),
                                      d_state["wm_norm"])
        frozen = jnp.where((d_state["norm_frozen"] == 0) & ~in_phase, 1, d_state["norm_frozen"]).astype(jnp.int32)

        # replay + negative buffer
        ring0 = d_state["replay"]
        ring = ring_write(ring0, {f: recs[f] for f in ROW_FIELDS})
        neg, n_new = neg_write(d_state["neg"], ring, ring0["pos"], ring0["size"], recs["terminated"] > 0.5, k_neg,
                               c.neg_new_per_iteration)

        # WM updates
        def one(carry, kk):
            params, opt = carry
            batch = sample_training_batch(ring, neg, kk, c.wm_batch, c.k_train, c.neg_ratio)
            (loss, aux), grads = jax.value_and_grad(lambda p: self.wm_loss(p, norm, batch), has_aux=True)(params)
            gn = optax.global_norm(grads)
            upd, opt = self.tx.update(grads, opt, params)
            return (optax.apply_updates(params, upd), opt), {"loss": loss, "valid_frac": aux["valid_frac"],
                                                             "grad_norm": gn}

        (params, opt), m = jax.lax.scan(one, (d_state["wm_params"], d_state["wm_opt"]),
                                        jax.random.split(k_train, c.wm_updates_per_iteration))
        m = jax.tree_util.tree_map(jnp.mean, m)

        # warm start
        gate_auc = metrics.get("d/wm_auc_replay" if c.warm_start_metric == "replay" else "d/wm_auc_imagination",
                               jnp.asarray(jnp.nan))
        ready, streak, switched = warm_start_gate(d_state["wm_ready"] > 0, d_state["auc_streak"], gate_auc,
                                                  metrics.get("d/wm_auc_nr_falls", jnp.asarray(0.0)), update, c,
                                                  self.nr_updates)
        switch_update = jnp.where(switched, update, d_state["switch_update"]).astype(jnp.int32)

        new = {**d_state, "wm_params": params, "wm_opt": opt, "wm_norm": norm, "norm_stats": stats,
               "norm_frozen": frozen, "replay": ring, "neg": neg, "wm_ready": ready.astype(jnp.int32),
               "auc_streak": streak.astype(jnp.int32), "switch_update": switch_update,
               "nr_wm_updates": d_state["nr_wm_updates"] + c.wm_updates_per_iteration}
        diag = recs["diag"]
        n_shield = jnp.maximum(jnp.sum(diag["shield_on"]), 1.0)
        metrics.update({
            "d/wm_loss": m["loss"], "d/wm_valid_frac": m["valid_frac"], "d/wm_grad_norm": m["grad_norm"],
            "d/wm_ready": ready.astype(jnp.float32), "d/shield_switch_update": switch_update.astype(jnp.float32),
            "d/shield_switched_now": switched.astype(jnp.float32), "d/auc_streak": streak.astype(jnp.float32),
            "d/norm_frozen": frozen.astype(jnp.float32), "d/norm_count": stats["count"],
            "d/replay_size": ring["size"].astype(jnp.float32), "d/neg_count": neg["count"].astype(jnp.float32),
            "d/neg_new": n_new.astype(jnp.float32),
            "d/shield_strength": self.strength(d_state, update),
            "d/intervention_rate": jnp.mean(diag["active"]),
            "d/intervention_rate_when_on": jnp.sum(diag["active"]) / n_shield,
            "d/delta_u_norm_mean": jnp.mean(diag["delta_u_norm"]),
            "d/delta_u_norm_active_mean": jnp.sum(diag["delta_u_norm"]) / jnp.maximum(jnp.sum(diag["active"]), 1.0),
            "d/cap_binding_rate_of_interventions": jnp.sum(diag["cap_bind"]) / jnp.maximum(jnp.sum(diag["active"]), 1.0),
            "d/reach_safe_rate_of_interventions": jnp.sum(diag["reach_safe"]) / jnp.maximum(jnp.sum(diag["active"]), 1.0),
            "d/v0_mean_when_on": jnp.sum(diag["v0"] * diag["shield_on"]) / n_shield,
            "d/stale_rate": jnp.mean(recs["stale"].astype(jnp.float32)),
        })
        return new, metrics

    # --- files ---------------------------------------------------------------------------------------------
    def save_bundle(self, path, persist, extra=None):
        bundle = {"params": persist["wm_params"], "norm": persist["wm_norm"]}
        WE.save_bundle(path, self.wm, bundle, self.schema, extra={"source": "d_training online", **(extra or {})})
