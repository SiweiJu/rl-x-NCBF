"""World model f_psi(x+_t, u~_t, xi) -> x-_{t+1} for PNCBF (plan v3.2 sec. 4.2-4.5, P1 / T13).

The state x is the "option 1" vector built by `wm_state.extract_state` (agent wm-data). This module only needs its
layout: the state/action/xi sizes and the named groups (`WMLayout.from_schema`). Three groups get special treatment
when the schema has them:
- `last_action`: the env's previous-action observation equals the chosen (undelayed) action, so it is set from u,
  never predicted (`last_action_source="chosen"`; "applied" uses u~, "predict" learns it like any other group).
- `base_xy` (2 dims) and `base_yaw` (1 dim): the network never sees them (translation / yaw invariance). It predicts
  their change in the heading frame of the current state, which is rotated into the state's frame. So the model works
  for any origin convention (world frame or window-start frame).

Everything is a pure function of (params, norm, arrays), jit- and grad-safe. Ensembles: `params` carry a leading
member axis M (M=1 by default); `step` averages the members' deltas, `step_members` returns all of them (for sigma).
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------------------------------------------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WMLayout:
    state_dim: int
    action_dim: int
    xi_dim: int
    groups: Tuple[Tuple[str, int, int], ...]          # (name, start, end) over the state vector
    xy_idx: Tuple[int, ...] = ()                      # 2 dims or empty
    yaw_idx: Tuple[int, ...] = ()                     # 1 dim or empty
    last_action_idx: Tuple[int, ...] = ()             # action_dim dims or empty

    @staticmethod
    def from_schema(schema, xy_group="base_xy", yaw_group="base_yaw", last_action_group="last_action"):
        groups = schema["groups"]
        if isinstance(groups, dict):
            groups = [(name, int(se[0]), int(se[1])) for name, se in groups.items()]
        groups = tuple((str(n), int(s), int(e)) for n, s, e in groups)
        by_name = {n: tuple(range(s, e)) for n, s, e in groups}
        xy = by_name.get(xy_group, ())
        yaw = by_name.get(yaw_group, ())
        la = by_name.get(last_action_group, ())
        if xy and len(xy) != 2:
            raise ValueError(f"{xy_group} must have 2 dims, got {len(xy)}")
        if yaw and len(yaw) != 1:
            raise ValueError(f"{yaw_group} must have 1 dim, got {len(yaw)}")
        if bool(xy) != bool(yaw):
            raise ValueError("base_xy and base_yaw must both be present or both absent")
        if la and len(la) != int(schema["action_dim"]):
            raise ValueError(f"{last_action_group} has {len(la)} dims, action_dim is {schema['action_dim']}")
        return WMLayout(int(schema["state_dim"]), int(schema["action_dim"]), int(schema["xi_dim"]), groups, xy, yaw, la)

    def group_index(self):
        return {n: np.arange(s, e) for n, s, e in self.groups}


@dataclass(frozen=True)
class WMConfig:
    width: int = 512
    depth: int = 3
    layer_norm: bool = True
    activation: str = "elu"
    ensemble_size: int = 1
    use_xi: bool = True
    last_action_source: str = "chosen"     # chosen | applied | predict
    std_floor: float = 1e-6                # normalization std floor (inputs and deltas)
    matmul_dtype: str = "float32"          # "bfloat16": bf16 matmul inputs with f32 accumulation/output
    err_std_floor: float = 1e-3            # floor of the per-dim loss scale (delta std), physical units


def _pose_mode(layout):
    return bool(layout.xy_idx)


def input_state_idx(layout):
    """State dims the network sees: all but base_xy / base_yaw."""
    drop = set(layout.xy_idx) | set(layout.yaw_idx)
    return np.array([i for i in range(layout.state_dim) if i not in drop], dtype=np.int32)


def pred_idx(layout, cfg):
    """State dims the network predicts a delta for: all but last_action (unless last_action_source == predict)."""
    drop = set(layout.last_action_idx) if cfg.last_action_source != "predict" else set()
    return np.array([i for i in range(layout.state_dim) if i not in drop], dtype=np.int32)


def wrap_angle(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def _rot(yaw, v, inverse=False):
    """Rotate 2-vectors v (..., 2) by +yaw (or -yaw)."""
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    if inverse:
        s = -s
    return jnp.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], axis=-1)


def state_delta(layout, cfg, x, x_next):
    """Delta over pred_idx; base_xy in the heading frame of x, base_yaw wrapped."""
    pidx = pred_idx(layout, cfg)
    d = x_next - x
    if _pose_mode(layout):
        xy, yaw = np.array(layout.xy_idx), layout.yaw_idx[0]
        d = d.at[..., xy].set(_rot(x[..., yaw], d[..., xy], inverse=True))
        d = d.at[..., yaw].set(wrap_angle(d[..., yaw]))
    return d[..., pidx]


def apply_delta(layout, cfg, x, delta, u_chosen=None, u_applied=None):
    """Inverse of state_delta, plus the last_action overwrite."""
    pidx = pred_idx(layout, cfg)
    full = jnp.zeros(x.shape[:-1] + (layout.state_dim,), x.dtype).at[..., pidx].set(delta)
    if _pose_mode(layout):
        xy, yaw = np.array(layout.xy_idx), layout.yaw_idx[0]
        full = full.at[..., xy].set(_rot(x[..., yaw], full[..., xy]))
    x_next = x + full
    if _pose_mode(layout):
        x_next = x_next.at[..., yaw].set(wrap_angle(x_next[..., yaw]))
    if layout.last_action_idx and cfg.last_action_source != "predict":
        la = np.array(layout.last_action_idx)
        u = u_chosen if cfg.last_action_source == "chosen" else u_applied
        if u is None:
            raise ValueError(f"last_action_source={cfg.last_action_source} needs that action")
        x_next = x_next.at[..., la].set(u)
    return x_next


def state_error(layout, x_hat, x_true):
    """x_hat - x_true with the yaw difference wrapped. Shape (..., state_dim)."""
    e = x_hat - x_true
    if _pose_mode(layout):
        yaw = layout.yaw_idx[0]
        e = e.at[..., yaw].set(wrap_angle(e[..., yaw]))
    return e


# ---------------------------------------------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------------------------------------------
_ACT = {"elu": nn.elu, "silu": nn.silu, "relu": nn.relu, "gelu": nn.gelu, "tanh": jnp.tanh}


def _f32_acc_dot_general(lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None, **kw):
    return jax.lax.dot_general(lhs, rhs, dimension_numbers, precision=precision, preferred_element_type=jnp.float32)


class DeltaMLP(nn.Module):
    out_dim: int
    width: int = 512
    depth: int = 3
    layer_norm: bool = True
    activation: str = "elu"
    matmul_dtype: str = "float32"

    @nn.compact
    def __call__(self, z):
        act = _ACT[self.activation]
        low = self.matmul_dtype != "float32"
        cast = (lambda a: a.astype(jnp.float32)) if low else (lambda a: a)
        if not low:
            dense = nn.Dense
        else:   # params stay f32; inputs/kernels are cast for the matmul, accumulation and output in f32
            dt = jnp.dtype(self.matmul_dtype)
            dense = lambda n, **kw: nn.Dense(n, dtype=dt, param_dtype=jnp.float32, dot_general=_f32_acc_dot_general, **kw)
        for _ in range(self.depth):
            z = dense(self.width)(z)
            z = cast(z)
            if self.layer_norm:
                z = nn.LayerNorm()(z)
            z = act(z)
        out = dense(self.out_dim, kernel_init=nn.initializers.variance_scaling(0.01, "fan_in", "truncated_normal"))(z)
        return cast(out)


# ---------------------------------------------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------------------------------------------
def identity_norm(layout, cfg):
    n_in = len(input_state_idx(layout)) + layout.action_dim + (layout.xi_dim if cfg.use_xi else 0)
    n_out = len(pred_idx(layout, cfg))
    return {"in_mean": jnp.zeros(n_in), "in_std": jnp.ones(n_in), "d_mean": jnp.zeros(n_out), "d_std": jnp.ones(n_out),
            "err_scale": jnp.ones(n_out)}


def raw_input(layout, cfg, x, u_tilde, xi):
    parts = [x[..., input_state_idx(layout)], u_tilde]
    if cfg.use_xi:
        parts.append(xi)
    return jnp.concatenate(parts, axis=-1)


def fit_norm(layout, cfg, x, u_tilde, xi, x_next):
    """Fit input and target-delta statistics on training transitions (arrays (S, ...), numpy or jax).
    Only non-shock transitions should be passed for the delta statistics (x_next = x-_{t+1})."""
    z = np.asarray(raw_input(layout, cfg, jnp.asarray(x), jnp.asarray(u_tilde), jnp.asarray(xi)), np.float64)
    d = np.asarray(state_delta(layout, cfg, jnp.asarray(x), jnp.asarray(x_next)), np.float64)
    in_std = np.maximum(z.std(0), cfg.std_floor)
    d_std = np.maximum(d.std(0), cfg.std_floor)
    return {"in_mean": jnp.asarray(z.mean(0), jnp.float32), "in_std": jnp.asarray(in_std, jnp.float32),
            "d_mean": jnp.asarray(d.mean(0), jnp.float32), "d_std": jnp.asarray(d_std, jnp.float32),
            "err_scale": jnp.asarray(np.maximum(d.std(0), cfg.err_std_floor), jnp.float32)}


def normalize_delta(norm, d):
    return (d - norm["d_mean"]) / norm["d_std"]


def denormalize_delta(norm, dn):
    return dn * norm["d_std"] + norm["d_mean"]


# ---------------------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------------------
class WorldModel:
    """Holds the static layout/config and the flax module; all methods are pure in (params, norm, arrays).
    A "bundle" is {"params": member-stacked flax params (leading axis M), "norm": normalization dict}."""

    def __init__(self, layout: WMLayout, cfg: WMConfig = WMConfig()):
        self.layout, self.cfg = layout, cfg
        self.n_out = len(pred_idx(layout, cfg))
        self.n_in = len(input_state_idx(layout)) + layout.action_dim + (layout.xi_dim if cfg.use_xi else 0)
        self.net = DeltaMLP(self.n_out, cfg.width, cfg.depth, cfg.layer_norm, cfg.activation, cfg.matmul_dtype)

    def init(self, key, norm=None):
        keys = jax.random.split(key, self.cfg.ensemble_size)
        params = jax.vmap(lambda k: self.net.init(k, jnp.zeros((1, self.n_in))))(keys)
        return {"params": params, "norm": norm if norm is not None else identity_norm(self.layout, self.cfg)}

    def member(self, bundle, i):
        return {"params": jax.tree_util.tree_map(lambda p: p[i:i + 1], bundle["params"]), "norm": bundle["norm"]}

    # --- single member -------------------------------------------------------------------------------------------
    def delta_norm_single(self, params_1, norm, x, u_tilde, xi):
        z = (raw_input(self.layout, self.cfg, x, u_tilde, xi) - norm["in_mean"]) / norm["in_std"]
        return self.net.apply(params_1, z)

    def step_single(self, params_1, norm, x, u_tilde, xi, u_chosen=None):
        d = denormalize_delta(norm, self.delta_norm_single(params_1, norm, x, u_tilde, xi))
        return apply_delta(self.layout, self.cfg, x, d, u_chosen=u_chosen, u_applied=u_tilde)

    # --- ensemble ------------------------------------------------------------------------------------------------
    def step_members(self, bundle, x, u_tilde, xi, u_chosen=None):
        """(M, ..., state_dim): every member's x-_{t+1}."""
        return jax.vmap(lambda p: self.step_single(p, bundle["norm"], x, u_tilde, xi, u_chosen))(bundle["params"])

    def step(self, bundle, x, u_tilde, xi, u_chosen=None):
        """x-_{t+1} from the members' mean delta."""
        norm = bundle["norm"]
        dn = jax.vmap(lambda p: self.delta_norm_single(p, norm, x, u_tilde, xi))(bundle["params"]).mean(0)
        return apply_delta(self.layout, self.cfg, x, denormalize_delta(norm, dn), u_chosen=u_chosen, u_applied=u_tilde)


# ---------------------------------------------------------------------------------------------------------------
# k-step autoregressive loss (plan sec. 4.3 / 4.4)
# ---------------------------------------------------------------------------------------------------------------
def window_mask_method_b(shock_next):
    """Method B: a window (B, k) is dropped if a shock happens between two of its steps (shock_next[:, :k-1])."""
    return jnp.logical_not(jnp.any(shock_next[:, :-1] > 0, axis=1))


def k_step_rollout_on_window(wm: WorldModel, step_fn, batch, teacher_forcing=True):
    """Autoregressive prediction over a window. batch fields (B, k, ...):
    x_plus (only [:, 0] used), u_applied, u_chosen, xi, x_plus_next, shock_next.
    step_fn(x, u_tilde, xi, u_chosen) -> x-_{t+1}.
    Method A: after a step whose transition has a shock / DR resampling (shock_next), continue from the recorded x+.
    Returns predicted x-_{t+1..t+k}, shape (B, k, D)."""
    def body(x, inp):
        u, uc, xi, xp_next, shock = inp
        x_hat = step_fn(x, u, xi, uc)
        if teacher_forcing:
            x_in = jnp.where(shock[:, None] > 0, xp_next, x_hat)
        else:
            x_in = x_hat
        return x_in, x_hat

    tm = lambda a: jnp.swapaxes(a, 0, 1)
    xs = (tm(batch["u_applied"]), tm(batch["u_chosen"]), tm(batch["xi"]), tm(batch["x_plus_next"]),
          tm(batch["shock_next"]))
    _, preds = jax.lax.scan(body, batch["x_plus"][:, 0], xs)
    return jnp.swapaxes(preds, 0, 1)


def k_step_loss_single(wm: WorldModel, params_1, norm, batch, method="A"):
    """Mean over valid windows, steps and predicted dims of ((x_hat - x-)/err_scale)^2.
    method "A": teacher forcing at shocks, all windows count. method "B": no teacher forcing; windows with a shock
    inside are masked. Optional batch["weight"] (B,) multiplies the window mask."""
    step_fn = lambda x, u, xi, uc: wm.step_single(params_1, norm, x, u, xi, uc)
    preds = k_step_rollout_on_window(wm, step_fn, batch, teacher_forcing=(method == "A"))
    err = state_error(wm.layout, preds, batch["x_minus_next"])[..., pred_idx(wm.layout, wm.cfg)] / norm["err_scale"]
    per_window = jnp.mean(err ** 2, axis=(1, 2))
    mask = jnp.ones(per_window.shape)
    if method == "B":
        mask = window_mask_method_b(batch["shock_next"]).astype(jnp.float32)
    elif method != "A":
        raise ValueError(f"method must be A or B, got {method}")
    if "weight" in batch:
        mask = mask * batch["weight"]
    loss = jnp.sum(per_window * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    return loss, {"loss": loss, "valid_frac": jnp.mean(mask)}


def k_step_loss(wm: WorldModel, params, norm, batch, method="A"):
    """Ensemble: sum of the members' losses (each member rolls out on its own predictions)."""
    losses, aux = jax.vmap(lambda p: k_step_loss_single(wm, p, norm, batch, method))(params)
    return jnp.sum(losses), jax.tree_util.tree_map(jnp.mean, aux)


# ---------------------------------------------------------------------------------------------------------------
# Imagination rollout (plan sec. 4.5): for T14 / P2
# ---------------------------------------------------------------------------------------------------------------
def no_delay_step(buffer, u):
    return u, buffer


def rollout(wm: WorldModel, bundle, x0, xi, delay_buffer0, u0, policy_fn: Callable, H: int,
            delay_step_fn: Optional[Callable] = None, open_loop: bool = False, return_actions: bool = False):
    """H-step imagination from x0 = x+_t (batched, (N, D)) with xi (N, P).

    Step t applies the candidate action u0 (N, A) through the delay buffer: (u~_t, buf) = delay_step_fn(buf, u0);
    steps t+1.. follow u_j = policy_fn(x_hat_j) (policy_fn wraps policy_obs_from_state + the deterministic policy).
    Returns x_hat_{t+1..t+H}, shape (H, N, D) (and the chosen actions (H, N, A) if return_actions).

    jax.grad w.r.t. u0 flows through the dynamics and the delay buffer, and (closed loop, default) through the policy's
    reaction to the imagined states. open_loop=True stops the gradient at the policy's outputs, so the later actions
    keep their values but are treated as constants (gradient through the dynamics only).
    """
    delay_step_fn = delay_step_fn or no_delay_step
    u_tilde, buf = delay_step_fn(delay_buffer0, u0)
    x1 = wm.step(bundle, x0, u_tilde, xi, u_chosen=u0)

    def body(carry, _):
        x, buf = carry
        u = policy_fn(x)
        if open_loop:
            u = jax.lax.stop_gradient(u)
        ut, buf = delay_step_fn(buf, u)
        xn = wm.step(bundle, x, ut, xi, u_chosen=u)
        return (xn, buf), (xn, u)

    if H > 1:
        _, (xs, us) = jax.lax.scan(body, (x1, buf), None, length=H - 1)
        traj = jnp.concatenate([x1[None], xs], axis=0)
        actions = jnp.concatenate([u0[None], us], axis=0)
    else:
        traj, actions = x1[None], u0[None]
    return (traj, actions) if return_actions else traj


def gated_value_and_grad(value_fn: Callable, u0, args, k: int):
    """Forward value for all N envs, gradient only for the k envs with the lowest value (static shapes).

    value_fn(u0, *args) -> (N,) values (e.g. the H-step safety value V of `rollout`); args are arrays with leading
    axis N. Returns (v (N,), grad (N, A) with zeros outside the selected envs, selected indices (k,)).
    For the shield: the constraint can only be active where V is low, so the VJP is spent there.
    """
    v = value_fn(u0, *args)
    _, idx = jax.lax.top_k(-jax.lax.stop_gradient(v), k)
    sub = [a[idx] for a in args]
    g_sub = jax.grad(lambda u: jnp.sum(value_fn(u, *sub)))(u0[idx])
    return v, jnp.zeros_like(u0).at[idx].set(g_sub), idx
