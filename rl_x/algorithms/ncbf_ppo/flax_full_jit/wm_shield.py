"""Training-time shield of method D on the per-event H-step safety value (plan v3.2 sec. 4.6, P2 / T15).

    g_e(u) = V_e(x_t, u) - (1 - α̃) V_e^prev - α̃ γ_c,        γ_c = logit(p_c)
    G(u)   = -τ log Σ_e exp(-g_e(u) / τ)                       (combine="residual", default; τ = 0: hard min)
             or, combine="value" (the old "soft-min on h", F8): G = g(V_s), V_s = softmin_τ(V_e), V_s^prev likewise
    a = ∇_u G(u⁰),  c = aᵀu⁰ - G(u⁰),  u* = u⁰ + [c - aᵀu⁰]_+ / ‖a‖² · a     (paper eq. 13, hard projection)

then ‖u* - u⁰‖ capped at max_delta_u, optional relinearization (recompute G at u*, project again if still violated),
optional clipping to the action bounds, and u_safe = u⁰ + strength · (u* - u⁰) (strength: shield phase-out, T16).

The VJP is gated by default: G(u⁰) is computed for all envs, the gradient (and the relinearization and post-check)
only for the k = ceil(gate_frac · N) envs with the lowest G (static shapes). Envs outside the gate keep u⁰.
`gate="full"` computes value and gradient for every env in one pass.

V^prev (per env and event) is the value of the action that was finally applied at the previous step, V_e(x_{t-1},
u_safe_{t-1}) (from the post-check where the shield acted, else V(u⁰)). On the first step of an episode, and after a
step that skipped the shield because its state was stale, V^prev = γ_c (the constraint is then V_e ≥ γ_c).
"""
import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from rl_x.algorithms.ncbf_ppo.flax_full_jit.safety_value import ShieldInputs

COMBINES = ("residual", "value")
GATES = ("gated", "full")
PROJECTIONS = ("hard_projection", "soft_slack")


@dataclass(frozen=True)
class ShieldConfig:
    p_c: float = 0.5                  # γ_c = logit(p_c)
    alpha: float = 0.5                # α̃
    tau: float = 0.1                  # cross-event soft-min temperature; 0 = hard min
    combine: str = "residual"         # residual (plan 4.6) | value (soft-min on V first, F8)
    gate: str = "gated"               # gated | full
    gate_frac: float = 0.25           # fraction of envs (lowest G(u⁰)) that get the VJP when gate="gated"
    relinearize: int = 1              # 0, 1, 2, ...
    projection: str = "hard_projection"   # hard_projection | soft_slack (gain δ / (‖a‖² + 1/λ_s))
    lambda_s: float = 1.0
    max_delta_u: float = 0.5          # cap on ‖u_safe - u⁰‖ (L2); <= 0 disables
    min_grad_norm: float = 0.0        # no correction where ‖a‖ < this (0 = off)
    action_clipping: bool = False     # clip u_safe to the action bounds given to make_shield
    skip_stale: bool = True           # no intervention where inputs.stale (stale IMU groups, F13)
    post_check: bool = True           # recompute G at the final action of the gated envs (G_after, V^prev)

    @property
    def gamma_c(self):
        return math.log(self.p_c) - math.log1p(-self.p_c)

    def validate(self):
        if self.combine not in COMBINES:
            raise ValueError(f"combine must be one of {COMBINES}")
        if self.gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}")
        if self.projection not in PROJECTIONS:
            raise ValueError(f"projection must be one of {PROJECTIONS}")
        if not 0.0 < self.p_c < 1.0:
            raise ValueError("p_c must be in (0, 1)")
        if self.relinearize < 0:
            raise ValueError("relinearize must be >= 0")
        return self

    def nr_gated(self, nr_envs):
        if self.gate == "full":
            return nr_envs
        return int(min(nr_envs, max(1, math.ceil(self.gate_frac * nr_envs))))


def softmin(x, tau, axis=-1):
    """-τ log Σ exp(-x/τ) (τ > 0) or min (τ = 0). Note: for E events softmin <= min, = min for E = 1."""
    if tau <= 0.0:
        return jnp.min(x, axis=axis)
    return -tau * jax.nn.logsumexp(-x / tau, axis=axis)


def combine_residual(cfg: ShieldConfig, v, v_prev):
    """G from V (..., E) and V^prev (..., E)."""
    g_c = cfg.gamma_c
    if cfg.combine == "residual":
        g = v - (1.0 - cfg.alpha) * v_prev - cfg.alpha * g_c
        return softmin(g, cfg.tau)
    vs, vs_prev = softmin(v, cfg.tau), softmin(v_prev, cfg.tau)
    return vs - (1.0 - cfg.alpha) * vs_prev - cfg.alpha * g_c


def project_halfspace(u, a, G, cfg: ShieldConfig):
    """One closed-form step onto the linearization G + aᵀ(v - u) ≥ 0 at u (batched (..., A)); returns (v, ok) where
    ok marks finite linearizations above the low-gradient guard. No cap here."""
    ok = jnp.all(jnp.isfinite(a), axis=-1) & jnp.isfinite(G)
    a = jnp.where(ok[..., None], a, 0.0)
    G = jnp.where(ok, G, 0.0)
    aTa = jnp.sum(a * a, axis=-1)
    if cfg.min_grad_norm > 0.0:
        ok = ok & (jnp.sqrt(aTa) >= cfg.min_grad_norm)
    delta = jnp.where(ok, jnp.maximum(0.0, -G), 0.0)       # = [c - aᵀu]_+ with c = aᵀu - G
    denom = aTa + (1.0 / max(cfg.lambda_s, 1e-12) if cfg.projection == "soft_slack" else 0.0) + 1e-12
    return u + (delta / denom)[..., None] * a, ok


def cap_correction(u0, u, max_delta_u):
    if max_delta_u <= 0.0:
        return u
    d = u - u0
    n = jnp.linalg.norm(d, axis=-1)
    scale = jnp.minimum(1.0, max_delta_u / jnp.maximum(n, 1e-12))
    return u0 + scale[..., None] * d


class Shield:
    """Built by `make_shield`. State: {"v_prev": (N, E)}.

    apply(params, inputs, u0, state, strength=1.0) -> (u_safe (N, A), new_state, out) with out:
      active (N,) bool       the shield changed the action (G(u⁰) < 0, in the gate, not stale, finite)
      delta_u (N,)           ‖u_safe - u⁰‖
      G_before (N,)          G(u⁰) (all envs)
      G_after (N,)           G(u_safe) (post-check for the gated envs; = G_before elsewhere and if post_check=False:
                             G at the last linearization point)
      V (N, E)               V_e(x_t, u⁰);  V_after (N, E): V_e at the final action (same convention as G_after)
      v_prev (N, E)          the V^prev used in this step
      gate_mask (N,) bool    envs that got the VJP
      grad_norm (N,)         ‖∇_u G(u⁰)‖ (0 outside the gate)
      nr_projections (N,)    projection steps taken (0 .. 1 + relinearize)
      logp, min_margin (N, E) from the value function at u⁰
    u_safe is differentiable w.r.t. u0 (incl. through a, c); wrap the correction in stop_gradient if not wanted.
    """

    def __init__(self, value_fn: Callable, cfg: ShieldConfig, nr_events: int, act_low=None, act_high=None):
        self.value_fn, self.cfg, self.nr_events = value_fn, cfg.validate(), int(nr_events)
        self.act_low, self.act_high = act_low, act_high
        if cfg.action_clipping and (act_low is None or act_high is None):
            raise ValueError("action_clipping needs act_low / act_high")

    def init_state(self, nr_envs, dtype=jnp.float32):
        return {"v_prev": jnp.full((nr_envs, self.nr_events), self.cfg.gamma_c, dtype)}

    def reset_state(self, state, mask):
        """Set V^prev = γ_c where mask (e.g. done). apply() already does this via inputs.first_step."""
        return {"v_prev": jnp.where(mask[:, None], self.cfg.gamma_c, state["v_prev"])}

    # --- pieces -------------------------------------------------------------------------------------------------
    def G_and_V(self, params, inputs, u, v_prev):
        v, aux = self.value_fn(params, inputs, u)
        return combine_residual(self.cfg, v, v_prev), v, aux

    def G_value_and_grad(self, params, inputs, u, v_prev):
        """(G (n,), V (n, E), aux), ∇_u G (n, A); per-env gradients (envs are independent)."""
        def f(u_):
            G, v, aux = self.G_and_V(params, inputs, u_, v_prev)
            return jnp.sum(G), (G, v, aux)
        (_, (G, v, aux)), a = jax.value_and_grad(f, has_aux=True)(u)
        return (G, v, aux), a

    def _correct(self, params, inputs, u0, v_prev, G0, a0, strength):
        """Projection + relinearization + post-check on a (sub)batch whose G(u⁰), ∇G(u⁰) are known."""
        cfg = self.cfg
        u, ok0 = project_halfspace(u0, a0, G0, cfg)
        u = cap_correction(u0, u, cfg.max_delta_u)
        need0 = ok0 & (G0 < 0.0)
        n_proj = need0.astype(jnp.int32)
        G_last, v_last = G0, None
        for _ in range(cfg.relinearize):
            (G_u, v_u, _), a_u = self.G_value_and_grad(params, inputs, u, v_prev)
            G_last, v_last = G_u, v_u
            u_new, ok = project_halfspace(u, a_u, G_u, cfg)
            again = need0 & ok & (G_u < 0.0)
            u = jnp.where(again[:, None], cap_correction(u0, u_new, cfg.max_delta_u), u)
            n_proj = n_proj + again.astype(jnp.int32)
        u = jnp.where(need0[:, None], u, u0)
        if cfg.action_clipping:
            u = jnp.clip(u, self.act_low, self.act_high)
        u = u0 + jnp.asarray(strength, u0.dtype) * (u - u0)
        if cfg.post_check:
            G_after, v_after, _ = self.G_and_V(params, inputs, u, v_prev)
        else:
            G_after = G_last
            v_after = v_last
        return u, need0, n_proj, G_after, v_after

    # --- main ---------------------------------------------------------------------------------------------------
    def apply(self, params, inputs: ShieldInputs, u0, state, strength=1.0):
        cfg = self.cfg
        n = u0.shape[0]
        g_c = jnp.asarray(cfg.gamma_c, u0.dtype)
        first = jnp.asarray(inputs.first_step, bool)
        stale = jnp.asarray(inputs.stale, bool) if cfg.skip_stale else jnp.zeros((n,), bool)
        v_prev = jnp.where(first[:, None], g_c, state["v_prev"])
        k = cfg.nr_gated(n)

        if cfg.gate == "full":
            (G0, v0, aux0), a0 = self.G_value_and_grad(params, inputs, u0, v_prev)
            idx = jnp.arange(n)
        else:
            G0, v0, aux0 = self.G_and_V(params, inputs, u0, v_prev)
            score = jnp.where(stale | ~jnp.isfinite(G0), jnp.inf, jax.lax.stop_gradient(G0))
            _, idx = jax.lax.top_k(-score, k)
            sub = jax.tree_util.tree_map(lambda x: x[idx], inputs)
            (_, _, _), a0_sub = self.G_value_and_grad(params, sub, u0[idx], v_prev[idx])
            a0 = jnp.zeros_like(u0).at[idx].set(a0_sub)
        gate = jnp.zeros((n,), bool).at[idx].set(True)

        sub = jax.tree_util.tree_map(lambda x: x[idx], inputs) if cfg.gate == "gated" else inputs
        G0s = jnp.where(stale[idx], jnp.inf, G0[idx])   # stale rows: no correction
        u_s, act_s, nproj_s, Ga_s, va_s = self._correct(params, sub, u0[idx], v_prev[idx], G0s, a0[idx], strength)
        if va_s is None:   # relinearize = 0 and no post-check: last linearization point is u⁰
            va_s = v0[idx]
            Ga_s = G0[idx]
        Ga_s = jnp.where(act_s, Ga_s, G0[idx])
        va_s = jnp.where(act_s[:, None], va_s, v0[idx])

        u_safe = u0.at[idx].set(u_s)
        active = jnp.zeros((n,), bool).at[idx].set(act_s)
        G_after = G0.at[idx].set(Ga_s)
        v_after = v0.at[idx].set(va_s)
        nproj = jnp.zeros((n,), jnp.int32).at[idx].set(nproj_s)
        delta_u = jnp.linalg.norm(u_safe - u0, axis=-1)
        active = active & (delta_u > 0.0)

        v_prev_next = jnp.where(stale[:, None], g_c, jax.lax.stop_gradient(v_after))
        out = {"active": active, "delta_u": delta_u, "G_before": G0, "G_after": G_after, "V": v0, "V_after": v_after,
               "v_prev": v_prev, "gate_mask": gate, "grad_norm": jnp.linalg.norm(a0, axis=-1) * gate,
               "nr_projections": nproj, "stale": stale, "logp": aux0["logp"], "min_margin": aux0["min_margin"]}
        return u_safe, {"v_prev": v_prev_next}, out


def make_shield(value_fn: Callable, cfg: ShieldConfig = ShieldConfig(), nr_events: int = 1, act_low=None,
                act_high=None) -> Shield:
    """value_fn(params, inputs, u) -> (V (N, E), aux) from `safety_value.make_safety_value_fn`."""
    return Shield(value_fn, cfg, nr_events, act_low, act_high)


def make_shield_fn(shield_cfg: ShieldConfig, value_fn: Callable, nr_events: int = 1, act_low=None, act_high=None):
    """Functional form for the training loop (agent d-train):

        shield_fn(params, inputs, u0 (N, A), v_prev (N, E), strength=1.0) -> (u (N, A), v_exec (N, E), diag)

    u = u0 + strength · Δu. v_exec = V at the executed action, already set to γ_c on stale rows: store it as the next
    step's v_prev (first steps are handled inside via inputs.first_step). diag: the `Shield.apply` outputs plus
    "delta_u_norm" (= delta_u) and "v0" (min_e V(u⁰), (N,)). Differentiable w.r.t. u0; params are stop_gradient'ed
    by the value function. Initial v_prev: `jnp.full((N, E), shield_cfg.gamma_c)`."""
    shield = Shield(value_fn, shield_cfg, nr_events, act_low, act_high)

    def shield_fn(params, inputs, u0, v_prev, strength=1.0):
        u, st, out = shield.apply(params, inputs, u0, {"v_prev": v_prev}, strength)
        out = dict(out, delta_u_norm=out["delta_u"], v0=jnp.min(out["V"], axis=-1))
        return u, st["v_prev"], out

    shield_fn.shield = shield
    return shield_fn
