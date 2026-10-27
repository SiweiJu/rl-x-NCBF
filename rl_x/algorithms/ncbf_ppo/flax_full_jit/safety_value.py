"""Per-event H-step safety value V_e(x_t, u) for method D (plan v3.2 sec. 4.5, P2 / T15).

    x̂_{t+1} = f_ψ(x⁺_t, ũ_t(u), ξ),   x̂_{j+1} = f_ψ(x̂_j, ũ_j(μ_θ(o(x̂_j))), ξ),  j = t+1 .. t+H-1
    log(1 - d_e(x)) = -softplus(-(ℓ_e(x) - b_e) / T_ℓ,e),   log P̂_e = Σ_{j=1..H} log(1 - d_e(x̂_{t+j}))
    V_e = log P̂_e - log1mexp(log P̂_e)                     (= logit P̂_e, numerically stable)

The candidate u enters at step t through the env's action delay (`wm_state.delay_step`); the later actions come from
the *unshielded* deterministic policy μ_θ on the noise-free observation o(x̂) (`wm_state.policy_obs_from_state`), with
stop_gradient on θ (and on ψ): the value is differentiable in u only (closed loop: through the dynamics, the delay
buffer and the policy's reaction; open loop: the later actions are constants).

Events (`SafetyValueConfig.events`): "fall" = `wm_state.margin_fall` (the env termination, default),
"fall_physical" = `wm_state.margin_fall_physical`, "collision" = `ImaginationSpec.collision_margin_fn` (P5 hook).

Sources (`SafetyValueConfig.source`), all returning V of shape (N, E) in logit units:
  "wm"               world-model imagination (above; main path)
  "critic"           the existing safety critic: V = logit h(s, u) (one event), via `critic_fn`
  "mjx_oracle"       true MJX H-step rollout with the real ξ (shocks / DR / command resampling off), small-scale checks
  "mjx_nominal_1step" (later, plan 4.11 (i)) not implemented yet

Everything is a pure function of (params, inputs, u), jit / vmap / grad safe. Batched over a leading env axis N.
"""
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_state
from rl_x.algorithms.ncbf_ppo.flax_full_jit import world_model as W

SOURCES = ("wm", "critic", "mjx_oracle", "mjx_nominal_1step")
EVENTS = ("fall", "fall_physical", "collision")


# ---------------------------------------------------------------------------------------------------------------
# Config / inputs
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SafetyValueConfig:
    source: str = "wm"
    H: int = 25
    events: Tuple[str, ...] = ("fall",)
    T_l: Tuple[float, ...] = (1.0,)              # per event (T14: 1.0 for fall); a 1-tuple is broadcast
    margin_offset: Tuple[float, ...] = (0.0,)    # b_e per event, ℓ_e - b_e (b_e > 0: more conservative)
    open_loop: bool = False
    matmul_dtype: str = "float32"                # WM matmuls: "float32" | "bfloat16" (f32 accumulation)
    margin_softmin_temperature: float = 0.0      # inside margin_fall / margin_fall_physical (0 = hard min over terms)
    delay_mode: str = "xi"                       # "xi": delay_steps_from_xi (fixed; worst case if mixed) | "given"
    logp_max: float = -1e-12                     # clamp of log P̂ before the logit (V <= ~27.6 at the default)
    scan_unroll: int = 1                         # lax.scan unroll of the imagination (speed only; same values)

    def per_event(self, values, name):
        values = tuple(float(v) for v in values)
        if len(values) == 1:
            values = values * len(self.events)
        if len(values) != len(self.events):
            raise ValueError(f"{name} has {len(values)} entries for {len(self.events)} events")
        return values

    def validate(self):
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source}")
        for e in self.events:
            if e not in EVENTS:
                raise ValueError(f"event must be one of {EVENTS}, got {e}")
        if self.delay_mode not in ("xi", "given"):
            raise ValueError("delay_mode must be 'xi' or 'given'")
        self.per_event(self.T_l, "T_l")
        self.per_event(self.margin_offset, "margin_offset")
        if self.source == "critic" and len(self.events) != 1:
            raise ValueError("source='critic' has one event")
        return self


class ShieldInputs(NamedTuple):
    """Per-env inputs of the safety value / shield at step t (leading axis N). A pytree; gather with tree_map.
    x0: x⁺_t (N, S) from `wm_state.extract_state`; xi_full: (N, P_full) from `wm_state.extract_xi` (the WM gets
    xi_full[..., spec.xi_active]); delay_buffer: (N, B, A) env action_history before step t; delay_steps: (N,) int
    (used when delay_mode="given"); command: (N, 3) velocity command; first_step: (N,) bool (episode_step == 0);
    stale: (N,) bool, IMU groups of x0 not recomputed (episode step 0, right after a shock / DR step; F13): the
    shield does not intervene there. extra: source-specific pytree (critic: {"obs", "latent_z"}; mjx_oracle: the
    batched env_state), else None."""
    x0: Any
    xi_full: Any
    delay_buffer: Any
    delay_steps: Any
    command: Any
    first_step: Any
    stale: Any
    extra: Any = None


@dataclass
class ImaginationSpec:
    """Static description of the state / ξ / observation used in imagination (from the env or hand-built in tests)."""
    lay: wm_state.Layout
    xl: wm_state.Layout
    obs_fn: Callable                       # (x, command, xi_full) -> policy-network input (full obs vector)
    xi_active: Optional[np.ndarray] = None # indices of xi_full the WM sees (bundle meta["schema"]["xi_active"])
    margin_cfg: wm_state.MarginConfig = wm_state.MarginConfig()
    phys_cfg: wm_state.PhysicalFallConfig = wm_state.PhysicalFallConfig()
    max_nr_delay_steps: int = 1
    collision_margin_fn: Optional[Callable] = None   # (x_traj (H,N,S), xi_full (N,P), extra) -> ℓ_col (H,N); P5

    @staticmethod
    def from_env(env, xi_active=None, margin_cfg=None, phys_cfg=None, collision_margin_fn=None):
        _, full_obs_fn = wm_state.make_policy_obs_fn(env)
        fn = env.domain_randomization_action_delay_function
        return ImaginationSpec(
            lay=wm_state.state_layout(env), xl=wm_state.xi_layout(env), obs_fn=full_obs_fn,
            xi_active=None if xi_active is None else np.asarray(xi_active),
            margin_cfg=margin_cfg or wm_state.margin_config_from_env(env),
            phys_cfg=phys_cfg or wm_state.PhysicalFallConfig(),
            max_nr_delay_steps=int(getattr(fn, "max_nr_delay_steps", 0) or 0),
            collision_margin_fn=collision_margin_fn)

    def wm_xi(self, xi_full):
        return xi_full if self.xi_active is None else xi_full[..., self.xi_active]


def inputs_from_env_state(env, env_state, spec: ImaginationSpec, extra=None) -> ShieldInputs:
    """ShieldInputs for the batched env_state about to be stepped (x⁺_t, ξ_t, delay buffer, command, flags)."""
    internal = env_state.internal_state
    x0 = jax.vmap(lambda d, i: wm_state.extract_state(env, d, i, i["last_action"]))(env_state.data, internal)
    xi_full = jax.vmap(lambda m, i: wm_state.extract_xi(env, m, i))(env_state.mjx_model, internal)
    buf = jax.vmap(lambda i: wm_state.delay_buffer(env, i))(internal)
    first = env_state.info_episode_store["episode_step"] == 0
    shock = (env_state.info["shock/perturbed"] > 0.5) | (env_state.info["shock/dr_resampled"] > 0.5)
    ds = wm_state.delay_steps_from_xi(xi_full, spec.xl, max_nr_delay_steps=max(spec.max_nr_delay_steps, 0))
    return ShieldInputs(x0=x0, xi_full=xi_full, delay_buffer=buf, delay_steps=ds,
                        command=internal["goal_velocities"], first_step=first, stale=first | shock, extra=extra)


def make_imagined_policy(policy_apply, process_fn=None):
    """μ_θ for imagination: policy_fn(policy_params, obs_full) -> processed deterministic action.
    policy_apply(params, obs, latent, decoder_out) -> (mean, logstd), as `PPO.policy.apply`."""
    def policy_fn(params, obs):
        mean, _ = policy_apply(params, obs, None, None)
        return process_fn(mean) if process_fn is not None else mean
    return policy_fn


# ---------------------------------------------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------------------------------------------
def log1mexp(x):
    """log(1 - exp(x)) for x < 0, stable on both ends."""
    ln2 = -0.6931471805599453
    near = x > ln2
    xa = jnp.where(near, x, ln2)        # safe inputs per branch, so the unused branch has no inf / NaN gradient
    xb = jnp.where(near, ln2, x)
    return jnp.where(near, jnp.log(-jnp.expm1(xa)), jnp.log1p(-jnp.exp(xb)))


def logit_from_logp(logp, logp_max=-1e-12):
    """V = logit P̂ from log P̂ (clamped to logp_max < 0)."""
    logp = jnp.minimum(logp, logp_max)
    return logp - log1mexp(logp)


def log_survival(margins, T_l, offset=0.0):
    """margins (H, ...) -> log P̂ (...) = Σ_j -softplus(-(ℓ_j - b) / T)."""
    return jnp.sum(-jax.nn.softplus(-(margins - offset) / T_l), axis=0)


# ---------------------------------------------------------------------------------------------------------------
# Value function
# ---------------------------------------------------------------------------------------------------------------
def event_margins(cfg: SafetyValueConfig, spec: ImaginationSpec, traj, xi_full, extra=None):
    """traj (H, N, S) -> margins (H, N, E) in normalized units (> 0 safe)."""
    xb = jnp.broadcast_to(xi_full, traj.shape[:-1] + xi_full.shape[-1:])
    out = []
    for e in cfg.events:
        if e == "fall":
            out.append(wm_state.margin_fall(traj, xb, spec.margin_cfg, spec.lay, spec.xl, cfg.margin_softmin_temperature))
        elif e == "fall_physical":
            out.append(wm_state.margin_fall_physical(traj, xb, spec.phys_cfg, spec.lay, spec.xl,
                                                     cfg.margin_softmin_temperature))
        elif e == "collision":
            if spec.collision_margin_fn is None:
                raise ValueError("event 'collision' needs ImaginationSpec.collision_margin_fn")
            out.append(spec.collision_margin_fn(traj, xi_full, extra))
    return jnp.stack(out, axis=-1)


def values_from_margins(cfg: SafetyValueConfig, margins):
    """margins (H, N, E) -> (V (N, E), log P̂ (N, E))."""
    T = jnp.asarray(cfg.per_event(cfg.T_l, "T_l"), margins.dtype)
    b = jnp.asarray(cfg.per_event(cfg.margin_offset, "margin_offset"), margins.dtype)
    logp = log_survival(margins, T, b)
    return logit_from_logp(logp, cfg.logp_max), logp


def make_safety_value_fn(cfg: SafetyValueConfig, spec: ImaginationSpec, wm: Optional[W.WorldModel] = None,
                         policy_fn: Optional[Callable] = None, critic_fn: Optional[Callable] = None,
                         mjx_rollout_fn: Optional[Callable] = None, policy_grad: bool = False):
    """Returns value_fn(params, inputs: ShieldInputs, u (N, A)) -> (V (N, E), aux).

    policy_grad=True (D2, plan 4.7/4.8): params["policy"] is NOT stop_gradient'ed, so the value is differentiable
      w.r.t. the imagined policy's parameters (closed loop); everything else (WM, critic) still is.

    params: {"wm": WM bundle, "policy": policy params, "critic": critic params}; only the entries the source needs.
      All of params is stop_gradient'ed inside (the value is differentiable in u only).
    policy_fn(policy_params, obs_full) -> u: the deterministic imagined policy (`make_imagined_policy`).
    critic_fn(critic_params, extra, u) -> p_safe (N,): source "critic".
    mjx_rollout_fn(params, inputs, u, H) -> x⁺ trajectory (H, N, S): source "mjx_oracle" (`make_mjx_oracle_rollout`).
    aux: {"logp": (N, E), "min_margin": (N, E)} (min_margin is NaN for the critic source).
    """
    cfg.validate()
    if cfg.source == "wm" and wm is not None and wm.cfg.matmul_dtype != cfg.matmul_dtype:
        wm = W.WorldModel(wm.layout, dataclasses.replace(wm.cfg, matmul_dtype=cfg.matmul_dtype))

    def delay_fn(inputs):
        if cfg.delay_mode == "given":
            ds = inputs.delay_steps
        else:
            ds = wm_state.delay_steps_from_xi(inputs.xi_full, spec.xl, max_nr_delay_steps=spec.max_nr_delay_steps)
        return lambda b, u: wm_state.delay_step(b, u, ds)

    def from_traj(traj, inputs):
        m = event_margins(cfg, spec, traj, inputs.xi_full, inputs.extra)
        v, logp = values_from_margins(cfg, m)
        return v, {"logp": logp, "min_margin": jnp.min(m, axis=0)}

    if cfg.source == "wm":
        if wm is None or policy_fn is None:
            raise ValueError("source='wm' needs wm and policy_fn")

        def value_fn(params, inputs, u):
            pol_params = params["policy"] if policy_grad else jax.lax.stop_gradient(params["policy"])
            params = dict(jax.lax.stop_gradient({k: v for k, v in params.items() if k != "policy"}), policy=pol_params)
            pol = lambda x: policy_fn(params["policy"], spec.obs_fn(x, inputs.command, inputs.xi_full))
            traj = W.rollout(wm, params["wm"], inputs.x0, spec.wm_xi(inputs.xi_full), inputs.delay_buffer, u, pol,
                             cfg.H, delay_step_fn=delay_fn(inputs), open_loop=cfg.open_loop, unroll=cfg.scan_unroll)
            return from_traj(traj, inputs)
    elif cfg.source == "critic":
        if critic_fn is None:
            raise ValueError("source='critic' needs critic_fn")

        def value_fn(params, inputs, u):
            params = jax.lax.stop_gradient(params)
            p = jnp.clip(critic_fn(params["critic"], inputs.extra, u), 1e-7, 1.0 - 1e-7)
            v = (jnp.log(p) - jnp.log1p(-p))[:, None]
            return v, {"logp": jnp.log(p)[:, None], "min_margin": jnp.full_like(v, jnp.nan)}
    elif cfg.source == "mjx_oracle":
        if mjx_rollout_fn is None:
            raise ValueError("source='mjx_oracle' needs mjx_rollout_fn")

        def value_fn(params, inputs, u):
            params = jax.lax.stop_gradient(params)
            return from_traj(mjx_rollout_fn(params, inputs, u, cfg.H), inputs)
    else:
        raise NotImplementedError("source='mjx_nominal_1step' is not implemented yet (plan 4.11 (i))")
    return value_fn


def make_mjx_oracle_rollout(env, policy_fn, spec: ImaginationSpec):
    """True MJX H-step rollout from the batched env_state in inputs.extra (shocks, DR and command resampling off,
    noise-free observations; same patches as the T12 oracle). The env must record the WM state
    (`record_wm_state`), whose info["wm/x_plus"] is the pre-reset x⁺ of each step. After a termination the env
    resets inside the step; the margins of the remaining steps are set to min(margin, margin at termination) so the
    terminal hazard is kept. Small-scale checks only (differentiating through mjx.step is slow)."""
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.wm_collect import _oracle_env_patches
    raw_step = type(env)._step.__wrapped__
    pidx = jnp.asarray(env.policy_observation_indices)

    def rollout_fn(params, inputs, u, H):
        state0 = inputs.extra
        with _oracle_env_patches(env, False):
            step = jax.vmap(lambda s, a: raw_step(env, s, a))

            def body(carry, _):
                state, u, dead = carry
                state = step(state, u)
                x = state.info["wm/x_plus"]
                obs = state.next_observation.at[:, pidx].set(
                    spec.obs_fn(x, state.internal_state["goal_velocities"], inputs.xi_full)[:, pidx])
                u_next = policy_fn(params["policy"], obs)
                dead_next = dead | state.terminated.astype(bool)
                return (state, u_next, dead_next), (x, dead)

            _, (xs, dead) = jax.lax.scan(body, (state0, u, jnp.zeros(u.shape[0], bool)), None, length=H)
        # freeze the trajectory at the terminal state (states after a reset would look safe)
        def freeze(carry, inp):
            x, d = inp
            x = jnp.where(d[:, None], carry, x)
            return x, x
        _, xs = jax.lax.scan(freeze, xs[0], (xs, dead))
        return xs

    return rollout_fn
