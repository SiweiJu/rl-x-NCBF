"""P3O penalty terms (Zhang et al., IJCAI 2022, arXiv 2205.11814). Pure, jittable.

P3O minimizes (their eqs. 5-7, one constraint)

    L(theta) = L_R^CLIP(theta) + kappa * max(0, L_C^CLIP(theta))
    L_R^CLIP = E[-min(r A_R, clip(r, 1-eps, 1+eps) A_R)]                        (PPO)
    L_C^CLIP = E[ max(r A_C, clip(r, 1-eps, 1+eps) A_C) ] + (1 - gamma)(J_C(pi_k) - d)

with r = pi_theta(a|s) / pi_k(a|s). The cost surrogate takes the *max* (pessimistic for cost): a cost decrease
beyond the trust region earns nothing, a cost increase is always charged. The penalty is a ReLU of the
minibatch *mean* (exact penalty, Theorem 1: equivalent to the constrained problem for kappa >= lambda*).

Units of the constant term (``constraint_normalization``): the surrogate is a per-step expectation, so J_C - d
must be in per-step units. The paper uses the discounted cost return from the initial state, hence (1 - gamma).
Our J_c is the undiscounted cost per episode (``cost_measure=episode``: failures per episode); by the
performance-difference lemma for episodic undiscounted returns, Delta J_ep ~= T * Delta E_t[r A_C] with T the
mean episode length (same argument as CPO's ``cost_violation_normalization=per_step``, T07 report section 3).
  * ``per_step``        (J_c - d) / T  (T = mean episode length from the rollout; nr_steps for cost_measure=rollout)
  * ``one_minus_gamma`` (1 - gamma)(J_c - d)  (paper formula, literal)
  * ``none``            J_c - d  (OmniSafe p3o.py: ``Jc = EpCost - cost_limit``)

Cost-advantage normalization (``cost_advantage_normalization``), per minibatch like the reward advantages:
  * ``scaled``     (default) A~ = (A - mu)/sigma and the constant becomes offset/sigma. Units as ``consistent``, but the
                   minibatch mean mu is dropped (E_{pi_k}[A^{pi_k}] = 0 by definition, so mu is critic bias / noise):
                   at pi = pi_k the relu argument is exactly offset/sigma, so the penalty is active iff J_c > d.
                   (T09 smoke: with ``consistent`` the critic bias mu < -offset switched the penalty off at J_c = 0.98,
                   d = 0.01.)
  * ``consistent`` N-P3O, Lee et al. 2023 (arXiv 2309.15430) eq. 16: A~ = (A - mu)/sigma and the constant
                   becomes (offset + mu)/sigma, so relu(.) is the unnormalized condition divided by sigma > 0:
                   the activation decision is unchanged by the normalization.
  * ``standardize`` A~ = (A - mu)/sigma, constant unchanged (P3O paper's "advantage normalization trick").
  * ``center``      A~ = A - mu, constant unchanged (OmniSafe buffer ``standardized_adv_c``).
  * ``none``
"""

import jax
import jax.numpy as jnp


CONSTRAINT_NORMALIZATIONS = ("per_step", "one_minus_gamma", "none")
COST_ADVANTAGE_NORMALIZATIONS = ("scaled", "consistent", "standardize", "center", "none")


def constraint_offset(measured_cost, cost_limit, horizon, gamma, normalization):
    """Constant term of L_C^CLIP in per-step units. horizon: mean episode length (or nr_steps)."""
    violation = measured_cost - cost_limit
    if normalization == "per_step":
        return violation / jnp.maximum(horizon, 1.0)
    if normalization == "one_minus_gamma":
        return (1.0 - gamma) * violation
    if normalization == "none":
        return violation
    raise ValueError(f"Unknown constraint_normalization {normalization!r}, expected one of {CONSTRAINT_NORMALIZATIONS}")


def normalize_cost_advantages(cost_advantages, offset, mode, eps=1e-8):
    """Returns (normalized advantages, offset in the same units)."""
    if mode == "none":
        return cost_advantages, offset
    mean = jnp.mean(cost_advantages)
    if mode == "center":
        return cost_advantages - mean, offset
    std = jnp.std(cost_advantages) + eps
    if mode == "standardize":
        return (cost_advantages - mean) / std, offset
    if mode == "scaled":
        return (cost_advantages - mean) / std, offset / std
    if mode == "consistent":
        return (cost_advantages - mean) / std, (offset + mean) / std
    raise ValueError(f"Unknown cost_advantage_normalization {mode!r}, expected one of {COST_ADVANTAGE_NORMALIZATIONS}")


def clipped_cost_surrogate(ratio, cost_advantages, clip_range, clip=True):
    """Per-sample cost surrogate: max(r A_C, clip(r) A_C) (paper eq. 7), or r A_C (clip=False, OmniSafe)."""
    unclipped = ratio * cost_advantages
    if not clip:
        return unclipped
    clipped = jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * cost_advantages
    return jnp.maximum(unclipped, clipped)


def penalty(ratio, cost_advantages, offset, kappa, clip_range, clip=True):
    """kappa * relu(mean(cost surrogate) + offset). Returns (penalty, violation estimate L_C^CLIP)."""
    violation = jnp.mean(clipped_cost_surrogate(ratio, cost_advantages, clip_range, clip)) + offset
    return kappa * jax.nn.relu(violation), violation
