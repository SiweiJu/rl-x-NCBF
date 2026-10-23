"""Constrained Policy Optimization (Achiam, Held, Tamar, Abbeel, ICML 2017), policy step in JAX.

The step follows OmniSafe's implementation (omnisafe/algorithms/on_policy/second_order/cpo.py,
``CPO._update_actor`` and ``CPO._cpo_search_step``; conjugate gradients from
omnisafe/utils/math.py; Fisher-vector product from ``NaturalPG._fvp``):

  g = grad of the reward surrogate  E[ratio * A_r]
  b = grad of the cost surrogate    E[ratio * A_c]
  F = Hessian of the mean KL(pi_old || pi_theta) at theta_old (+ damping * I)
  x = F^-1 g, p = F^-1 b (conjugate gradients),  q = g^T F^-1 g, s = b^T F^-1 b
  c = J_c - d (constraint violation, see cpo.py for the normalization)

  optim_case 4: b ~ 0 and c < 0                       -> TRPO step  sqrt(2 delta / q) x
  optim_case 3: c < 0 and B < 0 (whole region feasible) -> TRPO step
  optim_case 2: c < 0 and B >= 0 (boundary intersects)  -> dual solution (lambda*, nu*)
  optim_case 1: c >= 0 and B >= 0 (infeasible, recoverable) -> dual solution
  optim_case 0: c >= 0 and B < 0 (infeasible)          -> recovery step -sqrt(2 delta / s) p (only reduces cost)
  with A = q - r^2 / s, B = 2 delta - c^2 / s, r = -g^T F^-1 b (see cpo_step_direction for the sign).

Then a backtracking line search (step_frac *= decay) accepts the first step with
  KL <= delta, cost-surrogate increase <= max(-c, 0), and (cases 2-4) reward-surrogate improvement >= 0;
if no step is accepted the policy is not changed.

Differences to OmniSafe (documented in experiments/pncbf/reports/T07_T08.md):
  * r = -g^T F^-1 b (Achiam's reference-code convention) instead of OmniSafe's r = g^T F^-1 b; with the latter
    the dual step of cases 1/2 is not the solution of the trust-region subproblem.
  * CG divides with a guarded denominator instead of adding eps=1e-6 (our surrogate gradients can be
    ~1e-3, where eps=1e-6 would bias the solution).
  * A non-finite loss/KL rejects the trial step and shrinks it (OmniSafe skips the decay on a
    non-finite KL).
"""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


EPS = 1e-8


def gaussian_log_prob(mean, logstd, action):
    std = jnp.exp(logstd)
    return jnp.sum(-0.5 * ((action - mean) / std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - logstd, axis=-1)


def gaussian_kl(mean_p, logstd_p, mean_q, logstd_q):
    """KL(p || q) of diagonal Gaussians, summed over the last axis."""
    var_p = jnp.exp(2.0 * logstd_p)
    var_q = jnp.exp(2.0 * logstd_q)
    kl = logstd_q - logstd_p + (var_p + (mean_p - mean_q) ** 2) / (2.0 * var_q) - 0.5
    return jnp.sum(kl, axis=-1)


def conjugate_gradient(matrix_vector_product, b, nr_iterations, residual_tol=1e-10):
    """Solves A x = b for symmetric positive definite A, given only x -> A x."""
    x = jnp.zeros_like(b)
    r = b
    p = b
    rdotr = jnp.dot(r, r)
    done = jnp.sqrt(rdotr) < residual_tol

    def body(_, carry):
        x, r, p, rdotr, done = carry
        z = matrix_vector_product(p)
        pz = jnp.dot(p, z)
        alpha = jnp.where(done | (pz <= 0.0), 0.0, rdotr / jnp.where(pz > 0.0, pz, 1.0))
        x = x + alpha * p
        r = r - alpha * z
        new_rdotr = jnp.dot(r, r)
        mu = new_rdotr / jnp.where(rdotr > 0.0, rdotr, 1.0)
        new_done = done | (jnp.sqrt(new_rdotr) < residual_tol) | (pz <= 0.0)
        p = jnp.where(new_done, p, r + mu * p)
        rdotr = jnp.where(done, rdotr, new_rdotr)
        return x, r, p, rdotr, new_done

    x, *_ = jax.lax.fori_loop(0, nr_iterations, body, (x, r, p, rdotr, done))
    return x


def make_fisher_vector_product(policy_apply, unravel, theta_old, observations, damping):
    """v -> (H_KL + damping I) v, H_KL = Hessian of mean KL(pi_old || pi_theta) at theta_old."""
    mean_old, logstd_old = policy_apply(unravel(theta_old), observations)
    mean_old = jax.lax.stop_gradient(mean_old)
    logstd_old = jax.lax.stop_gradient(logstd_old)

    def mean_kl(theta):
        mean, logstd = policy_apply(unravel(theta), observations)
        return jnp.mean(gaussian_kl(mean_old, logstd_old, mean, logstd))

    grad_kl = jax.grad(mean_kl)

    def fisher_vector_product(v):
        return jax.jvp(grad_kl, (theta_old,), (v,))[1] + damping * v

    return fisher_vector_product


def project(value, low, high):
    return jnp.maximum(jnp.minimum(value, high), low)


def cpo_step_direction(g, b, x, p, xHx, cost_violation, target_kl):
    """Dual solution of the CPO trust-region subproblem (OmniSafe CPO._update_actor)."""
    c = cost_violation
    q = xHx
    # Sign convention of Achiam's reference code (rllab ConjugateConstraintOptimizer), where the gradient is
    # that of the *loss* -g: r = b^T H^-1 (-g). The paper / OmniSafe formulas below (nu = (lambda c - r)_+ / s,
    # f_a, the projection bounds r / c) are correct only with this r; using r = g^T H^-1 b with the ascent
    # gradient g (as OmniSafe does) gives steps that leave the trust region (checked against a numerical QP
    # solver in tests/test_constrained_baselines.py::test_cpo_dual_matches_numerical_qp).
    r = -jnp.dot(g, p)
    s = jnp.dot(b, p)
    two_delta = 2.0 * target_kl

    A = q - r ** 2 / (s + EPS)
    B = two_delta - c ** 2 / (s + EPS)

    trivial_cost_gradient = (jnp.dot(b, b) <= 1e-6) & (c < 0.0)
    optim_case = jnp.where(
        trivial_cost_gradient, 4,
        jnp.where((c < 0.0) & (B < 0.0), 3,
                  jnp.where((c < 0.0) & (B >= 0.0), 2,
                            jnp.where((c >= 0.0) & (B >= 0.0), 1, 0))))

    # Cases 3, 4: plain TRPO step.
    alpha = jnp.sqrt(two_delta / (xHx + EPS))
    step_trpo = alpha * x
    lambda_trpo = 1.0 / (alpha + EPS)

    # Cases 1, 2: analytical solution of the dual.
    lambda_a = jnp.sqrt(jnp.maximum(A, 0.0) / jnp.maximum(B, EPS))
    lambda_b = jnp.sqrt(jnp.maximum(q, 0.0) / two_delta)
    eps_cost = c + EPS
    r_over_c = r / eps_cost
    lambda_a_star = jnp.where(c < 0.0, project(lambda_a, 0.0, r_over_c), project(lambda_a, r_over_c, jnp.inf))
    lambda_b_star = jnp.where(c < 0.0, project(lambda_b, r_over_c, jnp.inf), project(lambda_b, 0.0, r_over_c))

    def f_a(lam):
        return -0.5 * (A / (lam + EPS) + B * lam) - r * c / (s + EPS)

    def f_b(lam):
        return -0.5 * (q / (lam + EPS) + two_delta * lam)

    lambda_dual = jnp.where(f_a(lambda_a_star) >= f_b(lambda_b_star), lambda_a_star, lambda_b_star)
    nu_dual = jnp.maximum(lambda_dual * c - r, 0.0) / (s + EPS)
    step_dual = (x - nu_dual * p) / (lambda_dual + EPS)

    # Case 0: recovery, purely decrease the cost.
    nu_recovery = jnp.sqrt(two_delta / (jnp.maximum(s, 0.0) + EPS))
    step_recovery = -nu_recovery * p

    is_trpo = optim_case >= 3
    is_dual = (optim_case == 1) | (optim_case == 2)
    step = jnp.where(is_trpo, step_trpo, jnp.where(is_dual, step_dual, step_recovery))
    lambda_star = jnp.where(is_trpo, lambda_trpo, jnp.where(is_dual, lambda_dual, 0.0))
    nu_star = jnp.where(is_trpo, 0.0, jnp.where(is_dual, nu_dual, nu_recovery))

    info = {"q": q, "r": r, "s": s, "A": A, "B": B, "lambda_star": lambda_star, "nu_star": nu_star}
    return step, optim_case, info


def cpo_line_search(evaluate, theta_old, step, optim_case, cost_violation, target_kl, nr_steps, decay):
    """Backtracking line search of OmniSafe CPO._cpo_search_step.

    evaluate(theta) -> (loss_reward, loss_cost, kl) with loss_reward = -E[ratio A_r], loss_cost = E[ratio A_c].
    Returns (step_frac, accepted, acceptance_step, kl, loss_reward, loss_cost) of the accepted step
    (step_frac = 0 and the old losses if nothing was accepted).
    """
    loss_reward_old, loss_cost_old, _ = evaluate(theta_old)
    cost_tolerance = jnp.maximum(-cost_violation, 0.0)

    def cond(carry):
        i, _, accepted, *_ = carry
        return (~accepted) & (i < nr_steps)

    def body(carry):
        i, step_frac, _, _, _, _ = carry
        loss_reward, loss_cost, kl = evaluate(theta_old + step_frac * step)
        finite = jnp.isfinite(loss_reward) & jnp.isfinite(loss_cost) & jnp.isfinite(kl)
        reward_ok = jnp.where(optim_case > 1, (loss_reward_old - loss_reward) >= 0.0, True)
        cost_ok = (loss_cost - loss_cost_old) <= cost_tolerance
        kl_ok = kl <= target_kl
        accepted = finite & reward_ok & cost_ok & kl_ok
        step_frac = jnp.where(accepted, step_frac, step_frac * decay)
        return i + 1, step_frac, accepted, kl, loss_reward, loss_cost

    init = (jnp.int32(0), jnp.float32(1.0), jnp.bool_(False), jnp.float32(0.0), loss_reward_old, loss_cost_old)
    i, step_frac, accepted, kl, loss_reward, loss_cost = jax.lax.while_loop(cond, body, init)
    step_frac = jnp.where(accepted, step_frac, 0.0)
    acceptance_step = jnp.where(accepted, i, 0)
    kl = jnp.where(accepted, kl, 0.0)
    loss_reward = jnp.where(accepted, loss_reward, loss_reward_old)
    loss_cost = jnp.where(accepted, loss_cost, loss_cost_old)
    return step_frac, accepted, acceptance_step, kl, loss_reward, loss_cost


def cpo_policy_update(policy_apply, params, observations, actions, old_log_probs, reward_advantages, cost_advantages,
                      cost_violation, target_kl, cg_iterations, cg_damping, line_search_steps, line_search_decay,
                      fvp_observations=None):
    """One CPO policy step on a batch. Returns (new_params, metrics)."""
    theta_old, unravel = ravel_pytree(params)
    if fvp_observations is None:
        fvp_observations = observations

    def forward(theta):
        mean, logstd = policy_apply(unravel(theta), observations)
        ratio = jnp.exp(gaussian_log_prob(mean, logstd, actions) - old_log_probs)
        loss_reward = -jnp.mean(ratio * reward_advantages)
        loss_cost = jnp.mean(ratio * cost_advantages)
        return loss_reward, loss_cost, mean, logstd

    g = -jax.grad(lambda theta: forward(theta)[0])(theta_old)
    b = jax.grad(lambda theta: forward(theta)[1])(theta_old)

    fvp = make_fisher_vector_product(policy_apply, unravel, theta_old, fvp_observations, cg_damping)
    x = conjugate_gradient(fvp, g, cg_iterations)
    xHx = jnp.dot(x, fvp(x))
    p = conjugate_gradient(fvp, b, cg_iterations)

    step, optim_case, info = cpo_step_direction(g, b, x, p, xHx, cost_violation, target_kl)

    _, _, mean_old, logstd_old = forward(theta_old)
    mean_old = jax.lax.stop_gradient(mean_old)
    logstd_old = jax.lax.stop_gradient(logstd_old)

    def evaluate(theta):
        loss_reward, loss_cost, mean, logstd = forward(theta)
        kl = jnp.mean(gaussian_kl(mean_old, logstd_old, mean, logstd))
        return loss_reward, loss_cost, kl

    step_frac, accepted, acceptance_step, kl, loss_reward, loss_cost = cpo_line_search(
        evaluate, theta_old, step, optim_case, cost_violation, target_kl, line_search_steps, line_search_decay
    )
    new_params = unravel(theta_old + step_frac * step)

    metrics = {
        "cpo/optim_case": optim_case.astype(jnp.float32),
        "cpo/accepted": accepted.astype(jnp.float32),
        "cpo/acceptance_step": acceptance_step.astype(jnp.float32),
        "cpo/step_frac": step_frac,
        "cpo/kl": kl,
        "cpo/xHx": xHx,
        "cpo/cost_violation": jnp.asarray(cost_violation, dtype=jnp.float32),
        "cpo/loss_reward": loss_reward,
        "cpo/loss_cost": loss_cost,
        "cpo/g_norm": jnp.linalg.norm(g),
        "cpo/b_norm": jnp.linalg.norm(b),
        "cpo/step_norm": jnp.linalg.norm(step_frac * step),
        **{f"cpo/{key}": value for key, value in info.items()},
    }
    return new_params, metrics
