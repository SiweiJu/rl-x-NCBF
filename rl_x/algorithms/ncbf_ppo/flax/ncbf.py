from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState


def get_ncbf(config, env):
    ncbf_type = config.algorithm.ncbf.type
    ncbf_observation_indices = getattr(env, "ncbf_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    NCBF = NCBF_FFNN(config.algorithm.nr_hidden_units, ncbf_observation_indices)
    safety_layer_function = make_get_safe_action(NCBF)

    return (NCBF, safety_layer_function)


class NCBF_FFNN(nn.Module):
    nr_hidden_units: int
    observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x[..., self.policy_observation_indices]
        policy_mean = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        policy_mean = nn.tanh(policy_mean)
        policy_mean = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(policy_mean)
        policy_mean = nn.tanh(policy_mean)
        policy_mean = nn.Dense(np.prod(self.as_shape).item(), kernel_init=orthogonal(0.01), bias_init=constant(0.0))(policy_mean)
        policy_logstd = self.param("policy_logstd", constant(jnp.log(self.std_dev)), (1, np.prod(self.as_shape).item()))
        return policy_mean, policy_logstd


def make_get_safe_action(
    ncbf_apply: Callable[[Array, Array], Array],  # h_phi(x)
    *,
    # Option A: known control-affine dynamics x_{t+1} = f(x_t) + g(x_t) u_t
    f: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    g: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    # Option B: learned differentiable one-step predictor x_{t+1} = fhat(x_t, u_t)
    fhat: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
    gamma_c: float = 0.0,
    eta_cbf: float = 1.0,      # \tilde alpha(s) = eta_cbf * s   Kappa function
    lambda_s: float = 1e3      # slack penalty (large -> hard projection)
):
    """
    Factory that returns a JIT-able safety layer:
        get_safe_action(action_raw, x_t) -> u_safe, info

    Args:
      ncbf: h_phi(x): R^n -> R, predictive neural CBF (differentiable)
      f, g: known control-affine dynamics pieces (set both or none)
      fhat: learned predictor x_{t+1} = fhat(x, u) (set if f,g not provided)
      gamma_c: CBF margin \gamma_c
      eta_cbf: CBF gain \tilde{alpha}(s) = eta_cbf * s
      lambda_s: slack penalty in QP (Eq. closed-form); large -> hard projection

    Returns:
      get_safe_action: (action_raw, x_t) -> (u_safe, info)
        where info is a dict with useful diagnostics.
    """

    assert (f is not None and g is not None) ^ (fhat is not None), \
        "Provide either (f and g) for known dynamics OR fhat for learned dynamics, but not both."

    def alpha(s: jnp.ndarray) -> jnp.ndarray:
        # kappa function for CBF constraint
        return eta_cbf * s

    # --------- Option A: Known control-affine dynamics ---------
    if f is not None and g is not None:
        # a = (∇h(x_t))^T g(x_t)
        def compute_a_c_known(x_t: jnp.ndarray, phi: jnp.ndarray) -> (jnp.ndarray, jnp.ndarray):
            grad_h = jax.grad(ncbf_apply)(phi, x_t)                    # (n,)
            gx = g(x_t)                                      # (n, m)
            fx = f(x_t)                                      # (n,)
            a = gx.T @ grad_h                                # (m,)
            c = - grad_h @ fx - alpha(ncbf(x_t) - gamma_c)   # scalar
            return a, c

        @jax.jit
        def get_safe_action(action_raw: jnp.ndarray, x_t: jnp.ndarray, phi:jnp.ndarray):
            a, c = compute_a_c_known(x_t, phi)
            aTa = jnp.dot(a, a) + 1e-12                      # numeric safety
            aTu = jnp.dot(a, action_raw)
            # Use ReLU on (c - a^T u_raw) to keep identity when constraint inactive
            delta = jnp.maximum(0.0, c - aTu)
            gain = delta / (aTa + (1.0 / lambda_s))
            u_safe = action_raw + gain * a
            # epsilon* as in the piecewise expression
            eps_star = delta / (1.0 + lambda_s * aTa)
            info = {
                "a": a, "c": c,
                "delta": delta,
                "gain": gain,
                "epsilon_star": eps_star,
                "constraint_active": (delta > 0.0)
            }
            return u_safe, info

        return get_safe_action
    else:
        @jax.jit
        def get_safe_action(action_raw: jnp.ndarray, x_t: jnp.ndarray):
            # No safety layer implemented for learned dynamics yet
            return action_raw, {}
        return get_safe_action

    # TODO, decide how to handle Option B, learn overal funciton or control affine, and how to parse it

    # # --------- Option B: Learned dynamics with linearization around action_raw ---------
    # else:
    #     # We will linearize h(x_{t+1}(u)) at u0 = action_raw:
    #     #   h(x_{t+1}(u)) ≈ h(u0) + a^T (u - u0),
    #     #   where a = d/du h(x_{t+1}(u))|_{u0} = (∇_x h)(x_{t+1}) @ (∂x_{t+1}/∂u).
    #     def a_and_c_from_linearization(x_t: jnp.ndarray, u0: jnp.ndarray):
    #         def h_of_u(u):
    #             x_next = fhat(x_t, u)                       # (n,)
    #             return ncbf(x_next)                         # scalar
    #
    #         # a = ∂/∂u h(fhat(x,u)) |_{u0}
    #         a = jax.jacrev(h_of_u)(u0)                      # (m,)
    #         h_u0 = h_of_u(u0)
    #         h_x  = ncbf(x_t)
    #         # From: h(u) - h(x_t) + alpha(h(x_t)-gamma_c) >= 0
    #         # Linearized: a^T(u-u0) + h(u0) - h(x_t) + alpha(...) >= 0
    #         # => a^T u >= [ -h(u0) + h(x_t) - alpha(...) + a^T u0 ] =: c_lin
    #         c_lin = - h_u0 + h_x - alpha(h_x - gamma_c) + jnp.dot(a, u0)
    #         return a, c_lin
    #
    #     @jax.jit
    #     def get_safe_action(action_raw: jnp.ndarray, x_t: jnp.ndarray):
    #         a, c = a_and_c_from_linearization(x_t, action_raw)
    #         aTa = jnp.dot(a, a) + 1e-12
    #         aTu = jnp.dot(a, action_raw)
    #         delta = jnp.maximum(0.0, c - aTu)
    #         gain = delta / (aTa + (1.0 / lambda_s))
    #         u_safe = action_raw + gain * a
    #         eps_star = delta / (1.0 + lambda_s * aTa)
    #         info = {
    #             "a": a, "c": c,
    #             "delta": delta,
    #             "gain": gain,
    #             "epsilon_star": eps_star,
    #             "constraint_active": (delta > 0.0)
    #         }
    #         return u_safe, info
    #
    #     return get_safe_action