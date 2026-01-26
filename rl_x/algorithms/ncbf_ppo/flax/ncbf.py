from copy import deepcopy
from functools import partial
from typing import Sequence, Callable, Optional, Tuple, Dict

import mujoco
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from mujoco import mjx

import jax.scipy as jsp

Array = jnp.ndarray


def get_ncbf(config, env):
    # TODO: make different types of ncbf
    n_ncbf_ensemble = config.algorithm.ncbf.n_enssemble
    use_safety_layer = config.algorithm.ncbf.use_safety_layer
    gamma_c = config.algorithm.ncbf.gamma_c
    eta_cbf = config.algorithm.ncbf.eta_cbf
    lambda_slack = config.algorithm.ncbf.lambda_slack
    ncbf_clipping = config.algorithm.ncbf.action_clipping

    use_robot_safety_layer = config.algorithm.ncbf.use_robust_safety_layer


    NCBF = [NCBF_FFNN(config.algorithm.ncbf.nr_hidden_units) for _ in range(n_ncbf_ensemble)]
    NCBF_apply = get_ensemble_forward_pass(NCBF[0].apply)

    act_low = jnp.array(env.single_action_space.low)
    act_high = jnp.array(env.single_action_space.high)

    if config.algorithm.ncbf.use_safety_layer:
        safety_layer_function = make_get_safe_action_q(
            ncbf_apply=NCBF_apply,
            ncbf_obs_from_obs_idx=env.ncbf_observation_indices,
            use_safety_layer=use_safety_layer,
            act_low=act_low,
            act_high=act_high,
            gamma_c=gamma_c,
            eta_cbf=eta_cbf,
            lambda_s=lambda_slack,
            use_robust_safety_layer=use_robot_safety_layer,
            action_clipping=ncbf_clipping,
        )

    # dummy
    if ncbf_clipping:
        dummy_safety_layer_function = lambda action_raw, obs_t, action_last, obs_last, phi: (jnp.clip(action_raw, act_low, act_high),
                                                                      jnp.array(False), jnp.array(0.0), jnp.array(0.0),
                                                                      jnp.array(0.0))
    else:
        dummy_safety_layer_function = lambda action_raw, obs_t, action_last, obs_last, phi: (action_raw, jnp.array(False), jnp.array(0.0), jnp.array(0.0),
                                                                      jnp.array(0.0))

    if config.algorithm.ncbf.use_safety_layer:
        safety_layer_function_for_batch = safety_layer_function
    else:
        safety_layer_function_for_batch = dummy_safety_layer_function

    # Vectorize over env axis 0: (N_env, act_dim), (N_env, obs_dim), phi -> (N_env, act_dim), (N_env, info_struct)
    batched_get_safe_action = jax.jit(
        jax.vmap(
            safety_layer_function_for_batch,
            in_axes=(0, 0, 0, 0, None),  # action_raw[env], obs_t[env], same phi for items in the batch
            out_axes=(0, 0, 0, 0, 0)  # batched u_safe, constraint_active, delta_u, h_u0
        )
    )
    return NCBF, NCBF_apply, batched_get_safe_action, safety_layer_function_for_batch

def get_ensemble_forward_pass(apply_fn):
    alpha = 0.6
    E = 5  # number of ensemble members, hardcoded for now
    k = max(1, int(np.ceil((1.0 - alpha) * E)))
    # print("cvar consider least k:", k)

    @jax.jit
    def ensemble_forward_pass(params_stack, input):
        def single_forward(params):
            return apply_fn(params, input)

        # Use tree_leaves to iterate over actual parameter dicts
        predictions = jax.vmap(single_forward, in_axes=0, out_axes=0)(params_stack)

        def aggregate_predictions(preds):
            # return preds[0]
            # return jnp.mean(preds, axis=0)
            # # def cvar_soft(losses):
            # #     eta = jax.lax.stop_gradient(jnp.quantile(losses, alpha))
            # #     tail = jnp.maximum(losses - eta, 0.0)
            # #     return eta + jnp.mean(tail) / (1 - alpha)
            # #
            def cvar_firstk(x):
                tail = jnp.sort(x, axis=0)[:k, ...]
                return jnp.mean(tail, axis=0)


            # def cvar_gaussian(losses, eps=1e-6):
            #     """
            #     Gaussian-fitted lower-tail CVaR at level alpha.
            #     losses: (E, ...) with larger=worse.
            #     Returns: CVaR_alpha(losses) with same shape as losses[0].
            #     """
            #     mu = jnp.mean(losses, axis=0)
            #     # Use ddof=0 for stability with small E; add eps to avoid sigma=0 issues.
            #     sigma = jnp.std(losses, axis=0) + eps
            #
            #     # z_alpha = Phi^{-1}(alpha)
            #     z = jsp.special.ndtri(alpha)
            #
            #     # phi(z) = standard normal pdf
            #     phi = jnp.exp(-0.5 * z * z) / jnp.sqrt(2.0 * jnp.pi)
            #
            #     # lower-tail CVaR (worst tail)
            #     return mu - sigma * (phi / alpha)

            #

            return cvar_firstk(preds)

        return aggregate_predictions(predictions.squeeze()), jnp.std(predictions, axis=0), predictions
    return ensemble_forward_pass



class NCBF_FFNN(nn.Module):
    nr_hidden_units: int

    @nn.compact
    def __call__(self, x):
        # Two hidden layers
        x = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        # Scalar CBF output h(x)
        h = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)

        # h = nn.sigmoid(h)
        # This output is used in the safety layer to calculate the gradient only, for prediction, use self.predict_with_sigmoid
        return jnp.squeeze(h, -1)  # shape ()


def make_get_safe_action_q(
    ncbf_apply: Callable[[dict, Array], Array],   # h_phi(obs)
    ncbf_obs_from_obs_idx: Array,
    use_safety_layer: bool,
    act_low: Array,
    act_high: Array,
    gamma_c: float,
    eta_cbf: float,      # \tilde alpha(s) = eta_cbf * s
    lambda_s: float,     # slack penalty
    action_clipping: bool = False,
    lambda_h:float = 0,
    use_robust_safety_layer: bool = False,
):
    """
    Returns a JIT-able safety layer:
        get_safe_action(action_raw, x_t, t, contact, phi) -> (u_safe, info)

    Args:
      ncbf_apply: flax apply function input: apply_fn, parameters_stack, input
      system_forward_dynamics_function: x_next = f(x, u, t, contact)
      state_from_obs_id: indices to get dynamics state from observation [pos(3), quat(4), joint_pos, joint_vel]
        use_safety_layer: whether to use safety layer (if False, returns raw action, but still calculates constraint violations)
      gamma_c: conservative margin
      eta_cbf: class-K gain
      lambda_s: slack penalty (large -> hard projection)
      obs_from_state: optional mapping if your NCBF uses an observation (not full state).
                      If None, we assume NCBF takes the raw state x_t / x_next directly.

    Notes:
      - Uses linearization:
          h_phi(x_{t+1}(u)) ≈ h_phi(x_{t+1}(u0)) + a^T (u - u0)
        where a = d/du h_phi(x_{t+1}(u))|_{u0}
      - No explicit g(x) needed.
    """

    def alpha(s: Array) -> Array:
        return eta_cbf * s

    @jax.jit
    def a_and_c_from_linearization(obs_t: Array, u0: Array, obs_last, u_last, phi: dict):
        """
        Compute:
          a = d/du h_phi(x_{t+1}(u)) | u0
          c = linearized RHS so constraint is a^T u >= c
        Args:
            obs_t: current observation (full observation matrix, need to get state from it)
        """

        x_t = obs_t[ncbf_obs_from_obs_idx]
        x_last = obs_last[ncbf_obs_from_obs_idx]

        def q_of_u(u):
            q_input = jnp.concatenate([x_t, u], axis=-1)
            q, _, _ = ncbf_apply(phi, q_input)
            return q # scalar-ish # note here phis is parameter stack for the ensemble

        a = jax.grad(q_of_u)(u0)

        h_x, h_x_std, _ = ncbf_apply(phi, jnp.concatenate([x_last, u_last], axis=-1))
        h_u0 = q_of_u(u0)

        c_lin = h_x - alpha(h_x - gamma_c)  + jnp.dot(a, u0) - h_u0
        return a, c_lin, h_x, h_u0, h_x_std

    @jax.jit
    def get_safe_action(
        action_raw: Array,
        obs_t: Array,
        last_action: Array,
        last_obs: Array,
        phis: dict
    ) -> Tuple[Array, Array, Array, Array, Array]:

        # clip action raw first
        # action_raw = jnp.clip(action_raw, act_low, act_high)
        action_raw = jax.lax.cond(action_clipping, lambda x: jnp.clip(x, act_low, act_high), lambda x: x, action_raw)

        a, c, h_x, h_u0, h_x_std = a_and_c_from_linearization(obs_t, action_raw, last_obs, last_action, phis)

        aTa = jnp.dot(a, a) + 1e-12
        aTu = jnp.dot(a, action_raw)

        # closed-form QP solution (soft slack)
        delta = jnp.maximum(0.0, c - aTu)
        gain = delta / (aTa + (1.0 * h_x_std**2  / lambda_s))
        u_safe = action_raw + gain * a

        # if last_action is all zeros, set use_safety_layer to False to avoid initial jank
        last_action_zero = jnp.all(last_action == 0.0)
        use_sl = jnp.logical_and(use_safety_layer, jnp.logical_not(last_action_zero))

        u_processed = jax.lax.cond(use_sl, lambda _: u_safe, lambda _: action_raw, operand=None)
        u_clipped = jax.lax.cond(
            action_clipping,
            lambda x: jnp.clip(x, act_low, act_high),
            lambda x: x,
            u_processed
        )
        delta_u = jnp.linalg.norm(u_clipped - action_raw)

        x_next = 0  # dummy for compatibility
        return u_clipped, delta, delta_u, x_next, h_u0

    return get_safe_action
