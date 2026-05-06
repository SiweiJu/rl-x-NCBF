from copy import deepcopy
from functools import partial
from typing import Sequence, Callable, Optional, Tuple, Dict
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax.scipy.special import logsumexp
from mujoco import mjx

Array = jnp.ndarray

_LOGISTIC_NORMAL_GH_NODES, _LOGISTIC_NORMAL_GH_WEIGHTS = np.polynomial.hermite.hermgauss(9)
_LOGISTIC_NORMAL_GH_PROB_WEIGHTS = _LOGISTIC_NORMAL_GH_WEIGHTS / np.sqrt(np.pi)
_RESIDUAL_MC_RNG = np.random.default_rng(0)


def split_ncbf_output(
    output: Array,
    output_distribution: str,
    min_log_std: float,
    max_log_std: float,
) -> Tuple[Array, Array]:
    if output_distribution == "logistic_normal":
        mean, raw_log_std = jnp.split(output, 2, axis=-1)
        log_std = jnp.clip(raw_log_std, min_log_std, max_log_std)
        return mean, log_std
    return output, jnp.zeros_like(output)


def _reshape_quadrature_values(values: Array, reference: Array) -> Array:
    shape = (1,) * (reference.ndim - 1) + (values.shape[0], 1)
    return values.reshape(shape)


def logistic_normal_binary_cross_entropy(mean_logits: Array, log_std: Array, targets: Array) -> Array:
    nodes = jnp.asarray(_LOGISTIC_NORMAL_GH_NODES, dtype=mean_logits.dtype)
    log_weights = jnp.log(jnp.asarray(_LOGISTIC_NORMAL_GH_PROB_WEIGHTS, dtype=mean_logits.dtype))
    nodes = _reshape_quadrature_values(nodes, mean_logits)
    log_weights = _reshape_quadrature_values(log_weights, mean_logits)

    logits = mean_logits[..., None, :] + jnp.sqrt(jnp.asarray(2.0, dtype=mean_logits.dtype)) * jnp.exp(log_std[..., None, :]) * nodes
    targets = targets[..., None, :]
    log_likelihood = jnp.where(
        targets > 0.5,
        -jax.nn.softplus(-logits),
        -jax.nn.softplus(logits),
    )
    return -logsumexp(log_weights + log_likelihood, axis=-2)


def logistic_normal_safe_probability(mean_logits: Array, log_std: Array) -> Array:
    nodes = jnp.asarray(_LOGISTIC_NORMAL_GH_NODES, dtype=mean_logits.dtype)
    weights = jnp.asarray(_LOGISTIC_NORMAL_GH_PROB_WEIGHTS, dtype=mean_logits.dtype)
    nodes = _reshape_quadrature_values(nodes, mean_logits)
    weights = _reshape_quadrature_values(weights, mean_logits)

    logits = mean_logits[..., None, :] + jnp.sqrt(jnp.asarray(2.0, dtype=mean_logits.dtype)) * jnp.exp(log_std[..., None, :]) * nodes
    return jnp.sum(weights * jax.nn.sigmoid(logits), axis=-2)


def get_ncbf(config, env):
    # ...existing imports and comments...
    n_ncbf_ensemble = config.algorithm.ncbf.n_ensemble
    n_ncbf_output = len(env.ncbf_target_indices)
    use_safety_layer = config.algorithm.ncbf.use_safety_layer
    ncbf_observation_indices = env.ncbf_observation_indices
    gamma_c = config.algorithm.ncbf.gamma_c
    ncbf_clipping = config.algorithm.ncbf.action_clipping
    eta_cbf = config.algorithm.ncbf.eta_cbf
    lambda_s = config.algorithm.ncbf.lambda_slack
    output_distribution = getattr(config.algorithm.ncbf, "output_distribution", "deterministic")
    min_log_std = getattr(config.algorithm.ncbf, "min_log_std", -5.0)
    max_log_std = getattr(config.algorithm.ncbf, "max_log_std", 2.0)
    residual_mc_samples = getattr(config.algorithm.ncbf, "residual_mc_samples", 16)

    act_low = jnp.array(env.single_action_space.low)
    act_high = jnp.array(env.single_action_space.high)

    NCBF = [
        NCBF_FFNN(
            config.algorithm.ncbf.nr_hidden_units,
            n_ncbf_output,
            output_distribution=output_distribution,
        )
        for _ in range(n_ncbf_ensemble)
    ]
    NCBF_apply = get_ensemble_forward_pass(
        NCBF[0].apply,
        n_ncbf_ensemble,
        output_distribution=output_distribution,
        min_log_std=min_log_std,
        max_log_std=max_log_std,
        residual_mc_samples=residual_mc_samples,
    )


    safety_layer_function = make_get_safe_action(
        ncbf_apply=NCBF_apply,
        ncbf_obs_from_obs_idx=ncbf_observation_indices,
        use_safety_layer=use_safety_layer,
        act_low=act_low,
        act_high=act_high,
        gamma_c=gamma_c,
        eta_cbf=eta_cbf,
        lambda_s=lambda_s,
        action_clipping=ncbf_clipping,
    )

    # dummy
    if ncbf_clipping:
        dummy_safety_layer_function = lambda action_raw, obs_t, last_action, last_obs, latent_z, safety_layer_curriculum_coeff, phi: (jnp.clip(action_raw, act_low, act_high), jnp.array(False), jnp.array(0.0))
    else:
        dummy_safety_layer_function = lambda action_raw, obs_t, last_action, last_obs, latent_z, safety_layer_curriculum_coeff, phi: (action_raw, jnp.array(False), jnp.array(0.0))

    if config.algorithm.ncbf.use_safety_layer:
        safety_layer_function_for_batch = safety_layer_function
    else:
        safety_layer_function_for_batch = dummy_safety_layer_function

    # Vectorize over env axis 0: (N_env, act_dim), (N_env, obs_dim), phi -> (N_env, act_dim), (N_env, info_struct)
    batched_get_safe_action = jax.jit(
        jax.vmap(
            safety_layer_function_for_batch,
            in_axes=(0, 0, 0, 0, 0, None, None),  # action_raw[env], obs_t[env], same coeff/phi for items in the batch
            out_axes=(0, 0, 0)  # batched u_safe, constraint_active, delta_u
        )
    )
    return NCBF, NCBF_apply, batched_get_safe_action, safety_layer_function_for_batch

def get_ensemble_forward_pass(
    apply_fn,
    n_ensemble: int,
    output_distribution: str,
    min_log_std: float,
    max_log_std: float,
    residual_mc_samples: int,
):
    alpha = 0.4
    E = n_ensemble
    k = max(1, int(np.ceil((1.0 - alpha) * E)))
    softmin_beta = 10.0
    residual_mc_samples = max(2, int(residual_mc_samples))
    residual_eps_current = _RESIDUAL_MC_RNG.standard_normal(residual_mc_samples).astype(np.float32)
    residual_eps_next = _RESIDUAL_MC_RNG.standard_normal(residual_mc_samples).astype(np.float32)

    def component_softmin(preds):
        weights = jax.nn.softmax(-softmin_beta * preds, axis=-1)
        return jnp.sum(weights * preds, axis=-1)

    @jax.jit
    def ensemble_forward_pass(params_stack, input):
        """
        one step forward pass through an ensemble of networks, 1 input
        input is a tuple of (observation, and history latent)
        """
        # Vectorized apply over ensemble, then take mean
        predictions = jax.vmap(lambda p: apply_fn(p, input))(params_stack)
        mean_predictions, log_std_predictions = split_ncbf_output(
            predictions,
            output_distribution,
            min_log_std,
            max_log_std,
        )

        def aggregate_predictions(preds):
            # return jnp.mean(preds, axis=0)
            # def cvar_soft(losses, alpha=0.90):
            #     eta = jax.lax.stop_gradient(jnp.quantile(losses, alpha))
            #     tail = jnp.maximum(losses - eta, 0.0)
            #     return eta + jnp.mean(tail) / (1 - alpha)
            #
            def cvar_bottomk(preds):
                # losses: (E,) or (E, ...) , larger = worse
                tail = jnp.sort(preds, axis=0)[:k, ...]
                return jnp.mean(tail, axis=0)

            return cvar_bottomk(component_softmin(preds))

        std_predictions = jnp.where(
            output_distribution == "logistic_normal",
            jnp.exp(log_std_predictions),
            jnp.zeros_like(mean_predictions),
        )

        def make_prediction_samples():
            eps_shape = (residual_mc_samples,) + (1,) * mean_predictions.ndim
            eps_current = jnp.asarray(residual_eps_current, dtype=mean_predictions.dtype).reshape(eps_shape)
            logits = mean_predictions[None, ...] + std_predictions[None, ...] * eps_current
            return component_softmin(jax.nn.sigmoid(logits))

        prediction_samples = make_prediction_samples()
        prediction_mean = aggregate_predictions(jax.nn.sigmoid(mean_predictions))
        n_samples = jnp.asarray(prediction_samples.size, dtype=mean_predictions.dtype)
        prediction_sample_mean = jnp.mean(prediction_samples)
        prediction_std = jnp.sqrt(
            jnp.sum(jnp.square(prediction_samples - prediction_sample_mean)) /
            jnp.maximum(n_samples - 1.0, 1.0)
        )

        prediction_details = {
            "mean_logits": mean_predictions,
            "std_logits": std_predictions,
            "residual_eps_current": jnp.asarray(residual_eps_current, dtype=mean_predictions.dtype),
            "residual_eps_next": jnp.asarray(residual_eps_next, dtype=mean_predictions.dtype),
        }

        return prediction_mean, prediction_std, prediction_details
    return ensemble_forward_pass



class NCBF_FFNN(nn.Module):
    nr_hidden_units: int
    n_outputs: int
    output_distribution: str = "deterministic"

    @nn.compact
    def __call__(self, x):
        # Two hidden layers
        x = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        # One CBF output per configured safety target.
        output_dim = self.n_outputs * (2 if self.output_distribution == "logistic_normal" else 1)
        h1 = nn.Dense(output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)

        # clip the output to be in [0, 1]
        # h1 = nn.sigmoid(h1)
        # to get the gradient without the sigmoid, we do not use sigmoid here, add a sigmoid function when preedicting

        # h1 = jnp.squeeze(h1, -1)  # shape ()
        return h1


def make_get_safe_action(
    ncbf_apply: Callable[[dict, Array], Array],   # h_phi(obs)
    ncbf_obs_from_obs_idx: Array,
    use_safety_layer: bool,
    act_low: Array,
    act_high: Array,
    gamma_c: float,
    eta_cbf: float,      # \tilde alpha(s) = eta_cbf * s
    lambda_s: float,     # slack penalty
    action_clipping: bool
    ):
    """
    Returns a JIT-able safety layer:
        get_safe_action(action_raw, x_t, t, contact, safety_layer_curriculum_coeff, phi) -> (u_safe, info)

    Args:
      ncbf_apply: flax apply function h_phi(obs)
      system_forward_dynamics_function: x_next = f(x, u, t, contact) (your MJX WBD step)
      state_from_obs_id: indices to get dynamics state from observation [pos(3), quat(4), joint_pos, joint_vel, contact (4)]
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

    def component_softmin(preds: Array) -> Array:
        beta = 10.0
        weights = jax.nn.softmax(-beta * preds, axis=-1)
        return jnp.sum(weights * preds, axis=-1)

    def prediction_details_to_samples(details: Dict[str, Array], eps_name: str) -> Array:
        mean_logits = details["mean_logits"]
        std_logits = details["std_logits"]
        eps = details[eps_name]
        eps_shape = (eps.shape[0],) + (1,) * mean_logits.ndim
        eps = eps.reshape(eps_shape)
        logits = mean_logits[None, ...] + std_logits[None, ...] * eps
        return component_softmin(jax.nn.sigmoid(logits))

    def residual_mean_and_std(current_details: Dict[str, Array], next_details: Dict[str, Array]) -> Tuple[Array, Array]:
        h_current_samples = prediction_details_to_samples(current_details, "residual_eps_current")
        h_next_samples = prediction_details_to_samples(next_details, "residual_eps_next")
        residual_samples = h_next_samples - (1.0 - eta_cbf) * h_current_samples - gamma_c
        residual_mean = jnp.mean(residual_samples)
        n_samples = jnp.asarray(residual_samples.size, dtype=residual_samples.dtype)
        residual_std = jnp.sqrt(
            jnp.sum(jnp.square(residual_samples - residual_mean)) /
            jnp.maximum(n_samples - 1.0, 1.0)
        )
        return residual_mean, residual_std

    @jax.jit
    def a_and_c_from_linearization(obs_t: Array, u0: Array, obs_last, u_last, latent_z, safety_layer_curriculum_coeff, phi: dict):
        """
        Compute:
          a = d/du h_phi(x_{t+1}(u)) | u0
          c = linearized RHS so constraint is a^T u >= c
        Args:
            obs_t: current observation (full observation matrix, need to get state from it)
        """

        x_t = obs_t[ncbf_obs_from_obs_idx]
        x_last = obs_last[ncbf_obs_from_obs_idx]

        std_coeff = jnp.asarray(safety_layer_curriculum_coeff, dtype=u0.dtype)

        def robust_residual_of_u(u):
            _, _, current_details = ncbf_apply(phi, jnp.concatenate([x_last, u_last, latent_z], axis=-1))
            _, _, next_details = ncbf_apply(phi, jnp.concatenate([x_t, u, latent_z], axis=-1))
            residual_mean, residual_std = residual_mean_and_std(current_details, next_details)
            robust_residual = residual_mean - std_coeff * residual_std
            return robust_residual, (residual_mean, residual_std)

        (robust_residual, (residual_mean, residual_std)), a = jax.value_and_grad(robust_residual_of_u, has_aux=True)(u0)

        c_lin = jnp.dot(a, u0) - robust_residual
        return a, c_lin, residual_mean, robust_residual, residual_std

    @jax.jit
    def get_safe_action(
        action_raw: Array,
        obs_t: Array,
        last_action: Array,
        last_obs: Array,
        latent_z: Array,
        safety_layer_curriculum_coeff: Array,
        phis: dict
    ) -> Tuple[Array, Array, Array]:

        action_raw = jax.lax.cond(action_clipping, lambda x: jnp.clip(x, act_low, act_high), lambda x: x, action_raw)

        a, c, residual_mean, robust_residual, residual_std = a_and_c_from_linearization(
            obs_t,
            action_raw,
            last_obs,
            last_action,
            latent_z,
            safety_layer_curriculum_coeff,
            phis,
        )

        aTa = jnp.dot(a, a) + 1e-12
        aTu = jnp.dot(a, action_raw)

        # constraint violation amount
        delta = jnp.maximum(0.0, c - aTu)

        # closed-form QP solution with soft slack
        gain = delta / (aTa + (1.0 / lambda_s))
        u_safe = action_raw + gain * a

        # eps_star = delta / (1.0 + lambda_s * aTa)

        # info = {
        #     # "a": a,
        #     # "c": c,
        #     # "delta": delta,
        #     # "gain": gain,
        #     # "epsilon_star": eps_star,
        #     "constraint_active": (delta > 0.0),
        #     # "h_x": h_x,
        #     # "h_u0": h_u0,
        # }
        constraint_active = jnp.array(delta > 0.0)

        u_processed = jax.lax.cond(use_safety_layer, lambda _: u_safe, lambda _: action_raw, operand=None)

        u_processed = jax.lax.cond(
            action_clipping,
            lambda x: jnp.clip(x, act_low, act_high),
            lambda x: x,
            u_processed
        )
        delta_u = jnp.linalg.norm(u_processed - action_raw)

        return u_processed, constraint_active, delta_u

    return get_safe_action
