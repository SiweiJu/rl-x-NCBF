
from typing import Callable, Tuple, Dict

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from jax.scipy.special import logsumexp

Array = jnp.ndarray

_LOGISTIC_NORMAL_GH_NODES, _LOGISTIC_NORMAL_GH_WEIGHTS = np.polynomial.hermite.hermgauss(9)
_LOGISTIC_NORMAL_GH_PROB_WEIGHTS = _LOGISTIC_NORMAL_GH_WEIGHTS / np.sqrt(np.pi)


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

    logits = (
        mean_logits[..., None, :]
        + jnp.sqrt(jnp.asarray(2.0, dtype=mean_logits.dtype))
        * jnp.exp(log_std[..., None, :])
        * nodes
    )
    targets = targets[..., None, :]
    log_likelihood = jnp.where(
        targets > 0.5,
        -jax.nn.softplus(-logits),
        -jax.nn.softplus(logits),
    )
    return -logsumexp(log_weights + log_likelihood, axis=-2)


def logistic_normal_safe_probability(mean_logits: Array, log_std: Array) -> Array:
    """E[sigmoid(z)] for z ~ N(mean_logits, exp(log_std)^2), using Gauss-Hermite."""
    nodes = jnp.asarray(_LOGISTIC_NORMAL_GH_NODES, dtype=mean_logits.dtype)
    weights = jnp.asarray(_LOGISTIC_NORMAL_GH_PROB_WEIGHTS, dtype=mean_logits.dtype)
    nodes = _reshape_quadrature_values(nodes, mean_logits)
    weights = _reshape_quadrature_values(weights, mean_logits)

    logits = (
        mean_logits[..., None, :]
        + jnp.sqrt(jnp.asarray(2.0, dtype=mean_logits.dtype))
        * jnp.exp(log_std[..., None, :])
        * nodes
    )
    return jnp.sum(weights * jax.nn.sigmoid(logits), axis=-2)


def get_ncbf(config, env):
    n_ncbf_ensemble = config.algorithm.ncbf.n_ensemble
    n_ncbf_output = len(env.ncbf_target_indices)
    use_safety_layer = config.algorithm.ncbf.use_safety_layer
    gamma_c = config.algorithm.ncbf.gamma_c
    ncbf_clipping = config.algorithm.ncbf.action_clipping
    eta_cbf = config.algorithm.ncbf.eta_cbf
    lambda_s = config.algorithm.ncbf.lambda_slack
    max_delta_u = config.algorithm.ncbf.max_delta_u
    output_distribution = getattr(config.algorithm.ncbf, "output_distribution", "deterministic")
    min_log_std = getattr(config.algorithm.ncbf, "min_log_std", -5.0)
    max_log_std = getattr(config.algorithm.ncbf, "max_log_std", 2.0)
    residual_mc_samples = getattr(config.algorithm.ncbf, "residual_mc_samples", 16)

    act_low = jnp.array(env.single_action_space.low)
    act_high = jnp.array(env.single_action_space.high)

    def dummy_safety_diagnostics(action):
        zero = jnp.asarray(0.0, dtype=action.dtype)
        one = jnp.asarray(1.0, dtype=action.dtype)
        return {
            "residual_mean": zero,
            "residual_std": zero,
            "robust_residual": zero,
            "constraint_delta": zero,
            "constraint_grad_norm": zero,
            "qp_gain": zero,
            "correction_norm": zero,
            "correction_scale": one,
            "correction_clipped": zero,
            "raw_action_norm": jnp.linalg.norm(action),
            "processed_action_norm": jnp.linalg.norm(action),
        }

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

    if config.algorithm.ncbf.use_safety_layer:
        safety_layer_function = make_get_safe_action(
            ncbf_apply=NCBF_apply,
            ncbf_obs_from_obs_idx=env.ncbf_observation_indices,
            use_safety_layer=use_safety_layer,
            act_low=act_low,
            act_high=act_high,
            gamma_c=gamma_c,
            eta_cbf=eta_cbf,
            lambda_s=lambda_s,
            max_delta_u=max_delta_u,
            action_clipping=ncbf_clipping,
        )

    # dummy
    if ncbf_clipping:
        dummy_safety_layer_function = lambda action_raw, obs_t, last_action, last_obs, latent_z, safety_layer_curriculum_coeff, phi: (
            jnp.clip(action_raw, act_low, act_high),
            jnp.array(False),
            jnp.array(0.0),
            dummy_safety_diagnostics(jnp.clip(action_raw, act_low, act_high)),
        )
    else:
        dummy_safety_layer_function = lambda action_raw, obs_t, last_action, last_obs, latent_z, safety_layer_curriculum_coeff, phi: (
            action_raw,
            jnp.array(False),
            jnp.array(0.0),
            dummy_safety_diagnostics(action_raw),
        )

    if config.algorithm.ncbf.use_safety_layer:
        safety_layer_function_for_batch = safety_layer_function
    else:
        safety_layer_function_for_batch = dummy_safety_layer_function

    # Vectorize over env axis 0.
    batched_get_safe_action = jax.jit(
        jax.vmap(
            safety_layer_function_for_batch,
            in_axes=(0, 0, 0, 0, 0, None, None),
            out_axes=(0, 0, 0, 0),
        )
    )
    return NCBF, NCBF_apply, batched_get_safe_action, safety_layer_function_for_batch


def get_ensemble_forward_pass(
    apply_fn,
    n_ensemble,
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

    def component_softmin(preds: Array) -> Array:
        """
        Softargmin-weighted average over safety dimensions.

        This preserves the aggregation used in your current implementation:
            sum_j softmax(-beta * h_j) h_j.
        If the paper defines log-sum-exp softmin instead, replace this function
        in both get_ensemble_forward_pass and make_get_safe_action.
        """
        weights = jax.nn.softmax(-softmin_beta * preds, axis=-1)
        return jnp.sum(weights * preds, axis=-1)

    @jax.jit
    def ensemble_forward_pass(params_stack, input):
        """
        One-step forward pass through an ensemble of NCBF networks.

        Returns:
            aggregated_prediction: scalar robust aggregate over components and ensemble
            prediction_std: scalar std of the aggregated safety value
            prediction_details: logits/stds and fixed MC noise for residual-level uncertainty
        """
        predictions = jax.vmap(lambda p: apply_fn(p, input))(params_stack)
        mean_logits, log_std_predictions = split_ncbf_output(
            predictions,
            output_distribution,
            min_log_std,
            max_log_std,
        )

        std_logits = jnp.where(
            output_distribution == "logistic_normal",
            jnp.exp(log_std_predictions),
            jnp.zeros_like(mean_logits),
        )

        def aggregate_predictions(preds: Array) -> Array:
            def cvar_bottomk(values: Array) -> Array:
                tail = jnp.sort(values, axis=0)[:k, ...]
                return jnp.mean(tail, axis=0)

            preds_softmin = component_softmin(preds)  # [E]
            return cvar_bottomk(preds_softmin)        # scalar

        if output_distribution == "logistic_normal":
            safe_predictions = logistic_normal_safe_probability(mean_logits, log_std_predictions)
        else:
            safe_predictions = jax.nn.sigmoid(mean_logits)

        # Fixed common-random-number MC samples. Shape is [M, E, d], not [M, 1, 1],
        # so each safety dimension gets its own diagonal logistic-normal noise.
        eps_shape = (residual_mc_samples,) + mean_logits.shape
        eps_current = jax.random.normal(
            jax.random.PRNGKey(0),
            shape=eps_shape,
            dtype=mean_logits.dtype,
        )
        eps_next = jax.random.normal(
            jax.random.PRNGKey(1),
            shape=eps_shape,
            dtype=mean_logits.dtype,
        )

        prediction_samples = component_softmin(
            jax.nn.sigmoid(mean_logits[None, ...] + std_logits[None, ...] * eps_current)
        )  # [M, E]
        prediction_sample_mean = jnp.mean(prediction_samples)
        n_samples = jnp.asarray(prediction_samples.size, dtype=mean_logits.dtype)
        prediction_var = (
            jnp.sum(jnp.square(prediction_samples - prediction_sample_mean))
            / jnp.maximum(n_samples - 1.0, 1.0)
        )
        prediction_std = jnp.sqrt(jnp.maximum(prediction_var, 0.0) + 1e-8)

        prediction_details = {
            "mean_logits": mean_logits,     # [E, d]
            "std_logits": std_logits,       # [E, d], already exp(log_std)
            "residual_eps_current": eps_current,  # [M, E, d]
            "residual_eps_next": eps_next,        # [M, E, d]
        }

        return aggregate_predictions(safe_predictions), prediction_std, prediction_details

    return ensemble_forward_pass


class NCBF_FFNN(nn.Module):
    nr_hidden_units: int
    n_outputs: int
    output_distribution: str = "deterministic"
    softmin_beta: int = 10

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.nr_hidden_units,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.tanh(x)
        x = nn.Dense(
            self.nr_hidden_units,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.tanh(x)

        # Output logits. For logistic_normal, the output is [mean_logits, raw_log_std].
        output_dim = self.n_outputs * (2 if self.output_distribution == "logistic_normal" else 1)
        h = nn.Dense(
            output_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        return h


def make_get_safe_action(
    ncbf_apply: Callable[[dict, Array], Array],
    ncbf_obs_from_obs_idx: Array,
    use_safety_layer: bool,
    act_low: Array,
    act_high: Array,
    gamma_c: float,
    eta_cbf: float,
    lambda_s: float,
    max_delta_u: float,
    action_clipping: bool,
):
    """
    Returns a JIT-able safety layer.

    The safety layer uses residual-level MC uncertainty:
        R = smin(h_next) - (1 - eta_cbf) smin(h_current) - gamma_c.

    The curriculum coefficient passed into get_safe_action is assumed to already
    be beta_k, not a probability. Your annealing from -2 to 1 can therefore be
    passed directly.
    """

    def component_softmin(preds: Array) -> Array:
        beta = 10.0
        weights = jax.nn.softmax(-beta * preds, axis=-1)
        return jnp.sum(weights * preds, axis=-1)

    def prediction_details_to_samples(details: Dict[str, Array], eps_name: str) -> Array:
        """
        Converts ensemble logistic-normal details into sampled scalar safety values.

        mean_logits: [E, d]
        std_logits:  [E, d]
        eps:         [M, E, d]
        returns:     [M, E], after sigmoid and component softmin over d
        """
        mean_logits = details["mean_logits"]
        std_logits = details["std_logits"]
        eps = details[eps_name]

        logits = mean_logits[None, ...] + std_logits[None, ...] * eps
        h_samples = jax.nn.sigmoid(logits)
        return component_softmin(h_samples)

    def residual_mean_and_std(
        current_details: Dict[str, Array],
        next_details: Dict[str, Array],
    ) -> Tuple[Array, Array]:
        """
        Estimate mean/std of the scalar CBF residual using MC samples.

        The std is computed after sampling the full safety vector, applying the
        component softmin, and forming the scalar CBF residual. This preserves
        the coupling among safety dimensions.
        """
        h_current_samples = prediction_details_to_samples(
            current_details,
            "residual_eps_current",
        )  # [M, E]
        h_next_samples = prediction_details_to_samples(
            next_details,
            "residual_eps_next",
        )  # [M, E]

        residual_samples = (
            h_next_samples
            - (1.0 - eta_cbf) * h_current_samples
            - gamma_c
        )  # [M, E]

        residual_mean = jnp.mean(residual_samples)
        n_samples = jnp.asarray(residual_samples.size, dtype=residual_samples.dtype)
        residual_var = (
            jnp.sum(jnp.square(residual_samples - residual_mean))
            / jnp.maximum(n_samples - 1.0, 1.0)
        )
        residual_std = jnp.sqrt(jnp.maximum(residual_var, 0.0) + 1e-8)
        return residual_mean, residual_std

    @jax.jit
    def a_and_c_from_linearization(
        obs_t: Array,
        u0: Array,
        obs_last: Array,
        u_last: Array,
        latent_z: Array,
        safety_layer_curriculum_coeff: Array,
        phi: dict,
    ):
        """
        Compute the affine chance-constrained CBF approximation
            a^T u >= c.

        We linearize only the residual mean and freeze the residual std at u0:
            R_mean(u) ≈ R_mean(u0) + a^T (u - u0),
            R_robust(u) = R_mean(u) - beta * stop_gradient(R_std(u0)).

        Therefore,
            c = a^T u0 - R_mean(u0) + beta * R_std(u0).
        """
        x_t = obs_t[ncbf_obs_from_obs_idx]
        x_last = obs_last[ncbf_obs_from_obs_idx]

        beta_coeff = jnp.asarray(safety_layer_curriculum_coeff, dtype=u0.dtype)

        # Current term does not depend on candidate action u, so keep it outside
        # the differentiated function.
        _, _, current_details = ncbf_apply(
            phi,
            jnp.concatenate([x_last, u_last, latent_z], axis=-1),
        )

        def residual_mean_of_u(u: Array):
            _, _, next_details = ncbf_apply(
                phi,
                jnp.concatenate([x_t, u, latent_z], axis=-1),
            )
            residual_mean, residual_std = residual_mean_and_std(
                current_details,
                next_details,
            )
            return residual_mean, residual_std

        (residual_mean, residual_std), a = jax.value_and_grad(
            residual_mean_of_u,
            has_aux=True,
        )(u0)

        # Freeze the std in the affine QP; beta is your annealed coefficient, e.g. -2 -> 1.
        residual_std_sg = jax.lax.stop_gradient(residual_std)
        robust_residual = residual_mean - beta_coeff * residual_std_sg
        c_lin = jnp.dot(a, u0) - residual_mean + beta_coeff * residual_std_sg

        return a, c_lin, residual_mean, robust_residual, residual_std_sg

    @jax.jit
    def get_safe_action(
        action_raw: Array,
        obs_t: Array,
        last_action: Array,
        last_obs: Array,
        latent_z: Array,
        safety_layer_curriculum_coeff: Array,
        phis: dict,
    ) -> Tuple[Array, Array, Array]:
        action_raw = jax.lax.cond(
            action_clipping,
            lambda x: jnp.clip(x, act_low, act_high),
            lambda x: x,
            action_raw,
        )

        a, c, residual_mean, robust_residual, residual_std = a_and_c_from_linearization(
            obs_t,
            action_raw,
            last_obs,
            last_action,
            latent_z,
            safety_layer_curriculum_coeff,
            phis,
        )

        linearization_is_finite = jnp.all(jnp.isfinite(a)) & jnp.isfinite(c)
        a = jnp.where(jnp.isfinite(a), a, jnp.zeros_like(a))
        c = jnp.where(jnp.isfinite(c), c, jnp.asarray(0.0, dtype=action_raw.dtype))
        residual_mean = jnp.nan_to_num(residual_mean, nan=0.0, posinf=0.0, neginf=0.0)
        robust_residual = jnp.nan_to_num(robust_residual, nan=0.0, posinf=0.0, neginf=0.0)
        residual_std = jnp.nan_to_num(residual_std, nan=0.0, posinf=0.0, neginf=0.0)

        aTa = jnp.dot(a, a) + 1e-12
        aTu = jnp.dot(a, action_raw)

        # Constraint violation amount for a^T u >= c.
        delta = jnp.where(linearization_is_finite, jnp.maximum(0.0, c - aTu), 0.0)

        # Closed-form one-constraint QP solution with soft slack:
        #   min ||u - u0||^2 + lambda_s * xi^2
        #   s.t. a^T u + xi >= c, xi >= 0.
        gain = delta / (aTa + (1.0 / lambda_s))
        correction = gain * a

        # Optional correction clipping. Note: if max_delta_u clips the correction,
        # the final action may no longer exactly satisfy the affine constraint.
        correction_norm = jnp.linalg.norm(correction) + 1e-8
        max_delta = jnp.asarray(max_delta_u, dtype=action_raw.dtype)
        correction_scale = jnp.where(
            max_delta > 0.0,
            jnp.minimum(1.0, max_delta / correction_norm),
            1.0,
        )
        u_safe = action_raw + correction_scale * correction

        constraint_active = jnp.array(delta > 0.0)

        u_processed = jax.lax.cond(
            use_safety_layer,
            lambda _: u_safe,
            lambda _: action_raw,
            operand=None,
        )

        # Optional action clipping. Note: clipping after the QP can also break exact
        # satisfaction of the affine constraint.
        u_processed = jax.lax.cond(
            action_clipping,
            lambda x: jnp.clip(x, act_low, act_high),
            lambda x: x,
            u_processed,
        )
        u_processed_is_finite = jnp.all(jnp.isfinite(u_processed))
        u_processed = jnp.where(u_processed_is_finite, u_processed, action_raw)
        constraint_active = constraint_active & linearization_is_finite & u_processed_is_finite
        delta_u = jnp.linalg.norm(u_processed - action_raw)

        diagnostics = {
            "linearization_is_finite": linearization_is_finite.astype(action_raw.dtype),
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "robust_residual": robust_residual,
            "constraint_delta": delta,
            "constraint_grad_norm": jnp.sqrt(aTa),
            "qp_gain": gain,
            "correction_norm": correction_norm,
            "correction_scale": correction_scale,
            "correction_clipped": ((max_delta > 0.0) & (correction_scale < 0.999)).astype(action_raw.dtype),
            "raw_action_norm": jnp.linalg.norm(action_raw),
            "processed_action_norm": jnp.linalg.norm(u_processed),
        }

        return u_processed, constraint_active, delta_u, diagnostics

    return get_safe_action
