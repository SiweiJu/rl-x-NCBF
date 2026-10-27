import os
import shutil
import json
from copy import deepcopy
import logging
import time
from functools import partial

import tree
import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as nn
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.ncbf_ppo.flax_full_jit.general_properties import GeneralProperties
from rl_x.algorithms.ncbf_ppo.flax_full_jit.policy import get_policy
from rl_x.algorithms.ncbf_ppo.flax_full_jit.critic import get_critic
from rl_x.algorithms.ncbf_ppo.flax_full_jit.ncbf_targets import (
    LABEL_SOURCES,
    TEMPORAL_MODES,
    TEMPORAL_PAIRS,
    TERMINAL_CRITIC_INPUTS,
    critic_training_inputs,
    episode_first_step,
    legacy_raw_vs_executed_temporal_loss,
    receding_min_target,
    successor_pairs,
    termination_window_labels,
    temporal_consistency_loss,
)
from rl_x.algorithms.ncbf_ppo.flax_full_jit.ncbf import (
    get_ncbf,
    logistic_normal_binary_cross_entropy,
    logistic_normal_safe_probability,
    split_ncbf_output,
)
from rl_x.algorithms.ncbf_ppo.flax_full_jit.history_encoder import get_history_encoder
from rl_x.algorithms.ncbf_ppo.flax_full_jit.paper_compat import (
    ANTICIPATION_STYLES,
    anticipation_squared_error,
    gae_advantages,
    maybe_clip_logstd,
)
from rl_x.algorithms.ncbf_ppo.flax_full_jit.decoder import get_decoder
from rl_x.algorithms.ncbf_ppo.flax_full_jit.d_training import DTrainer, METHODS as D_METHODS


rlx_logger = logging.getLogger("rl_x")


def _replace_optimizer_learning_rate(train_state: TrainState, learning_rate):
    inject_state = train_state.opt_state[1]
    hyperparams = {**inject_state.hyperparams, "learning_rate": learning_rate}
    new_inject_state = inject_state._replace(hyperparams=hyperparams)
    new_opt_state = tuple(
        train_state.opt_state[i] if i != 1 else new_inject_state
        for i in range(len(train_state.opt_state))
    )
    return train_state.replace(opt_state=new_opt_state)


def _adapt_learning_rate_from_kl(
    current_lr,
    approx_kl,
    target_kl,
    kl_margin,
    kl_lr_scale,
    kl_lr_gain,
    lr_min,
    lr_max,
):
    current_lr = jnp.asarray(current_lr)
    eps = jnp.asarray(1e-8, dtype=current_lr.dtype)
    approx_kl = jnp.maximum(jnp.asarray(approx_kl, dtype=current_lr.dtype), eps)
    target_kl = jnp.maximum(jnp.asarray(target_kl, dtype=current_lr.dtype), eps)
    kl_margin = jnp.maximum(jnp.asarray(kl_margin, dtype=current_lr.dtype), 1.0)
    kl_lr_scale = jnp.maximum(jnp.asarray(kl_lr_scale, dtype=current_lr.dtype), 1.0)
    kl_lr_gain = jnp.maximum(jnp.asarray(kl_lr_gain, dtype=current_lr.dtype), 0.0)

    log_kl_error = jnp.log(target_kl / approx_kl)
    deadband = jnp.log(kl_margin)
    excess_error = jnp.maximum(jnp.abs(log_kl_error) - deadband, 0.0)
    max_log_step = jnp.log(kl_lr_scale)
    log_lr_step = jnp.sign(log_kl_error) * jnp.minimum(kl_lr_gain * excess_error, max_log_step)

    next_lr = current_lr * jnp.exp(log_lr_step)
    return jnp.clip(next_lr, lr_min, lr_max)


def _nan_to_none(value):
    """Replace NaN floats (empty groups) by None so the summary is valid JSON."""
    if isinstance(value, dict):
        return {key: _nan_to_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(item) for item in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _binary_auroc(score, positive):
    """AUROC of score for the positive class (Mann-Whitney with average ranks); nan if one class is missing."""
    positive = positive.astype(bool)
    nr_positive, nr_negative = int(positive.sum()), int((~positive).sum())
    if nr_positive == 0 or nr_negative == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    sorted_score = score[order]
    # average ranks over ties
    unique, start_index, counts = np.unique(sorted_score, return_index=True, return_counts=True)
    for first, count in zip(start_index, counts):
        if count > 1:
            ranks[order[first:first + count]] = first + (count + 1) / 2.0
    return float((ranks[positive].sum() - nr_positive * (nr_positive + 1) / 2.0) / (nr_positive * nr_negative))


def summarize_counterfactual(records, head_names, nr_calibration_bins=5):
    """Turn the per-decision records of PPO.counterfactual_check into rates.

    records: active / delta_u / p_* with shape [D, N] (or [D, N, K] per head), and *_raw / *_safe outcomes with
    shape [D, R, N] (or [D, R, N, K]); D decision points, R repeats, N envs.
    """
    nr_repeats = records["failed_raw"].shape[1]

    def per_outcome(name):  # [D, N] -> [D, R, N]
        return np.repeat(records[name][:, None], nr_repeats, axis=1)

    active = per_outcome("active").astype(bool)
    valid = ~(records["censored_raw"] | records["censored_safe"])
    fail_raw, fail_safe = records["failed_raw"].astype(bool), records["failed_safe"].astype(bool)

    def rate(numerator, denominator):
        count = int(denominator.sum())
        return float((numerator & denominator).sum() / count) if count > 0 else float("nan")

    on, off = valid & active, valid & ~active
    summary = {
        "nr_outcomes": int(valid.sum()),
        "censored_fraction": float(1.0 - valid.mean()),
        "intervention_rate": float(active[valid].mean()) if valid.any() else float("nan"),
        "mean_delta_u_when_active": float(per_outcome("delta_u")[on].mean()) if on.any() else float("nan"),
        # Interventions: was the raw action going to fail, and did the shielded action avoid it?
        "active": {
            "count": int(on.sum()),
            "fail_rate_raw": rate(fail_raw, on),
            "fail_rate_shielded": rate(fail_safe, on),
            "necessary_and_effective": rate(fail_raw & ~fail_safe, on),  # raw fails, shielded does not
            "unnecessary": rate(~fail_raw & ~fail_safe, on),  # neither fails
            "ineffective": rate(fail_raw & fail_safe, on),  # both fail
            "harmful": rate(~fail_raw & fail_safe, on),  # only the shielded action fails
        },
        # No intervention: failures the shield did not act on (both branches take the same first action).
        "inactive": {
            "count": int(off.sum()),
            "fail_rate": rate(fail_raw, off),
            "branch_mismatch_rate": rate(fail_raw != fail_safe, off),  # should be 0: the branches are identical
        },
    }
    raw_failures = int((fail_raw & valid).sum())
    flagged = int((fail_raw & on).sum())
    summary["recall"] = float(flagged / raw_failures) if raw_failures > 0 else float("nan")  # upcoming raw failures the shield acted on
    summary["precision"] = summary["active"]["fail_rate_raw"]  # interventions where the raw action would have failed

    # Critic calibration: predicted P(safe) for the first action against the realized outcome.
    calibration = {}
    for branch, predicted, failed in (("raw", per_outcome("p_raw"), fail_raw), ("shielded", per_outcome("p_safe"), fail_safe)):
        p, safe = predicted[valid], (~failed)[valid].astype(np.float64)
        bins = []
        edges = np.linspace(0.0, 1.0, nr_calibration_bins + 1)
        for low, high in zip(edges[:-1], edges[1:]):
            in_bin = (p >= low) & ((p < high) if high < 1.0 else (p <= high))
            bins.append({"range": [float(low), float(high)], "count": int(in_bin.sum()),
                         "mean_predicted": float(p[in_bin].mean()) if in_bin.any() else float("nan"),
                         "realized_safe_rate": float(safe[in_bin].mean()) if in_bin.any() else float("nan")})
        calibration[branch] = {
            "auroc_failure": _binary_auroc(1.0 - p, 1.0 - safe),
            "brier": float(np.mean(np.square(p - safe))) if len(p) else float("nan"),
            "mean_predicted_safe": float(p.mean()) if len(p) else float("nan"),
            "realized_safe_rate": float(safe.mean()) if len(p) else float("nan"),
            "bins": bins,
        }
    summary["calibration"] = calibration

    # Per safety head: realized failure rates of the two branches when the shield is active.
    heads_raw, heads_safe = records["failed_heads_raw"].astype(bool), records["failed_heads_safe"].astype(bool)
    summary["heads"] = {
        name: {"fail_rate_raw_when_active": rate(heads_raw[..., k], on), "fail_rate_shielded_when_active": rate(heads_safe[..., k], on),
               "fail_rate_when_inactive": rate(heads_raw[..., k], off)}
        for k, name in enumerate(head_names)
    }
    return summary


def checkpoint_code_family(loaded_algorithm_config):
    """Which code wrote a checkpoint, from its saved algorithm config:
    "paper"     - ncbf-q / plots (cacffe7): has ncbf.pretrain and ncbf_buffer.pos_buffer_size;
    "multistep" - multistep before the 2026-10-14 safety-critic fixes: no ncbf.independent_ensemble;
    "current"   - written after the fixes."""
    ncbf = loaded_algorithm_config.get("ncbf", {}) or {}
    buffers = loaded_algorithm_config.get("ncbf_buffer", {}) or {}
    if "pretrain" in ncbf or "pos_buffer_size" in buffers:
        return "paper"
    if "independent_ensemble" not in ncbf:
        return "multistep"
    return "current"


def legacy_ncbf_defaults(family):
    """Values for ncbf keys that a checkpoint's config lacks, so it runs with the behavior of the code that wrote it.
    The paper code already trained independent members, paired (s_t, a_t) and used the eta-scaled gamma_c; only the
    pre-fix multistep code had the B1/B2/B3 regressions. Both used h(reset input) on the first step."""
    if family == "paper":
        return {"independent_ensemble": True, "critic_action_index": "current", "gamma_c_scaled_by_eta": True,
                "first_step_prev_value": "input", "label_source": "termination_window"}
    if family == "multistep":
        return {"independent_ensemble": False, "critic_action_index": "previous", "gamma_c_scaled_by_eta": False,
                "first_step_prev_value": "input"}
    return {"first_step_prev_value": "input"}  # written after B1-B3 but before the F1 fix


class PPO:
    SUPPORTS_RESUME = True
    def __init__(self, config, env, run_path, writer):
        self.config = config
        self.env = env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        # Requeue-safe resume (runner.resume): see _save_resume_checkpoint / _restore_resume_checkpoint.
        self.resume = bool(config.runner.get("resume", False))
        self.resume_path = os.path.join(run_path, "resume")
        self.save_interval_iterations = int(config.runner.get("save_interval_iterations", 0))
        self.save_interval_minutes = float(config.runner.get("save_interval_minutes", 20.0))
        self.resume_start_update = 0
        self.resume_env_curriculum = None
        self.resume_last_save_update = 0
        self.resume_last_save_time = time.time()
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.nr_parallel_seeds = config.algorithm.nr_parallel_seeds
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.render = config.environment.render
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.adaptive_lr = bool(config.algorithm.get("adaptive_lr", False))
        self.adaptive_lr_target_kl = float(config.algorithm.get("adaptive_lr_target_kl", 0.0))
        if self.adaptive_lr_target_kl <= 0.0:
            self.adaptive_lr_target_kl = float(config.algorithm.target_kl)
        self.adaptive_lr_kl_margin = float(config.algorithm.get("kl_margin", 2.0))
        self.adaptive_lr_scale = float(config.algorithm.get("kl_lr_scale", 1.25))
        self.adaptive_lr_gain = float(config.algorithm.get("kl_lr_gain", 0.25))
        self.adaptive_lr_min = float(config.algorithm.get("lr_min", 1e-6))
        self.adaptive_lr_max = float(config.algorithm.get("lr_max", 0.0))
        if self.adaptive_lr_max <= 0.0:
            self.adaptive_lr_max = self.learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.clip_range = config.algorithm.clip_range
        self.target_kl = config.algorithm.target_kl
        self.entropy_coef = config.algorithm.entropy_coef
        self.critic_coef = config.algorithm.critic_coef
        self.anticipation_coef = config.algorithm.ncbf.policy_loss_coef
        self.aux_prediction_coef = config.algorithm.next_step_predictor.aux_loss_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = config.algorithm.total_timesteps // self.batch_size
        self.nr_minibatches = self.batch_size // self.minibatch_size

        self.next_step_predictor_lr = config.algorithm.next_step_predictor.lr
        self.next_step_predictor_output_indices = env.next_state_indices
        single_env = getattr(env, "envs", [env])[0]
        self.ball_plate_obs_indices = jnp.asarray(
            np.asarray(getattr(single_env, "ball_plate_obs_idx", np.array([], dtype=int)), dtype=int),
            dtype=int,
        )
        ball_plate_dropped_idx = np.asarray(
            getattr(single_env, "ball_plate_dropped_obs_idx", np.array([], dtype=int)),
            dtype=int,
        )
        self.ball_plate_dropped_obs_indices = jnp.asarray(ball_plate_dropped_idx, dtype=int)
        self.ball_plate_dropped_obs_index = int(ball_plate_dropped_idx[0]) if ball_plate_dropped_idx.size > 0 else 0
        decoder_ball_plate_indices = np.nonzero(np.isin(
            np.asarray(self.next_step_predictor_output_indices, dtype=int),
            np.asarray(self.ball_plate_obs_indices, dtype=int),
        ))[0]
        self.decoder_output_ball_plate_indices = jnp.asarray(decoder_ball_plate_indices, dtype=int)
        self.has_decoder_ball_plate_output = decoder_ball_plate_indices.size > 0
        self.next_step_predictor_nr_minibatches = config.algorithm.next_step_predictor.nr_minibatches
        self.next_step_predictor_minibatch_size = config.algorithm.minibatch_size
        self.use_decoder_output_for_policy = bool(getattr(config.algorithm, "use_decoder_output_for_policy", False))

        self.ncbf_n_ensemble = config.algorithm.ncbf.n_ensemble
        self.ncbf_H = config.algorithm.ncbf.H
        self.ncbf_gamma_c = config.algorithm.ncbf.gamma_c
        self.ncbf_w_clf = config.algorithm.ncbf.w_clf
        self.ncbf_w_cbf = config.algorithm.ncbf.w_cbf
        self.ncbf_w_lip = config.algorithm.ncbf.w_lip
        self.ncbf_w_wd = config.algorithm.ncbf.w_wd
        self.ncbf_eta_cbf = config.algorithm.ncbf.eta_cbf
        self.ncbf_L_target = config.algorithm.ncbf.L_max
        self.ncbf_loss_coef = config.algorithm.ncbf.loss_coef
        self.ncbf_stop_encoder_gradient = config.algorithm.ncbf.stop_encoder_gradient
        self.ncbf_use_safety_layer = bool(config.algorithm.ncbf.use_safety_layer)
        self.ncbf_output_distribution = getattr(config.algorithm.ncbf, "output_distribution", "deterministic")
        self.ncbf_min_log_std = getattr(config.algorithm.ncbf, "min_log_std")
        self.ncbf_max_log_std = getattr(config.algorithm.ncbf, "max_log_std")
        self.ncbf_safety_layer_std_coeff_start = getattr(config.algorithm.ncbf, "safety_layer_std_coeff_start")
        self.ncbf_safety_layer_std_coeff_final = getattr(config.algorithm.ncbf, "safety_layer_std_coeff_final")
        self.ncbf_lambda_slack = getattr(config.algorithm.ncbf, "lambda_slack", 1000.0)
        self.ncbf_max_delta_u = getattr(config.algorithm.ncbf, "max_delta_u", 0.0)
        self.ncbf_safety_layer_projection = getattr(config.algorithm.ncbf, "safety_layer_projection", "soft_slack")
        self.ncbf_safety_layer_min_grad_norm = getattr(config.algorithm.ncbf, "safety_layer_min_grad_norm", 0.0)
        self.ncbf_post_check_actual_residual = bool(getattr(config.algorithm.ncbf, "post_check_actual_residual", False))
        self.ncbf_minibatch_size = config.algorithm.minibatch_size
        self.ncbf_coef_decay_lambda = config.algorithm.ncbf.coef_decay_lambda
        self.ncbf_independent_ensemble = bool(config.algorithm.ncbf.get("independent_ensemble", True))
        self.ncbf_bootstrap_prob = float(config.algorithm.ncbf.get("ensemble_bootstrap_prob", 1.0))
        self.ncbf_critic_action_index = config.algorithm.ncbf.get("critic_action_index", "current")
        if self.ncbf_critic_action_index not in ("current", "previous"):
            raise ValueError("algorithm.ncbf.critic_action_index must be 'current' or 'previous'")
        self.ncbf_label_discount = bool(config.algorithm.ncbf.get("label_discount", False))
        self.ncbf_w_temporal = float(config.algorithm.ncbf.get("w_temporal", 0.0))
        # F1: "pre_reset" trains the episode-ending step on its real (s_t, a_t); "zeros" = paper code (0, 0).
        self.ncbf_terminal_critic_input = config.algorithm.ncbf.get("terminal_critic_input", "pre_reset")
        if self.ncbf_terminal_critic_input not in TERMINAL_CRITIC_INPUTS:
            raise ValueError(f"algorithm.ncbf.terminal_critic_input must be one of {TERMINAL_CRITIC_INPUTS}")
        # F2: "adjacent" = paper eq. 4-6 on (s_t, a_t) -> (s_{t+1}, a_{t+1}); "legacy_raw_vs_executed" = paper code.
        self.ncbf_temporal_mode = config.algorithm.ncbf.get("temporal_mode", "adjacent")
        if self.ncbf_temporal_mode not in TEMPORAL_MODES:
            raise ValueError(f"algorithm.ncbf.temporal_mode must be one of {TEMPORAL_MODES}")
        self.ncbf_temporal_pairs = config.algorithm.ncbf.get("temporal_pairs", "safe_safe")
        if self.ncbf_temporal_pairs not in TEMPORAL_PAIRS:
            raise ValueError(f"algorithm.ncbf.temporal_pairs must be one of {TEMPORAL_PAIRS}")
        self.ncbf_lipschitz_state_only = bool(config.algorithm.ncbf.get("lipschitz_state_only", True))
        self.ncbf_gamma_c_scaled_by_eta = bool(config.algorithm.ncbf.get("gamma_c_scaled_by_eta", True))
        self.anticipation_through_shield = bool(config.algorithm.ncbf.get("anticipation_through_shield", False))
        # T06c switches (paper_compat.py): default = pncbf-v2 behavior.
        self.anticipation_style = config.algorithm.ncbf.get("anticipation_style", "mean_action")
        if self.anticipation_style not in ANTICIPATION_STYLES:
            raise ValueError(f"algorithm.ncbf.anticipation_style must be one of {ANTICIPATION_STYLES}")
        self.gae_stop_at_truncation = bool(config.algorithm.get("gae_stop_at_truncation", True))
        self.policy_logstd_clip = bool(config.algorithm.get("policy_logstd_clip", True))
        # Critic labels: "env_targets" = min of the env safety indicator dims over s_{t+1..t+H}, one head each;
        # "termination_window" = paper code (ncbf-q): one head, no termination at transitions t..t+H;
        # ncbf.safe_fall_H >= 0 switches the latter to the SafeFall rule (branch ncbf-q-safefall).
        self.ncbf_label_source = config.algorithm.ncbf.get("label_source", "env_targets")
        if self.ncbf_label_source not in LABEL_SOURCES:
            raise ValueError(f"algorithm.ncbf.label_source must be one of {LABEL_SOURCES}")
        self.ncbf_safe_fall_H = int(config.algorithm.ncbf.get("safe_fall_H", -1))
        if self.ncbf_safe_fall_H >= 0 and self.ncbf_label_source != "termination_window":
            raise ValueError("algorithm.ncbf.safe_fall_H needs algorithm.ncbf.label_source=termination_window")
        if self.ncbf_label_source == "termination_window":
            self.ncbf_target_names = ["termination"]
        else:
            self.ncbf_target_names = list(getattr(env, "ncbf_target_names", [f"target_{i}" for i in range(len(env.ncbf_target_indices))]))
        self.ncbf_nr_heads = len(self.ncbf_target_names)

        self.ncbf_neg_buffer_size = config.algorithm.ncbf_buffer.neg_buffer_size * self.nr_steps * self.nr_envs
        self.ncbf_neg_sampling_ratio = config.algorithm.ncbf_buffer.neg_sampling_ratio
        self.ncbf_replay_neg_minibatch_size = int(self.minibatch_size * self.ncbf_neg_sampling_ratio)

        self.ncbf_observation_indices = env.ncbf_observation_indices

        rlx_logger.info(f"INFO - NCBF neg buffer size:{self.ncbf_neg_buffer_size}")


        # assert ncbf nr_steps * nr_envs must be a multiple of ncbf batchsize
        if (self.nr_steps * self.nr_envs) % self.ncbf_minibatch_size != 0:
            raise ValueError("NCBF batch size must divide evenly into nr_steps * nr_envs.")

        if config.algorithm.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.batch_size * (self.total_timesteps // self.batch_size)
        self.nr_multi_learning_and_eval_save_iterations = int(self.total_timesteps // self.evaluation_and_save_frequency)
        self.nr_updates_per_multi_learning_iteration = int(self.evaluation_and_save_frequency // self.batch_size)
        self.os_shape = env.single_observation_space.shape
        self.as_shape = env.single_action_space.shape
        self.horizon = env.horizon
        counterfactual_config = config.algorithm.get("counterfactual", None) or {}
        self.counterfactual_enabled = bool(counterfactual_config.get("enabled", False))
        self.counterfactual_nr_decision_points = int(counterfactual_config.get("nr_decision_points", 8))
        self.counterfactual_interval = int(counterfactual_config.get("interval", 50))
        self.counterfactual_horizon = int(counterfactual_config.get("horizon", 0))
        self.counterfactual_continuation = counterfactual_config.get("continuation", "shielded")
        self.counterfactual_nr_repeats = int(counterfactual_config.get("nr_repeats", 1))
        if self.counterfactual_continuation not in ("shielded", "raw"):
            raise ValueError("algorithm.counterfactual.continuation must be 'shielded' or 'raw'")
        self.nr_history_steps = env.nr_history_steps

        if self.evaluation_and_save_frequency % self.batch_size != 0:
            self.evaluation_and_save_frequency = self.batch_size * (self.evaluation_and_save_frequency // self.batch_size)

        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet. This is mainly limited by not being able to log mutliple wandb runs at the same time.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, reset_key, ncbf_key, encoder_key, decoder_key = jax.random.split(self.key, 7)
        reset_key = jax.random.split(reset_key, 1)

        self.policy, self.get_processed_action = get_policy(self.config, self.env)
        self.critic = get_critic(self.config, self.env)
        self.encoder = get_history_encoder(self.config, self.env)
        self.decoder = get_decoder(self.config, self.env)

        self.ncbf, self.ncbf_forward, self.batched_ncbf_safety_layer, self.ncbf_safety_layer = get_ncbf(config, env)
        # self.ncbf_apply = self.ncbf[0].apply # TODO: fix this
        self.ncbf_apply = self.ncbf_forward

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        learning_rate = self.learning_rate if self.adaptive_lr else (linear_schedule if self.anneal_learning_rate else self.learning_rate)

        env_state = self.env.reset(reset_key, False)

        dummy_latent = jnp.zeros((1, self.encoder.hidden_size))
        dummy_decoder_output = None
        if self.use_decoder_output_for_policy:
            dummy_decoder_output = jnp.zeros((1, self.next_step_predictor_output_indices.shape[0]))

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, env_state.next_observation, dummy_latent, dummy_decoder_output),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, env_state.next_observation, dummy_latent, dummy_decoder_output),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        def encoder_linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.next_step_predictor_lr * fraction
        next_step_predictor_lr = encoder_linear_schedule

        self.encoder_state = TrainState.create(
            apply_fn=self.encoder.apply,
            params=self.encoder.init(encoder_key, env_state.history_stack),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=next_step_predictor_lr),
            )
        )

        self.decoder_state = TrainState.create(
            apply_fn=self.decoder.apply,
            params=self.decoder.init(decoder_key, jnp.zeros((1, self.encoder.hidden_size)), env_state.next_observation, jnp.zeros((1, self.as_shape[0]))),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.next_step_predictor_lr),
            )
        )

        # dummy_ncbf_input is observation and action
        dummy_action = jnp.zeros((env_state.next_observation.shape[0],) + self.as_shape)
        dummy_ncbf_input = jnp.concatenate((jnp.concatenate([env_state.next_observation[..., env.ncbf_observation_indices], dummy_action], axis=-1), dummy_latent), axis=-1)
        ncbf_keys = jax.random.split(ncbf_key, self.ncbf_n_ensemble)
        if self.ncbf_independent_ensemble:
            # One TrainState whose params carry a leading ensemble axis [E, ...]; each member has its own init.
            ncbf_params = jax.vmap(lambda k: self.ncbf[0].init(k, dummy_ncbf_input))(ncbf_keys)
        else:
            # Legacy: a single network that is repeated n_ensemble times at inference.
            ncbf_params = self.ncbf[0].init(ncbf_keys[0], dummy_ncbf_input)
        self.ncbf_state = TrainState.create(
            apply_fn=self.ncbf[0].apply,
            params=ncbf_params,
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=config.algorithm.ncbf.lr),
            )
        )

        # P2 / T17: method D (online world model + WM shield, d_training.py); "default" = the code path above unchanged.
        self.method = str(config.algorithm.get("method", "default"))
        if self.method not in D_METHODS:
            raise ValueError(f"algorithm.method must be one of {D_METHODS}, got {self.method}")
        self.d = DTrainer(self, config, env) if self.method == "d" else None

        if self.save_model:
            os.makedirs(self.save_path, exist_ok=self.resume)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    def _ncbf_params_stack(self, ncbf_params):
        if self.ncbf_independent_ensemble:
            return ncbf_params
        return jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[None, ...], self.ncbf_n_ensemble, axis=0),
            ncbf_params,
        )


    def _ncbf_apply_members(self, ncbf_params, inputs):
        """Raw critic outputs for every ensemble member: [E, B, out] (E = 1 in legacy mode)."""
        if self.ncbf_independent_ensemble:
            return jax.vmap(self.ncbf_state.apply_fn, in_axes=(0, None))(ncbf_params, inputs)
        return self.ncbf_state.apply_fn(ncbf_params, inputs)[None]


    def _ncbf_probabilities(self, raw_output):
        """Safe probability per head from raw critic output (any leading shape)."""
        logits, log_std = split_ncbf_output(raw_output, self.ncbf_output_distribution, self.ncbf_min_log_std, self.ncbf_max_log_std)
        if self.ncbf_output_distribution == "logistic_normal":
            return logistic_normal_safe_probability(logits, log_std)
        return nn.sigmoid(logits)


    def _ball_plate_post_drop_mask(self, observation):
        if self.ball_plate_dropped_obs_indices.size == 0:
            return jnp.zeros(observation.shape[:-1], dtype=bool)
        return observation[..., self.ball_plate_dropped_obs_index] > 0.5


    def _mask_policy_ball_plate_inputs(self, observation):
        if self.ball_plate_dropped_obs_indices.size == 0 or self.ball_plate_obs_indices.size == 0:
            return observation
        post_drop = self._ball_plate_post_drop_mask(observation)
        ball_plate_observation = observation[..., self.ball_plate_obs_indices]
        ball_plate_observation = jnp.where(
            post_drop[..., None],
            jnp.zeros_like(ball_plate_observation),
            ball_plate_observation,
        )
        return observation.at[..., self.ball_plate_obs_indices].set(ball_plate_observation)


    def _get_policy_decoder_output(self, decoder_params, history_latent, state, action, mask_observation=None):
        if not self.use_decoder_output_for_policy:
            return None

        decoder_output = self.decoder.apply(
            decoder_params,
            jax.lax.stop_gradient(history_latent),
            state,
            action,
        )
        if (
            self.ball_plate_dropped_obs_indices.size > 0 and
            self.has_decoder_ball_plate_output and
            mask_observation is not None
        ):
            post_drop = self._ball_plate_post_drop_mask(mask_observation)
            ball_plate_output = decoder_output[..., self.decoder_output_ball_plate_indices]
            ball_plate_output = jnp.where(
                post_drop[..., None],
                jnp.zeros_like(ball_plate_output),
                ball_plate_output,
            )
            decoder_output = decoder_output.at[..., self.decoder_output_ball_plate_indices].set(ball_plate_output)
        return jax.lax.stop_gradient(decoder_output)


    def _bypass_post_drop_safety_layer(self, action_raw, safe_action, constraint_active, delta_u, diagnostics, observation):
        if self.ball_plate_dropped_obs_indices.size == 0:
            return safe_action, constraint_active, delta_u, diagnostics

        post_drop = self._ball_plate_post_drop_mask(observation)
        safe_action = jnp.where(post_drop[..., None], action_raw, safe_action)
        constraint_active = jnp.where(post_drop, jnp.zeros_like(constraint_active, dtype=bool), constraint_active)
        delta_u = jnp.where(post_drop, jnp.zeros_like(delta_u), delta_u)

        raw_norm = jnp.linalg.norm(action_raw, axis=-1)
        neutral_diagnostics = {}
        for key, value in diagnostics.items():
            if key in ("raw_action_norm", "processed_action_norm"):
                neutral_value = raw_norm
            elif key in ("correction_scale", "linearization_is_finite", "post_constraint_satisfied", "actual_post_is_finite"):
                neutral_value = jnp.ones_like(value)
            else:
                neutral_value = jnp.zeros_like(value)
            neutral_diagnostics[key] = jnp.where(post_drop, neutral_value, value)
        return safe_action, constraint_active, delta_u, neutral_diagnostics


    def train(self):
        def jitable_train_function(key, parallel_seed_id):
            def repeat_ncbf_params_from_params(ncbf_params):
                # Returns the [E, ...] parameter stack the ensemble forward pass expects.
                return self._ncbf_params_stack(ncbf_params)

            def repeat_ncbf_params(ncbf_state):
                return repeat_ncbf_params_from_params(ncbf_state.params)

            # Single step rollout, shared by policy and NCBF training.
            def single_rollout(single_rollout_carry, _):
                policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, key, safety_layer_curriculum_coeff, d_roll = single_rollout_carry

                key, subkey = jax.random.split(key)
                observation = env_state.next_observation
                history_stack = env_state.history_stack
                last_action = env_state.last_action
                last_state = env_state.last_state
                latent_z = encoder_state.apply_fn(encoder_state.params, history_stack)
                observation_input = self._mask_policy_ball_plate_inputs(observation)
                policy_decoder_output = self._get_policy_decoder_output(decoder_state.params, latent_z, last_state, last_action, observation)
                action_mean, action_logstd = self.policy.apply(policy_state.params, observation_input, latent_z, policy_decoder_output)
                action_logstd = maybe_clip_logstd(action_logstd, self.policy_logstd_clip)
                action_std = jnp.exp(action_logstd)
                action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                log_prob = (-0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(
                    2.0 * jnp.pi) - action_logstd).sum(1)
                raw_processed_action = self.get_processed_action(action)

                params_stack = repeat_ncbf_params(ncbf_state)

                first_step = episode_first_step(env_state)
                processed_action, constraint_active, delta_u, safety_diagnostics = self.batched_ncbf_safety_layer(raw_processed_action, observation, last_action, last_state, latent_z,
                                                                                      safety_layer_curriculum_coeff,
                                                                                      params_stack, first_step)
                processed_action, constraint_active, delta_u, safety_diagnostics = self._bypass_post_drop_safety_layer(
                    raw_processed_action,
                    processed_action,
                    constraint_active,
                    delta_u,
                    safety_diagnostics,
                    observation,
                )
                # contraint_active = 0
                # delta_u = 0
                value = self.critic.apply(critic_state.params, observation_input, latent_z, policy_decoder_output).squeeze(-1)

                env_action = processed_action
                if self.d is not None:
                    # method D: the WM shield replaces the critic shield (use_safety_layer is False in D runs)
                    env_action, d_inputs, d_v_prev, d_v_exec, d_diag = self.d.shield_step(
                        d_roll, policy_state.params, env_state, env_action)
                    constraint_active = d_diag["active"] > 0.5
                    delta_u = d_diag["delta_u_norm"]
                env_state = self.env.step(env_state, env_action)
                done = env_state.terminated | env_state.truncated
                d_rec = {}
                if self.d is not None:
                    d_rec = self.d.record(d_inputs, env_action, d_v_prev, env_state, d_diag)
                    d_roll = {**d_roll, "v_prev": d_v_exec}
                transition = (observation, env_state.actual_next_observation, action, env_action, env_state.reward, value,
                              env_state.terminated, done, log_prob, env_state.info, constraint_active, delta_u, last_state, last_action, history_stack, safety_diagnostics, first_step, d_rec)

                if self.render:
                    def render(env_state):
                        return self.env.render(env_state)

                    env_state = jax.experimental.io_callback(render, env_state, env_state)

                return (policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, key, safety_layer_curriculum_coeff, d_roll), transition

            @jax.jit
            @jax.jit
            def get_receding_min_target(future_states: jnp.ndarray,
                                        dones: jnp.ndarray,
                                        terminates: jnp.ndarray):
                """Labels for the transitions of one rollout; see ncbf_targets.receding_min_target.
                Returns target [T, N, K], y [T, N], mask [T, N], dists_to_next_term [T, N]."""
                if self.ncbf_label_source == "termination_window":
                    return termination_window_labels(dones, terminates, self.ncbf_H, self.ncbf_safe_fall_H)
                return receding_min_target(future_states, dones, terminates, self.ncbf_H, self.env.ncbf_target_indices)

            @partial(jax.jit, static_argnums=(8,))
            def train_next_step_predictor(encoder_state, decoder_state, states, next_states, actions, masks, history_stacks, key: jax.random.PRNGKey, nr_minibatches: int):
                @jax.jit
                def loss_fn(encoder_params, decoder_params, minib_obs, minib_next_obs, minib_actions, minib_masks, minib_history_stack):
                    latent = encoder_state.apply_fn(encoder_params, minib_history_stack)
                    pred_next_obs = decoder_state.apply_fn(decoder_params, latent, minib_obs, minib_actions)

                    # Reshape masks from (batch,) to (batch, 1) for proper broadcasting with obs differences
                    minib_masks = minib_masks[..., None].astype(jnp.float32)
                    mse_loss = jnp.sum(minib_masks * jnp.square(pred_next_obs - minib_next_obs[..., self.next_step_predictor_output_indices])) / (jnp.sum(minib_masks) + 1e-8)
                    return mse_loss

                grad_predictor_loss_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))

                @jax.jit
                def minibatch_update(carry, _):
                    encoder_state, decoder_state, key = carry
                    predictor_key, key = jax.random.split(key)

                    # sample minibatch
                    key, sample_key = jax.random.split(key)
                    indices = jax.random.randint(sample_key, (self.next_step_predictor_minibatch_size), 0, states.shape[0])
                    minib_obs = states[indices]
                    minib_next_obs = next_states[indices]
                    minib_history_stack = history_stacks[indices]
                    minib_masks = masks[indices]
                    minib_actions = actions[indices]

                    loss, grads = grad_predictor_loss_fn(
                        encoder_state.params,
                        decoder_state.params,
                        minib_obs,
                        minib_next_obs,
                        minib_actions,
                        minib_masks,
                        minib_history_stack,
                    )
                    new_encoder_state = encoder_state.apply_gradients(grads=grads[0])
                    new_decoder_state = decoder_state.apply_gradients(grads=grads[1])
                    carry = (new_encoder_state, new_decoder_state, key)
                    metrics = {"next_step_predictor/loss": loss}
                    return carry, metrics

                init_carry = (encoder_state, decoder_state, key)
                (encoder_state, decoder_state, key), metrics = jax.lax.scan(minibatch_update, init_carry, jnp.arange(nr_minibatches))
                safe_mean = lambda x: jnp.mean(x) if x is not None else x
                mean_metrics = {f"next_step_predictor/{k}": safe_mean(v)
                                for k, v in metrics.items()}
                return encoder_state, decoder_state, mean_metrics, key

            key, reset_key = jax.random.split(key, 2)
            reset_keys = jax.random.split(reset_key, self.nr_envs)
            env_state = self.env.reset(reset_keys, False)
            # Resume: the env state is re-reset (episodes restart), only the env curriculum is carried over.
            if self.resume_env_curriculum is not None:
                restored_internal_state = dict(env_state.internal_state)
                for curriculum_key, curriculum_value in self.resume_env_curriculum.items():
                    if curriculum_key in restored_internal_state:
                        restored_internal_state[curriculum_key] = jnp.asarray(curriculum_value, dtype=restored_internal_state[curriculum_key].dtype)
                env_state = env_state.replace(internal_state=restored_internal_state)

            policy_state = self.policy_state
            critic_state = self.critic_state
            ncbf_state = self.ncbf_state
            encoder_state = self.encoder_state
            decoder_state = self.decoder_state

            def _init_buffer(capacity):
                buffer = {
                    "states": jnp.zeros((capacity, ) + (self.os_shape[0], ), dtype=jnp.float32),
                    "next_states": jnp.zeros((capacity, ) + (self.os_shape[0], ), dtype=jnp.float32),
                    "actions": jnp.zeros((capacity, ) + (self.as_shape[0], ), dtype=jnp.float32),
                    "dones": jnp.zeros((capacity, ), dtype=jnp.bool),
                    "terminations": jnp.zeros((capacity, ), dtype=jnp.bool),
                    "y_target": jnp.zeros((capacity, self.ncbf_nr_heads), dtype=jnp.float32),
                    "masks": jnp.zeros((capacity, ), dtype=jnp.bool),
                    "indices_to_term": jnp.zeros((capacity, ), dtype=jnp.int32),
                    "last_state": jnp.zeros((capacity, ) + (self.os_shape[0], ), dtype=jnp.float32),
                    "last_action": jnp.zeros((capacity, ) + (self.as_shape[0], ), dtype=jnp.float32),
                    "history_stack": jnp.zeros((capacity, self.nr_history_steps) + (self.os_shape[0], ), dtype=jnp.float32),
                    "pos": jnp.zeros((), dtype=jnp.int32),
                    "size": jnp.zeros((), dtype=jnp.int32)
                }
                return buffer

            ncbf_replay_buffer = _init_buffer(self.ncbf_neg_buffer_size)

            def multi_learning_and_eval_save_iteration(multi_learning_and_eval_save_iteration_carry, multi_learning_iteration_step, first_learning_iteration_step=0):
                policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, ncbf_replay_buffer, key, d_state = multi_learning_and_eval_save_iteration_carry

                def learning_iteration(learning_iteration_carry, learning_iteration_step):
                    policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, ncbf_replay_buffer, key, d_state = learning_iteration_carry
                    ncbf_neg_buffer = ncbf_replay_buffer
                    current_learning_update = (multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step
                    curriculum_denominator = jnp.maximum(jnp.asarray(self.nr_updates - 1, dtype=jnp.float32), 1.0)
                    safety_layer_curriculum_fraction = jnp.clip(
                        jnp.asarray(current_learning_update, dtype=jnp.float32) / curriculum_denominator,
                        0.0,
                        1.0,
                    )
                    safety_layer_curriculum_coeff = (
                        jnp.asarray(self.ncbf_safety_layer_std_coeff_start, dtype=jnp.float32) +
                        safety_layer_curriculum_fraction *
                        jnp.asarray(self.ncbf_safety_layer_std_coeff_final - self.ncbf_safety_layer_std_coeff_start, dtype=jnp.float32)
                    )
                    safety_layer_curriculum_coeff = jnp.where(
                        self.ncbf_use_safety_layer,
                        safety_layer_curriculum_coeff,
                        jnp.asarray(0.0, dtype=jnp.float32),
                    )
                    d_roll = self.d.rollout_carry(d_state, current_learning_update) if self.d is not None else {}
                    rollout_carry = (policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, key, safety_layer_curriculum_coeff, d_roll)
                    single_rollout_carry, batch = jax.lax.scan(single_rollout, rollout_carry, None, self.nr_steps)
                    policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, key, _, d_roll = single_rollout_carry
                    states, next_states, actions, env_actions, rewards, values, terminations, dones, log_probs, infos, constraints_active, delta_u, last_states, last_actions, history_stacks, safety_diagnostics, first_steps, d_recs = batch
                    d_metrics, d_ppo_batch, d_strength = {}, {}, jnp.asarray(0.0, jnp.float32)
                    if self.d is not None:
                        # D (plan 4.9 step 2): eval on the new rollout, normalizers, replay, WM updates, warm start.
                        d_state = {**d_state, "v_prev": d_roll["v_prev"]}
                        d_strength = d_roll["s"]
                        key, d_key = jax.random.split(key)
                        d_state, d_metrics = self.d.iteration_update(d_state, policy_state.params, d_recs,
                                                                     current_learning_update, d_key)
                        d_ppo_batch = self.d.ppo_batch(d_recs)
                        del d_recs
                    # Per-step physics-state records (env option record_transition_physics_state) are data, not metrics.
                    infos = {k: v for k, v in infos.items() if not k.startswith("transition/")}

                    # process the batch data to get mask and y_target
                    y_target, safe_y, masks, indices_to_term = get_receding_min_target(next_states, dones, terminations)
                    next_state_mask = ~ dones

                    # Successor pairs for the temporal-consistency loss: (s_t, a_t) -> (s_{t+1}, a_{t+1}) within one episode.
                    next_env_actions, next_y_target, temporal_valid = successor_pairs(env_actions, y_target, masks, dones)

                    ncbf_metrics = {}
                    ncbf_metrics["ncbf/safety_layer_enabled"] = jnp.asarray(self.ncbf_use_safety_layer, dtype=jnp.float32)
                    ncbf_metrics["ncbf/safety_layer_curriculum_coeff"] = safety_layer_curriculum_coeff
                    ncbf_metrics["ncbf/safety_layer_std_coeff"] = safety_layer_curriculum_coeff
                    ncbf_metrics["ncbf/constraints_active_rate"] = jnp.mean(constraints_active)
                    ncbf_metrics["ncbf/mean_delta_u"] = jnp.mean(jnp.abs(delta_u))
                    active_mask = constraints_active.astype(jnp.float32)
                    nonzero_correction = (delta_u > 1e-6).astype(jnp.float32)
                    n_active = jnp.sum(active_mask)
                    ncbf_metrics["ncbf/safety_layer_active_rate"] = jnp.mean(active_mask)
                    ncbf_metrics["ncbf/safety_layer_delta_u_mean"] = jnp.mean(delta_u)
                    ncbf_metrics["ncbf/safety_layer_delta_u_max"] = jnp.max(delta_u)
                    ncbf_metrics["ncbf/safety_layer_delta_u_active_mean"] = jnp.sum(delta_u * active_mask) / (n_active + 1e-8)
                    ncbf_metrics["ncbf/safety_layer_nonzero_correction_rate"] = jnp.mean(nonzero_correction)
                    ncbf_metrics["ncbf/safety_layer_active_but_zero_rate"] = jnp.mean(active_mask * (1.0 - nonzero_correction))
                    ncbf_metrics["ncbf/mean_y"] = jnp.mean(y_target)
                    ncbf_metrics["ncbf/mask_rate"] = jnp.mean(masks.astype(jnp.float32))
                    ncbf_metrics["ncbf/safe_y_rate"] = jnp.mean(safe_y.astype(jnp.float32))
                    ncbf_metrics["ncbf/nr_rollout_neg_samples"] = jnp.sum(~safe_y)
                    for head_index, head_name in enumerate(self.ncbf_target_names[:2]):
                        ncbf_metrics[f"ncbf/target_{head_name}_safe_rate"] = jnp.mean(y_target[..., head_index])
                        ncbf_metrics[f"ncbf/nr_{head_name}_unsafe_targets"] = jnp.sum((masks & (y_target[..., head_index] < 0.5)).astype(jnp.float32))
                    ncbf_metrics["rollout/done_rate"] = jnp.mean(dones.astype(jnp.float32))
                    ncbf_metrics["rollout/termination_rate"] = jnp.mean(terminations.astype(jnp.float32))
                    ncbf_metrics["rollout/nr_terminations"] = jnp.sum(terminations.astype(jnp.float32))
                    ncbf_metrics["rollout/truncation_rate"] = jnp.mean((dones & ~terminations).astype(jnp.float32))
                    ncbf_metrics["rollout/reward_mean"] = jnp.mean(rewards)
                    action_low = jnp.asarray(self.env.single_action_space.low)
                    action_high = jnp.asarray(self.env.single_action_space.high)
                    ncbf_metrics["policy/raw_action_abs_mean"] = jnp.mean(jnp.abs(actions))
                    ncbf_metrics["policy/raw_action_abs_max"] = jnp.max(jnp.abs(actions))
                    ncbf_metrics["policy/raw_action_out_of_bounds_fraction"] = jnp.mean(
                        ((actions < action_low) | (actions > action_high)).astype(jnp.float32)
                    )
                    ncbf_metrics["policy/executed_action_abs_mean"] = jnp.mean(jnp.abs(env_actions))
                    raw_processed_actions = self.get_processed_action(actions)
                    raw_processed_norm = jnp.linalg.norm(raw_processed_actions, axis=-1)
                    executed_norm = jnp.linalg.norm(env_actions, axis=-1)
                    shield_correction_norm = jnp.linalg.norm(env_actions - raw_processed_actions, axis=-1)
                    ncbf_metrics["ncbf/safety_layer_raw_processed_action_norm"] = jnp.mean(raw_processed_norm)
                    ncbf_metrics["ncbf/safety_layer_executed_action_norm"] = jnp.mean(executed_norm)
                    ncbf_metrics["ncbf/safety_layer_action_correction_norm"] = jnp.mean(shield_correction_norm)
                    ncbf_metrics["ncbf/safety_layer_action_correction_norm_max"] = jnp.max(shield_correction_norm)
                    ncbf_metrics["ncbf/safety_layer_correction_to_action_ratio"] = jnp.mean(shield_correction_norm / (raw_processed_norm + 1e-8))
                    ncbf_metrics["ncbf/safety_layer_executed_action_out_of_bounds_fraction"] = jnp.mean(
                        ((env_actions < action_low) | (env_actions > action_high)).astype(jnp.float32)
                    )
                    ncbf_metrics["ncbf/safety_layer_action_bound_saturation_rate"] = jnp.mean(
                        ((env_actions <= action_low + 1e-5) | (env_actions >= action_high - 1e-5)).astype(jnp.float32)
                    )
                    ncbf_metrics["ncbf/safety_layer_lambda_slack_config"] = jnp.asarray(self.ncbf_lambda_slack, dtype=jnp.float32)
                    ncbf_metrics["ncbf/safety_layer_max_delta_u_config"] = jnp.asarray(self.ncbf_max_delta_u, dtype=jnp.float32)
                    ncbf_metrics["ncbf/safety_layer_projection_is_hard"] = jnp.asarray(
                        self.ncbf_safety_layer_projection == "hard_projection",
                        dtype=jnp.float32,
                    )
                    ncbf_metrics["ncbf/safety_layer_min_grad_norm_config"] = jnp.asarray(
                        self.ncbf_safety_layer_min_grad_norm,
                        dtype=jnp.float32,
                    )
                    ncbf_metrics["ncbf/safety_layer_actual_post_check_enabled"] = jnp.asarray(
                        self.ncbf_post_check_actual_residual,
                        dtype=jnp.float32,
                    )
                    for diag_key, diag_value in safety_diagnostics.items():
                        ncbf_metrics[f"ncbf_safety_layer/{diag_key}"] = jnp.mean(diag_value)
                    ncbf_metrics["ncbf/safety_layer_constraint_delta_max"] = jnp.max(safety_diagnostics["constraint_delta"])
                    ncbf_metrics["ncbf/safety_layer_robust_residual_min"] = jnp.min(safety_diagnostics["robust_residual"])
                    ncbf_metrics["ncbf/safety_layer_correction_clipped_rate"] = jnp.mean(safety_diagnostics["correction_clipped"])
                    post_violation = safety_diagnostics["post_constraint_violation"]
                    ncbf_metrics["ncbf/safety_layer_post_violation_rate"] = jnp.mean(post_violation)
                    ncbf_metrics["ncbf/safety_layer_active_post_violation_rate"] = jnp.sum(post_violation * active_mask) / (n_active + 1e-8)
                    ncbf_metrics["ncbf/safety_layer_post_constraint_delta_max"] = jnp.max(safety_diagnostics["post_constraint_delta"])
                    ncbf_metrics["ncbf/safety_layer_post_linearized_margin_min"] = jnp.min(safety_diagnostics["post_linearized_margin"])
                    ncbf_metrics["ncbf/safety_layer_nonfinite_linearization_rate"] = 1.0 - jnp.mean(safety_diagnostics["linearization_is_finite"])
                    ncbf_metrics["ncbf/safety_layer_unavailable_or_post_violation_rate"] = jnp.mean(
                        jnp.maximum(1.0 - safety_diagnostics["linearization_is_finite"], post_violation)
                    )
                    required_correction_norm = safety_diagnostics["required_correction_norm"]
                    ncbf_metrics["ncbf/safety_layer_required_correction_norm"] = jnp.mean(required_correction_norm)
                    ncbf_metrics["ncbf/safety_layer_required_correction_norm_active_mean"] = (
                        jnp.sum(required_correction_norm * active_mask) / (n_active + 1e-8)
                    )
                    ncbf_metrics["ncbf/safety_layer_required_correction_norm_max"] = jnp.max(required_correction_norm)
                    ncbf_metrics["ncbf/safety_layer_required_exceeds_cap_rate"] = jnp.mean(safety_diagnostics["required_exceeds_cap"])
                    ncbf_metrics["ncbf/safety_layer_active_required_exceeds_cap_rate"] = (
                        jnp.sum(safety_diagnostics["required_exceeds_cap"] * active_mask) / (n_active + 1e-8)
                    )
                    ncbf_metrics["ncbf/safety_layer_capped_active_rate"] = jnp.mean(safety_diagnostics["capped_active"])
                    ncbf_metrics["ncbf/safety_layer_low_grad_linearization_rate"] = jnp.mean(
                        safety_diagnostics["low_grad_linearization"]
                    )
                    ncbf_metrics["ncbf/safety_layer_low_grad_active_rate"] = jnp.mean(
                        safety_diagnostics["low_grad_active"]
                    )
                    ncbf_metrics["ncbf/safety_layer_active_low_grad_fraction"] = (
                        jnp.sum(safety_diagnostics["low_grad_active"] * active_mask) / (n_active + 1e-8)
                    )
                    ncbf_metrics["ncbf/safety_layer_low_grad_guarded_rate"] = jnp.mean(
                        safety_diagnostics["low_grad_guarded"]
                    )
                    ncbf_metrics["ncbf/safety_layer_soft_residual_fraction"] = jnp.mean(safety_diagnostics["soft_residual_fraction"])
                    ncbf_metrics["ncbf/safety_layer_soft_residual_fraction_active_mean"] = (
                        jnp.sum(safety_diagnostics["soft_residual_fraction"] * active_mask) / (n_active + 1e-8)
                    )
                    ncbf_metrics["ncbf/safety_layer_post_clip_changed_action_rate"] = jnp.mean(safety_diagnostics["post_clip_changed_action"])
                    ncbf_metrics["ncbf/safety_layer_post_clip_action_delta_norm"] = jnp.mean(safety_diagnostics["post_clip_action_delta_norm"])
                    actual_post_violation = safety_diagnostics["actual_post_violation"]
                    ncbf_metrics["ncbf/safety_layer_actual_post_violation_rate"] = jnp.mean(actual_post_violation)
                    ncbf_metrics["ncbf/safety_layer_active_actual_post_violation_rate"] = (
                        jnp.sum(actual_post_violation * active_mask) / (n_active + 1e-8)
                    )
                    ncbf_metrics["ncbf/safety_layer_actual_post_robust_residual_min"] = jnp.min(
                        safety_diagnostics["actual_post_robust_residual"]
                    )
                    ncbf_metrics["ncbf/safety_layer_actual_post_nonfinite_rate"] = (
                        1.0 - jnp.mean(safety_diagnostics["actual_post_is_finite"])
                    )

                    @jax.jit
                    def append_to_buffer(buffer,
                                         states,
                                         next_states,
                                         actions,
                                         dones,
                                         terminations,
                                         y_target,
                                         masks,
                                         indices_to_term,
                                         last_states,
                                         last_actions,
                                         history_stacks):
                        """
                        Append valid samples to a single ring buffer.

                        Inputs:
                          states:          (T, E, D)
                          next_states:     (T, E, D)
                          actions:         (T, E, A)
                          dones:           (T, E)
                          terminations:    (T, E)
                          y_target:        (T, E, K)
                          masks:           (T, E)      bool, True = keep
                          indices_to_term: (T, E)
                          last_states:     (T, E, D)
                          last_actions:    (T, E, A)
                          history_stacks:  (T, E, Hs, D)

                        Buffer fields:
                          states
                          next_states
                          actions
                          dones
                          terminations
                          y_target
                          masks
                          indices_to_term
                          last_state
                          last_action
                          history_stack
                          write_pos
                          size
                        """
                        # flatten rollout dims
                        states = states.reshape(-1, states.shape[-1])  # (B, D)
                        next_states = next_states.reshape(-1, next_states.shape[-1])  # (B, D)
                        actions = actions.reshape(-1, actions.shape[-1])  # (B, A)
                        dones = dones.reshape(-1)  # (B,)
                        terminations = terminations.reshape(-1)  # (B,)
                        y_target = y_target.reshape(-1, y_target.shape[-1])  # (B, K)
                        masks = masks.reshape(-1).astype(jnp.bool_)  # (B,)
                        indices_to_term = indices_to_term.reshape(-1)  # (B,)
                        last_states = last_states.reshape(-1, last_states.shape[-1])  # (B, D)
                        last_actions = last_actions.reshape(-1, last_actions.shape[-1])  # (B, A)
                        history_stacks = history_stacks.reshape(
                            -1, history_stacks.shape[-2], history_stacks.shape[-1]
                        )  # (B, Hs, D)

                        B = masks.shape[0]
                        capacity = buffer["states"].shape[0]

                        # valid indices, padded to fixed size
                        valid_idx = jnp.where(masks, size=B, fill_value=-1)[0]  # (B,)
                        valid_mask = valid_idx >= 0
                        n_valid = jnp.sum(valid_mask, dtype=jnp.int32)

                        gather_idx = jnp.clip(valid_idx, 0)

                        states_v = states[gather_idx]
                        next_states_v = next_states[gather_idx]
                        actions_v = actions[gather_idx]
                        dones_v = dones[gather_idx]
                        terminations_v = terminations[gather_idx]
                        y_target_v = y_target[gather_idx]
                        indices_to_term_v = indices_to_term[gather_idx]
                        last_states_v = last_states[gather_idx]
                        last_actions_v = last_actions[gather_idx]
                        history_stacks_v = history_stacks[gather_idx]
                        masks_v = masks[gather_idx]

                        start = buffer["pos"]
                        write_offsets = jnp.arange(B, dtype=jnp.int32)
                        write_slots = (start + write_offsets) % capacity

                        new_buffer = dict(buffer)

                        def write_1d(field, values):
                            old = buffer[field][write_slots]
                            new = jnp.where(valid_mask, values, old)
                            new_buffer[field] = buffer[field].at[write_slots].set(new)

                        def write_2d(field, values):
                            old = buffer[field][write_slots]
                            new = jnp.where(valid_mask[:, None], values, old)
                            new_buffer[field] = buffer[field].at[write_slots].set(new)

                        def write_3d(field, values):
                            old = buffer[field][write_slots]
                            new = jnp.where(valid_mask[:, None, None], values, old)
                            new_buffer[field] = buffer[field].at[write_slots].set(new)

                        write_2d("states", states_v)
                        write_2d("next_states", next_states_v)
                        write_2d("actions", actions_v)
                        write_1d("dones", dones_v)
                        write_1d("terminations", terminations_v)
                        write_2d("y_target", y_target_v)  # always 2D now
                        write_1d("masks", masks_v)
                        write_1d("indices_to_term", indices_to_term_v)
                        write_2d("last_state", last_states_v)
                        write_2d("last_action", last_actions_v)
                        write_3d("history_stack", history_stacks_v)

                        new_buffer["pos"] = (start + n_valid) % capacity
                        new_buffer["size"] = jnp.minimum(capacity, buffer["size"] + n_valid)

                        return new_buffer

                    # write negative and valid data samples to the buffer.
                    # A window is negative if it ends in a termination or if any safety target drops to 0 inside it;
                    # the second case covers non-terminal failures (ball drop without termination, non-terminal collisions).
                    target_unsafe = jnp.any(y_target < 0.5, axis=-1)
                    neg_masks = jnp.logical_and(masks, jnp.logical_or(~safe_y, target_unsafe))
                    ncbf_neg_buffer = append_to_buffer(ncbf_neg_buffer,
                        states, next_states, env_actions, dones, terminations, y_target, neg_masks, indices_to_term, last_states, last_actions, history_stacks)
                    ncbf_metrics["ncbf/current_neg_samples"] = jnp.sum(neg_masks.astype(jnp.float32))
                    ncbf_metrics["ncbf/neg_buffer_size"] = ncbf_neg_buffer["size"].astype(jnp.float32)
                    ncbf_metrics["ncbf/replay_neg_target_batch_size"] = jnp.asarray(self.ncbf_replay_neg_minibatch_size, dtype=jnp.float32)

                    # sample and update

                    # Calculating advantages and returns
                    def calculate_gae_advantages(critic_state, encoder_state, decoder_state,
                                                 states, next_states, env_actions,
                                                 history_stacks, rewards, values, terminations, dones):
                        terminations = terminations.astype(jnp.float32)
                        dones = dones.astype(jnp.float32)

                        next_history_stacks = jnp.roll(history_stacks, -1, axis=2)
                        next_history_stacks = next_history_stacks.at[:, :, -1, :].set(next_states)
                        next_latents = jax.lax.stop_gradient(encoder_state.apply_fn(encoder_state.params, next_history_stacks))
                        next_state_inputs = self._mask_policy_ball_plate_inputs(next_states)
                        next_decoder_output = self._get_policy_decoder_output(decoder_state.params, next_latents, states, env_actions, next_states)
                        next_values = self.critic.apply(critic_state.params, next_state_inputs, next_latents, next_decoder_output).squeeze(-1)
                        delta = rewards + self.gamma * next_values * (1.0 - terminations) - values
                        # T06c: algorithm.gae_stop_at_truncation=False reproduces the paper code (cut at terminations only)
                        advantages = gae_advantages(delta, terminations, dones, self.gamma, self.gae_lambda,
                                                    self.gae_stop_at_truncation)
                        returns = advantages + values
                        return advantages, returns

                    advantages, returns = calculate_gae_advantages(
                        critic_state,
                        encoder_state,
                        decoder_state,
                        states,
                        next_states,
                        env_actions,
                        history_stacks,
                        rewards,
                        values,
                        terminations,
                        dones,
                    )

                    # Optimizing
                    def ppo_loss_fn(policy_params, critic_params, encoder_params, decoder_params,
                                    ncbf_params,
                                    state_b, action_b, log_prob_b, return_b, advantage_b,
                                    last_state_b, last_action_b, history_stack_b, first_step_b, d_b=None):
                        # Policy loss
                        latent_b = jax.lax.stop_gradient(self.encoder.apply(encoder_params, history_stack_b))
                        decoder_output_b = self._get_policy_decoder_output(
                            decoder_params,
                            latent_b,
                            last_state_b,
                            last_action_b,
                            state_b,
                        )
                        state_input_b = self._mask_policy_ball_plate_inputs(state_b)
                        action_mean, action_logstd = self.policy.apply(policy_params, state_input_b, latent_b, decoder_output_b)
                        action_logstd = maybe_clip_logstd(action_logstd, self.policy_logstd_clip)
                        action_std = jnp.exp(action_logstd)
                        new_log_prob = -0.5 * ((action_b - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
                        new_log_prob = new_log_prob.sum(1)
                        entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
                        entropy = jnp.sum(entropy, axis=-1)

                        logratio = new_log_prob - log_prob_b
                        ratio = jnp.exp(logratio)
                        approx_kl_div = jnp.mean((ratio - 1) - logratio)
                        clip_fraction = jnp.mean(jnp.float32((jnp.abs(ratio - 1) > self.clip_range)))

                        pg_loss1 = -advantage_b * ratio
                        pg_loss2 = -advantage_b * jnp.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
                        pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

                        entropy_loss = jnp.mean(entropy)

                        # Critic loss
                        new_value = self.critic.apply(critic_params, state_input_b, latent_b, decoder_output_b)
                        critic_loss = 0.5 * jnp.mean((new_value.squeeze(-1) - return_b) ** 2)

                        # anticipation loss
                        d2_loss = jnp.asarray(0.0, dtype=pg_loss.dtype)
                        d_anticipation_active = jnp.asarray(0.0, dtype=pg_loss.dtype)
                        if self.d is not None:
                            # method D (T16): anticipation through the WM shield (plan 4.7.1) and D2 (4.7.3)
                            action_mean_processed = self.get_processed_action(action_mean)
                            if self.d.cfg.anticipation_coef > 0.0:
                                anticipation_loss, d_anticipation_active = self.d.anticipation_loss(
                                    d_state, policy_params, d_b, action_mean_processed, d_strength, self.anticipation_style)
                            else:
                                anticipation_loss = jnp.asarray(0.0, dtype=pg_loss.dtype)
                            if self.d.cfg.d2_coef > 0.0:
                                d2_loss = self.d.d2_loss(d_state, policy_params, d_b, action_mean_processed)
                            anticipation_nonfinite_rate = jnp.asarray(0.0, dtype=pg_loss.dtype)
                        elif self.anticipation_coef > 0.0:
                            action_mean_processed = self.get_processed_action(action_mean)
                            # T06c: the paper code shielded the sampled rollout action (anticipation_style=sampled_action)
                            anticipation_input = (self.get_processed_action(action_b) if self.anticipation_style == "sampled_action"
                                                  else action_mean_processed)
                            safe_action_b, constraint_active_b, delta_u_b, safety_diagnostics_b = self.batched_ncbf_safety_layer(
                                anticipation_input,
                                state_b,
                                last_action_b,
                                last_state_b,
                                latent_b,
                                safety_layer_curriculum_coeff,
                                repeat_ncbf_params_from_params(ncbf_params),
                                first_step_b,
                            )
                            safe_action_b, _, _, _ = self._bypass_post_drop_safety_layer(
                                anticipation_input,
                                safe_action_b,
                                constraint_active_b,
                                delta_u_b,
                                safety_diagnostics_b,
                                state_b,
                            )
                            safe_action_is_finite = jnp.all(jnp.isfinite(safe_action_b), axis=-1)
                            safe_action_b = jnp.where(safe_action_is_finite[:, None], safe_action_b, anticipation_input)
                            if not self.anticipation_through_shield:
                                # Behavior cloning toward the shielded action (no gradient through the shield).
                                safe_action_b = jax.lax.stop_gradient(safe_action_b)
                            anticipation_loss = 0.5 * jnp.mean(anticipation_squared_error(action_mean_processed, safe_action_b, self.anticipation_style))
                            anticipation_nonfinite_rate = 1.0 - jnp.mean(safe_action_is_finite.astype(jnp.float32))
                        else:
                            anticipation_loss = jnp.asarray(0.0, dtype=pg_loss.dtype)
                            anticipation_nonfinite_rate = jnp.asarray(0.0, dtype=pg_loss.dtype)

                        anticipation_coef = self.d.cfg.anticipation_coef if self.d is not None else self.anticipation_coef
                        loss = (
                            pg_loss -
                            self.entropy_coef * entropy_loss +
                            self.critic_coef * critic_loss +
                            anticipation_coef * anticipation_loss
                        )
                        if self.d is not None and self.d.cfg.d2_coef > 0.0:
                            loss = loss + self.d.cfg.d2_coef * d2_loss

                        metrics = {
                            "loss/policy_gradient_loss": pg_loss,
                            "loss/critic_loss": critic_loss,
                            "loss/entropy_loss": entropy_loss,
                            "loss/anticipation_loss": anticipation_loss,
                            "loss/anticipation_nonfinite_rate": anticipation_nonfinite_rate,
                            "loss/d2_wm_safe_loss": d2_loss,
                            "d/anticipation_shield_active_rate": d_anticipation_active,
                            "policy_ratio/approx_kl": approx_kl_div,
                            "policy_ratio/clip_fraction": clip_fraction,
                        }
                        return loss, metrics

                    def auxiliary_loss_fn(encoder_params, decoder_params, ncbf_params,
                                          state_b, next_state_b, env_action_b,
                                          last_action_b, next_state_mask_b, history_stack_b,
                                          y_target_b, indices_to_term_b, ncbf_masks_b,
                                          next_env_action_b, next_y_target_b, temporal_valid_b, raw_action_b,
                                          replay_state_b, replay_next_state_b, replay_action_b,
                                          replay_y_target_b, replay_indices_to_term_b, replay_ncbf_masks_b,
                                          replay_last_action_b, replay_history_stack_b, bootstrap_key):
                        latent_b = self.encoder.apply(encoder_params, history_stack_b)
                        next_history_stack_b = jnp.roll(history_stack_b, -1, axis=1)
                        next_history_stack_b = next_history_stack_b.at[:, -1, :].set(next_state_b)
                        next_latent_b = self.encoder.apply(encoder_params, next_history_stack_b)

                        # auxiliary next step prediction loss
                        pred_next_obs = self.decoder.apply(decoder_params, latent_b, state_b, env_action_b)
                        next_state_target = next_state_b[..., self.next_step_predictor_output_indices]
                        prediction_loss = jnp.sum(next_state_mask_b * jnp.mean(jnp.square(pred_next_obs - next_state_target), axis=-1)) / (jnp.sum(next_state_mask_b) + 1e-8)

                        replay_latent_b = self.encoder.apply(encoder_params, replay_history_stack_b)

                        # Safety critic input (state, action, latent). The label y_t covers s_{t+1..t+H}, so the action
                        # paired with s_t must be the action executed at t; this is also how the shield queries the
                        # critic, h(s_t, u). "previous" reproduces the legacy pairing with a_{t-1}.
                        # The episode-ending step keeps its pre-reset (s_t, a_t) unless terminal_critic_input="zeros" (F1).
                        critic_state_b, current_action_b = critic_training_inputs(
                            state_b, env_action_b, last_action_b, ~next_state_mask_b,
                            self.ncbf_critic_action_index, self.ncbf_terminal_critic_input)
                        replay_current_action_b = replay_action_b if self.ncbf_critic_action_index == "current" else replay_last_action_b
                        current_h_input = jnp.concatenate([critic_state_b[..., self.ncbf_observation_indices], current_action_b, latent_b], axis=-1)
                        replay_h_input = jnp.concatenate([replay_state_b[..., self.ncbf_observation_indices], replay_current_action_b, replay_latent_b], axis=-1)

                        h_input = jnp.concatenate((current_h_input, replay_h_input), axis=0)
                        ncbf_y_target = jnp.concatenate((y_target_b, replay_y_target_b), axis=0)
                        ncbf_masks = jnp.concatenate((ncbf_masks_b, replay_ncbf_masks_b), axis=0).astype(jnp.float32)
                        ncbf_indices_to_term = jnp.concatenate((indices_to_term_b, replay_indices_to_term_b), axis=0)

                        h_raw = self._ncbf_apply_members(ncbf_params, h_input)  # [E, B, out]
                        h_logits, h_log_std = split_ncbf_output(
                            h_raw,
                            self.ncbf_output_distribution,
                            self.ncbf_min_log_std,
                            self.ncbf_max_log_std,
                        )
                        ncbf_y_target = ncbf_y_target.astype(h_logits.dtype)
                        targets = jnp.broadcast_to(ncbf_y_target[None], h_logits.shape)

                        if self.ncbf_output_distribution == "logistic_normal":
                            bce = logistic_normal_binary_cross_entropy(h_logits, h_log_std, targets)
                            h_prob = logistic_normal_safe_probability(h_logits, h_log_std)
                            ncbf_mean_std = jnp.mean(jnp.exp(h_log_std))
                        else:
                            bce = optax.sigmoid_binary_cross_entropy(h_logits, targets)
                            h_prob = nn.sigmoid(h_logits)
                            ncbf_mean_std = jnp.array(0.0, dtype=h_logits.dtype)

                        # (1) horizon-safety classification, optionally with the label discount of paper eq. (2):
                        # negative heads are weighted by coef_decay_lambda ** (steps to failure).
                        if self.ncbf_label_discount:
                            steps_to_failure = jnp.clip(ncbf_indices_to_term, 0, self.ncbf_H).astype(h_logits.dtype)
                            label_weight = jnp.where(ncbf_y_target < 0.5, (self.ncbf_coef_decay_lambda ** steps_to_failure)[:, None], 1.0)
                        else:
                            label_weight = jnp.ones_like(ncbf_y_target)
                        head_loss = jnp.mean(label_weight[None] * bce, axis=-1)  # [E, B]

                        # Each member sees its own bootstrap resample so the ensemble disagreement is meaningful.
                        n_members = h_logits.shape[0]
                        if n_members > 1 and self.ncbf_bootstrap_prob < 1.0:
                            bootstrap = jax.random.bernoulli(bootstrap_key, self.ncbf_bootstrap_prob, (n_members, h_input.shape[0])).astype(h_logits.dtype)
                        else:
                            bootstrap = jnp.ones((n_members, h_input.shape[0]), dtype=h_logits.dtype)
                        sample_weight = bootstrap * ncbf_masks[None]
                        per_member_cls = jnp.sum(sample_weight * head_loss, axis=-1) / (jnp.sum(sample_weight, axis=-1) + 1e-8)
                        cls_loss = jnp.sum(per_member_cls)

                        # (2) DT-CBF temporal consistency (paper eq. 4-6) on on-policy successor pairs (F2 fix):
                        # residual h(s_{t+1}, a_{t+1}) - (1 - eta) h(s_t, a_t) - eta gamma_c >= 0 on safe->safe pairs
                        # of one episode, both actions on the executed-action scale. temporal_mode
                        # "legacy_raw_vs_executed" restores the paper code (h(s_t, a_exec) vs h(s_t, a_raw)).
                        if self.ncbf_w_temporal > 0.0:
                            n_current = state_b.shape[0]
                            h_cur = h_prob[:, :n_current]
                            if self.ncbf_temporal_mode == "adjacent":
                                next_h_input = jnp.concatenate([next_state_b[..., self.ncbf_observation_indices], next_env_action_b, next_latent_b], axis=-1)
                                h_next = self._ncbf_probabilities(self._ncbf_apply_members(ncbf_params, next_h_input))
                                temporal_loss = temporal_consistency_loss(
                                    h_cur, h_next, y_target_b, next_y_target_b, temporal_valid_b,
                                    self.ncbf_gamma_c, self.ncbf_eta_cbf, self.ncbf_gamma_c_scaled_by_eta,
                                    self.ncbf_temporal_pairs)
                            else:
                                raw_h_input = jnp.concatenate([state_b[..., self.ncbf_observation_indices], raw_action_b, latent_b], axis=-1)
                                h_raw = self._ncbf_probabilities(self._ncbf_apply_members(ncbf_params, raw_h_input))
                                temporal_loss = legacy_raw_vs_executed_temporal_loss(
                                    h_cur, h_raw, y_target_b, ncbf_masks_b, indices_to_term_b, self.ncbf_H)
                        else:
                            temporal_loss = jnp.asarray(0.0, dtype=h_logits.dtype)

                        # (3) Lipschitz penalty. By default only on the state part of the input: the shield needs an
                        # informative gradient with respect to the action, so the action columns are not penalized.
                        if self.ncbf_w_lip > 0.0:
                            def summed_logits(x):
                                raw = self._ncbf_apply_members(ncbf_params, x)
                                logits, _ = split_ncbf_output(raw, self.ncbf_output_distribution, self.ncbf_min_log_std, self.ncbf_max_log_std)
                                return jnp.sum(logits)
                            input_grad = jax.grad(summed_logits)(jax.lax.stop_gradient(h_input))
                            if self.ncbf_lipschitz_state_only:
                                n_state = len(self.ncbf_observation_indices)
                                input_grad = jnp.concatenate([input_grad[:, :n_state], input_grad[:, n_state + self.as_shape[0]:]], axis=-1)
                            grad_norm = jnp.linalg.norm(input_grad, axis=-1) / n_members
                            lip_loss = jnp.mean(jnp.square(jax.nn.relu(grad_norm - self.ncbf_L_target)))
                        else:
                            lip_loss = jnp.asarray(0.0, dtype=h_logits.dtype)

                        ncbf_loss = cls_loss + self.ncbf_w_temporal * temporal_loss + self.ncbf_w_lip * lip_loss

                        loss = (
                            self.aux_prediction_coef * prediction_loss +
                            self.ncbf_loss_coef * ncbf_loss
                        )

                        den = jnp.sum(ncbf_masks) + 1e-8
                        member_mean_bce = jnp.mean(bce, axis=0)  # [B, K]
                        member_mean_prob = jnp.mean(h_prob, axis=0)  # [B, K]
                        metrics = {
                            "loss/aux_prediction_loss": prediction_loss,
                            "loss/ncbf_loss": ncbf_loss,
                            "loss/ncbf_cls_loss": cls_loss,
                            "loss/ncbf_temporal_loss": temporal_loss,
                            "loss/ncbf_lip_loss": lip_loss,
                            "loss/weighted_aux_prediction_loss": self.aux_prediction_coef * prediction_loss,
                            "loss/weighted_ncbf_loss": self.ncbf_loss_coef * ncbf_loss,
                            "ncbf/current_samples_per_minibatch": jnp.sum(ncbf_masks_b.astype(jnp.float32)),
                            "ncbf/replay_neg_samples_per_minibatch": jnp.sum(replay_ncbf_masks_b.astype(jnp.float32)),
                            "ncbf/mean_std": ncbf_mean_std,
                            # Std of the safe probability across ensemble members (0 for a collapsed ensemble).
                            "ncbf/ensemble_disagreement": jnp.sum(ncbf_masks * jnp.mean(jnp.std(h_prob, axis=0), axis=-1)) / den,
                        }
                        for head_index, head_name in enumerate(self.ncbf_target_names):
                            metrics[f"ncbf/{head_name}_bce"] = jnp.sum(ncbf_masks * member_mean_bce[..., head_index]) / den
                            metrics[f"ncbf/pred_{head_name}_safe_prob"] = jnp.sum(ncbf_masks * member_mean_prob[..., head_index]) / den
                            metrics[f"ncbf/minibatch_target_{head_name}_safe_rate"] = jnp.sum(ncbf_masks * ncbf_y_target[..., head_index]) / den

                        return loss, metrics

                    batch_states = states.reshape((-1,) + self.os_shape)
                    batch_next_states = next_states.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)
                    batch_env_actions = env_actions.reshape((-1,) + self.as_shape)
                    batch_advantages = advantages.reshape(-1)
                    batch_returns = returns.reshape(-1)
                    batch_log_probs = log_probs.reshape(-1)
                    batch_last_states = last_states.reshape((-1,) + self.os_shape)
                    batch_last_actions = last_actions.reshape((-1,) + self.as_shape)
                    batch_history_stack = history_stacks.reshape((-1,) + (self.nr_history_steps, ) + self.os_shape)
                    batch_next_state_masks = next_state_mask.reshape(-1)
                    batch_y_targets = y_target.reshape((-1,) + (self.ncbf_nr_heads, ))
                    batch_ncbf_masks = masks.reshape(-1)
                    batch_indices_to_term = indices_to_term.reshape(-1)
                    batch_next_env_actions = next_env_actions.reshape((-1,) + self.as_shape)
                    batch_next_y_targets = next_y_target.reshape((-1,) + (self.ncbf_nr_heads, ))
                    batch_temporal_valid = temporal_valid.reshape(-1)
                    batch_first_steps = first_steps.reshape(-1)

                    def sample_neg_replay_buffer(buffer, replay_key):
                        sample_limit = jnp.maximum(buffer["size"], 1)
                        replay_indices = jax.random.randint(
                            replay_key,
                            (self.ncbf_replay_neg_minibatch_size,),
                            0,
                            sample_limit,
                        )
                        replay_masks = buffer["masks"][replay_indices] & (buffer["size"] > 0)
                        replay_states_raw = buffer["states"][replay_indices]
                        replay_actions_raw = buffer["actions"][replay_indices]
                        if self.ncbf_terminal_critic_input == "zeros":
                            replay_states_raw, replay_actions_raw = critic_training_inputs(
                                replay_states_raw, replay_actions_raw, replay_actions_raw,
                                buffer["dones"][replay_indices], "current", "zeros")
                        return (
                            replay_states_raw,
                            buffer["next_states"][replay_indices],
                            replay_actions_raw,
                            buffer["y_target"][replay_indices],
                            buffer["indices_to_term"][replay_indices],
                            replay_masks,
                            buffer["last_action"][replay_indices],
                            buffer["history_stack"][replay_indices],
                        )

                    initial_policy_latent = jax.lax.stop_gradient(encoder_state.apply_fn(encoder_state.params, batch_history_stack))
                    initial_policy_decoder_output = self._get_policy_decoder_output(
                        decoder_state.params,
                        initial_policy_latent,
                        batch_last_states,
                        batch_last_actions,
                        batch_states,
                    )
                    initial_policy_state_input = self._mask_policy_ball_plate_inputs(batch_states)
                    initial_action_mean, initial_action_logstd = self.policy.apply(
                        policy_state.params,
                        initial_policy_state_input,
                        initial_policy_latent,
                        initial_policy_decoder_output,
                    )
                    initial_action_logstd = maybe_clip_logstd(initial_action_logstd, self.policy_logstd_clip)
                    initial_action_std = jnp.exp(initial_action_logstd)
                    initial_new_log_probs = -0.5 * ((batch_actions - initial_action_mean) / initial_action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - initial_action_logstd
                    initial_new_log_probs = initial_new_log_probs.sum(1)
                    initial_logratio = initial_new_log_probs - batch_log_probs
                    initial_ratio = jnp.exp(initial_logratio)
                    initial_approx_kl = jnp.mean((initial_ratio - 1.0) - initial_logratio)
                    initial_abs_logratio = jnp.mean(jnp.abs(initial_logratio))

                    ppo_grad_loss_fn = jax.value_and_grad(ppo_loss_fn, argnums=(0, 1), has_aux=True)
                    auxiliary_grad_loss_fn = jax.value_and_grad(auxiliary_loss_fn, argnums=(0, 1, 2), has_aux=True)

                    key, subkey = jax.random.split(key)
                    batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
                    batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
                    batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

                    def minibatch_update(carry, minibatch_indices):
                        policy_state, critic_state, encoder_state, decoder_state, ncbf_state, key = carry
                        key, replay_key, bootstrap_key = jax.random.split(key, 3)
                        replay_states, replay_next_states, replay_actions, replay_y_targets, replay_indices_to_term, \
                            replay_ncbf_masks, replay_last_actions, replay_history_stacks = sample_neg_replay_buffer(ncbf_neg_buffer, replay_key)

                        minibatch_advantages = batch_advantages[minibatch_indices]
                        minibatch_advantages = (minibatch_advantages - jnp.mean(minibatch_advantages)) / (jnp.std(minibatch_advantages) + 1e-8)

                        (ppo_loss, ppo_metrics), (policy_gradients, critic_gradients) = ppo_grad_loss_fn(
                            policy_state.params,
                            critic_state.params,
                            encoder_state.params,
                            decoder_state.params,
                            ncbf_state.params,
                            batch_states[minibatch_indices],
                            batch_actions[minibatch_indices],
                            batch_log_probs[minibatch_indices],
                            batch_returns[minibatch_indices],
                            minibatch_advantages,
                            batch_last_states[minibatch_indices],
                            batch_last_actions[minibatch_indices],
                            batch_history_stack[minibatch_indices],
                            batch_first_steps[minibatch_indices],
                            jax.tree_util.tree_map(lambda a: a[minibatch_indices], d_ppo_batch),
                        )
                        (auxiliary_loss, auxiliary_metrics), (encoder_gradients, decoder_gradients, ncbf_gradients) = auxiliary_grad_loss_fn(
                            encoder_state.params,
                            decoder_state.params,
                            ncbf_state.params,
                            batch_states[minibatch_indices],
                            batch_next_states[minibatch_indices],
                            batch_env_actions[minibatch_indices],
                            batch_last_actions[minibatch_indices],
                            batch_next_state_masks[minibatch_indices],
                            batch_history_stack[minibatch_indices],
                            batch_y_targets[minibatch_indices],
                            batch_indices_to_term[minibatch_indices],
                            batch_ncbf_masks[minibatch_indices],
                            batch_next_env_actions[minibatch_indices],
                            batch_next_y_targets[minibatch_indices],
                            batch_temporal_valid[minibatch_indices],
                            batch_actions[minibatch_indices],
                            replay_states,
                            replay_next_states,
                            replay_actions,
                            replay_y_targets,
                            replay_indices_to_term,
                            replay_ncbf_masks,
                            replay_last_actions,
                            replay_history_stacks,
                            bootstrap_key,
                        )
                        metrics = {**ppo_metrics, **auxiliary_metrics}
                        metrics["loss/ppo_loss"] = ppo_loss
                        if self.d is not None:
                            # D safety net: a non-finite policy / value gradient (from the WM terms) skips this
                            # minibatch's policy and value step instead of poisoning the networks
                            grads_finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in
                                                              jax.tree_util.tree_leaves((policy_gradients, critic_gradients))]))
                            policy_gradients = jax.tree_util.tree_map(lambda g: jnp.where(grads_finite, g, 0.0), policy_gradients)
                            critic_gradients = jax.tree_util.tree_map(lambda g: jnp.where(grads_finite, g, 0.0), critic_gradients)
                            metrics["d/nonfinite_ppo_grad_rate"] = 1.0 - grads_finite.astype(jnp.float32)
                        metrics["loss/auxiliary_loss"] = auxiliary_loss

                        if self.adaptive_lr:
                            ppo_update_enabled = jnp.asarray(True)
                        else:
                            ppo_update_enabled = jnp.logical_or(
                                self.target_kl <= 0.0,
                                metrics["policy_ratio/approx_kl"] <= self.target_kl,
                            )

                        policy_raw_grad_norm = optax.global_norm(policy_gradients)
                        encoder_grad_norm = optax.global_norm(encoder_gradients)

                        policy_state = jax.lax.cond(
                            ppo_update_enabled,
                            lambda state: state.apply_gradients(grads=policy_gradients),
                            lambda state: state,
                            policy_state,
                        )
                        critic_state = critic_state.apply_gradients(grads=critic_gradients)
                        encoder_state = encoder_state.apply_gradients(grads=encoder_gradients)
                        decoder_state = decoder_state.apply_gradients(grads=decoder_gradients)
                        ncbf_state = ncbf_state.apply_gradients(grads=ncbf_gradients)

                        metrics["gradients/policy_grad_norm"] = jnp.where(ppo_update_enabled, policy_raw_grad_norm, 0.0)
                        metrics["gradients/policy_raw_grad_norm"] = policy_raw_grad_norm
                        metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)
                        metrics["gradients/encoder_grad_norm"] = encoder_grad_norm
                        metrics["gradients/decoder_grad_norm"] = optax.global_norm(decoder_gradients)
                        metrics["gradients/ncbf_grad_norm"] = optax.global_norm(ncbf_gradients)
                        metrics["policy_ratio/update_enabled"] = ppo_update_enabled.astype(jnp.float32)

                        carry = (policy_state, critic_state, encoder_state, decoder_state, ncbf_state, key)
                        return carry, metrics

                    init_carry = (policy_state, critic_state, encoder_state, decoder_state, ncbf_state, key)
                    carry, optimization_metrics = jax.lax.scan(minibatch_update, init_carry, batch_indices)
                    policy_state, critic_state, encoder_state, decoder_state, ncbf_state, key = carry

                    if self.adaptive_lr and self.adaptive_lr_target_kl > 0.0:
                        current_lr = policy_state.opt_state[1].hyperparams["learning_rate"]
                        next_lr = _adapt_learning_rate_from_kl(
                            current_lr,
                            jnp.mean(optimization_metrics["policy_ratio/approx_kl"]),
                            self.adaptive_lr_target_kl,
                            self.adaptive_lr_kl_margin,
                            self.adaptive_lr_scale,
                            self.adaptive_lr_gain,
                            self.adaptive_lr_min,
                            self.adaptive_lr_max,
                        )
                        policy_state = _replace_optimizer_learning_rate(policy_state, next_lr)
                        critic_state = _replace_optimizer_learning_rate(critic_state, next_lr)
                        optimization_metrics["lr/adaptive_learning_rate"] = next_lr
                        optimization_metrics["lr/adaptive_lr_multiplier"] = next_lr / (current_lr + 1e-12)
                    optimization_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
                    optimization_metrics["v_value/explained_variance"] = 1 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
                    optimization_metrics["policy/std_dev"] = jnp.mean(jnp.exp(maybe_clip_logstd(policy_state.params["params"]["policy_logstd"], self.policy_logstd_clip)))
                    optimization_metrics["policy_ratio/initial_approx_kl"] = initial_approx_kl
                    optimization_metrics["policy_ratio/initial_abs_logratio"] = initial_abs_logratio
                    optimization_metrics["rollout/advantages_mean"] = jnp.mean(advantages)
                    optimization_metrics["rollout/advantages_std"] = jnp.std(advantages)
                    optimization_metrics["rollout/returns_mean"] = jnp.mean(returns)
                    optimization_metrics["rollout/returns_std"] = jnp.std(returns)

                    # Logging
                    combined_learning_iteration_step = (multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step + 1
                    steps_metrics = {
                        "steps/nr_env_steps": combined_learning_iteration_step * self.nr_steps * self.nr_envs,
                        "steps/nr_updates": combined_learning_iteration_step * self.nr_epochs * self.nr_minibatches,
                    }

                    combined_metrics = {**infos, **steps_metrics, **optimization_metrics, **ncbf_metrics, **d_metrics}
                    combined_metrics = tree.map_structure(lambda x: jnp.mean(x), combined_metrics)

                    def callback(carry):
                        metrics, parallel_seed_id = carry
                        current_time = time.time()
                        metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (current_time - self.last_time[parallel_seed_id]))
                        metrics["time/iteration_seconds"] = current_time - self.last_time[parallel_seed_id]
                        self.last_time[parallel_seed_id] = current_time
                        global_step = int(metrics["steps/nr_env_steps"])
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        # wandb.log(metrics, step=global_step)
                        self.end_logging()

                    jax.debug.callback(callback, (combined_metrics, parallel_seed_id))

                    ncbf_replay_buffer = ncbf_neg_buffer

                    if self.resume:
                        def resume_callback(policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_curriculum, nr_completed_updates, d_persist):
                            self._maybe_save_resume_checkpoint(policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_curriculum, nr_completed_updates,
                                                               d_persist=d_persist if self.d is not None else None)
                        env_curriculum = {
                            k: env_state.internal_state[k]
                            for k in ("env_curriculum_coeff", "env_curriculum_levels_in_a_row")
                            if k in env_state.internal_state
                        }
                        jax.debug.callback(resume_callback, policy_state, critic_state, ncbf_state, encoder_state, decoder_state,
                                           env_curriculum, combined_learning_iteration_step,
                                           self.d.persistent(d_state) if self.d is not None else {})

                    return (policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, ncbf_replay_buffer, key, d_state), None

                key, subkey = jax.random.split(key)
                learning_iteration_carry, _ = jax.lax.scan(learning_iteration, (policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, ncbf_replay_buffer, subkey, d_state), jnp.arange(first_learning_iteration_step, self.nr_updates_per_multi_learning_iteration))
                policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, ncbf_replay_buffer, key, d_state = learning_iteration_carry

                # Evaluating
                if self.evaluation_active:
                    def single_eval_rollout(single_eval_rollout_carry, _):
                        policy_state, ncbf_state, eval_env_state, eval_stats = single_eval_rollout_carry
                        latent_z = encoder_state.apply_fn(encoder_state.params, eval_env_state.history_stack)
                        policy_decoder_output = self._get_policy_decoder_output(
                            decoder_state.params,
                            latent_z,
                            eval_env_state.last_state,
                            eval_env_state.last_action,
                            eval_env_state.next_observation,
                        )
                        eval_observation_input = self._mask_policy_ball_plate_inputs(eval_env_state.next_observation)
                        action_mean, _ = self.policy.apply(policy_state.params, eval_observation_input, latent_z, policy_decoder_output)
                        action = action_mean
                        raw_processed_action = self.get_processed_action(action)
                        params_stack = repeat_ncbf_params(ncbf_state)
                        processed_action, eval_constraint_active, eval_delta_u, _ = self.batched_ncbf_safety_layer(
                            raw_processed_action,
                            eval_env_state.next_observation,
                            eval_env_state.last_action,
                            eval_env_state.last_state,
                            latent_z,
                            jnp.asarray(1.0, dtype=raw_processed_action.dtype),
                            params_stack,
                            episode_first_step(eval_env_state),
                        )
                        processed_action, eval_constraint_active, eval_delta_u, _ = self._bypass_post_drop_safety_layer(
                            raw_processed_action,
                            processed_action,
                            eval_constraint_active,
                            eval_delta_u,
                            {},
                            eval_env_state.next_observation,
                        )
                        eval_env_state = self.env.step(eval_env_state, processed_action)
                        eval_done = eval_env_state.terminated | eval_env_state.truncated
                        eval_stats = {
                            "steps": eval_stats["steps"] + 1.0,
                            "terminations": eval_stats["terminations"] + jnp.sum(eval_env_state.terminated.astype(jnp.float32)),
                            "episodes": eval_stats["episodes"] + jnp.sum(eval_done.astype(jnp.float32)),
                            "interventions": eval_stats["interventions"] + jnp.mean(eval_constraint_active.astype(jnp.float32)),
                            "delta_u": eval_stats["delta_u"] + jnp.mean(eval_delta_u),
                            "collisions": eval_stats["collisions"] + jnp.sum(eval_env_state.info.get("env_info/obstacle_collision", jnp.zeros(self.nr_envs))),
                        }

                        return (policy_state, ncbf_state, eval_env_state, eval_stats), None

                    key, reset_key = jax.random.split(key)
                    reset_keys = jax.random.split(reset_key, self.nr_envs)
                    eval_env_state = self.env.reset(reset_keys, True)
                    zero = jnp.asarray(0.0, dtype=jnp.float32)
                    eval_stats = {"steps": zero, "terminations": zero, "episodes": zero, "interventions": zero, "delta_u": zero, "collisions": zero}
                    single_eval_rollout_carry, _ = jax.lax.scan(single_eval_rollout, (policy_state, ncbf_state, eval_env_state, eval_stats), jnp.arange(self.horizon))
                    _, ncbf_state, eval_env_state, eval_stats = single_eval_rollout_carry

                    nr_eval_steps = jnp.maximum(eval_stats["steps"], 1.0)
                    eval_metrics = {
                        "eval/episode_return": jnp.mean(eval_env_state.info["rollout/episode_return"]),
                        "eval/episode_length": jnp.mean(eval_env_state.info["rollout/episode_length"]),
                        # Fraction of finished episodes that ended in a constraint violation (termination).
                        "eval/failure_rate": eval_stats["terminations"] / jnp.maximum(eval_stats["episodes"], 1.0),
                        "eval/terminations_per_env": eval_stats["terminations"] / self.nr_envs,
                        "eval/intervention_rate": eval_stats["interventions"] / nr_eval_steps,
                        "eval/mean_delta_u": eval_stats["delta_u"] / nr_eval_steps,
                        "eval/collisions_per_env": eval_stats["collisions"] / self.nr_envs,
                    }

                    def callback(metrics_and_global_step):
                        metrics, global_step = metrics_and_global_step
                        global_step = int(global_step)
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging()

                    global_step = (multi_learning_iteration_step + 1) * self.nr_updates_per_multi_learning_iteration * self.nr_steps * self.nr_envs
                    jax.debug.callback(callback, (eval_metrics, global_step))


                # Saving
                if self.save_model:
                    def save_with_check(policy_state, critic_state, ncbf_state, encoder_state, decoder_state, d_persist):
                        self.save(policy_state, critic_state, ncbf_state, encoder_state, decoder_state)
                        if self.d is not None:
                            self.d.save_bundle(f"{self.save_path}/wm_latest.msgpack", d_persist)
                    jax.debug.callback(save_with_check, policy_state, critic_state, ncbf_state, encoder_state, decoder_state,
                                       self.d.persistent(d_state) if self.d is not None else {})


                return (policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, ncbf_replay_buffer, key, d_state), None

            d_state = self.d.init_state() if self.d is not None else {}
            carry = (policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, ncbf_replay_buffer, key, d_state)
            # Resume: finish the partially done multi iteration first, then scan over the remaining ones.
            start_multi, start_learning = divmod(self.resume_start_update, self.nr_updates_per_multi_learning_iteration)
            if start_learning > 0:
                carry, _ = multi_learning_and_eval_save_iteration(carry, jnp.asarray(start_multi), start_learning)
                start_multi += 1
            if start_multi < self.nr_multi_learning_and_eval_save_iterations:
                jax.lax.scan(multi_learning_and_eval_save_iteration, carry, jnp.arange(start_multi, self.nr_multi_learning_and_eval_save_iterations))


        if self.resume:
            self._restore_resume_checkpoint()
            if self.resume_start_update >= self.nr_multi_learning_and_eval_save_iterations * self.nr_updates_per_multi_learning_iteration:
                rlx_logger.info("Resume: training already finished, nothing to do.")
                return
        self.key, subkey = jax.random.split(self.key)
        if self.resume_start_update > 0:
            # Do not replay the random stream of the first attempt.
            subkey = jax.random.fold_in(subkey, self.resume_start_update)
        seed_keys = jax.random.split(subkey, self.nr_parallel_seeds)
        train_function = jax.jit(jax.vmap(jitable_train_function))
        self.last_time = [time.time() for _ in range(self.nr_parallel_seeds)]
        self.start_time = deepcopy(self.last_time)
        jax.block_until_ready(train_function(seed_keys, jnp.arange(self.nr_parallel_seeds)))
        rlx_logger.info(f"Average time: {max([time.time() - t for t in self.start_time]):.2f} s")


    def log(self, name, value, step):
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)



    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)

    def start_logging(self, step):
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self):
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")


    def save(self, policy_state, critic_state, ncbf_state, encoder_state, decoder_state):
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state,
            "ncbf": ncbf_state,
            "encoder": encoder_state,
            "decoder": decoder_state,
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.latest_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as f:
            json.dump(self.config.algorithm.to_dict(), f)
        shutil.make_archive(f"{self.save_path}/{self.latest_model_file_name}", "zip", f"{self.save_path}/tmp")
        os.rename(f"{self.save_path}/{self.latest_model_file_name}.zip", f"{self.save_path}/{self.latest_model_file_name}")
        shutil.rmtree(f"{self.save_path}/tmp")

        if self.track_wandb:
            wandb.save(f"{self.save_path}/{self.latest_model_file_name}", base_path=self.save_path)


    def _resume_checkpoint_target(self):
        target = {
            "policy": self.policy_state,
            "critic": self.critic_state,
            "ncbf": self.ncbf_state,
            "encoder": self.encoder_state,
            "decoder": self.decoder_state,
        }
        if getattr(self, "d", None) is not None:
            target["d"] = self.d.persist0
        return target


    def _maybe_save_resume_checkpoint(self, policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_curriculum, nr_completed_updates, d_persist=None):
        nr_completed_updates = int(np.asarray(nr_completed_updates))
        nr_total_updates = self.nr_multi_learning_and_eval_save_iterations * self.nr_updates_per_multi_learning_iteration
        due_by_iterations = self.save_interval_iterations > 0 and nr_completed_updates - self.resume_last_save_update >= self.save_interval_iterations
        due_by_time = self.save_interval_minutes > 0 and (time.time() - self.resume_last_save_time) >= 60.0 * self.save_interval_minutes
        if not (due_by_iterations or due_by_time or nr_completed_updates >= nr_total_updates):
            return
        self._save_resume_checkpoint(policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_curriculum, nr_completed_updates, d_persist=d_persist)


    def _save_resume_checkpoint(self, policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_curriculum, nr_completed_updates, d_persist=None):
        """Atomic resume checkpoint: write a new step directory, then atomically repoint resume/LATEST to it.
        A kill during the save leaves LATEST on the previous complete checkpoint. Older step dirs are removed
        only after the pointer has been switched."""
        os.makedirs(self.resume_path, exist_ok=True)
        step_dir_name = f"step_{nr_completed_updates:09d}"
        step_dir = os.path.join(self.resume_path, step_dir_name)
        partial_dir = step_dir + ".partial"
        if os.path.exists(partial_dir):
            shutil.rmtree(partial_dir)
        os.makedirs(partial_dir)
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state,
            "ncbf": ncbf_state,
            "encoder": encoder_state,
            "decoder": decoder_state,
        }
        if d_persist is not None:   # method D: WM params, Adam state, normalizers, warm-start state (not the replay)
            checkpoint["d"] = d_persist
        save_args = orbax_utils.save_args_from_target(checkpoint)
        orbax.checkpoint.PyTreeCheckpointer().save(os.path.join(partial_dir, "states"), checkpoint, save_args=save_args)
        np.savez(os.path.join(partial_dir, "env_curriculum.npz"), **{k: np.asarray(v) for k, v in env_curriculum.items()})
        global_step = nr_completed_updates * self.nr_steps * self.nr_envs
        meta = {
            "nr_completed_updates": nr_completed_updates,
            "global_step": global_step,
            "time": time.time(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0),
        }
        with open(os.path.join(partial_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        with open(os.path.join(partial_dir, "config_algorithm.json"), "w") as f:
            json.dump(self.config.algorithm.to_dict(), f)
        if os.path.exists(step_dir):
            shutil.rmtree(step_dir)
        os.rename(partial_dir, step_dir)
        pointer_tmp = os.path.join(self.resume_path, "LATEST.tmp")
        with open(pointer_tmp, "w") as f:
            f.write(step_dir_name)
            f.flush()
            os.fsync(f.fileno())
        os.replace(pointer_tmp, os.path.join(self.resume_path, "LATEST"))
        for name in os.listdir(self.resume_path):
            if name.startswith("step_") and name != step_dir_name:
                shutil.rmtree(os.path.join(self.resume_path, name), ignore_errors=True)
        self.resume_last_save_update = nr_completed_updates
        self.resume_last_save_time = time.time()
        rlx_logger.info(f"Resume checkpoint saved: {step_dir} (update {nr_completed_updates}, step {global_step})")


    def _restore_resume_checkpoint(self):
        """Restore params + optimizer states, the update counter and the env curriculum from resume/LATEST.
        Not restored: the env state (all envs are re-reset), the NCBF negative replay buffer (refilled by the
        next rollout with the default neg_buffer_size=1) and the RNG stream (re-derived from the seed + step).
        Method D: the WM state ("d": params, Adam, normalizers, warm-start state) is restored; its replay ring,
        negative buffer and V^prev are not (they refill from the next rollouts)."""
        restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0)
        pointer = os.path.join(self.resume_path, "LATEST")
        if not os.path.exists(pointer):
            rlx_logger.info(f"Resume: no checkpoint in {self.resume_path}, starting fresh (SLURM_RESTART_COUNT={restart_count})")
            return
        with open(pointer) as f:
            step_dir = os.path.join(self.resume_path, f.read().strip())
        target = self._resume_checkpoint_target()
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpoint = orbax.checkpoint.PyTreeCheckpointer().restore(os.path.join(step_dir, "states"), item=target, restore_args=restore_args)
        self.policy_state = checkpoint["policy"]
        self.critic_state = checkpoint["critic"]
        self.ncbf_state = checkpoint["ncbf"]
        self.encoder_state = checkpoint["encoder"]
        self.decoder_state = checkpoint["decoder"]
        if getattr(self, "d", None) is not None:
            self.d.persist = checkpoint["d"]
        with open(os.path.join(step_dir, "meta.json")) as f:
            meta = json.load(f)
        curriculum_file = os.path.join(step_dir, "env_curriculum.npz")
        if os.path.exists(curriculum_file):
            with np.load(curriculum_file) as data:
                self.resume_env_curriculum = {k: data[k] for k in data.files}
        self.resume_start_update = int(meta["nr_completed_updates"])
        self.resume_last_save_update = self.resume_start_update
        self.resume_last_save_time = time.time()
        rlx_logger.info(f"Resume: restored {step_dir} (update {self.resume_start_update}, step {meta['global_step']}, "
                        f"SLURM_RESTART_COUNT={restart_count})")
        if self.track_wandb:
            # Config, not a logged metric: wandb drops metrics whose step is below the run's last logged step.
            history = list(wandb.config.get("resume_history", []) or [])
            history.append({"restored_step": int(meta["global_step"]), "slurm_restart_count": restart_count,
                            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "")})
            wandb.config.update({"resume_history": history, "resume_nr_restores": len(history)}, allow_val_change=True)


    def load(config, env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(splitted_path[:-1]))
        checkpoint_file_name = splitted_path[-1]
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"

        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        runtime_ncbf_params = {
            "use_safety_layer",
            "gamma_c",
            "eta_cbf",
            "lambda_slack",
            "action_clipping",
            "max_delta_u",
            "safety_layer_projection",
            "safety_layer_min_grad_norm",
            "post_check_actual_residual",
            "safety_layer_std_coeff_start",
            "safety_layer_std_coeff_final",
            "gamma_c_scaled_by_eta",
            "tighten_uncertainty",
            "slack_uncertainty",
        }
        for key, value in loaded_algorithm_config.items():
            if isinstance(value, dict):
                if key not in config.algorithm or key in ("eval_sweep", "wm_collect"):  # eval / data settings come from the command line only
                    continue
                parent = config.algorithm[key]
                for sub_key, sub_value in value.items():
                    full_param = f"algorithm.{key}.{sub_key}"
                    if full_param in explicitly_set_algorithm_params:
                        continue
                    if key == "ncbf" and sub_key in runtime_ncbf_params:
                        continue
                    if sub_key in parent:
                        parent[sub_key] = sub_value
                continue
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value

        if "policy_layer_norm" not in loaded_algorithm_config and "algorithm.policy_layer_norm" not in explicitly_set_algorithm_params:
            # Checkpoints written before the T06c switch (2026-10-26) have a policy without LayerNorm; the default is now True.
            config.algorithm.policy_layer_norm = False
        loaded_ncbf_config = loaded_algorithm_config.get("ncbf", {})
        checkpoint_family = checkpoint_code_family(loaded_algorithm_config)
        if checkpoint_family == "paper":
            raise ValueError(
                "This checkpoint was written by the paper code (ncbf-q family: 5 separate critic TrainStates, no "
                "history encoder/decoder); its parameters cannot be restored into this PPO. Load it with the paper "
                "code, e.g. eval_sweep's PaperAgent. Its behavior corresponds to legacy_ncbf_defaults('paper'): "
                f"{legacy_ncbf_defaults(checkpoint_family)}")
        for sub_key, legacy_value in legacy_ncbf_defaults(checkpoint_family).items():
            if sub_key not in loaded_ncbf_config and f"algorithm.ncbf.{sub_key}" not in explicitly_set_algorithm_params:
                rlx_logger.info(f"Legacy checkpoint: setting algorithm.ncbf.{sub_key}={legacy_value}")
                config.algorithm.ncbf[sub_key] = legacy_value
        # Method D checkpoints load as plain policies (mode A eval, wm_collect): the WM / shield is training-only and its
        # bundle is models/wm_latest.msgpack. --algorithm.method=d on the command line keeps it (needs record_wm_state).
        if config.algorithm.get("method", "default") == "d" and "algorithm.method" not in explicitly_set_algorithm_params:
            config.algorithm.method = "default"
        model = PPO(config, env, run_path, writer)

        target_ncbf = model.ncbf_state
        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "ncbf": target_ncbf,
            "encoder": model.encoder_state,
            "decoder": model.decoder_state,
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.ncbf_state = checkpoint["ncbf"]
        model.encoder_state = checkpoint["encoder"]
        model.decoder_state = checkpoint["decoder"]

        shutil.rmtree(checkpoint_dir)

        return model


    def _policy_and_shield(self, env_state):
        """Deterministic policy action before and after the safety shield (evaluation, beta = 1)."""
        latent_z = self.encoder_state.apply_fn(self.encoder_state.params, env_state.history_stack)
        policy_decoder_output = self._get_policy_decoder_output(
            self.decoder_state.params,
            latent_z,
            env_state.last_state,
            env_state.last_action,
            env_state.next_observation,
        )
        observation_input = self._mask_policy_ball_plate_inputs(env_state.next_observation)
        action_mean, _ = self.policy.apply(self.policy_state.params, observation_input, latent_z, policy_decoder_output)
        raw_processed_action = self.get_processed_action(action_mean)
        processed_action, constraint_active, delta_u, _ = self.batched_ncbf_safety_layer(
            raw_processed_action,
            env_state.next_observation,
            env_state.last_action,
            env_state.last_state,
            latent_z,
            jnp.asarray(1.0, dtype=raw_processed_action.dtype),
            self._ncbf_params_stack(self.ncbf_state.params),
            episode_first_step(env_state),
        )
        processed_action, constraint_active, delta_u, _ = self._bypass_post_drop_safety_layer(
            raw_processed_action,
            processed_action,
            constraint_active,
            delta_u,
            {},
            env_state.next_observation,
        )
        return raw_processed_action, processed_action, constraint_active, delta_u, latent_z


    def _shielded_action(self, env_state):
        """Deterministic policy action passed through the safety shield (evaluation, beta = 1)."""
        _, processed_action, constraint_active, delta_u, _ = self._policy_and_shield(env_state)
        return processed_action, constraint_active, delta_u


    def _safety_prediction(self, observation, action, latent_z):
        """Critic prediction for (s, a): per-head safe probability [B, K] (ensemble mean), and the mean and std
        over ensemble members of the probability that every head stays safe (min over heads) [B]."""
        inputs = jnp.concatenate([observation[..., self.ncbf_observation_indices], action, latent_z], axis=-1)
        probabilities = self._ncbf_probabilities(self._ncbf_apply_members(self.ncbf_state.params, inputs))  # [E, B, K]
        all_heads = jnp.min(probabilities, axis=-1)  # [E, B]
        return jnp.mean(probabilities, axis=0), jnp.mean(all_heads, axis=0), jnp.std(all_heads, axis=0)


    def _branch_outcome(self, env_state, first_action, horizon):
        """Take first_action, then follow the continuation policy for horizon - 1 steps. Returns, per env,
        whether a safety target dropped below 0.5 within the episode (overall and per head), the step of the
        first failure (horizon if none), and whether the episode was truncated first (censored)."""
        target_indices = jnp.asarray(self.env.ncbf_target_indices)
        nr_heads = target_indices.shape[0]
        batch = first_action.shape[0]

        def record(env_state, flags, step_index):
            failed, failed_heads, first_failure, alive, censored = flags
            unsafe_heads = (env_state.actual_next_observation[..., target_indices] < 0.5) & alive[:, None]
            unsafe = jnp.any(unsafe_heads, axis=-1)
            first_failure = jnp.where(unsafe & ~failed, step_index, first_failure)
            failed = failed | unsafe
            failed_heads = failed_heads | unsafe_heads
            truncated_only = env_state.truncated & ~env_state.terminated & ~failed
            censored = censored | (alive & truncated_only & (step_index < horizon - 1))
            alive = alive & ~(env_state.terminated | env_state.truncated)
            return failed, failed_heads, first_failure, alive, censored

        flags = (jnp.zeros(batch, dtype=bool), jnp.zeros((batch, nr_heads), dtype=bool),
                 jnp.full(batch, horizon, dtype=jnp.int32), jnp.ones(batch, dtype=bool), jnp.zeros(batch, dtype=bool))
        env_state = self.env.step(env_state, first_action)
        flags = record(env_state, flags, 0)

        def continue_step(carry, step_index):
            env_state, flags = carry
            raw_action, safe_action, _, _, _ = self._policy_and_shield(env_state)
            action = safe_action if self.counterfactual_continuation == "shielded" else raw_action
            env_state = self.env.step(env_state, action)
            return (env_state, record(env_state, flags, step_index)), None

        (_, flags), _ = jax.lax.scan(continue_step, (env_state, flags), jnp.arange(1, horizon))
        failed, failed_heads, first_failure, _, censored = flags
        return {"failed": failed, "failed_heads": failed_heads, "first_failure_step": first_failure, "censored": censored}


    def counterfactual_check(self):
        """E1: is each intervention of the shield necessary and effective, and does the critic predict the outcome?

        Along a shielded rollout, every `interval` steps, branch from the same state (same env randomness) with the
        raw policy action and with the shielded action, continue with the continuation policy, and record failures
        within `horizon` steps. Writes counterfactual_summary.json and counterfactual_records.npz next to the models.
        """
        horizon = self.counterfactual_horizon if self.counterfactual_horizon > 0 else self.ncbf_H
        nr_decisions, interval, nr_repeats = self.counterfactual_nr_decision_points, self.counterfactual_interval, self.counterfactual_nr_repeats
        rlx_logger.info(f"Counterfactual check: {self.nr_envs} envs x {nr_decisions} decision points, interval {interval}, "
                        f"horizon {horizon}, {nr_repeats} repeat(s), continuation {self.counterfactual_continuation}, "
                        f"use_safety_layer={self.ncbf_use_safety_layer}.")
        self.key, subkey = jax.random.split(self.key)
        env_state = self.env.reset(jax.random.split(subkey, self.nr_envs), True)

        def shielded_step(env_state, _):
            _, safe_action, _, _, _ = self._policy_and_shield(env_state)
            return self.env.step(env_state, safe_action), None

        def with_repeat_key(env_state, repeat_index):
            return env_state.replace(key=jax.vmap(lambda k: jax.random.fold_in(k, repeat_index))(env_state.key))

        def decision_point(env_state, _):
            env_state, _ = jax.lax.scan(shielded_step, env_state, None, length=interval)
            raw_action, safe_action, active, delta_u, latent_z = self._policy_and_shield(env_state)
            heads_raw, p_raw, p_raw_std = self._safety_prediction(env_state.next_observation, raw_action, latent_z)
            heads_safe, p_safe, p_safe_std = self._safety_prediction(env_state.next_observation, safe_action, latent_z)

            def repeat(_, repeat_index):
                branch_state = with_repeat_key(env_state, repeat_index)
                return None, (self._branch_outcome(branch_state, raw_action, horizon), self._branch_outcome(branch_state, safe_action, horizon))

            _, (outcome_raw, outcome_safe) = jax.lax.scan(repeat, None, jnp.arange(nr_repeats))
            record = {
                "active": active, "delta_u": delta_u,
                "p_raw": p_raw, "p_raw_std": p_raw_std, "p_heads_raw": heads_raw,
                "p_safe": p_safe, "p_safe_std": p_safe_std, "p_heads_safe": heads_safe,
                **{f"{name}_raw": value for name, value in outcome_raw.items()},
                **{f"{name}_safe": value for name, value in outcome_safe.items()},
            }
            return env_state, record

        @jax.jit
        def run(env_state):
            _, records = jax.lax.scan(decision_point, env_state, None, length=nr_decisions)
            return records

        records = {name: np.asarray(value) for name, value in jax.device_get(run(env_state)).items()}
        summary = summarize_counterfactual(records, self.ncbf_target_names)
        summary.update({
            "nr_envs": self.nr_envs, "nr_decision_points": nr_decisions, "interval": interval, "horizon": horizon,
            "nr_repeats": nr_repeats, "continuation": self.counterfactual_continuation,
            "use_safety_layer": self.ncbf_use_safety_layer,
            "gamma_c": float(self.config.algorithm.ncbf.gamma_c), "eta_cbf": float(self.config.algorithm.ncbf.eta_cbf),
        })
        summary = _nan_to_none(summary)
        rlx_logger.info("Counterfactual summary: " + json.dumps(summary))
        run_path = os.path.dirname(self.save_path)
        os.makedirs(run_path, exist_ok=True)
        with open(os.path.join(run_path, "counterfactual_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        np.savez_compressed(os.path.join(run_path, "counterfactual_records.npz"), **records)
        return summary


    def test(self, episodes):
        self.key, subkey = jax.random.split(self.key)
        reset_keys = jax.random.split(subkey, self.nr_envs)
        env_state = self.env.reset(reset_keys, True)

        if self.render:
            rlx_logger.info("Rendering runs infinitely. The episodes parameter is ignored.")

            @jax.jit
            def rollout(env_state):
                processed_action, _, _ = self._shielded_action(env_state)
                return self.env.step(env_state, processed_action)

            while True:
                env_state = rollout(env_state)
                env_state = self.env.render(env_state)

        if self.counterfactual_enabled:
            return self.counterfactual_check()

        if self.config.algorithm.get("wm_collect", {}).get("enabled", False):
            from rl_x.algorithms.ncbf_ppo.flax_full_jit.wm_collect import run_wm_collect
            return run_wm_collect(self)

        if self.config.algorithm.get("eval_sweep", {}).get("enabled", False):
            from rl_x.algorithms.ncbf_ppo.flax_full_jit.eval_sweep import run_eval_sweep
            return run_eval_sweep(self)

        # Batch evaluation: run episodes * horizon steps on all parallel envs and write a summary.
        nr_eval_steps = int(max(1, episodes)) * self.horizon
        rlx_logger.info(f"Evaluating {self.nr_envs} envs for {nr_eval_steps} steps (use_safety_layer={self.ncbf_use_safety_layer}).")

        def eval_step(carry, _):
            env_state, stats = carry
            processed_action, constraint_active, delta_u = self._shielded_action(env_state)
            env_state = self.env.step(env_state, processed_action)
            done = (env_state.terminated | env_state.truncated).astype(jnp.float32)
            terminated = env_state.terminated.astype(jnp.float32)
            collision = env_state.info.get("env_info/obstacle_collision", jnp.zeros(self.nr_envs))
            stats = {
                "steps": stats["steps"] + 1.0,
                "episodes": stats["episodes"] + jnp.sum(done),
                "terminations": stats["terminations"] + jnp.sum(terminated),
                "collisions": stats["collisions"] + jnp.sum(collision),
                "return_sum": stats["return_sum"] + jnp.sum(done * env_state.info["rollout/episode_return"]),
                "length_sum": stats["length_sum"] + jnp.sum(done * env_state.info["rollout/episode_length"]),
                "interventions": stats["interventions"] + jnp.mean(constraint_active.astype(jnp.float32)),
                "delta_u": stats["delta_u"] + jnp.mean(delta_u),
            }
            return (env_state, stats), None

        @jax.jit
        def run_evaluation(env_state):
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            stats = {name: zero for name in ("steps", "episodes", "terminations", "collisions", "return_sum", "length_sum", "interventions", "delta_u")}
            (_, stats), _ = jax.lax.scan(eval_step, (env_state, stats), None, length=nr_eval_steps)
            return stats

        stats = {name: float(value) for name, value in jax.device_get(run_evaluation(env_state)).items()}
        nr_episodes = max(stats["episodes"], 1.0)
        summary = {
            "nr_envs": self.nr_envs,
            "nr_steps": int(stats["steps"]),
            "nr_finished_episodes": int(stats["episodes"]),
            "failure_rate": stats["terminations"] / nr_episodes,
            "terminations_per_env": stats["terminations"] / self.nr_envs,
            "collisions_per_env": stats["collisions"] / self.nr_envs,
            "mean_episode_return": stats["return_sum"] / nr_episodes,
            "mean_episode_length": stats["length_sum"] / nr_episodes,
            "intervention_rate": stats["interventions"] / max(stats["steps"], 1.0),
            "mean_delta_u": stats["delta_u"] / max(stats["steps"], 1.0),
            "use_safety_layer": self.ncbf_use_safety_layer,
            "gamma_c": float(self.config.algorithm.ncbf.gamma_c),
            "eta_cbf": float(self.config.algorithm.ncbf.eta_cbf),
        }
        rlx_logger.info("Evaluation summary: " + json.dumps(summary))
        run_path = os.path.dirname(self.save_path)
        os.makedirs(run_path, exist_ok=True)
        with open(os.path.join(run_path, "eval_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary


    def general_properties():
        return GeneralProperties
