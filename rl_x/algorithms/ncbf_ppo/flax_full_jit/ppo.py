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
from rl_x.algorithms.ncbf_ppo.flax_full_jit.ncbf import get_ncbf
from rl_x.algorithms.ncbf_ppo.flax_full_jit.history_encoder import get_history_encoder
from rl_x.algorithms.ncbf_ppo.flax_full_jit.decoder import get_decoder


rlx_logger = logging.getLogger("rl_x")


class PPO:
    def __init__(self, config, env, run_path, writer):
        self.config = config
        self.env = env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
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

        self.next_step_predictor_pretrain_steps = config.algorithm.next_step_predictor.pretrain_nr_steps * self.nr_steps
        self.next_step_predictor_lr = config.algorithm.next_step_predictor.lr
        self.next_step_predictor_output_indices = env.next_state_indices
        self.next_step_predictor_pretrain_nr_minibatches = config.algorithm.next_step_predictor.pretrain_nr_minibatches
        self.next_step_predictor_nr_minibatches = config.algorithm.next_step_predictor.nr_minibatches
        self.next_step_predictor_minibatch_size = config.algorithm.minibatch_size

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
        self.ncbf_minibatch_size = config.algorithm.minibatch_size
        self.ncbf_nr_minibatches = config.algorithm.ncbf.nr_minibatches
        self.ncbf_coef_decay_lambda = config.algorithm.ncbf.coef_decay_lambda

        self.ncbf_pos_buffer_size = config.algorithm.ncbf_buffer.pos_buffer_size * self.nr_steps * self.nr_envs
        self.ncbf_neg_buffer_size = config.algorithm.ncbf_buffer.neg_buffer_size * self.nr_steps * self.nr_envs
        self.ncbf_neg_sampling_ratio = config.algorithm.ncbf_buffer.neg_sampling_ratio
        self.ncbf_replay_neg_minibatch_size = int(self.minibatch_size * self.ncbf_neg_sampling_ratio)

        self.ncbf_pretrain_steps = config.algorithm.ncbf.pretrain.nr_steps * self.nr_steps
        self.ncbf_observation_indices = env.ncbf_observation_indices

        self.nr_pretrain_steps = max(self.next_step_predictor_pretrain_steps, self.ncbf_pretrain_steps)

        rlx_logger.info(f"NCBF pretrain steps:{self.ncbf_pretrain_steps * self.nr_envs}")
        rlx_logger.info(f"INFO - NCBF pos buffer size:{self.ncbf_pos_buffer_size}")
        rlx_logger.info(f"INFO - NCBF neg buffer size:{self.ncbf_neg_buffer_size}")

        self.ncbf_pretrain_nr_minibatches = config.algorithm.ncbf.pretrain.nr_minibatches

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
        self.nr_history_steps = env.nr_history_steps

        if self.evaluation_and_save_frequency % self.batch_size != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of batch size")

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

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        env_state = self.env.reset(reset_key, False)

        dummy_latent = jnp.zeros((1, self.encoder.hidden_size))

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, env_state.next_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, env_state.next_observation),
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
        self.ncbf_state = [
            TrainState.create(
                apply_fn=self.ncbf[i].apply,
                params=self.ncbf[i].init(ncbf_keys[i], dummy_ncbf_input),
                tx=optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.inject_hyperparams(optax.adam)(learning_rate=config.algorithm.ncbf.lr),
                )
            )
            for i in range(self.ncbf_n_ensemble)
        ]

        if self.save_model:
            os.makedirs(self.save_path)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    def train(self):
        def jitable_train_function(key, parallel_seed_id):
            def repeat_ncbf_params(ncbf_state):
                return jax.tree_util.tree_map(
                    lambda x: jnp.repeat(x[None, ...], self.ncbf_n_ensemble, axis=0),
                    ncbf_state.params,
                )

            # Single step rollout, used for training and ncbf pretraining, therefore defined at first
            def single_rollout(single_rollout_carry, _):
                policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, key = single_rollout_carry

                key, subkey = jax.random.split(key)
                observation = env_state.next_observation
                history_stack = env_state.history_stack
                last_action = env_state.last_action
                last_state = env_state.last_state
                latent_z = encoder_state.apply_fn(encoder_state.params, history_stack)
                action_mean, action_logstd = self.policy.apply(policy_state.params, observation)
                action_logstd = jnp.clip(action_logstd, -5.0, 2.0)
                action_std = jnp.exp(action_logstd)
                action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                log_prob = (-0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(
                    2.0 * jnp.pi) - action_logstd).sum(1)
                raw_processed_action = self.get_processed_action(action)

                params_stack = repeat_ncbf_params(ncbf_state)

                processed_action, constraint_active, delta_u = self.batched_ncbf_safety_layer(raw_processed_action, observation, last_action, last_state, latent_z,
                                                                                      params_stack)
                # contraint_active = 0
                # delta_u = 0
                value = self.critic.apply(critic_state.params, observation).squeeze(-1)

                env_state = self.env.step(env_state, processed_action)
                done = env_state.terminated | env_state.truncated
                transition = (observation, env_state.actual_next_observation, action, env_state.reward, value,
                              env_state.terminated, done, log_prob, env_state.info, constraint_active, delta_u, last_state, last_action, history_stack)

                if self.render:
                    def render(env_state):
                        return self.env.render(env_state)

                    env_state = jax.experimental.io_callback(render, env_state, env_state)

                return (policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, key), transition

            @jax.jit
            def window_any_done_next_H(dones: jnp.ndarray, terminates: jnp.ndarray):
                """
                dones, terminates: [T, N_env] bool
                Returns:
                  y:    [T, N] bool   True if NO terminate in (t, t+H], else False
                  mask: [T, N] bool   True if NO (done & ~terminate) in (t, t+H], else False
                  dists_to_next_term: [T, N] int32  distance to next terminate at/after t within same episode
                                       0 means terminate at t; cap (=H+1) means none within episode horizon
                """
                H = self.ncbf_H  # must be a Python int for good jit behavior
                cap = jnp.int32(H + 10)

                dones = dones.at[-1, :].set(True)

                def _future_terms_within_H(events: jnp.ndarray, dones_boundary: jnp.ndarray):
                    def body(dist_prev, inp):
                        ev_t, done_t = inp

                        # Reset distance when crossing episode boundary
                        dist_prev = jnp.where(done_t, cap, dist_prev)

                        dist_t = jnp.where(ev_t, 0, jnp.minimum(dist_prev + 1, cap))
                        return dist_t, dist_t

                    init = jnp.full((events.shape[1],), cap, dtype=jnp.int32)
                    # Scan BACKWARD then reverse output
                    _, dists_rev = jax.lax.scan(body, init, (events[::-1], dones_boundary[::-1]))
                    dists = dists_rev[::-1]

                    any_next_H = jnp.logical_and(dists >= 0, dists <= H)
                    return any_next_H, dists

                any_term_next_H, dists_to_next_term = _future_terms_within_H(terminates, dones)
                y = ~any_term_next_H

                def _future_trunc_within_H(terms, truncs):
                    # set mask to false if any truncation (either timely trauncation or at the end of each env rollouts)
                    def body(dist_prev, inp):
                        term_t, trunc_t = inp

                        # Reset distance when crossing episode boundary
                        dist_prev = jnp.where(trunc_t, cap, dist_prev)

                        # if term, set to cap t maintain
                        dist_prev = jnp.where(term_t, cap, dist_prev)

                        dist_t = jnp.where(trunc_t, 0, jnp.minimum(dist_prev + 1, cap))
                        return dist_t, dist_t

                    init = jnp.full((truncs.shape[1],), cap, dtype=jnp.int32)
                    # Scan BACKWARD then reverse output
                    _, dists_rev = jax.lax.scan(body, init, (terms[::-1], truncs[::-1]))
                    dists = dists_rev[::-1]

                    any_next_H = jnp.logical_and(dists >= 0, dists <= H)
                    return any_next_H, dists

                trunc_done = dones & (~terminates)
                any_done_next_H, _ = _future_trunc_within_H(terminates, trunc_done)
                mask = ~any_done_next_H
                return y, mask, dists_to_next_term

            @jax.jit
            def get_receding_min_target(future_states: jnp.ndarray,
                                        dones: jnp.ndarray,
                                        terminates: jnp.ndarray):
                """
                future_states: [T, N, D], actual next observations for each sampled transition
                dones:         [T, N] bool
                terminates:    [T, N] bool

                Returns:
                  target: [T, N, K]   receding-horizon min target
                  safe_y: [T, N] if state is receding-horizon-safe
                  y:      [T, N] bool same as your current y
                  mask:   [T, N] bool same as your current mask
                """
                H = self.ncbf_H
                idx = jnp.asarray(self.env.ncbf_target_indices)

                # [T, N, K], where K = number of target state dims
                vals = future_states[..., idx]

                y, mask, dists_to_next_term = window_any_done_next_H(dones, terminates)

                T, N, K = vals.shape
                offsets = jnp.arange(H, dtype=jnp.int32)  # [H]
                time_idx = jnp.arange(T, dtype=jnp.int32)[:, None] + offsets[None, :]  # [T, H]

                # Pad with +inf so that windows going beyond T do not affect the min
                pad = jnp.full((H - 1, N, K), jnp.inf, dtype=vals.dtype)
                vals_pad = jnp.concatenate([vals, pad], axis=0)  # [T+H-1, N, K]

                # Gather rolling windows: [T, H, N, K]
                windows = vals_pad[time_idx]

                # Effective horizon length:
                # include the terminal state if termination occurs within the horizon
                eff_len = jnp.where(dists_to_next_term < H,
                                    dists_to_next_term + 1,
                                    H).astype(jnp.int32)  # [T, N]

                # valid[o, t, n] = whether offset o is inside the allowed horizon
                valid = offsets[None, :, None] < eff_len[:, None, :]  # [T, H, N]

                # Mask invalid entries with +inf so min ignores them
                windows = jnp.where(valid[..., None], windows, jnp.inf)

                # Min over time -> [T, N, K]
                target = jnp.min(windows, axis=(1))

                return target, y, mask, dists_to_next_term

            @jax.jit
            def ncbf_loss_fn(params, minib_obs, minib_nxt, minib_acts, minib_y, minib_mask, minib_indices_to_term,
                             last_states, last_actions, minib_latent, minib_latent_next):
                # vmap_apply = jax.vmap(ncbf_state.apply_fn, in_axes=(None, 0))
                h_input = jnp.concatenate([minib_obs[..., self.ncbf_observation_indices], last_actions], axis=-1)
                h_input_next = jnp.concatenate([minib_nxt[..., self.ncbf_observation_indices], minib_acts], axis=-1)

                h_input = jnp.concatenate((h_input, minib_latent), axis=-1)
                h_input_next = jnp.concatenate((h_input_next, minib_latent_next), axis=-1)

                h_x_logits = ncbf_state[0].apply_fn(params, h_input)  # [B]
                h_xn_logits = ncbf_state[0].apply_fn(params, h_input_next)  # [B]

                h_x = nn.sigmoid(h_x_logits)
                h_xn = nn.sigmoid(h_xn_logits)
                
                minib_y = minib_y.astype(h_x.dtype)
                minib_mask_head = minib_mask[..., None].astype(h_x.dtype)

                # (1) BCE classification
                k = jnp.clip(minib_indices_to_term, 0, self.ncbf_H)
                coef = self.ncbf_coef_decay_lambda ** k
                # use the decaying coef only for negtive samples
                coef = jnp.where(minib_y < 0.5, coef[..., None], 1.0)

                bce = optax.sigmoid_binary_cross_entropy(h_x_logits, minib_y)
                head_loss = jnp.mean(coef * bce, axis=-1)
                num = jnp.sum(minib_mask * head_loss, axis=-1)
                den = jnp.sum(minib_mask) + 1e-8
                cls_loss = num / den

                # (2) DT-CBF penalty: if not safe, h(x_next) should further decrease
                # check if this is true
                # cbf_decrease = h_xn - h_x
                # neg_mask = (1.0 - minib_y) * minib_mask  # only apply to unsafe samples
                #
                # violation = jnp.maximum(0.0, cbf_decrease)  # ReLU(h_xn - h_x)
                # neg_cbf_loss = jnp.sum(neg_mask * violation) / (jnp.sum(neg_mask) + 1e-8)
                neg_cbf_loss = 0

                # if both x and x_n is safe, apply cbf to encourage h(x) >= h(x_n)
                pos_mask = minib_y * minib_mask_head * (minib_indices_to_term > self.ncbf_H)[..., None]
                cbf_decrease = h_xn - h_x
                cbf_decrease = jnp.maximum(0.0, cbf_decrease)  # ReLU(h_x - h_xn)
                pos_cbf_loss = jnp.sum(pos_mask * cbf_decrease) / (jnp.sum(pos_mask) + 1e-8)
                cbf_loss = neg_cbf_loss + pos_cbf_loss

                # (3) Lipschitz regularizer: fixed-size pair sampling from valid positions
                @jax.jit
                def f_single(x_single):
                    # shape (output_dim,) -> reduce to scalar
                    x_single = x_single[None, ...]
                    y = ncbf_state[0].apply_fn(params, x_single)
                    return jnp.sum(y)

                # Vectorize grad over batch
                max_norm = self.ncbf_L_target
                grad_h = jax.grad(f_single)(h_input)  # [D]
                grad_norm = jnp.linalg.norm(grad_h)  # [1]
                lip_violation = jnp.maximum(0.0, grad_norm - max_norm)
                lip_loss = jnp.mean(lip_violation)

                # (4) weight decay
                wd_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params))

                total = (self.ncbf_w_clf * cls_loss +
                         self.ncbf_w_cbf * cbf_loss +
                         self.ncbf_w_lip * lip_loss +
                         self.ncbf_w_wd * wd_loss)

                # calculate mse of ncbf prediction for logging, also calculate mse for y is true or false separately
                valid = minib_mask_head.astype(jnp.float32)
                p_safe = h_x
                sq_err = jnp.square(p_safe - minib_y)
                mse_all = jnp.sum(valid * sq_err) / (jnp.sum(valid) + 1e-8)

                pos_mask = valid * (minib_y > 0.5)
                mse_pos = jnp.sum(pos_mask * sq_err) / (jnp.sum(pos_mask) + 1e-8)

                neg_mask = valid * (minib_y <= 0.5)
                mse_neg = jnp.sum(neg_mask * sq_err) / (jnp.sum(neg_mask) + 1e-8)

                # weighted mse neg
                mse_neg_weighted = jnp.sum(neg_mask * sq_err * coef) / (jnp.sum(neg_mask) + 1e-8)

                # for debug purposes
                coef_neg = jnp.where(minib_y <= 0.5, coef, 0.0)
                mean_coef_neg = jnp.sum(coef_neg) / (jnp.sum(neg_mask) + 1e-8)
                max_coef_neg = jnp.max(coef_neg)
                min_coef_neg = jnp.min(coef_neg)

                metrics = dict(
                    total_loss=total,
                    cls_loss=cls_loss,
                    cbf_loss=cbf_loss,
                    lip_loss=lip_loss,
                    wd_loss=wd_loss,
                    grad_norm=grad_norm,
                    mse_all=mse_all,
                    mse_pos=mse_pos,
                    mse_neg=mse_neg,
                    mse_neg_weighted=mse_neg_weighted,
                    neg_coef=mean_coef_neg,
                    max_neg_coef=max_coef_neg,
                    min_neg_coef=min_coef_neg,
                    n_neg_samples=jnp.sum(minib_y),
                )
                return total, metrics

            grad_ncbf_loss_fn = jax.value_and_grad(ncbf_loss_fn, argnums=0, has_aux=True)

            # @partial(jax.jit, static_argnums=(4,))
            # def train_ncbf(ncbf_state: TrainState, pos_buffer: dict, neg_buffer: dict,
            #                key: jax.random.PRNGKey, nr_minibatches: int):
            #     """
            #     ncbf_state: TrainState
            #     states: (T, E, D)
            #     next_states: (T, E, D)
            #     dones: (T, E)
            #     terminates: (T, E)
            #     """
            #     def do_train(carry):
            #         ncbf_state, key = carry
            #
            #         # ---------- one minibatch step ----------
            #         @jax.jit
            #         def ncbf_minibatch_update(carry, _):
            #             ncbf_state, key = carry
            #
            #             key, replay_buffer_key  = jax.random.split(key, 2)
            #
            #             @jax.jit
            #             def sample_and_merge(pos_buffer, neg_buffer, key):
            #                 batch_size = self.ncbf_minibatch_size
            #                 nr_neg_samples = int(batch_size * self.ncbf_neg_sampling_ratio)
            #                 nr_pos_samples = batch_size - nr_neg_samples
            #
            #                 key, subkey1, subkey2 = jax.random.split(key, 3)
            #                 pos_indices = jax.random.randint(subkey1, (nr_pos_samples,), 0, pos_buffer["size"])
            #                 neg_indices = jax.random.randint(subkey2, (nr_neg_samples,), 0, neg_buffer["size"])
            #
            #                 states_pos = pos_buffer["states"][pos_indices]
            #                 next_states_pos = pos_buffer["next_states"][pos_indices]
            #                 actions_pos = pos_buffer["actions"][pos_indices]
            #                 y_target_pos = pos_buffer["y_target"][pos_indices]
            #                 masks_pos = pos_buffer["masks"][pos_indices]
            #                 indices_to_term_pos = pos_buffer["indices_to_term"][pos_indices]
            #                 last_state_pos = pos_buffer["last_state"][pos_indices]
            #                 last_action_pos = pos_buffer["last_action"][pos_indices]
            #                 history_stack_pos = pos_buffer["history_stack"][pos_indices]
            #
            #                 states_neg = neg_buffer["states"][neg_indices]
            #                 next_states_neg = neg_buffer["next_states"][neg_indices]
            #                 actions_neg = neg_buffer["actions"][neg_indices]
            #                 y_target_neg = neg_buffer["y_target"][neg_indices]
            #                 masks_neg = neg_buffer["masks"][neg_indices]
            #                 indices_to_term_neg = neg_buffer["indices_to_term"][neg_indices]
            #                 last_state_neg = neg_buffer["last_state"][neg_indices]
            #                 last_action_neg = neg_buffer["last_action"][neg_indices]
            #                 history_stack_neg = neg_buffer["history_stack"][neg_indices]
            #
            #                 states = jnp.concatenate([states_pos, states_neg], axis=0)
            #                 next_states = jnp.concatenate([next_states_pos, next_states_neg], axis=0)
            #                 actions = jnp.concatenate([actions_pos, actions_neg], axis=0)
            #                 y_target = jnp.concatenate([y_target_pos, y_target_neg], axis=0)
            #                 masks = jnp.concatenate([masks_pos, masks_neg], axis=0)
            #                 indices_to_term = jnp.concatenate([indices_to_term_pos, indices_to_term_neg], axis=0)
            #                 last_states = jnp.concatenate([last_state_pos, last_state_neg], axis=0)
            #                 last_actions = jnp.concatenate([last_action_pos, last_action_neg], axis=0)
            #                 history_stacks = jnp.concatenate([history_stack_pos, history_stack_neg], axis=0)
            #
            #                 # shuffle
            #                 perm_key, _ = jax.random.split(key)
            #                 perm = jax.random.permutation(perm_key, states.shape[0])
            #                 states = states[perm]
            #                 next_states = next_states[perm]
            #                 y_target = y_target[perm]
            #                 masks = masks[perm]
            #                 indices_to_term = indices_to_term[perm]
            #                 last_states = last_states[perm]
            #                 last_actions = last_actions[perm]
            #                 history_stacks = history_stacks[perm]
            #                 return states, next_states, actions, y_target, masks, indices_to_term, last_states, last_actions, history_stacks
            #
            #             states, next_states, actions, y_targets, masks, indices_to_term, last_states, last_actions, history_stacks = sample_and_merge(pos_buffer, neg_buffer, replay_buffer_key)
            #
            #             latent = encoder_state.apply_fn(encoder_state.params, history_stacks)
            #
            #             # Remove the first observation in history and append the next observation
            #             minib_history_stack_next = jnp.concatenate(
            #                 [history_stacks[:, 1:, :], next_states[:, None, :]], axis=1)
            #             latent_next = encoder_state.apply_fn(encoder_state.params, minib_history_stack_next)
            #
            #             (loss, metrics), ncbf_grads = grad_ncbf_loss_fn(
            #                 ncbf_state.params,
            #                 states,
            #                 next_states,
            #                 actions,
            #                 y_targets,
            #                 masks,
            #                 indices_to_term,
            #                 last_states,
            #                 last_actions,
            #                 latent,
            #                 latent_next,
            #             )
            #             metrics["grad_norm"] = optax.global_norm(ncbf_grads)
            #             new_state = ncbf_state.apply_gradients(grads=ncbf_grads)
            #
            #             carry = (new_state, key)
            #             return carry, metrics
            #
            #         init_carry = (ncbf_state, key)
            #         (ncbf_state, key), metrics = jax.lax.scan(ncbf_minibatch_update, init_carry, jnp.arange(nr_minibatches))
            #
            #         safe_mean = lambda x: jnp.mean(x) if x is not None else x
            #         mean_metrics = {f"ncbf/{k}": safe_mean(v) for k, v in metrics.items()}
            #         mean_metrics["ncbf/lr"] = ncbf_state.opt_state[1].hyperparams["learning_rate"]
            #
            #         return (ncbf_state, key), mean_metrics
            #
            #     def skip_train(carry):
            #         ncbf_state, key = carry
            #         # filling dummy metrics with zeros for logging consistency
            #         zero_metrics = {
            #             "ncbf/clf_loss": jnp.array(0.0),
            #             "ncbf/cbf_loss": jnp.array(0.0),
            #             "ncbf/lip_loss": jnp.array(0.0),
            #             "ncbf/wd_loss": jnp.array(0.0),
            #             "ncbf/total_loss": jnp.array(0.0),
            #             "ncbf/grad_norm": jnp.array(0.0),
            #             "ncbf/mse_all": jnp.array(0.0),
            #             "ncbf/mse_pos": jnp.array(0.0),
            #             "ncbf/mse_neg": jnp.array(0.0),
            #             "ncbf/mse_neg_weighted": jnp.array(0.0),
            #             "ncbf/neg_coef": jnp.array(0.0),
            #             "ncbf/max_neg_coef": jnp.array(0.0),
            #             "ncbf/min_neg_coef": jnp.array(0.0),
            #             "ncbf/n_neg_samples": jnp.array(0.0),
            #             "ncbf/lr": ncbf_state.opt_state[1].hyperparams["learning_rate"],
            #         }
            #         return (ncbf_state, key), zero_metrics
            #
            #     (ncbf_state, key), mean_metrics = jax.lax.cond(
            #         nr_minibatches > 0,
            #         do_train,
            #         skip_train,
            #         operand=(ncbf_state, key),
            #     )
            #     return ncbf_state, mean_metrics, key

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

            policy_state = self.policy_state
            critic_state = self.critic_state
            ncbf_state = self.ncbf_state[0]
            encoder_state = self.encoder_state
            decoder_state = self.decoder_state

            def _init_buffer(capacity):
                buffer = {
                    "states": jnp.zeros((capacity, ) + (self.os_shape[0], ), dtype=jnp.float32),
                    "next_states": jnp.zeros((capacity, ) + (self.os_shape[0], ), dtype=jnp.float32),
                    "actions": jnp.zeros((capacity, ) + (self.as_shape[0], ), dtype=jnp.float32),
                    "dones": jnp.zeros((capacity, ), dtype=jnp.bool),
                    "terminations": jnp.zeros((capacity, ), dtype=jnp.bool),
                    "y_target": jnp.zeros((capacity, len(self.env.ncbf_target_indices)), dtype=jnp.float32),
                    "masks": jnp.zeros((capacity, ), dtype=jnp.bool),
                    "indices_to_term": jnp.zeros((capacity, ), dtype=jnp.int32),
                    "last_state": jnp.zeros((capacity, ) + (self.os_shape[0], ), dtype=jnp.float32),
                    "last_action": jnp.zeros((capacity, ) + (self.as_shape[0], ), dtype=jnp.float32),
                    "history_stack": jnp.zeros((capacity, self.nr_history_steps) + (self.os_shape[0], ), dtype=jnp.float32),
                    "pos": jnp.zeros((), dtype=jnp.int32),
                    "size": jnp.zeros((), dtype=jnp.int32)
                }
                return buffer

            # # initialize two ncbf replay buffers
            # ncbf_pos_buffer = _init_buffer(self.ncbf_pos_buffer_size)
            ncbf_neg_buffer = _init_buffer(self.ncbf_neg_buffer_size)

            ncbf_replay_buffer = (None, ncbf_neg_buffer)

            def multi_learning_and_eval_save_iteration(multi_learning_and_eval_save_iteration_carry, multi_learning_iteration_step):
                policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, ncbf_replay_buffer, key = multi_learning_and_eval_save_iteration_carry

                def learning_iteration(learning_iteration_carry, learning_iteration_step):
                    policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, ncbf_replay_buffer, key = learning_iteration_carry
                    ncbf_pos_buffer, ncbf_neg_buffer = ncbf_replay_buffer
                    rollout_carry = (policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, key)
                    single_rollout_carry, batch = jax.lax.scan(single_rollout, rollout_carry, None, self.nr_steps)
                    policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, key = single_rollout_carry
                    states, next_states, actions, rewards, values, terminations, dones, log_probs, infos, constraints_active, delta_u, last_states, last_actions, history_stacks = batch

                    # process the batch data to get mask and y_target
                    y_target, safe_y, masks, indices_to_term = get_receding_min_target(next_states, dones, terminations)
                    next_state_mask = ~ dones

                    ncbf_metrics = {}
                    ncbf_metrics["ncbf/constraints_active_rate"] = jnp.mean(constraints_active)
                    ncbf_metrics["ncbf/mean_delta_u"] = jnp.mean(jnp.abs(delta_u))
                    ncbf_metrics["ncbf/mean_y"] = jnp.mean(y_target)
                    ncbf_metrics["ncbf/mask_rate"] = jnp.mean(masks.astype(jnp.float32))
                    ncbf_metrics["ncbf/safe_y_rate"] = jnp.mean(safe_y.astype(jnp.float32))
                    ncbf_metrics["ncbf/nr_rollout_neg_samples"] = jnp.sum(~safe_y)
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

                    # write negative and valid data samples to the buffer
                    neg_masks = jnp.logical_and(masks, ~safe_y)
                    ncbf_neg_buffer = append_to_buffer(ncbf_neg_buffer,
                        states, next_states, actions, dones, terminations, y_target, neg_masks, indices_to_term, last_states, last_actions, history_stacks)
                    ncbf_metrics["ncbf/current_neg_samples"] = jnp.sum(neg_masks.astype(jnp.float32))
                    ncbf_metrics["ncbf/neg_buffer_size"] = ncbf_neg_buffer["size"].astype(jnp.float32)
                    ncbf_metrics["ncbf/replay_neg_target_batch_size"] = jnp.asarray(self.ncbf_replay_neg_minibatch_size, dtype=jnp.float32)

                    # sample and update

                    # Calculating advantages and returns
                    def calculate_gae_advantages(critic_state, next_states, rewards, values, terminations, dones):
                        terminations = terminations.astype(jnp.float32)
                        dones = dones.astype(jnp.float32)

                        def compute_advantages(carry, t):
                            prev_advantage = carry[0]
                            advantage = delta[t] + self.gamma * self.gae_lambda * (1.0 - dones[t]) * prev_advantage
                            return (advantage,), advantage

                        next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)
                        delta = rewards + self.gamma * next_values * (1.0 - terminations) - values
                        init_advantages = delta[-1]
                        _, advantages = jax.lax.scan(compute_advantages, (init_advantages,), jnp.arange(self.nr_steps - 2, -1, -1))
                        advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
                        returns = advantages + values
                        return advantages, returns

                    advantages, returns = calculate_gae_advantages(critic_state, next_states, rewards, values, terminations, dones)

                    # Optimizing
                    def loss_fn(policy_params, critic_params, encoder_params, decoder_params, ncbf_params,
                                state_b, next_state_b, action_b, log_prob_b,
                                return_b, advantage_b,
                                last_state_b, last_action_b, history_stack_b, next_state_mask_b,
                                y_target_b, indices_to_term_b, ncbf_masks_b,
                                replay_state_b, replay_next_state_b, replay_action_b,
                                replay_y_target_b, replay_indices_to_term_b, replay_ncbf_masks_b,
                                replay_last_action_b, replay_history_stack_b
                                ):
                        # Policy loss
                        latent_b = self.encoder.apply(encoder_params, history_stack_b)
                        next_history_stack_b = jnp.roll(history_stack_b, -1, axis=1)
                        next_history_stack_b = next_history_stack_b.at[:, -1, :].set(next_state_b)
                        next_latent_b = self.encoder.apply(encoder_params, next_history_stack_b)

                        action_mean, action_logstd = self.policy.apply(policy_params, state_b)
                        action_logstd = jnp.clip(action_logstd, -5.0, 2.0)
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
                        new_value = self.critic.apply(critic_params, state_b)
                        critic_loss = 0.5 * jnp.mean((new_value.squeeze(-1) - return_b) ** 2)

                        # anticipation loss
                        # ncbf_params_stack = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[s.params for s in ncbf_state])
                        # safe_action_b, _, _ = self.ncbf_safety_layer(action_mean, state_b, last_action_b, last_state_b, latent_b, ncbf_params_stack)
                        #
                        # anticipation_loss = 0.5 * jnp.mean(jnp.sum((action_mean - safe_action_b) ** 2, axis=-1))
                        anticipation_loss = 0

                        # auxiliary next step prediction loss
                        pred_next_obs = self.decoder.apply(decoder_params, latent_b, state_b, action_b)
                        next_state_target = next_state_b[..., self.next_step_predictor_output_indices]
                        prediction_loss = jnp.sum(next_state_mask_b * jnp.mean(jnp.square(pred_next_obs - next_state_target), axis=-1)) / (jnp.sum(next_state_mask_b) + 1e-8)

                        replay_latent_b = self.encoder.apply(encoder_params, replay_history_stack_b)
                        replay_next_history_stack_b = jnp.roll(replay_history_stack_b, -1, axis=1)
                        replay_next_history_stack_b = replay_next_history_stack_b.at[:, -1, :].set(replay_next_state_b)
                        replay_next_latent_b = self.encoder.apply(encoder_params, replay_next_history_stack_b)

                        # ncbf_loss
                        current_ncbf_latent_b = jax.lax.cond(
                            self.ncbf_stop_encoder_gradient,
                            jax.lax.stop_gradient,
                            lambda x: x,
                            latent_b,
                        )
                        current_ncbf_next_latent_b = jax.lax.cond(
                            self.ncbf_stop_encoder_gradient,
                            jax.lax.stop_gradient,
                            lambda x: x,
                            next_latent_b,
                        )
                        replay_ncbf_latent_b = jax.lax.cond(
                            self.ncbf_stop_encoder_gradient,
                            jax.lax.stop_gradient,
                            lambda x: x,
                            replay_latent_b,
                        )
                        replay_ncbf_next_latent_b = jax.lax.cond(
                            self.ncbf_stop_encoder_gradient,
                            jax.lax.stop_gradient,
                            lambda x: x,
                            replay_next_latent_b,
                        )
                        current_h_input = jnp.concatenate([state_b[..., self.ncbf_observation_indices], last_action_b], axis=-1)
                        current_h_input_next = jnp.concatenate([next_state_b[..., self.ncbf_observation_indices], action_b], axis=-1)
                        replay_h_input = jnp.concatenate([replay_state_b[..., self.ncbf_observation_indices], replay_last_action_b], axis=-1)
                        replay_h_input_next = jnp.concatenate([replay_next_state_b[..., self.ncbf_observation_indices], replay_action_b], axis=-1)

                        current_h_input = jnp.concatenate((current_h_input, current_ncbf_latent_b), axis=-1)
                        current_h_input_next = jnp.concatenate((current_h_input_next, current_ncbf_next_latent_b), axis=-1)
                        replay_h_input = jnp.concatenate((replay_h_input, replay_ncbf_latent_b), axis=-1)
                        replay_h_input_next = jnp.concatenate((replay_h_input_next, replay_ncbf_next_latent_b), axis=-1)

                        h_input = jnp.concatenate((current_h_input, replay_h_input), axis=0)
                        h_input_next = jnp.concatenate((current_h_input_next, replay_h_input_next), axis=0)
                        ncbf_y_target = jnp.concatenate((y_target_b, replay_y_target_b), axis=0)
                        ncbf_masks = jnp.concatenate((ncbf_masks_b, replay_ncbf_masks_b), axis=0).astype(jnp.float32)

                        h_x = ncbf_state.apply_fn(ncbf_params, h_input)  # [B]
                        h_xn = ncbf_state.apply_fn(ncbf_params, h_input_next)  # [B]
                        ncbf_y_target = ncbf_y_target.astype(h_x.dtype)

                        # decaying coefficient disabled
                        # k = jnp.clip(indices_to_term_b, 0, self.ncbf_H)
                        # coef = self.ncbf_coef_decay_lambda ** k
                        # # use the decaying coef only for negtive samples
                        # coef = jnp.where(~y_target_b, coef, 1.0)

                        # BCE with logits: softplus(z) - y*z
                        # probability form of BCE loss:
                        # minib_y = minib_y.astype(jnp.int32)
                        # p_safe = jnp.clip(h_x, 1e-6, 1 - 1e-6)
                        # bce =  -(minib_y * jnp.log(p_safe) + (1.0 - minib_y) * jnp.log(1.0 - p_safe))
                        # num = jnp.sum(minib_mask * coef * bce)
                        # den = jnp.sum(minib_mask) + 1e-8
                        # clf_loss = num / den

                        bce = optax.sigmoid_binary_cross_entropy(h_x, ncbf_y_target)
                        head_loss = jnp.mean(bce, axis=-1)
                        num = jnp.sum(ncbf_masks * head_loss, axis=-1)
                        den = jnp.sum(ncbf_masks) + 1e-8
                        cls_loss = num / den
                        ncbf_loss = cls_loss

                        # Combine losses
                        loss = (pg_loss -
                                self.entropy_coef * entropy_loss +
                                self.critic_coef * critic_loss +
                                self.anticipation_coef * anticipation_loss +
                                self.aux_prediction_coef * prediction_loss +
                                self.ncbf_loss_coef * ncbf_loss
                                )

                        # Create metrics
                        metrics = {
                            "loss/policy_gradient_loss": pg_loss,
                            "loss/critic_loss": critic_loss,
                            "loss/entropy_loss": entropy_loss,
                            "loss/anticipation_loss": anticipation_loss,
                            "loss/aux_prediction_loss": prediction_loss,
                            "loss/ncbf_loss": ncbf_loss,
                            "loss/weighted_ncbf_loss": self.ncbf_loss_coef * ncbf_loss,
                            "ncbf/current_samples_per_minibatch": jnp.sum(ncbf_masks_b.astype(jnp.float32)),
                            "ncbf/replay_neg_samples_per_minibatch": jnp.sum(replay_ncbf_masks_b.astype(jnp.float32)),
                            "policy_ratio/approx_kl": approx_kl_div,
                            "policy_ratio/clip_fraction": clip_fraction,
                        }

                        return loss, metrics

                    batch_states = states.reshape((-1,) + self.os_shape)
                    batch_next_states = next_states.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)
                    batch_advantages = advantages.reshape(-1)
                    batch_returns = returns.reshape(-1)
                    batch_log_probs = log_probs.reshape(-1)
                    batch_last_states = last_states.reshape((-1,) + self.os_shape)
                    batch_last_actions = last_actions.reshape((-1,) + self.as_shape)
                    batch_history_stack = history_stacks.reshape((-1,) + (self.nr_history_steps, ) + self.os_shape)
                    batch_next_state_masks = next_state_mask.reshape(-1)
                    batch_y_targets = y_target.reshape((-1,) + (len(self.env.ncbf_target_indices), ))
                    batch_ncbf_masks = masks.reshape(-1)
                    batch_indices_to_term = indices_to_term.reshape(-1)

                    def sample_neg_replay_buffer(buffer, replay_key):
                        sample_limit = jnp.maximum(buffer["size"], 1)
                        replay_indices = jax.random.randint(
                            replay_key,
                            (self.ncbf_replay_neg_minibatch_size,),
                            0,
                            sample_limit,
                        )
                        replay_masks = buffer["masks"][replay_indices] & (buffer["size"] > 0)
                        return (
                            buffer["states"][replay_indices],
                            buffer["next_states"][replay_indices],
                            buffer["actions"][replay_indices],
                            buffer["y_target"][replay_indices],
                            buffer["indices_to_term"][replay_indices],
                            replay_masks,
                            buffer["last_action"][replay_indices],
                            buffer["history_stack"][replay_indices],
                        )

                    initial_action_mean, initial_action_logstd = self.policy.apply(policy_state.params, batch_states)
                    initial_action_logstd = jnp.clip(initial_action_logstd, -5.0, 2.0)
                    initial_action_std = jnp.exp(initial_action_logstd)
                    initial_new_log_probs = -0.5 * ((batch_actions - initial_action_mean) / initial_action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - initial_action_logstd
                    initial_new_log_probs = initial_new_log_probs.sum(1)
                    initial_logratio = initial_new_log_probs - batch_log_probs
                    initial_ratio = jnp.exp(initial_logratio)
                    initial_approx_kl = jnp.mean((initial_ratio - 1.0) - initial_logratio)
                    initial_abs_logratio = jnp.mean(jnp.abs(initial_logratio))

                    # vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), out_axes=0)
                    # safe_mean = lambda x: jnp.mean(x) if x is not None else x
                    # mean_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
                    # grad_loss_fn = jax.value_and_grad(mean_loss_fn, argnums=(0, 1, 2, 3), has_aux=True)
                    grad_loss_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4), has_aux=True)

                    @jax.jit
                    def predictor_loss_fn(encoder_params, decoder_params, minib_obs, minib_next_obs, minib_actions, minib_masks,
                                minib_history_stack):
                        latent = encoder_state.apply_fn(encoder_params, minib_history_stack)
                        pred_next_obs = decoder_state.apply_fn(decoder_params, latent, minib_obs, minib_actions)

                        # Reshape masks from (batch,) to (batch, 1) for proper broadcasting with obs differences
                        minib_masks = minib_masks[..., None].astype(jnp.float32)
                        mse_loss = jnp.sum(minib_masks * jnp.square(
                            pred_next_obs - minib_next_obs[..., self.next_step_predictor_output_indices])) / (
                                               jnp.sum(minib_masks) + 1e-8)
                        return mse_loss

                    # grad_predictor_loss_fn = jax.value_and_grad(predictor_loss_fn, argnums=(0, 1))

                    key, subkey = jax.random.split(key)
                    batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
                    batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
                    batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

                    def minibatch_update(carry, minibatch_indices):
                        policy_state, critic_state, encoder_state, decoder_state, ncbf_state, key = carry
                        key, replay_key = jax.random.split(key)
                        replay_states, replay_next_states, replay_actions, replay_y_targets, replay_indices_to_term, \
                            replay_ncbf_masks, replay_last_actions, replay_history_stacks = sample_neg_replay_buffer(ncbf_neg_buffer, replay_key)

                        minibatch_advantages = batch_advantages[minibatch_indices]
                        minibatch_advantages = (minibatch_advantages - jnp.mean(minibatch_advantages)) / (jnp.std(minibatch_advantages) + 1e-8)

                        # minib_latent = encoder_state.apply_fn(encoder_state.params, batch_history_stack[minibatch_indices])
                        (loss, metrics), (policy_gradients, critic_gradients, encoder_gradients, decoder_gradients, ncbf_gradients) = grad_loss_fn(
                            policy_state.params,
                            critic_state.params,
                            encoder_state.params,
                            decoder_state.params,
                            ncbf_state.params,
                            batch_states[minibatch_indices],
                            batch_next_states[minibatch_indices],
                            batch_actions[minibatch_indices],
                            batch_log_probs[minibatch_indices],
                            batch_returns[minibatch_indices],
                            minibatch_advantages,
                            batch_last_states[minibatch_indices],
                            batch_last_actions[minibatch_indices],
                            batch_history_stack[minibatch_indices],
                            batch_next_state_masks[minibatch_indices],
                            batch_y_targets[minibatch_indices],
                            batch_indices_to_term[minibatch_indices],
                            batch_ncbf_masks[minibatch_indices],
                            replay_states,
                            replay_next_states,
                            replay_actions,
                            replay_y_targets,
                            replay_indices_to_term,
                            replay_ncbf_masks,
                            replay_last_actions,
                            replay_history_stacks,
                        )

                        ppo_update_enabled = jnp.logical_or(
                            self.target_kl <= 0.0,
                            metrics["policy_ratio/approx_kl"] <= self.target_kl,
                        )

                        policy_raw_grad_norm = optax.global_norm(policy_gradients)
                        encoder_raw_grad_norm = optax.global_norm(encoder_gradients)

                        policy_state = jax.lax.cond(
                            ppo_update_enabled,
                            lambda state: state.apply_gradients(grads=policy_gradients),
                            lambda state: state,
                            policy_state,
                        )
                        critic_state = critic_state.apply_gradients(grads=critic_gradients)
                        encoder_state = jax.lax.cond(
                            ppo_update_enabled,
                            lambda state: state.apply_gradients(grads=encoder_gradients),
                            lambda state: state,
                            encoder_state,
                        )
                        decoder_state = decoder_state.apply_gradients(grads=decoder_gradients)
                        # temporarily not updating ncbf
                        ncbf_state = ncbf_state.apply_gradients(grads=ncbf_gradients)

                        # update next step predictor
                        # predictor_loss, grads_predictor = grad_predictor_loss_fn(
                        #     encoder_state.params,
                        #     decoder_state.params,
                        #     batch_states[minibatch_indices],
                        #     batch_next_states[minibatch_indices],
                        #     batch_actions[minibatch_indices],
                        #     batch_next_state_masks[minibatch_indices],
                        #     batch_history_stack[minibatch_indices],
                        # )
                        # encoder_state = encoder_state.apply_gradients(grads=grads_predictor[0])
                        # decoder_state = decoder_state.apply_gradients(grads=grads_predictor[1])

                        # metrics["next_step_predictor/loss"] = predictor_loss
                        metrics["gradients/policy_grad_norm"] = jnp.where(ppo_update_enabled, policy_raw_grad_norm, 0.0)
                        metrics["gradients/policy_raw_grad_norm"] = policy_raw_grad_norm
                        metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)
                        metrics["gradients/encoder_grad_norm"] = jnp.where(ppo_update_enabled, encoder_raw_grad_norm, 0.0)
                        metrics["gradients/encoder_raw_grad_norm"] = encoder_raw_grad_norm
                        metrics["policy_ratio/update_enabled"] = ppo_update_enabled.astype(jnp.float32)

                        carry = (policy_state, critic_state, encoder_state, decoder_state, ncbf_state, key)
                        return carry, metrics

                    init_carry = (policy_state, critic_state, encoder_state, decoder_state, ncbf_state, key)
                    carry, optimization_metrics = jax.lax.scan(minibatch_update, init_carry, batch_indices)
                    policy_state, critic_state, encoder_state, decoder_state, ncbf_state, key = carry

                    optimization_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
                    optimization_metrics["v_value/explained_variance"] = 1 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
                    optimization_metrics["policy/std_dev"] = jnp.mean(jnp.exp(jnp.clip(policy_state.params["params"]["policy_logstd"], -5.0, 2.0)))
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

                    combined_metrics = {**infos, **steps_metrics, **optimization_metrics, **ncbf_metrics}
                    combined_metrics = tree.map_structure(lambda x: jnp.mean(x), combined_metrics)

                    def callback(carry):
                        metrics, parallel_seed_id = carry
                        current_time = time.time()
                        metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (current_time - self.last_time[parallel_seed_id]))
                        self.last_time[parallel_seed_id] = current_time
                        global_step = int(metrics["steps/nr_env_steps"])
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        # wandb.log(metrics, step=global_step)
                        self.end_logging()

                    jax.debug.callback(callback, (combined_metrics, parallel_seed_id))

                    ncbf_replay_buffer = (ncbf_pos_buffer, ncbf_neg_buffer)
                    return (policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, ncbf_replay_buffer, key), None

                key, subkey = jax.random.split(key)
                learning_iteration_carry, _ = jax.lax.scan(learning_iteration, (policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, ncbf_replay_buffer, subkey), jnp.arange(self.nr_updates_per_multi_learning_iteration))
                policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, ncbf_replay_buffer, key = learning_iteration_carry

                # Evaluating
                if self.evaluation_active:
                    def single_eval_rollout(single_eval_rollout_carry, _):
                        policy_state, ncbf_state, eval_env_state = single_eval_rollout_carry
                        action_mean, _ = self.policy.apply(policy_state.params, eval_env_state.next_observation)
                        action = action_mean
                        raw_processed_action = self.get_processed_action(action)
                        latent_z = encoder_state.apply_fn(encoder_state.params, eval_env_state.history_stack)
                        params_stack = repeat_ncbf_params(ncbf_state)
                        processed_action, _, _ = self.batched_ncbf_safety_layer(
                            raw_processed_action,
                            eval_env_state.next_observation,
                            eval_env_state.last_action,
                            eval_env_state.last_state,
                            latent_z,
                            params_stack,
                        )
                        eval_env_state = self.env.step(eval_env_state, processed_action)

                        return (policy_state, ncbf_state, eval_env_state), None

                    key, reset_key = jax.random.split(key)
                    reset_keys = jax.random.split(reset_key, self.nr_envs)
                    eval_env_state = self.env.reset(reset_keys, True)
                    single_eval_rollout_carry, _ = jax.lax.scan(single_eval_rollout, (policy_state, ncbf_state, eval_env_state), jnp.arange(self.horizon))
                    _, ncbf_state, eval_env_state = single_eval_rollout_carry

                    eval_metrics = {
                        "eval/episode_return": jnp.mean(eval_env_state.info["rollout/episode_return"]),
                        "eval/episode_length": jnp.mean(eval_env_state.info["rollout/episode_length"]),
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
                    def save_with_check(policy_state, critic_state, ncbf_state, encoder_state, decoder_state):
                        self.save(policy_state, critic_state, ncbf_state, encoder_state, decoder_state)
                    jax.debug.callback(save_with_check, policy_state, critic_state, ncbf_state, encoder_state, decoder_state)


                return (policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, ncbf_replay_buffer, key), None

            jax.lax.scan(multi_learning_and_eval_save_iteration, (policy_state, critic_state, ncbf_state, encoder_state, decoder_state, env_state, ncbf_replay_buffer, key), jnp.arange(self.nr_multi_learning_and_eval_save_iterations))


        self.key, subkey = jax.random.split(self.key)
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

    def start_logging_ncbf_pretrain(self, steps):
        rlx_logger.info(f"pretrained NCBF for {steps} steps")

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


    def load(config, env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(splitted_path[:-1]))
        checkpoint_file_name = splitted_path[-1]
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"

        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = PPO(config, env, run_path, writer)

        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "ncbf": model.ncbf_state,
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


    def test(self, episodes):
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

        def get_ncbf_params_stack():
            if isinstance(self.ncbf_state, list):
                return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[s.params for s in self.ncbf_state])
            return jax.tree_util.tree_map(
                lambda x: jnp.repeat(x[None, ...], self.ncbf_n_ensemble, axis=0),
                self.ncbf_state.params,
            )

        @jax.jit
        def rollout(env_state, key):
            # key, subkey = jax.random.split(key)
            action_mean, action_logstd = self.policy.apply(self.policy_state.params, env_state.next_observation)
            # action_std = jnp.exp(action_logstd)
            action = action_mean # + action_std * jax.random.normal(subkey, shape=action_mean.shape)
            raw_processed_action = self.get_processed_action(action)
            latent_z = self.encoder_state.apply_fn(self.encoder_state.params, env_state.history_stack)
            params_stack = get_ncbf_params_stack()
            processed_action, constraint_active, delta_u = self.batched_ncbf_safety_layer(
                raw_processed_action,
                env_state.next_observation,
                env_state.last_action,
                env_state.last_state,
                latent_z,
                params_stack,
            )
            env_state = self.env.step(env_state, processed_action)
            return env_state, key

        self.key, subkey = jax.random.split(self.key)
        reset_keys = jax.random.split(subkey, self.nr_envs)
        env_state = self.env.reset(reset_keys, True)
        while True:
            env_state, self.key = rollout(env_state, self.key)
            if self.render:
                env_state = self.env.render(env_state)


    def general_properties():
        return GeneralProperties
