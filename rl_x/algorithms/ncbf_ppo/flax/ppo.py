import datetime
import os
import pickle
import shutil
import json
import logging
import time
from collections import deque
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

from rl_x.algorithms.ncbf_ppo.flax.general_properties import GeneralProperties
from rl_x.algorithms.ncbf_ppo.flax.policy import get_policy
from rl_x.algorithms.ncbf_ppo.flax.critic import get_critic
from rl_x.algorithms.ncbf_ppo.flax.ncbf import (
    get_ncbf,
    logistic_normal_binary_cross_entropy,
    split_ncbf_output,
)
from rl_x.algorithms.ncbf_ppo.flax.batch import Batch
from rl_x.algorithms.ncbf_ppo.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.ncbf_ppo.flax.history_encoder import get_history_encoder
from rl_x.algorithms.ncbf_ppo.flax.decoder import get_decoder

rlx_logger = logging.getLogger("rl_x")


class PPO:
    def __init__(self, config, env, run_path, writer) -> None:
        self.config = config
        self.env = env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.clip_range = config.algorithm.clip_range
        self.entropy_coef = config.algorithm.entropy_coef
        self.critic_coef = config.algorithm.critic_coef
        self.anticipation_coef = config.algorithm.ncbf.policy_loss_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
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
        self.ncbf_minibatch_size = config.algorithm.minibatch_size
        self.ncbf_nr_minibatches = config.algorithm.ncbf.nr_minibatches
        self.ncbf_use_safety_layer = config.algorithm.ncbf.use_safety_layer
        self.ncbf_output_distribution = getattr(config.algorithm.ncbf, "output_distribution", "deterministic")
        self.ncbf_min_log_std = getattr(config.algorithm.ncbf, "min_log_std", -5.0)
        self.ncbf_max_log_std = getattr(config.algorithm.ncbf, "max_log_std", 2.0)

        self.ncbf_buffer_size = config.algorithm.ncbf_buffer.buffer_size
        self.ncbf_pretrain_steps = config.algorithm.ncbf.pretrain.nr_steps
        self.ncbf_pretrain_nr_minibatches = config.algorithm.ncbf.pretrain.nr_minibatches

        self.action_noise_sampling_ratio = config.algorithm.action_noise_sampling_ratio
        self.rollout_save_name = config.algorithm.rollout_save_name

        # assert ncbf nr_steps * nr_envs must be a multiple of ncbf batchsize
        if (self.nr_steps * self.nr_envs) % self.ncbf_minibatch_size != 0:
            raise ValueError("NCBF batch size must divide evenly into nr_steps * nr_envs.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, ncbf_key, encoder_key, decoder_key = jax.random.split(self.key, 6)

        self.os_shape = env.single_observation_space.shape
        self.as_shape = env.single_action_space.shape
        
        self.policy, self.get_processed_action = get_policy(config, env)
        self.ncbf, self.ncbf_apply, self.batched_ncbf_safety_layer, self.ncbf_safety_layer = get_ncbf(config, env)
        self.critic = get_critic(config, env)
        self.encoder = get_history_encoder(self.config, self.env)
        self.decoder = get_decoder(self.config, self.env)
        self.replay_buffer = ReplayBuffer(capacity=config.algorithm.ncbf_buffer.buffer_size, nr_envs=self.nr_envs, os_shape=self.os_shape, as_shape=self.as_shape, rng=ncbf_key)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)
        self.encoder.apply = jax.jit(self.encoder.apply)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        state = jnp.array([env.single_observation_space.sample()])
        dummy_latent = jnp.zeros((1, self.encoder.hidden_size))
        dummy_history_stack = jnp.zeros((1, self.env.nr_history_steps) + (self.os_shape[0],))

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, state, dummy_latent),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, state, dummy_latent),
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
            params=self.encoder.init(encoder_key, dummy_history_stack),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=next_step_predictor_lr),
            )
        )

        self.decoder_state = TrainState.create(
            apply_fn=self.decoder.apply,
            params=self.decoder.init(decoder_key, dummy_latent, state, jnp.zeros((1, self.as_shape[0]))),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.next_step_predictor_lr),
            )
        )

        dummy_action = jnp.zeros((state.shape[0], self.as_shape[0]))

        dummy_h_input = jnp.concatenate([state[..., env.ncbf_observation_indices], dummy_action, dummy_latent], axis=-1)
        ncbf_keys = jax.random.split(ncbf_key, self.ncbf_n_ensemble)
        self.ncbf_state = [
            TrainState.create(
                apply_fn=self.ncbf[i].apply,
                params=self.ncbf[i].init(ncbf_keys[i], dummy_h_input),
                tx=optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.inject_hyperparams(optax.adam)(learning_rate=config.algorithm.ncbf.lr),
                )
            )
            for i in range(self.ncbf_n_ensemble)
        ]
        self.ncbf_state = self.ncbf_state[0]

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "best.model"
            self.best_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    
    def train(self):
        @jax.jit
        def _future_event_within_H(events: jnp.array, H: int):
            """
            events: [T, N] bool (True if event occurs at time t in env n)
            Returns:
              any_next_H: [T, N] bool, True if there is an event in (t, t+H] for that env.
            """
            T, N = events.shape

            def body(carry, ev_t):
                # carry: [N] int, distance to next event seen so far (from future)
                dist_prev = carry
                # if event at t: distance = 0; else = dist_prev + 1 (capped at H+1)
                dist = jnp.where(ev_t, 0, jnp.minimum(dist_prev + 1, H + 1))
                return dist, dist

            init = jnp.full((N,), H + 1, dtype=jnp.int32)

            # scan backwards in time
            _, dists_rev = jax.lax.scan(body, init, events[::-1])  # [T,N], reversed
            dists = dists_rev[::-1]  # [T,N], distance to next event (0 if at t)

            # “next H steps” = strictly after t: 0 < dist <= H
            any_next_H = jnp.logical_and(dists > 0, dists <= H)
            return any_next_H

        @jax.jit
        def window_any_done_next_H(
                dones: jnp.array,
                terminates: jnp.array,
                H: int
        ):
            """
            dones, terminates: [T, N_env, 1] or [T, N_env] bool
              - dones: episode ends (terminated or truncated)
              - terminates: true termination (failure) events

            Returns:
              y: [T, N_env] bool
                 horizon-safe label: True if NO terminate in next H steps, else False
              mask: [T, N_env] bool
                 training mask: ~dones  (valid only on non-done steps)
            """
            # squeeze last dim if present
            # any terminate in (t, t+H] -> y[t] = False, else True
            # set last done to True to avoid counting beyond buffer end for each env
            dones = dones.at[-1, :].set(True)

            any_term_next_H = _future_event_within_H(terminates, H)  # [T, N]
            y = ~any_term_next_H  # [T, N] bool

            # mask is "no timely truncation in the next H steps"
            mask = ~_future_event_within_H(dones & ~terminates, H) # [T, N] bool

            return y, mask

        @jax.jit
        def get_action_and_value(policy_state: TrainState, critic_state: TrainState, state: np.ndarray, history_latent: np.ndarray, key: jax.random.PRNGKey):
            action_mean, action_logstd = self.policy.apply(policy_state.params, state, history_latent)
            action_logstd = jnp.clip(action_logstd, -5.0, 2.0)
            action_std = jnp.exp(action_logstd)
            key, subkey = jax.random.split(key)
            action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
            log_prob = -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
            value = self.critic.apply(critic_state.params, state, history_latent)
            processed_action = self.get_processed_action(action)
            return processed_action, action, value.reshape(-1), log_prob.sum(1), key

        @jax.jit
        def calculate_gae_advantages(critic_state: TrainState, encoder_state: TrainState, next_states: np.ndarray, history_stacks: np.ndarray, rewards: np.ndarray, terminations: np.ndarray, dones: np.ndarray, values: np.ndarray):
            terminations = terminations.astype(jnp.float32)
            dones = dones.astype(jnp.float32)

            def compute_advantages(carry, t):
                prev_advantage = carry[0]
                advantage = delta[t] + self.gamma * self.gae_lambda * (1.0 - dones[t]) * prev_advantage
                return (advantage,), advantage

            next_history_stacks = jnp.roll(history_stacks, -1, axis=2)
            next_history_stacks = next_history_stacks.at[:, :, -1, :].set(next_states)
            next_latents = encoder_state.apply_fn(encoder_state.params, next_history_stacks)
            next_values = self.critic.apply(critic_state.params, next_states, next_latents).squeeze(-1)
            delta = rewards + self.gamma * next_values * (1.0 - terminations) - values
            init_advantages = delta[-1]
            _, advantages = jax.lax.scan(compute_advantages, (init_advantages,), jnp.arange(self.nr_steps - 2, -1, -1))
            advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
            returns = advantages + values
            return advantages, returns
        

        @jax.jit
        def update(policy_state: TrainState, critic_state: TrainState, encoder_state: TrainState, ncbf_state: TrainState,
                   states: np.ndarray, actions: np.ndarray, advantages: np.ndarray, returns: np.ndarray, values: np.ndarray, log_probs: np.ndarray,
                   history_stacks: np.ndarray,
                   key: jax.random.PRNGKey):
            def loss_fn(policy_params, critic_params, ncbf_params, state_b, action_b, log_prob_b, return_b, advantage_b, history_stack_b):
                history_latent_b = encoder_state.apply_fn(encoder_state.params, history_stack_b)
                # Policy loss
                action_mean, action_logstd = self.policy.apply(policy_params, state_b, history_latent_b)
                action_logstd = jnp.clip(action_logstd, -5.0, 2.0)
                action_std = jnp.exp(action_logstd)
                new_log_prob = -0.5 * ((action_b - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
                new_log_prob = new_log_prob.sum(1)
                entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
                
                logratio = new_log_prob - log_prob_b
                ratio = jnp.exp(logratio)
                approx_kl_div = (ratio - 1) - logratio
                clip_fraction = jnp.float32((jnp.abs(ratio - 1) > self.clip_range))

                pg_loss1 = -advantage_b * ratio
                pg_loss2 = -advantage_b * jnp.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_loss = jnp.maximum(pg_loss1, pg_loss2)
                
                entropy_loss = entropy.sum(1)
                
                # Critic loss
                new_value = self.critic.apply(critic_params, state_b, history_latent_b)
                critic_loss = 0.5 * (new_value - return_b) ** 2

                # anticipation loss
                anticipation_loss = 0.0

                # Combine losses
                loss = pg_loss - self.entropy_coef * entropy_loss + self.critic_coef * critic_loss + self.anticipation_coef * anticipation_loss

                # Create metrics

                metrics = {
                    "loss/policy_gradient_loss": pg_loss,
                    "loss/critic_loss": critic_loss,
                    "loss/entropy_loss": entropy_loss,
                    "loss/anticipation_loss": anticipation_loss,
                    "policy_ratio/approx_kl": approx_kl_div,
                    "policy_ratio/clip_fraction": clip_fraction,
                }

                return loss, (metrics)
            

            batch_states = states.reshape((-1,) + self.os_shape)
            batch_actions = actions.reshape((-1,) + self.as_shape)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_log_probs = log_probs.reshape(-1)
            batch_history_stacks = history_stacks.reshape((-1, self.env.nr_history_steps) + self.os_shape)

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0), out_axes=0)
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
            grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

            key, subkey = jax.random.split(key)
            batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
            batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
            batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

            def minibatch_update(carry, minibatch_indices):
                policy_state, critic_state = carry

                minibatch_advantages = batch_advantages[minibatch_indices]
                minibatch_advantages = (minibatch_advantages - jnp.mean(minibatch_advantages)) / (jnp.std(minibatch_advantages) + 1e-8)

                # here do not update ncbf params
                (loss, (metrics)), (policy_gradients, critic_gradients) = grad_loss_fn(
                    policy_state.params,
                    critic_state.params,
                    ncbf_state.params,                                    # TODO : currently use only first ncbf in ensemble for ppo update
                    batch_states[minibatch_indices],
                    batch_actions[minibatch_indices],
                    batch_log_probs[minibatch_indices],
                    batch_returns[minibatch_indices],
                    minibatch_advantages,
                    batch_history_stacks[minibatch_indices],
                )

                policy_state = policy_state.apply_gradients(grads=policy_gradients)
                critic_state = critic_state.apply_gradients(grads=critic_gradients)

                metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)
                carry = (policy_state, critic_state)

                return carry, (metrics)
            
            init_carry = (policy_state, critic_state)
            carry, (metrics) = jax.lax.scan(minibatch_update, init_carry, batch_indices)
            policy_state, critic_state = carry

            # Calculate mean metrics
            mean_metrics = {key: jnp.mean(metrics[key]) for key in metrics}
            mean_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
            mean_metrics["v_value/explained_variance"] = 1 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
            mean_metrics["policy/std_dev"] = jnp.mean(jnp.exp(jnp.clip(policy_state.params["params"]["policy_logstd"], -5.0, 2.0)))

            return policy_state, critic_state, mean_metrics, key

        @partial(jax.jit, static_argnums=(7,))
        def train_ncbf(ncbf_state: TrainState, states: np.ndarray, next_states: np.ndarray, actions: np.ndarray,
                       y_targets: np.ndarray, masks: np.ndarray,
                       key: jax.random.PRNGKey, nr_minibatches: int):
            """
            ncbf_state: TrainState
            states: (T, E, D)
            next_states: (T, E, D)
            dones: (T, E)
            terminates: (T, E)
            """

            @jax.jit
            def make_h_input(obs, action):
                latent = jnp.zeros(obs.shape[:-1] + (self.encoder.hidden_size,), dtype=obs.dtype)
                return jnp.concatenate([obs[..., self.ncbf_observation_indices], action, latent], axis=-1)

            def loss_fn(params, minib_obs, minib_nxt, minib_action, minib_y, minib_mask):
                gamma_c = self.ncbf_gamma_c
                h_input = make_h_input(minib_obs, minib_action)
                h_input_next = make_h_input(minib_nxt, minib_action)
                h_x_raw = ncbf_state.apply_fn(params, h_input)
                h_xn_raw = ncbf_state.apply_fn(params, h_input_next)
                h_x, h_x_log_std = split_ncbf_output(
                    h_x_raw,
                    self.ncbf_output_distribution,
                    self.ncbf_min_log_std,
                    self.ncbf_max_log_std,
                )
                h_xn, _ = split_ncbf_output(
                    h_xn_raw,
                    self.ncbf_output_distribution,
                    self.ncbf_min_log_std,
                    self.ncbf_max_log_std,
                )
                minib_y = jnp.broadcast_to(minib_y[..., None], h_x.shape)
                minib_mask = jnp.broadcast_to(minib_mask[..., None], h_x.shape)

                # (1) classification: logits = h(x) - gamma_c
                logits = h_x - gamma_c
                if self.ncbf_output_distribution == "logistic_normal":
                    classification_loss = logistic_normal_binary_cross_entropy(logits, h_x_log_std, minib_y)
                    mean_std = jnp.mean(jnp.exp(h_x_log_std))
                else:
                    classification_loss = jax.nn.softplus(logits) - minib_y * logits
                    mean_std = jnp.array(0.0, dtype=h_x.dtype)

                num = jnp.sum(minib_mask * classification_loss)
                den = jnp.sum(minib_mask) + 1e-8
                clf_loss = num / den

                # (2) DT-CBF penalty: ReLU( -(h_{t+1}-h_t + eta*(h_t-gamma_c)) )
                alpha = self.ncbf_eta_cbf * (h_x - gamma_c)
                cbf_ineq = h_xn - h_x + alpha
                cbf_loss = jnp.sum(minib_mask * jnp.maximum(0.0, -cbf_ineq)) / (jnp.sum(minib_mask) + 1e-8)

                # (3) Lipschitz regularizer: fixed-size pair sampling from valid positions
                @jax.jit
                def grad_norm_penalty(params, x, max_norm):
                    """
                    x: (batch, input_dim)
                    apply_fn: model.apply
                    params: model parameters

                    Returns: scalar penalty = E[(max(0, ||∇_x f(x)||_2 - target_norm), 0))^2]
                    """

                    def f_single(x_single):
                        # shape (output_dim,) -> reduce to scalar
                        y_raw = self.ncbf[0].apply(params, x_single)
                        y, _ = split_ncbf_output(
                            y_raw,
                            self.ncbf_output_distribution,
                            self.ncbf_min_log_std,
                            self.ncbf_max_log_std,
                        )
                        return jnp.sum(y)

                    # Vectorize grad over batch
                    grad_h = jax.vmap(jax.grad(f_single))(x)  # [B,T,D]
                    grad_norm = jnp.linalg.norm(grad_h, axis=-1)  # [B,T]
                    lip_violation = jnp.maximum(0.0, grad_norm - max_norm)
                    lip_loss = jnp.mean(lip_violation)
                    return lip_loss

                #
                lip_loss = grad_norm_penalty(ncbf_state.params, h_input, self.ncbf_L_target)

                # (4) weight decay
                wd_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params))

                total = (self.ncbf_w_clf * clf_loss +
                         self.ncbf_w_cbf * cbf_loss +
                         self.ncbf_w_lip * lip_loss +
                         self.ncbf_w_wd * wd_loss)

                metrics = dict(
                    loss_total=total,
                    loss_clf=clf_loss,
                    loss_cbf=cbf_loss,
                    loss_lip=lip_loss,
                    loss_wd=wd_loss,
                    mean_y=jnp.mean(minib_y),
                    mean_std=mean_std,
                )
                return total, metrics

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0, 0), out_axes=0)
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
            grad_ncbf_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=0, has_aux=True)

            key, subkey = jax.random.split(key)
            # Create [nr_minibatches, buffer_size] by vmapping a permutation call
            subkeys = jax.random.split(subkey, nr_minibatches)
            def perm_fn(k):
                return jax.random.permutation(k, self.replay_buffer.size)
            # Shape: [nr_minibatches, buffer_size]
            all_perms = jax.vmap(perm_fn)(subkeys)
            batch_indices = all_perms[:, :self.ncbf_minibatch_size]

            # ---------- one minibatch step ----------
            @jax.jit
            def ncbf_minibatch_update(
                    carry,
                    minibatch_indices
            ):
                ncbf_state = carry
                minib_obs = states[minibatch_indices]
                minib_nxt = next_states[minibatch_indices]
                minib_action = actions[minibatch_indices]
                minib_y = y_targets[minibatch_indices]
                minib_mask = masks[minibatch_indices]


                (loss, metrics), ncbf_grads = grad_ncbf_loss_fn(
                    ncbf_state.params,
                    minib_obs,
                    minib_nxt,
                    minib_action,
                    minib_y,
                    minib_mask
                )
                metrics["grad_norm"] = optax.global_norm(ncbf_grads)
                new_state = ncbf_state.apply_gradients(grads=ncbf_grads)

                return new_state, metrics

            init_carry = ncbf_state
            ncbf_state, metrics = jax.lax.scan(ncbf_minibatch_update, init_carry, batch_indices)

            mean_metrics = {'ncbf/' + key: jnp.mean(metrics[key]) for key in metrics}
            mean_metrics['ncbf/lr'] = ncbf_state.opt_state[1].hyperparams['learning_rate']
            return ncbf_state, mean_metrics, key

        @jax.jit
        def get_deterministic_action(policy_state: TrainState, state: np.ndarray, history_latent: np.ndarray):
            action_mean, action_logstd = self.policy.apply(policy_state.params, state, history_latent)
            return self.get_processed_action(action_mean)

        def get_safety_layer_curriculum_coeff(step):
            if not self.ncbf_use_safety_layer:
                return np.float32(0.0)
            denominator = max(float(self.total_timesteps) - float(self.nr_envs), 1.0)
            return np.float32(np.clip(float(step) / denominator, 0.0, 1.0))

        self.set_train_mode()

        batch = Batch(
            states=np.zeros((self.nr_steps, self.nr_envs) + self.os_shape),
            next_states=np.zeros((self.nr_steps, self.nr_envs) + self.os_shape),
            actions=np.zeros((self.nr_steps, self.nr_envs) + self.as_shape),
            rewards=np.zeros((self.nr_steps, self.nr_envs)),
            values=np.zeros((self.nr_steps, self.nr_envs)),
            terminations=np.zeros((self.nr_steps, self.nr_envs), dtype=bool),
            dones=np.zeros((self.nr_steps, self.nr_envs), dtype=bool),
            log_probs=np.zeros((self.nr_steps, self.nr_envs)),
            advantages=np.zeros((self.nr_steps, self.nr_envs)),
            returns=np.zeros((self.nr_steps, self.nr_envs)),
            masks=np.zeros((self.nr_steps, self.nr_envs), dtype=bool),
            y_targets=np.zeros((self.nr_steps, self.nr_envs)),
            constraint_violated=np.zeros((self.nr_steps, self.nr_envs), dtype=bool),
            delta_u=np.zeros((self.nr_steps, self.nr_envs)),
            history_stacks=np.zeros((self.nr_steps, self.nr_envs, self.env.nr_history_steps) + self.os_shape)
        )

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        # pre-sampling and pretrain ncbf
        state, info = self.env.reset()
        if self.ncbf_pretrain_steps > 0:
            # initialize a temporary buffer to collect ncbf pretrian data
            ncbf_batch = Batch(
                states=np.zeros((self.ncbf_pretrain_steps, self.nr_envs) + self.os_shape),
                next_states=np.zeros((self.ncbf_pretrain_steps, self.nr_envs) + self.os_shape),
                actions=np.zeros((self.ncbf_pretrain_steps, self.nr_envs) + self.as_shape),
                rewards=np.zeros((self.ncbf_pretrain_steps, self.nr_envs)),
                values=np.zeros((self.ncbf_pretrain_steps, self.nr_envs)),
                terminations=np.zeros((self.ncbf_pretrain_steps, self.nr_envs), dtype=bool),
                dones=np.zeros((self.ncbf_pretrain_steps, self.nr_envs), dtype=bool),
                log_probs=np.zeros((self.ncbf_pretrain_steps, self.nr_envs)),
                advantages=np.zeros((self.ncbf_pretrain_steps, self.nr_envs)),
                returns=np.zeros((self.ncbf_pretrain_steps, self.nr_envs)),
                masks=np.zeros((self.ncbf_pretrain_steps, self.nr_envs), dtype=bool),
                y_targets=np.zeros((self.ncbf_pretrain_steps, self.nr_envs)),
                constraint_violated=np.zeros((self.ncbf_pretrain_steps, self.nr_envs)),
                delta_u=np.zeros((self.ncbf_pretrain_steps, self.nr_envs)),
                history_stacks=np.zeros((self.ncbf_pretrain_steps, self.nr_envs, self.env.nr_history_steps) + self.os_shape)
            )

            # Initialize history stacks for pretraining
            pretrain_history_stacks = np.tile(state[:, None, :], (1, self.env.nr_history_steps, 1))
            if "history_stack" in info:
                pretrain_history_stacks = np.asarray(info["history_stack"])
            pretrain_last_states = state.copy()
            pretrain_last_actions = np.zeros((self.nr_envs,) + self.as_shape)

            for step in range(self.ncbf_pretrain_steps):
                latent_z = self.encoder.apply(self.encoder_state.params, pretrain_history_stacks)
                raw_processed_action, action, value, log_prob, self.key = get_action_and_value(self.policy_state, self.critic_state, state, latent_z, self.key)
                params_stack = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[s.params for s in self.ncbf_state])
                processed_action, constraint_active, delta_u = self.batched_ncbf_safety_layer(
                    raw_processed_action,
                    state,
                    pretrain_last_actions,
                    pretrain_last_states,
                    latent_z,
                    np.float32(0.0),
                    params_stack,
                )
                processed_action = jax.device_get(processed_action)
                next_state, reward, terminated, truncated, info = self.env.step(processed_action)
                done = terminated | truncated
                actual_next_state = next_state.copy()
                for i, single_done in enumerate(done):
                    if single_done:
                        actual_next_state[i] = np.array(self.env.get_final_observation_at_index(info, i))
                        saving_return_buffer.append(self.env.get_final_info_value_at_index(info, "episode_return", i))

                ncbf_batch.states[step] = state
                ncbf_batch.next_states[step] = actual_next_state
                ncbf_batch.actions[step] = processed_action
                ncbf_batch.rewards[step] = reward
                ncbf_batch.values[step] = value
                ncbf_batch.history_stacks[step] = pretrain_history_stacks
                
                # Update history stacks for next step (sliding window with new observation)
                next_pretrain_history_stacks = np.roll(pretrain_history_stacks, shift=-1, axis=1)
                next_pretrain_history_stacks[:, -1] = next_state
                reset_history_stacks = np.tile(next_state[:, None, :], (1, self.env.nr_history_steps, 1))
                pretrain_history_stacks = np.where(done[:, None, None], reset_history_stacks, next_pretrain_history_stacks)
                pretrain_last_states = np.where(done[:, None], next_state, state)
                pretrain_last_actions = np.where(done[:, None], np.zeros_like(pretrain_last_actions), processed_action)
                ncbf_batch.terminations[step] = terminated
                ncbf_batch.dones[step] = done
                ncbf_batch.log_probs[step] = log_prob
                state = next_state

            # calculate y labels and masks
            y_bool, masks = window_any_done_next_H(ncbf_batch.dones, ncbf_batch.terminations, self.ncbf_H)
            y = y_bool.astype(jnp.float32)
            ncbf_batch.masks = masks
            ncbf_batch.y_targets = y

            # add batch to buffer
            self.replay_buffer.add_batch(
                states=ncbf_batch.states,
                next_states=ncbf_batch.next_states,
                actions=ncbf_batch.actions,
                rewards=ncbf_batch.rewards,
                terminations=ncbf_batch.terminations,
                masks=ncbf_batch.masks,
                y_targets=ncbf_batch.y_targets,
            )
            # pretrain ncbf
            keys = jax.random.split(self.key, 5 + 1)
            self.key = keys[0]

            self.ncbf_state[0], ncbf_metrics, self.key = train_ncbf(self.ncbf_state[0],
                                                                 self.replay_buffer.states,
                                                                 self.replay_buffer.next_states,
                                                                 self.replay_buffer.actions,
                                                                 self.replay_buffer.y_targets,
                                                                 self.replay_buffer.masks,
                                                                 keys[1],
                                                                 self.ncbf_pretrain_nr_minibatches)

            self.ncbf_state[1], ncbf_metrics, self.key = train_ncbf(self.ncbf_state[1],
                                                                 self.replay_buffer.states,
                                                                 self.replay_buffer.next_states,
                                                                 self.replay_buffer.actions,
                                                                 self.replay_buffer.y_targets,
                                                                 self.replay_buffer.masks,
                                                                 keys[2],
                                                                 self.ncbf_pretrain_nr_minibatches)

            self.ncbf_state[2], ncbf_metrics, self.key = train_ncbf(self.ncbf_state[2],
                                                                 self.replay_buffer.states,
                                                                 self.replay_buffer.next_states,
                                                                 self.replay_buffer.actions,
                                                                 self.replay_buffer.y_targets,
                                                                 self.replay_buffer.masks,
                                                                 keys[2],
                                                                 self.ncbf_pretrain_nr_minibatches)

            self.ncbf_state[3], ncbf_metrics, self.key = train_ncbf(self.ncbf_state[3],
                                                                 self.replay_buffer.states,
                                                                 self.replay_buffer.next_states,
                                                                 self.replay_buffer.actions,
                                                                 self.replay_buffer.y_targets,
                                                                 self.replay_buffer.masks,
                                                                 keys[3],
                                                                 self.ncbf_pretrain_nr_minibatches)

            self.ncbf_state[4], ncbf_metrics, self.key = train_ncbf(self.ncbf_state[4],
                                                                 self.replay_buffer.states,
                                                                 self.replay_buffer.next_states,
                                                                 self.replay_buffer.actions,
                                                                 self.replay_buffer.y_targets,
                                                                 self.replay_buffer.masks,
                                                                 keys[4],
                                                                 self.ncbf_pretrain_nr_minibatches)

            # get scalar mean from ncbf_metrics
            for key, value in ncbf_metrics.items():
                ncbf_metrics[key] = value.item()

            mean_y = jnp.mean(ncbf_batch.y_targets)
            ncbf_metrics['ncbf/mean_y'] = mean_y.item()

            # log ncbf pretrain metrics
            for key, value in ncbf_metrics.items():
                self.log(key, value, 0)

            self.start_logging_ncbf_pretrain(self.ncbf_pretrain_steps)

        state, info = self.env.reset()
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        steps_metrics = {}
        
        history_stacks = np.tile(state[:, None, :], (1, self.env.nr_history_steps, 1))
        if "history_stack" in info:
            history_stacks = np.asarray(info["history_stack"])
        last_states = state.copy()
        last_actions = np.zeros((self.nr_envs,) + self.as_shape)
        
        while global_step < self.total_timesteps:
            start_time = time.time()
            time_metrics = {}

            # Acting
            dones_this_rollout = 0
            step_info_collection = {}
            safety_layer_curriculum_coeffs = np.zeros(self.nr_steps, dtype=np.float32)
            for step in range(self.nr_steps):
                latent_z = self.encoder.apply(self.encoder_state.params, history_stacks)
                _, action, value, log_prob, self.key = get_action_and_value(self.policy_state, self.critic_state, state, latent_z, self.key)
                params_stack = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[s.params for s in self.ncbf_state])

                safety_layer_curriculum_coeff = get_safety_layer_curriculum_coeff(global_step)
                safe_action, constraint_active, delta_u = self.batched_ncbf_safety_layer(
                    action,
                    state,
                    last_actions,
                    last_states,
                    latent_z,
                    safety_layer_curriculum_coeff,
                    params_stack,
                )
                processed_action = safe_action
                processed_action = jax.device_get(processed_action)
                next_state, reward, terminated, truncated, info = self.env.step(processed_action)
                done = terminated | truncated
                actual_next_state = next_state.copy()
                for i, single_done in enumerate(done):
                    if single_done:
                        actual_next_state[i] = np.array(self.env.get_final_observation_at_index(info, i))
                        saving_return_buffer.append(self.env.get_final_info_value_at_index(info, "episode_return", i))
                        dones_this_rollout += 1
                for key, info_value in self.env.get_logging_info_dict(info).items():
                    step_info_collection.setdefault(key, []).extend(info_value)

                batch.states[step] = state
                batch.next_states[step] = actual_next_state
                batch.actions[step] = action
                batch.rewards[step] = reward
                batch.values[step] = value
                batch.history_stacks[step] = history_stacks
                batch.terminations[step] = terminated
                batch.log_probs[step] = log_prob
                batch.constraint_violated[step] = constraint_active
                batch.delta_u[step] = delta_u
                batch.dones[step] = done
                safety_layer_curriculum_coeffs[step] = safety_layer_curriculum_coeff

                next_history_stacks = np.roll(history_stacks, shift=-1, axis=1)
                next_history_stacks[:, -1] = next_state
                reset_history_stacks = np.tile(next_state[:, None, :], (1, self.env.nr_history_steps, 1))
                history_stacks = np.where(done[:, None, None], reset_history_stacks, next_history_stacks)
                last_states = np.where(done[:, None], next_state, state)
                last_actions = np.where(done[:, None], np.zeros_like(last_actions), processed_action)
                state = next_state
                global_step += self.nr_envs

            nr_episodes += dones_this_rollout
            
            acting_end_time = time.time()
            time_metrics["time/acting_time"] = acting_end_time - start_time

            batch_mean_constraint_violated = jnp.mean(batch.constraint_violated)
            batch_mean_delta_u = jnp.mean(batch.delta_u)

            # Calculating advantages and returns
            batch.advantages, batch.returns = calculate_gae_advantages(self.critic_state, self.encoder_state, batch.next_states, batch.history_stacks, batch.rewards, batch.terminations, batch.dones, batch.values)

            calc_adv_return_end_time = time.time()
            time_metrics["time/calc_adv_and_return_time"] = calc_adv_return_end_time - acting_end_time

            # updating the ncbf
            y_bool, mask_valid = window_any_done_next_H(batch.dones, batch.terminations, self.ncbf_H)  # get true if any done in next H steps for each env
            y = y_bool.astype(jnp.float32)
            batch.masks = mask_valid
            batch.y_targets = y

            # add batch to buffer
            self.replay_buffer.add_batch(
                states=batch.states,
                next_states=batch.next_states,
                actions=batch.actions,
                rewards=batch.rewards,
                terminations=batch.terminations,
                masks=batch.masks,
                y_targets=batch.y_targets
            )

            if self.ncbf_nr_minibatches > 0:
                self.ncbf_state[0], ncbf_metrics, self.key = train_ncbf(self.ncbf_state[0], self.replay_buffer.states, self.replay_buffer.next_states, self.replay_buffer.actions, self.replay_buffer.y_targets, self.replay_buffer.masks, self.key, self.ncbf_nr_minibatches)
                self.ncbf_state[1], ncbf_metrics, self.key = train_ncbf(self.ncbf_state[1], self.replay_buffer.states, self.replay_buffer.next_states, self.replay_buffer.actions, self.replay_buffer.y_targets, self.replay_buffer.masks, self.key, self.ncbf_nr_minibatches)
                self.ncbf_state[2], ncbf_metrics, self.key = train_ncbf(self.ncbf_state[2], self.replay_buffer.states, self.replay_buffer.next_states, self.replay_buffer.actions, self.replay_buffer.y_targets, self.replay_buffer.masks, self.key, self.ncbf_nr_minibatches)
                self.ncbf_state[3], ncbf_metrics, self.key = train_ncbf(self.ncbf_state[3], self.replay_buffer.states, self.replay_buffer.next_states, self.replay_buffer.actions, self.replay_buffer.y_targets, self.replay_buffer.masks, self.key, self.ncbf_nr_minibatches)
                self.ncbf_state[4], ncbf_metrics, self.key = train_ncbf(self.ncbf_state[4], self.replay_buffer.states, self.replay_buffer.next_states, self.replay_buffer.actions, self.replay_buffer.y_targets, self.replay_buffer.masks, self.key, self.ncbf_nr_minibatches)
            else:
                ncbf_metrics = {}
            # get scalar mean from ncbf_metrics
            for key, value in ncbf_metrics.items():
                ncbf_metrics[key] = value.item()

            mean_y = jnp.mean(batch.y_targets)
            ncbf_metrics['ncbf/mean_y'] = mean_y.item()
            ncbf_metrics['ncbf/mean_constraint_violated'] = batch_mean_constraint_violated.item()
            ncbf_metrics['ncbf/mean_delta_u'] = batch_mean_delta_u.item()
            ncbf_metrics['ncbf/safety_layer_curriculum_coeff'] = float(np.mean(safety_layer_curriculum_coeffs))

            # Optimizing
            self.policy_state, self.critic_state, optimization_metrics, self.key = update(
                self.policy_state, self.critic_state, self.encoder_state, self.ncbf_state[0],
                batch.states, batch.actions, batch.advantages, batch.returns, batch.values, batch.log_probs,
                batch.history_stacks,
                self.key
            )
            optimization_metrics = {key: value.item() for key, value in optimization_metrics.items()}
            nr_updates += self.nr_epochs * self.nr_minibatches

            optimizing_end_time = time.time()
            time_metrics["time/optimizing_time"] = optimizing_end_time - calc_adv_return_end_time


            # Evaluating
            evaluation_metrics = {}
            if global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1:
                self.set_eval_mode()
                state, info = self.env.reset()
                eval_history_stacks = np.tile(state[:, None, :], (1, self.env.nr_history_steps, 1))
                if "history_stack" in info:
                    eval_history_stacks = np.asarray(info["history_stack"])
                eval_last_states = state.copy()
                eval_last_actions = np.zeros((self.nr_envs,) + self.as_shape)
                eval_nr_episodes = 0
                evaluation_metrics = {"eval/episode_return": [], "eval/episode_length": []}
                while True:
                    latent_z = self.encoder.apply(self.encoder_state.params, eval_history_stacks)
                    raw_processed_action = get_deterministic_action(self.policy_state, state, latent_z)
                    params_stack = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[s.params for s in self.ncbf_state])
                    safe_action, constraint_active, delta_u = self.batched_ncbf_safety_layer(
                        raw_processed_action,
                        state,
                        eval_last_actions,
                        eval_last_states,
                        latent_z,
                        np.float32(1.0),
                        params_stack,
                    )
                    processed_action = safe_action
                    processed_action = jax.device_get(processed_action)
                    next_state, reward, terminated, truncated, info = self.env.step(processed_action)
                    done = terminated | truncated
                    for i, single_done in enumerate(done):
                        if single_done:
                            eval_nr_episodes += 1
                            evaluation_metrics["eval/episode_return"].append(self.env.get_final_info_value_at_index(info, "episode_return", i))
                            evaluation_metrics["eval/episode_length"].append(self.env.get_final_info_value_at_index(info, "episode_length", i))
                            if eval_nr_episodes == self.evaluation_episodes:
                                break
                    if eval_nr_episodes == self.evaluation_episodes:
                        break
                    next_eval_history_stacks = np.roll(eval_history_stacks, shift=-1, axis=1)
                    next_eval_history_stacks[:, -1] = next_state
                    reset_eval_history_stacks = np.tile(next_state[:, None, :], (1, self.env.nr_history_steps, 1))
                    eval_history_stacks = np.where(done[:, None, None], reset_eval_history_stacks, next_eval_history_stacks)
                    eval_last_states = np.where(done[:, None], next_state, state)
                    eval_last_actions = np.where(done[:, None], np.zeros_like(eval_last_actions), processed_action)
                    state = next_state
                evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics.items()}
                state, _ = self.env.reset()
                self.set_train_mode()
            
            evaluating_end_time = time.time()
            time_metrics["time/evaluating_time"] = evaluating_end_time - optimizing_end_time
            

            # Saving
            # Also only save when there were finished episodes this update
            if self.save_model and dones_this_rollout > 0:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()
            
            saving_end_time = time.time()
            time_metrics["time/saving_time"] = saving_end_time - evaluating_end_time
            time_metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (saving_end_time - start_time))


            # Logging
            self.start_logging(global_step)

            steps_metrics["steps/nr_env_steps"] = global_step
            steps_metrics["steps/nr_updates"] = nr_updates
            steps_metrics["steps/nr_episodes"] = nr_episodes

            rollout_info_metrics = {}
            env_info_metrics = {}
            if step_info_collection:
                info_names = list(step_info_collection.keys())
                for info_name in info_names:
                    metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                    metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                    mean_value = np.mean(step_info_collection[info_name])
                    if mean_value == mean_value:  # Check if mean_value is NaN
                        metric_dict[f"{metric_group}/{info_name}"] = mean_value
            
            combined_metrics = {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics, **ncbf_metrics}
            for key, value in combined_metrics.items():
                self.log(f"{key}", value, global_step)

            self.end_logging()


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

    def start_logging_ncbf_pretrain(self, steps):
        rlx_logger.info(f"pretrained NCBF for {steps} steps")

    def end_logging(self):
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

    
    def save(self):
        checkpoint = {
            "policy": self.policy_state,
            "critic": self.critic_state,
            "ncbf": self.ncbf_state,
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.best_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as f:
            json.dump(self.config.algorithm.to_dict(), f)
        shutil.make_archive(f"{self.save_path}/{self.best_model_file_name}", "zip", f"{self.save_path}/tmp")
        os.rename(f"{self.save_path}/{self.best_model_file_name}.zip", f"{self.save_path}/{self.best_model_file_name}")
        shutil.rmtree(f"{self.save_path}/tmp")

        if self.track_wandb:
            wandb.save(f"{self.save_path}/{self.best_model_file_name}", base_path=self.save_path)


    def load(config, env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(splitted_path[:-1]))
        checkpoint_file_name = splitted_path[-1]
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"

        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():

            if isinstance(value, dict):
                # apply nested keys unless explicitly set by the caller
                if not (key in config.algorithm or hasattr(config.algorithm, key)):
                    continue
                for sub_key, sub_value in value.items():
                    full_param = f"algorithm.{key}.{sub_key}"
                    if full_param in explicitly_set_algorithm_params:
                        continue
                    try:
                        parent = getattr(config.algorithm, key)
                        if isinstance(parent, dict):
                            parent[sub_key] = sub_value
                        else:
                            setattr(parent, sub_key, sub_value)
                    except Exception:
                        try:
                            config.algorithm[key][sub_key] = sub_value
                        except Exception:
                            pass
                continue
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
    
    def validate_dynamics_model_on_the_real_robot(self):
        # quick and dirty validation of the system dynamics function and the ncbf using real data collected from the robot
        file_list = "/home/siwei/Documents/repos/rl-x-NCBF/misc/data_paths.txt"
        target_path = "/home/siwei/Documents/repos/rl-x-NCBF/misc/real_robot_rollouts_validation"

        os.makedirs(target_path, exist_ok=True)

        def _load_rollout_paths(self, file_list_path: str) -> list[str]:
            """
            Read newline-separated rollout paths from `file_list_path`.
            Lines starting with # or blank lines are ignored.
            """
            if not os.path.isfile(file_list_path):
                raise FileNotFoundError(f"Rollout file list not found: {file_list_path}")

            with open(file_list_path, "r", encoding="utf-8") as file_handle:
                paths = [
                    os.path.abspath(line.strip())
                    for line in file_handle
                    if line.strip() and not line.lstrip().startswith("#")
                ]

            if not paths:
                raise ValueError(f"No rollout paths found in: {file_list_path}")

            return paths

        rollout_paths = _load_rollout_paths(self, file_list)

        def _load_rollouts_from_json(self, file_paths: list[str]) -> list[list]:
            """
            Load rollout datasets stored as JSON lists.
            """
            datasets = []
            for path in file_paths:
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Rollout file not found: {path}")
                try:
                    with open(path, "r", encoding="utf-8") as file_handle:
                        rollout_data = json.load(file_handle)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse rollout file: {path}") from exc
                if not isinstance(rollout_data, dict):
                    raise ValueError(f"Rollout file must contain a JSON dict: {path}")
                datasets.append(rollout_data)
            return datasets

        rollouts = _load_rollouts_from_json(self, rollout_paths)

        for idx, rollout in enumerate(rollouts):
            pos = np.array(rollout["pos"])
            qpos = np.array(rollout["joint_positions"])
            qvel = np.array(rollout["joint_velocities"])
            actions = np.array(rollout["action"])

            x_true = np.concatenate([qpos, qvel], axis=-1)
            x_pred = np.stack([self.system_dynamics_function(pos_row, x_row, a_row) for pos_row, x_row, a_row in zip(pos[:-1], x_true[:-1], actions[:-1])])
            pred_error = x_true[1:, ...] - x_pred[..., self.env.ncbf_obs_in_dynamics_state_idx]
            pred_error_norm = np.linalg.norm(pred_error, axis=-1)
            print("prediction error: ", np.mean(pred_error_norm))

            h_input = x_true
            prediction_mean, prediction_std, _ = self.ncbf_apply(
                jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[s.params for s in self.ncbf_state]),
                h_input[:-1]
            )

    def test(self, episodes):
        # self.validate_dynamics_model_on_the_real_robot()
        # return

        # @jax.jit
        def get_action(policy_state: TrainState, state: np.ndarray, last_state: np.ndarray, last_action: np.ndarray, history_latent: np.ndarray):
            action_mean, action_logstd = self.policy.apply(policy_state.params, state, history_latent)
            raw_processed_action = self.get_processed_action(action_mean)

            sampling_ratio = self.action_noise_sampling_ratio
            self.key, sampling_key, action_offset_key = jax.random.split(self.key, 3)
            if_sampling = jax.random.uniform(sampling_key, (1,)) < sampling_ratio
            # print("raw action: ", raw_processed_action)
            # add action noise

            raw_processed_action = raw_processed_action + jax.random.normal(action_offset_key,
                                                                    raw_processed_action.shape) * 1 * if_sampling

            # if last_action is not None:
            #     params_stack = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[s.params for s in self.ncbf_state])
            #     safe_action, constraint_active, delta_u = self.batched_ncbf_safety_layer(raw_processed_action, state, last_action, last_state, latent_z, params_stack)
            # else:
            #     safe_action = raw_processed_action
            #     constraint_active = jnp.array([0])
            #     delta_u = jnp.array([0])
            #     h_u0 = jnp.array([0])
            # temporarily disabled
            safe_action = raw_processed_action
            constraint_active = 0
            delta_u = 0
            return safe_action, raw_processed_action, constraint_active, delta_u

        rollout_path = self.save_path.replace("models", "rollout")
        os.makedirs(rollout_path, exist_ok=True)

        rollouts = []
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, info = self.env.reset()
            history_stack = info["history_stack"][0]
            self.env.envs[0].internal_state["safe_prediction"] = 1
            previous_state = np.stack(info["last_state"])
            last_action = np.stack(info["last_action"])

            # hard coded load commands
            # self.env.envs[0].command_function._load_random_trajectory(i)
            rollout_dict = dict(states=[], actions=[], rewards=[], dones=[], safe_prediction=[], predictions=[],
                                delta_u=[], constraint_active=[], raw_action=[], safe_action=[], joint_position_obs=[],
                                h_u0=[], x_next_true=[], next_step_pred=[], ret=[])

            while not done:
                latent_z = self.encoder.apply(self.encoder_state.params, history_stack[None, ...])

                processed_action, raw_action, constraint_active, delta_u = get_action(self.policy_state, state, previous_state, last_action, latent_z)
                # params_stack = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[s.params for s in self.ncbf_state])
                h_input = jnp.concatenate([state[:, self.env.envs[0].ncbf_observation_indices], processed_action, latent_z], axis=-1)
                # prediction_mean, prediction_std, predictions = self.ncbf_state.apply_fn(self.ncbf_state.params, h_input)
                prediction_raw = self.ncbf_state.apply_fn(self.ncbf_state.params, h_input)
                prediction_mean, _ = split_ncbf_output(
                    prediction_raw,
                    self.ncbf_output_distribution,
                    self.ncbf_min_log_std,
                    self.ncbf_max_log_std,
                )

                def component_softmin(preds):
                    """
                    preds: (E, ..., K)
                    softmin over the last dimension K for each ensemble member
                    returns: (E, ...)
                    """
                    beta = 10
                    weights = jax.nn.softmax(-beta * preds, axis=-1)
                    preds_softmin = jnp.sum(weights * preds, axis=-1)
                    return preds_softmin



                predictions = 0

                next_step_prediction = self.decoder.apply(self.decoder_state.params, latent_z, state, processed_action)
                prediction_denormed = next_step_prediction / (1 + next_step_prediction)

                self.env.envs[0].internal_state["safe_prediction"] = component_softmin(prediction_mean) # prediction_mean
                previous_state = state
                state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                history_stack = info["history_stack"][0]

                done = terminated | truncated
                episode_return += reward

                if terminated:
                    print("ncbf prediction: ", prediction_mean)

                last_action = processed_action

                rollout_dict["predictions"].append(predictions)
                rollout_dict["dones"].append(done)
                rollout_dict["safe_prediction"].append(prediction_mean)
                rollout_dict["delta_u"].append(delta_u)
                rollout_dict["constraint_active"].append(constraint_active)
                rollout_dict["raw_action"].append(raw_action)
                rollout_dict["safe_action"].append(processed_action)
                rollout_dict["next_step_pred"].append(next_step_prediction)

                joint_pos = (state[0, self.env.envs[0].joint_positions_obs_idx] * 3.14) + self.env.envs[0].internal_state["actuator_joint_nominal_positions"]
                rollout_dict["joint_position_obs"].append(joint_pos)

            rollout_dict["ret"].append(episode_return)
            rollouts.append(rollout_dict)
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")


        # save rollout file
        rollout_file = os.path.join(rollout_path, f"{self.rollout_save_name}.pkl")
        with open(rollout_file, "wb") as f:
            pickle.dump(rollouts, f)


    
            
    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
