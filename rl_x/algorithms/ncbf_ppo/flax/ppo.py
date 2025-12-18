import os
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
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.ncbf_ppo.flax.general_properties import GeneralProperties
from rl_x.algorithms.ncbf_ppo.flax.policy import get_policy
from rl_x.algorithms.ncbf_ppo.flax.critic import get_critic
from rl_x.algorithms.ncbf_ppo.flax.ncbf import get_ncbf
from rl_x.algorithms.ncbf_ppo.flax.batch import Batch
from rl_x.algorithms.ncbf_ppo.flax.replay_buffer import ReplayBuffer

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

        self.ncbf_n_ensemble = config.algorithm.ncbf.n_enssemble
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

        self.ncbf_buffer_size = config.algorithm.ncbf_buffer.buffer_size
        self.ncbf_pretrain_steps = config.algorithm.ncbf.pretrain.nr_steps
        self.ncbf_pretrain_nr_minibatches = config.algorithm.ncbf.pretrain.nr_minibatches

        # assert ncbf nr_steps * nr_envs must be a multiple of ncbf batchsize
        if (self.nr_steps * self.nr_envs) % self.ncbf_minibatch_size != 0:
            raise ValueError("NCBF batch size must divide evenly into nr_steps * nr_envs.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, ncbf_key = jax.random.split(self.key, 4)

        self.os_shape = env.single_observation_space.shape
        self.as_shape = env.single_action_space.shape
        
        self.policy, self.get_processed_action = get_policy(config, env)
        self.ncbf, self.ncbf_apply, self.batched_ncbf_safety_layer, self.ncbf_safety_layer = get_ncbf(config, env)
        self.critic = get_critic(config, env)
        self.replay_buffer = ReplayBuffer(capacity=config.algorithm.ncbf_buffer.buffer_size, nr_envs=self.nr_envs, os_shape=self.os_shape, as_shape=self.as_shape, rng=ncbf_key)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        state = jnp.array([env.single_observation_space.sample()])

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, state),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, state),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        ncbf_keys = jax.random.split(ncbf_key, self.ncbf_n_ensemble)
        self.ncbf_state = [
            TrainState.create(
                apply_fn=self.ncbf[i].apply,
                params=self.ncbf[i].init(ncbf_keys[i], state),
                tx=optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.inject_hyperparams(optax.adam)(learning_rate=config.algorithm.ncbf.lr),
                )
            )
            for i in range(self.ncbf_n_ensemble)
        ]

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
        def get_action_and_value(policy_state: TrainState, critic_state: TrainState, state: np.ndarray, key: jax.random.PRNGKey):
            action_mean, action_logstd = self.policy.apply(policy_state.params, state)
            action_std = jnp.exp(action_logstd)
            key, subkey = jax.random.split(key)
            action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
            log_prob = -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
            value = self.critic.apply(critic_state.params, state)
            processed_action = self.get_processed_action(action)
            return processed_action, action, value.reshape(-1), log_prob.sum(1), key

        @jax.jit
        def calculate_gae_advantages(critic_state: TrainState, next_states: np.ndarray, rewards: np.ndarray, terminations: np.ndarray, values: np.ndarray):
            def compute_advantages(carry, t):
                prev_advantage = carry[0]
                advantage = delta[t] + self.gamma * self.gae_lambda * (1 - terminations[t]) * prev_advantage
                return (advantage,), advantage

            next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)
            delta = rewards + self.gamma * next_values * (1.0 - terminations) - values
            init_advantages = delta[-1]
            _, advantages = jax.lax.scan(compute_advantages, (init_advantages,), jnp.arange(self.nr_steps - 2, -1, -1))
            advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
            returns = advantages + values
            return advantages, returns
        

        @jax.jit
        def update(policy_state: TrainState, critic_state: TrainState, ncbf_state: TrainState,
                   states: np.ndarray, actions: np.ndarray, advantages: np.ndarray, returns: np.ndarray, values: np.ndarray, log_probs: np.ndarray,
                   key: jax.random.PRNGKey):
            def loss_fn(policy_params, critic_params, ncbf_params, state_b, action_b, log_prob_b, return_b, advantage_b):
                # Policy loss
                action_mean, action_logstd = self.policy.apply(policy_params, state_b)
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
                new_value = self.critic.apply(critic_params, state_b)
                critic_loss = 0.5 * (new_value - return_b) ** 2

                # anticipation loss
                safe_action_b, _, _ = self.ncbf_safety_layer(action_b, state_b, ncbf_params)
                anticipation_loss = 0.5 * (action_mean - safe_action_b) ** 2

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

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, None, 0, 0, 0, 0, 0), out_axes=0)
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
                    minibatch_advantages
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
            mean_metrics["policy/std_dev"] = jnp.mean(jnp.exp(policy_state.params["params"]["policy_logstd"]))

            return policy_state, critic_state, mean_metrics, key

        @partial(jax.jit, static_argnums=(6,))
        def train_ncbf(ncbf_state: TrainState, states: np.ndarray, next_states: np.ndarray, y_targets: np.ndarray, masks: np.ndarray,
                       key: jax.random.PRNGKey, nr_minibatches: int):
            """
            ncbf_state: TrainState
            states: (T, E, D)
            next_states: (T, E, D)
            dones: (T, E)
            terminates: (T, E)
            """

            @jax.jit
            def loss_fn(params, minib_obs, minib_nxt, minib_y, minib_mask):
                gamma_c = self.ncbf_gamma_c
                h_x = ncbf_state.apply_fn(params, minib_obs)  # [B,T]
                h_xn = ncbf_state.apply_fn(params, minib_nxt)

                # (1) BCE classification: logits = h(x) - gamma_c
                logits = h_x - gamma_c
                # BCE with logits: softplus(z) - y*z
                num = jnp.sum(minib_mask * (jax.nn.softplus(logits) - minib_y * logits))
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
                        y = self.ncbf[0].apply(params, x_single)
                        return jnp.sum(y)

                    # Vectorize grad over batch
                    grad_h = jax.vmap(jax.grad(f_single))(x)  # [B,T,D]
                    grad_norm = jnp.linalg.norm(grad_h, axis=-1)  # [B,T]
                    lip_violation = jnp.maximum(0.0, grad_norm - max_norm)
                    lip_loss = jnp.mean(lip_violation)
                    return lip_loss

                #
                lip_loss = grad_norm_penalty(ncbf_state.params, minib_obs, self.ncbf_L_target)

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
                )
                return total, metrics

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0), out_axes=0)
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
                minib_y = y_targets[minibatch_indices]
                minib_mask = masks[minibatch_indices]


                (loss, metrics), ncbf_grads = grad_ncbf_loss_fn(
                    ncbf_state.params,
                    minib_obs,
                    minib_nxt,
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
        def get_deterministic_action(policy_state: TrainState, state: np.ndarray):
            action_mean, action_logstd = self.policy.apply(policy_state.params, state)
            return self.get_processed_action(action_mean)

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
            delta_u=np.zeros((self.nr_steps, self.nr_envs))
        )

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        # pre-sampling and pretrain ncbf
        state, _ = self.env.reset()
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
                delta_u=np.zeros((self.ncbf_pretrain_steps, self.nr_envs))
            )

            for step in range(self.nr_steps):
                raw_processed_action, action, value, log_prob, self.key = get_action_and_value(self.policy_state, self.critic_state, state, self.key)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated | truncated
                actual_next_state = next_state.copy()
                for i, single_done in enumerate(done):
                    if single_done:
                        actual_next_state[i] = np.array(self.env.get_final_observation_at_index(info, i))
                        saving_return_buffer.append(self.env.get_final_info_value_at_index(info, "episode_return", i))

                ncbf_batch.states[step] = state
                ncbf_batch.next_states[step] = actual_next_state
                ncbf_batch.actions[step] = action
                ncbf_batch.rewards[step] = reward
                ncbf_batch.values[step] = value
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
            keys = jax.random.split(self.key, 2 + 1)
            self.key = keys[0]

            self.ncbf_state[0], ncbf_metrics, self.key = train_ncbf(self.ncbf_state[0],
                                                                 self.replay_buffer.states,
                                                                 self.replay_buffer.next_states,
                                                                 self.replay_buffer.y_targets,
                                                                 self.replay_buffer.masks,
                                                                 keys[1],
                                                                 self.ncbf_pretrain_nr_minibatches)

            self.ncbf_state[1], ncbf_metrics, self.key = train_ncbf(self.ncbf_state[1],
                                                                 self.replay_buffer.states,
                                                                 self.replay_buffer.next_states,
                                                                 self.replay_buffer.y_targets,
                                                                 self.replay_buffer.masks,
                                                                 keys[2],
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

        state, _ = self.env.reset()
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        steps_metrics = {}
        while global_step < self.total_timesteps:
            start_time = time.time()
            time_metrics = {}

            # Acting
            dones_this_rollout = 0
            step_info_collection = {}
            for step in range(self.nr_steps):
                _, action, value, log_prob, self.key = get_action_and_value(self.policy_state, self.critic_state, state, self.key)
                safe_action, constraint_active, delta_u = self.batched_ncbf_safety_layer(action, state, self.ncbf_state[0].params)
                processed_action = safe_action
                next_state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
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
                batch.terminations[step] = terminated
                batch.log_probs[step] = log_prob
                batch.constraint_violated[step] = constraint_active
                batch.delta_u[step] = delta_u
                batch.dones[step] = done

                state = next_state
                global_step += self.nr_envs

            nr_episodes += dones_this_rollout
            
            acting_end_time = time.time()
            time_metrics["time/acting_time"] = acting_end_time - start_time

            batch_mean_constraint_violated = jnp.mean(batch.constraint_violated)
            batch_mean_delta_u = jnp.mean(batch.delta_u)

            # Calculating advantages and returns
            batch.advantages, batch.returns = calculate_gae_advantages(self.critic_state, batch.next_states, batch.rewards, batch.terminations, batch.values)
            
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
                self.ncbf_state[0], ncbf_metrics, self.key = train_ncbf(self.ncbf_state[0], self.replay_buffer.states, self.replay_buffer.next_states, self.replay_buffer.y_targets, self.replay_buffer.masks, self.key, self.ncbf_nr_minibatches)
                self.ncbf_state[1], ncbf_metrics, self.key = train_ncbf(self.ncbf_state[1], self.replay_buffer.states, self.replay_buffer.next_states, self.replay_buffer.y_targets, self.replay_buffer.masks, self.key, self.ncbf_nr_minibatches)
            else:
                ncbf_metrics = {}
            # get scalar mean from ncbf_metrics
            for key, value in ncbf_metrics.items():
                ncbf_metrics[key] = value.item()

            mean_y = jnp.mean(batch.y_targets)
            ncbf_metrics['ncbf/mean_y'] = mean_y.item()
            ncbf_metrics['ncbf/mean_constraint_violated'] = batch_mean_constraint_violated.item()
            ncbf_metrics['ncbf/mean_delta_u'] = batch_mean_delta_u.item()

            # Optimizing
            self.policy_state, self.critic_state, optimization_metrics, self.key = update(
                self.policy_state, self.critic_state, self.ncbf_state[0],
                batch.states, batch.actions, batch.advantages, batch.returns, batch.values, batch.log_probs,
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
                state, _ = self.env.reset()
                eval_nr_episodes = 0
                evaluation_metrics = {"eval/episode_return": [], "eval/episode_length": []}
                while True:
                    raw_processed_action = get_deterministic_action(self.policy_state, state)
                    safe_action, constraint_active, delta_u = self.ncbf_safety_layer(raw_processed_action, state, self.ncbf_state[0].params)
                    processed_action = safe_action
                    state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
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
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.ncbf_state = checkpoint["ncbf"]

        shutil.rmtree(checkpoint_dir)

        return model
    

    def test(self, episodes):
        @jax.jit
        def get_action(policy_state: TrainState, state: np.ndarray):
            action_mean, action_logstd = self.policy.apply(policy_state.params, state)
            raw_processed_action = self.get_processed_action(action_mean)
            safe_action, constraint_active, delta_u = self.batched_ncbf_safety_layer(raw_processed_action, state, self.ncbf_state[0].params)
            return safe_action, raw_processed_action, constraint_active, delta_u
        
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.env.reset()
            self.env.envs[0].internal_state["safe_prediction"] = 1

            while not done:
                processed_action, raw_action, constraint_active, delta_u = get_action(self.policy_state, state)
                prediction_mean, prediction_std = self.ncbf_apply(self.ncbf_state, state)
                # processed_action, constraint_active, delta_u = self.batched_ncbf_safety_layer(processed_action,
                #                                                                               state,
                #                                                                               self.ncbf_state.params)

                # print("prediction: ", prediction, "constraint_active: ", constraint_active, "delta_u: ", delta_u)
                print("prediction mean: ", prediction_mean, "prediction std: ", prediction_std)
                self.env.envs[0].internal_state["safe_prediction"] = prediction_mean
                state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")
    
            
    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
