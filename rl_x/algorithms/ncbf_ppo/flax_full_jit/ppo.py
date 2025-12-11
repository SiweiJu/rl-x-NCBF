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
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb
from jax import lax
from tensorflow.python.training.training_util import global_step

from rl_x.algorithms.ncbf_ppo.flax_full_jit.general_properties import GeneralProperties
from rl_x.algorithms.ncbf_ppo.flax_full_jit.policy import get_policy
from rl_x.algorithms.ncbf_ppo.flax_full_jit.critic import get_critic
from rl_x.algorithms.ncbf_ppo.flax_full_jit.ncbf import get_ncbf

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
        self.entropy_coef = config.algorithm.entropy_coef
        self.critic_coef = config.algorithm.critic_coef
        self.anticipation_coef = config.algorithm.ncbf.policy_loss_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = config.algorithm.total_timesteps // self.batch_size
        self.nr_minibatches = self.batch_size // self.minibatch_size

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

        self.ncbf_buffer_size = config.algorithm.ncbf_buffer.buffer_size * self.nr_steps * self.nr_envs
        self.ncbf_pretrain_steps = config.algorithm.ncbf.pretrain.nr_steps * self.nr_steps
        rlx_logger.info(f"NCBF pretrain steps:{self.ncbf_pretrain_steps * self.nr_envs}")
        rlx_logger.info(f"INFO - NCBF buffer size:{self.ncbf_buffer_size}")

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

        if self.evaluation_and_save_frequency % self.batch_size != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of batch size")
        
        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet. This is mainly limited by not being able to log mutliple wandb runs at the same time.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, reset_key, ncbf_key = jax.random.split(self.key, 5)
        reset_key = jax.random.split(reset_key, 1)

        self.policy, self.get_processed_action = get_policy(self.config, self.env)
        self.critic = get_critic(self.config, self.env)

        self.ncbf, self.batched_ncbf_safety_layer, self.ncbf_safety_layer = get_ncbf(config, env)
        self.ncbf.apply = jax.jit(self.ncbf.apply)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        env_state = self.env.reset(reset_key, False)

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

        self.ncbf_state = TrainState.create(
            apply_fn=self.ncbf.apply,
            params=self.ncbf.init(ncbf_key, env_state.next_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=config.algorithm.ncbf.lr),
            )
        )

        if self.save_model:
            os.makedirs(self.save_path)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

 
    def train(self):
        def jitable_train_function(key, parallel_seed_id):

            # Single step rollout, used for training and ncbf pretraining, therefore defined at first
            def single_rollout(single_rollout_carry, _):
                policy_state, critic_state, ncbf_state, env_state, key = single_rollout_carry

                key, subkey = jax.random.split(key)
                observation = env_state.next_observation
                action_mean, action_logstd = self.policy.apply(policy_state.params, observation)
                action_std = jnp.exp(action_logstd)
                action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                log_prob = (-0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(
                    2.0 * jnp.pi) - action_logstd).sum(1)
                raw_processed_action = self.get_processed_action(action)

                processed_action, constraint_active, delta_u = self.batched_ncbf_safety_layer(raw_processed_action, observation,
                                                                                      ncbf_state.params)
                value = self.critic.apply(critic_state.params, observation).squeeze(-1)

                env_state = self.env.step(env_state, processed_action)
                done = env_state.terminated | env_state.truncated
                transition = (observation, env_state.actual_next_observation, action, env_state.reward, value,
                              env_state.terminated, done, log_prob, env_state.info, constraint_active, delta_u)

                if self.render:
                    def render(env_state):
                        return self.env.render(env_state)

                    env_state = jax.experimental.io_callback(render, env_state, env_state)

                return (policy_state, critic_state, ncbf_state, env_state, key), transition

            @jax.jit
            def window_any_done_next_H(dones: jnp.array, terminates: jnp.array, H: int):
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
                # any terminate in (t, t+H] -> y[t] = False, else True

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

                any_term_next_H = _future_event_within_H(terminates, H)  # [T, N]
                y = ~any_term_next_H  # [T, N] bool

                # mask is simply "not done for next H"
                any_done_next_H = _future_event_within_H(dones, H)  # [T, N]
                mask = ~any_done_next_H  # [T, N] bool

                return y, mask

            # @jax.jit
            # def add_to_ncbf_replay_buffer(ncbf_replay_buffer, new_data):
            #     """
            #     ncbf_replay_buffer: dict with
            #       - 'states'       : [C, N_env, ...]
            #       - 'next_states'  : [C, N_env, ...]
            #       - 'actions'      : [C, N_env, ...]
            #       - 'dones'        : [C, N_env]
            #       - 'terminations' : [C, N_env]
            #       - 'y_target'     : [C, N_env]
            #       - 'masks'        : [C, N_env]
            #       - 'pos'          : scalar int32
            #       - 'size'         : scalar int32
            #
            #     new_data: dict with
            #       - 'states'       : [T, N_env, ...]
            #       - 'next_states'  : [T, N_env, ...]
            #       - 'actions'      : [T, N_env, ...]
            #       - 'dones'        : [T, N_env]
            #       - 'terminations' : [T, N_env]
            #       - 'y_target'     : [T, N_env]
            #       - 'masks'        : [T, N_env]
            #     """
            #     capacity = ncbf_replay_buffer["states"].shape[0]
            #     pos = ncbf_replay_buffer["pos"]
            #     size = ncbf_replay_buffer["size"]
            #
            #     batch_size = new_data["states"].shape[0]
            #
            #     first_chunk = jnp.minimum(batch_size, capacity - pos)
            #     second_chunk = batch_size - first_chunk
            #
            #     def write_chunk(buf, src, start, length):
            #         # length is traced, but slice on src is fine if leading dim is static
            #         src_part = src.at[:length]
            #         # pad start indices for all dims: (start, 0, 0, ...)
            #         start_indices = (start,) + (0,) * (buf.ndim - 1)
            #         return lax.dynamic_update_slice(buf, src_part, start_indices)
            #
            #
            #     def write_chunk_wrap(buf, src, start, first_len, second_len):
            #         # tail part
            #         buf = write_chunk(buf, src, start, first_len)
            #         # head part
            #         src_head = src[first_len:first_len + second_len]
            #         start_indices = (0,) + (0,) * (buf.ndim - 1)
            #         return lax.dynamic_update_slice(buf, src_head, start_indices)
            #
            #     def write_all_fields(rb, nd, pos, first_chunk, second_chunk):
            #         wrap = second_chunk > 0
            #
            #         def write_no_wrap(rb_inner):
            #             rb_inner["states"] = write_chunk(rb_inner["states"], nd["states"], pos, first_chunk)
            #             rb_inner["next_states"] = write_chunk(rb_inner["next_states"], nd["next_states"], pos,
            #                                                   first_chunk)
            #             rb_inner["actions"] = write_chunk(rb_inner["actions"], nd["actions"], pos, first_chunk)
            #             rb_inner["dones"] = write_chunk(rb_inner["dones"], nd["dones"], pos, first_chunk)
            #             rb_inner["terminations"] = write_chunk(rb_inner["terminations"], nd["terminations"], pos,
            #                                                    first_chunk)
            #             rb_inner["y_target"] = write_chunk(rb_inner["y_target"], nd["y_target"], pos, first_chunk)
            #             rb_inner["masks"] = write_chunk(rb_inner["masks"], nd["masks"], pos, first_chunk)
            #             return rb_inner
            #
            #         def write_with_wrap(rb_inner):
            #             rb_inner["states"] = write_chunk_wrap(
            #                 rb_inner["states"], nd["states"], pos, first_chunk, second_chunk
            #             )
            #             rb_inner["next_states"] = write_chunk_wrap(
            #                 rb_inner["next_states"], nd["next_states"], pos, first_chunk, second_chunk
            #             )
            #             rb_inner["actions"] = write_chunk_wrap(
            #                 rb_inner["actions"], nd["actions"], pos, first_chunk, second_chunk
            #             )
            #             rb_inner["dones"] = write_chunk_wrap(
            #                 rb_inner["dones"], nd["dones"], pos, first_chunk, second_chunk
            #             )
            #             rb_inner["terminations"] = write_chunk_wrap(
            #                 rb_inner["terminations"], nd["terminations"], pos, first_chunk, second_chunk
            #             )
            #             rb_inner["y_target"] = write_chunk_wrap(
            #                 rb_inner["y_target"], nd["y_target"], pos, first_chunk, second_chunk
            #             )
            #             rb_inner["masks"] = write_chunk_wrap(
            #                 rb_inner["masks"], nd["masks"], pos, first_chunk, second_chunk
            #             )
            #             return rb_inner
            #
            #         rb = jax.lax.cond(wrap, write_with_wrap, write_no_wrap, rb)
            #         return rb
            #
            #     ncbf_replay_buffer = write_all_fields(
            #         ncbf_replay_buffer, new_data, pos, first_chunk, second_chunk
            #     )
            #
            #     new_pos = (pos + batch_size) % capacity
            #     new_size = jnp.minimum(capacity, size + batch_size)
            #
            #     ncbf_replay_buffer["pos"] = new_pos
            #     ncbf_replay_buffer["size"] = new_size
            #
            #     return ncbf_replay_buffer

            @partial(jax.jit, static_argnums=(3,))
            def train_ncbf(ncbf_state: TrainState, replay_buffer: dict,
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
                        x: (input_dim)
                        apply_fn: model.apply
                        params: model parameters

                        Returns: scalar penalty = E[(max(0, ||∇_x f(x)||_2 - target_norm), 0))^2]
                        """

                        def f_single(x_single):
                            # shape (output_dim,) -> reduce to scalar
                            x_single = x_single[None, ...]  # [1, D]
                            y = self.ncbf.apply(params, x_single)
                            return jnp.sum(y)

                        # Vectorize grad over batch
                        grad_h = jax.grad(f_single)(x)  # [D]
                        grad_norm = jnp.linalg.norm(grad_h)  # [1]
                        lip_violation = jnp.maximum(0.0, grad_norm - max_norm)
                        lip_loss = jnp.mean(lip_violation)
                        return lip_loss, grad_norm

                    #
                    lip_loss, grad_norm = grad_norm_penalty(ncbf_state.params, minib_obs, self.ncbf_L_target)

                    # (4) weight decay
                    wd_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree.leaves(params))

                    total = (self.ncbf_w_clf * clf_loss +
                             self.ncbf_w_cbf * cbf_loss +
                             self.ncbf_w_lip * lip_loss +
                             self.ncbf_w_wd * wd_loss)

                    metrics = dict(
                        total_loss=total,
                        clf_loss=clf_loss,
                        cbf_loss=cbf_loss,
                        lip_loss=lip_loss,
                        wd_loss=wd_loss,
                        grad_norm=grad_norm,
                    )
                    return total, metrics

                vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0), out_axes=0)
                safe_mean = lambda x: jnp.mean(x) if x is not None else x
                mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
                grad_ncbf_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=0, has_aux=True)

                def do_train(carry):
                    ncbf_state, key = carry

                    # ---------- one minibatch step ----------
                    @jax.jit
                    def ncbf_minibatch_update(carry, _):
                        ncbf_state, key = carry

                        key, replay_buffer_key  = jax.random.split(key, 2)

                        idx1 = jax.random.randint(replay_buffer_key, (self.minibatch_size,), 0, replay_buffer["size"])
                        idx2 = jax.random.randint(replay_buffer_key, (self.minibatch_size,), 0, self.nr_envs)

                        states = replay_buffer["states"][idx1, idx2]
                        next_states = replay_buffer["next_states"][idx1, idx2]
                        y_targets = replay_buffer["y_target"][idx1, idx2]
                        masks = replay_buffer["masks"][idx1, idx2]

                        (loss, metrics), ncbf_grads = grad_ncbf_loss_fn(
                            ncbf_state.params,
                            states,
                            next_states,
                            y_targets,
                            masks,
                        )
                        metrics["grad_norm"] = optax.global_norm(ncbf_grads)
                        new_state = ncbf_state.apply_gradients(grads=ncbf_grads)

                        carry = (new_state, key)
                        return carry, metrics

                    init_carry = (ncbf_state, key)
                    (ncbf_state, key), metrics = jax.lax.scan(ncbf_minibatch_update, init_carry, jnp.arange(nr_minibatches))

                    mean_metrics = {  # fixed structure
                        "ncbf/clf_loss": jnp.mean(metrics["clf_loss"]),
                        "ncbf/cbf_loss": jnp.mean(metrics["cbf_loss"]),
                        "ncbf/lip_loss": jnp.mean(metrics["lip_loss"]),
                        "ncbf/wd_loss": jnp.mean(metrics["wd_loss"]),
                        "ncbf/total_loss": jnp.mean(metrics["total_loss"]),
                        "ncbf/grad_norm": jnp.mean(metrics["grad_norm"]),
                        "ncbf/lr": ncbf_state.opt_state[1].hyperparams["learning_rate"],
                    }

                    return (ncbf_state, key), mean_metrics

                def skip_train(carry):
                    ncbf_state, key = carry
                    zero_metrics = {
                        "ncbf/clf_loss": jnp.array(0.0),
                        "ncbf/cbf_loss": jnp.array(0.0),
                        "ncbf/lip_loss": jnp.array(0.0),
                        "ncbf/wd_loss": jnp.array(0.0),
                        "ncbf/total_loss": jnp.array(0.0),
                        "ncbf/grad_norm": jnp.array(0.0),
                        "ncbf/lr": ncbf_state.opt_state[1].hyperparams["learning_rate"],
                    }
                    return (ncbf_state, key), zero_metrics

                (ncbf_state, key), mean_metrics = jax.lax.cond(
                    nr_minibatches > 0,
                    do_train,
                    skip_train,
                    operand=(ncbf_state, key),
                )
                return ncbf_state, mean_metrics, key

            key, reset_key = jax.random.split(key, 2)
            reset_keys = jax.random.split(reset_key, self.nr_envs)
            env_state = self.env.reset(reset_keys, False)

            policy_state = self.policy_state
            critic_state = self.critic_state
            ncbf_state = self.ncbf_state

            # initialize ncbf replay buffer
            capacity = int(self.ncbf_buffer_size // self.nr_envs)
            ncbf_replay_buffer = {
                "states": jnp.zeros((capacity, self.nr_envs) + (self.os_shape[0],), dtype=jnp.float32),
                "next_states": jnp.zeros((capacity, self.nr_envs) + (self.os_shape[0],), dtype=jnp.float32),
                "actions": jnp.zeros((capacity, self.nr_envs) + (self.as_shape[0],), dtype=jnp.float32),
                "dones": jnp.zeros((capacity, self.nr_envs), dtype=jnp.bool),
                "terminations": jnp.zeros((capacity, self.nr_envs), dtype=jnp.bool),
                "y_target": jnp.zeros((capacity, self.nr_envs), dtype=jnp.float32),
                "masks": jnp.zeros((capacity, self.nr_envs), dtype=jnp.bool),
                "pos": jnp.zeros((), dtype=jnp.int32),
                "size": jnp.zeros((), dtype=jnp.int32)
            }

            # prefill ncbf buffer
            if self.ncbf_pretrain_steps > 0:
                # get all transitions to a batch and then add to buffer
                # need to get the entire batch first so that y_target for ncbf can be calculated properly
                init_carry = (policy_state, critic_state, ncbf_state, env_state, subkey)
                single_rollout_carry, batch = jax.lax.scan(single_rollout, init_carry, None,
                                                           self.ncbf_pretrain_steps)
                policy_state, critic_state, ncbf_state, env_state, key = single_rollout_carry
                states, next_states, actions, rewards, values, terminations, dones, log_probs, infos, constraints_active, delta_u = batch

                # process the batch data to get mask and y_target
                y_bool, masks = window_any_done_next_H(dones, terminations, self.ncbf_H)
                y_target = y_bool.astype(jnp.float32)

                ncbf_replay_buffer["states"] = ncbf_replay_buffer["states"].at[:self.ncbf_pretrain_steps, : ].set(states)
                ncbf_replay_buffer["next_states"] = ncbf_replay_buffer["next_states"].at[:self.ncbf_pretrain_steps, : ].set(next_states)
                ncbf_replay_buffer["actions"] = ncbf_replay_buffer["actions"].at[:self.ncbf_pretrain_steps, : ].set(actions)
                ncbf_replay_buffer["dones"] = ncbf_replay_buffer["dones"].at[:self.ncbf_pretrain_steps, : ].set(dones)
                ncbf_replay_buffer["terminations"] = ncbf_replay_buffer["terminations"].at[:self.ncbf_pretrain_steps, : ].set(terminations)
                ncbf_replay_buffer["y_target"] = ncbf_replay_buffer["y_target"].at[:self.ncbf_pretrain_steps, : ].set(y_target)
                ncbf_replay_buffer["masks"] = ncbf_replay_buffer["masks"].at[:self.ncbf_pretrain_steps, : ].set(masks)
                ncbf_replay_buffer["pos"] = jnp.array(self.ncbf_pretrain_steps)
                ncbf_replay_buffer["size"] = jnp.array(self.ncbf_pretrain_steps)

                ncbf_state, ncbf_metrics, key = train_ncbf(ncbf_state, ncbf_replay_buffer, key, self.ncbf_pretrain_nr_minibatches)

                ncbf_metrics = tree.map_structure(lambda x: jnp.mean(x), ncbf_metrics)
                mean_y = jnp.mean(y_bool)
                ncbf_metrics['ncbf/mean_y'] = mean_y
                ncbf_metrics['ncbf/constraints_active_rate'] = jnp.mean(constraints_active)
                ncbf_metrics['ncbf/mean_delta_u'] = jnp.mean(jnp.abs(delta_u))

                def pretrain_callback(carry):
                    # step set to zero for pretrian logging
                    self.start_logging(0)
                    metrics = carry
                    for key, value in metrics.items():
                        self.log(f"{key}", np.asarray(value), 0)
                    self.end_logging()

                jax.debug.callback(pretrain_callback, ncbf_metrics)

            # pregenerate index slices for the replay buffer in advance
            update_indices = jnp.arange(self.nr_updates_per_multi_learning_iteration * self.nr_multi_learning_and_eval_save_iterations)
            ncbf_write_start_pos_array = (self.ncbf_pretrain_steps + update_indices * self.nr_steps) % capacity
            ncbf_write_start_pos_array = ncbf_write_start_pos_array.reshape(self.nr_multi_learning_and_eval_save_iterations, self.nr_updates_per_multi_learning_iteration)

            def multi_learning_and_eval_save_iteration(multi_learning_and_eval_save_iteration_carry, multi_learning_iteration_step):
                policy_state, critic_state, ncbf_state, env_state, ncbf_replay_buffer, key = multi_learning_and_eval_save_iteration_carry

                def learning_iteration(learning_iteration_carry, learning_iteration_step):
                    policy_state, critic_state, ncbf_state, env_state, ncbf_replay_buffer, key = learning_iteration_carry
                    rollout_carry = (policy_state, critic_state, ncbf_state, env_state, key)
                    single_rollout_carry, batch = jax.lax.scan(single_rollout, rollout_carry, None, self.nr_steps)
                    policy_state, critic_state, ncbf_state, env_state, key = single_rollout_carry
                    states, next_states, actions, rewards, values, terminations, dones, log_probs, infos, constraints_active, delta_u = batch

                    # process the batch data to get mask and y_target
                    y_bool, masks = window_any_done_next_H(dones, terminations, self.ncbf_H)
                    y_target = y_bool.astype(jnp.float32)

                    buffer_start_pos = ncbf_write_start_pos_array[multi_learning_iteration_step, learning_iteration_step]
                    # add to ncbf replay buffer

                    def update_replay_buffer(states_buffer, states, buffer_start_pos, nr_steps):
                        # states_buffer: [T, ...]
                        # states: [nr_steps, ...]
                        # buffer_start_pos: scalar int32 tracer

                        # Build the full start indices vector: first dim is dynamic, others are 0
                        start_indices = jnp.concatenate([
                            jnp.array([buffer_start_pos], dtype=jnp.int32),
                            jnp.zeros(states_buffer.ndim - 1, dtype=jnp.int32),
                        ])

                        # Do the dynamic update
                        new_states_buffer = lax.dynamic_update_slice(
                            states_buffer,  # operand
                            states,  # update (shape [nr_steps, ...])
                            start_indices,  # start indices for each axis
                        )
                        return new_states_buffer

                    ncbf_replay_buffer["states"] = update_replay_buffer(ncbf_replay_buffer["states"], states, buffer_start_pos, self.nr_steps)
                    ncbf_replay_buffer["next_states"] = update_replay_buffer(ncbf_replay_buffer["next_states"], next_states, buffer_start_pos, self.nr_steps)
                    ncbf_replay_buffer["actions"] = update_replay_buffer(ncbf_replay_buffer["actions"], actions, buffer_start_pos, self.nr_steps)
                    ncbf_replay_buffer["dones"] = update_replay_buffer(ncbf_replay_buffer["dones"], dones, buffer_start_pos, self.nr_steps)
                    ncbf_replay_buffer["terminations"] = update_replay_buffer(ncbf_replay_buffer["terminations"], terminations, buffer_start_pos, self.nr_steps)
                    ncbf_replay_buffer["y_target"] = update_replay_buffer(ncbf_replay_buffer["y_target"], y_target, buffer_start_pos, self.nr_steps)
                    ncbf_replay_buffer["masks"] = update_replay_buffer(ncbf_replay_buffer["masks"], masks, buffer_start_pos, self.nr_steps)
                    ncbf_replay_buffer["pos"] = (buffer_start_pos + self.nr_steps) % capacity
                    ncbf_replay_buffer["size"] = jnp.minimum(capacity, ncbf_replay_buffer["size"] + self.nr_steps)

                    # train ncbf
                    ncbf_state, ncbf_metrics, key = train_ncbf(ncbf_state, ncbf_replay_buffer, key,
                                                                         self.ncbf_nr_minibatches)
                    ncbf_metrics["ncbf/constraints_active_rate"] = jnp.mean(constraints_active)
                    ncbf_metrics["ncbf/mean_delta_u"] = jnp.mean(jnp.abs(delta_u))
                    ncbf_metrics["ncbf/mean_y"] = jnp.mean(y_bool)

                    # Calculating advantages and returns
                    def calculate_gae_advantages(critic_state, next_states, rewards, values, terminations):
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

                    advantages, returns = calculate_gae_advantages(critic_state, next_states, rewards, values, terminations)


                    # Optimizing
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
                        loss = (pg_loss - self.entropy_coef * entropy_loss +
                                self.critic_coef * critic_loss +
                                self.anticipation_coef * anticipation_loss)

                        # Create metrics
                        metrics = {
                            "loss/policy_gradient_loss": pg_loss,
                            "loss/critic_loss": critic_loss,
                            "loss/entropy_loss": entropy_loss,
                            "loss/anticipation_loss": anticipation_loss,
                            "policy_ratio/approx_kl": approx_kl_div,
                            "policy_ratio/clip_fraction": clip_fraction,
                        }

                        return loss, metrics

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
                        policy_state, critic_state, ncbf_state = carry

                        minibatch_advantages = batch_advantages[minibatch_indices]
                        minibatch_advantages = (minibatch_advantages - jnp.mean(minibatch_advantages)) / (jnp.std(minibatch_advantages) + 1e-8)

                        (loss, metrics), (policy_gradients, critic_gradients) = grad_loss_fn(
                            policy_state.params,
                            critic_state.params,
                            ncbf_state.params,
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

                        carry = (policy_state, critic_state, ncbf_state)

                        return carry, metrics
                    
                    init_carry = (policy_state, critic_state, ncbf_state)
                    carry, optimization_metrics = jax.lax.scan(minibatch_update, init_carry, batch_indices)
                    policy_state, critic_state, ncbf_state = carry

                    optimization_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
                    optimization_metrics["v_value/explained_variance"] = 1 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
                    optimization_metrics["policy/std_dev"] = jnp.mean(jnp.exp(policy_state.params["params"]["policy_logstd"]))


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
                        self.end_logging()

                    jax.debug.callback(callback, (combined_metrics, parallel_seed_id))
                    
                    return (policy_state, critic_state, ncbf_state, env_state, ncbf_replay_buffer, key), None
                    
                key, subkey = jax.random.split(key)
                learning_iteration_carry, _ = jax.lax.scan(learning_iteration, (policy_state, critic_state, ncbf_state, env_state, ncbf_replay_buffer, subkey), jnp.arange(self.nr_updates_per_multi_learning_iteration))
                policy_state, critic_state, ncbf_state, env_state, ncbf_replay_buffer, key = learning_iteration_carry

                # Evaluating
                if self.evaluation_active:
                    def single_eval_rollout(single_eval_rollout_carry, _):
                        policy_state, ncbf_state, eval_env_state = single_eval_rollout_carry

                        action_mean, _ = self.policy.apply(policy_state.params, eval_env_state.next_observation)
                        action = action_mean
                        processed_action = self.get_processed_action(action)
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
                    def save_with_check(policy_state, critic_state, ncbf_state):
                        self.save(policy_state, critic_state, ncbf_state)
                    jax.debug.callback(save_with_check, policy_state, critic_state, ncbf_state)

                
                return (policy_state, critic_state, ncbf_state, env_state, ncbf_replay_buffer, key), None

            jax.lax.scan(multi_learning_and_eval_save_iteration, (policy_state, critic_state, ncbf_state, env_state, ncbf_replay_buffer, key), jnp.arange(self.nr_multi_learning_and_eval_save_iterations))
            

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


    def save(self, policy_state, critic_state, ncbf_state):
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state,
            "ncbf_state": ncbf_state,
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
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

        @jax.jit
        def rollout(env_state, key):
            # key, subkey = jax.random.split(key)
            action_mean, action_logstd = self.policy.apply(self.policy_state.params, env_state.next_observation)
            # action_std = jnp.exp(action_logstd)
            action = action_mean # + action_std * jax.random.normal(subkey, shape=action_mean.shape)
            raw_processed_action = self.get_processed_action(action)
            processed_action, constraint_active, delta_u = self.batched_ncbf_safety_layer(raw_processed_action, env_state.next_observation, self.ncbf_state.params)
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
