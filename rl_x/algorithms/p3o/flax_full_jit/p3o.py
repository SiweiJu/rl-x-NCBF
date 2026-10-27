import os
import shutil
import json
from copy import deepcopy
import logging
import time

"""P3O: Penalized Proximal Policy Optimization (Zhang et al., IJCAI 2022, arXiv 2205.11814), full-jit JAX.

Same rollout / GAE / minibatching / eval / checkpoint structure as ppo_lagrangian.flax_full_jit (T07). The policy loss
is the clipped PPO reward surrogate plus kappa * relu(clipped cost surrogate + constraint offset), see penalty.py.
Optional CaT (Constraints as Terminations, arXiv 2403.18765) on the reward side, see cat.py:
``--algorithm.p3o.enabled=False --algorithm.cat.enabled=True`` is PPO + CaT.
"""
import tree
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.p3o.flax_full_jit.general_properties import GeneralProperties
from rl_x.algorithms.p3o.flax_full_jit.policy import get_policy
from rl_x.algorithms.p3o.flax_full_jit.critic import get_critic
from rl_x.algorithms.p3o.flax_full_jit import penalty as p3o_penalty
from rl_x.algorithms.p3o.flax_full_jit import cat
from rl_x.algorithms.ppo_lagrangian.flax_full_jit import lagrange

rlx_logger = logging.getLogger("rl_x")


class P3O:
    eval_sweep_policy_only = True  # eval_sweep.make_agent: plain policy, mode A only

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
        self.total_timesteps = int(config.algorithm.total_timesteps)
        self.nr_envs = config.environment.nr_envs
        self.render = config.environment.render
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.cost_gamma = config.algorithm.cost_gamma
        self.cost_gae_lambda = config.algorithm.cost_gae_lambda
        self.clip_range = config.algorithm.clip_range
        self.entropy_coef = config.algorithm.entropy_coef
        self.critic_coef = config.algorithm.critic_coef
        self.cost_critic_coef = config.algorithm.cost_critic_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active
        self.cost_limit = config.algorithm.cost_limit
        self.cost_measure = config.algorithm.cost_measure
        if self.cost_measure not in lagrange.COST_MEASURES:
            raise ValueError(f"Unknown cost_measure {self.cost_measure!r}")
        p3o_config = config.algorithm.p3o
        self.p3o_enabled = bool(p3o_config.enabled)
        self.kappa = float(p3o_config.kappa)
        self.clip_cost_surrogate = bool(p3o_config.clip_cost_surrogate)
        self.constraint_normalization = p3o_config.constraint_normalization
        self.cost_advantage_normalization = p3o_config.cost_advantage_normalization
        if self.constraint_normalization not in p3o_penalty.CONSTRAINT_NORMALIZATIONS:
            raise ValueError(f"Unknown p3o.constraint_normalization {self.constraint_normalization!r}")
        if self.cost_advantage_normalization not in p3o_penalty.COST_ADVANTAGE_NORMALIZATIONS:
            raise ValueError(f"Unknown p3o.cost_advantage_normalization {self.cost_advantage_normalization!r}")
        cat_config = config.algorithm.cat
        self.cat_enabled = bool(cat_config.enabled)
        self.cat_source = cat_config.constraint_source
        self.cat_p_max = float(cat_config.p_max)
        self.cat_min_p = float(cat_config.min_p)
        self.cat_tau = float(cat_config.tau)
        self.cat_clip_reward_at_zero = bool(cat_config.clip_reward_at_zero)
        if self.cat_source not in cat.CONSTRAINT_SOURCES:
            raise ValueError(f"Unknown cat.constraint_source {self.cat_source!r}")
        if self.cat_enabled and self.cat_source in ("collision", "penetration"):
            obstacle_config = getattr(env, "obstacle_config", {})
            if not getattr(env, "use_obstacles", False):
                raise ValueError(f"cat.constraint_source={self.cat_source} needs environment.obstacles.enabled=True")
            if obstacle_config.get("terminate_on_collision", True):
                raise ValueError("CaT on collisions needs environment.obstacles.terminate_on_collision=False "
                                 "(otherwise every collision is already a hard termination)")
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = self.total_timesteps // self.batch_size
        self.nr_minibatches = self.batch_size // self.minibatch_size
        if config.algorithm.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.batch_size * self.nr_updates
        self.nr_multi_learning_and_eval_save_iterations = int(self.total_timesteps // self.evaluation_and_save_frequency)
        self.nr_updates_per_multi_learning_iteration = int(self.evaluation_and_save_frequency // self.batch_size)
        self.os_shape = env.single_observation_space.shape
        self.as_shape = env.single_action_space.shape
        self.horizon = env.horizon

        if self.evaluation_and_save_frequency % self.batch_size != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of batch size")

        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("Minibatch size must divide evenly into nr_envs * nr_steps")

        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet. This is mainly limited by not being able to log multiple wandb runs at the same time.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, reward_critic_key, cost_critic_key, reset_key = jax.random.split(self.key, 5)
        reset_key = jax.random.split(reset_key, 1)

        self.policy, self.get_processed_action = get_policy(self.config, self.env)
        self.reward_critic = get_critic(self.config, self.env)
        self.cost_critic = get_critic(self.config, self.env)

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

        self.reward_critic_state = TrainState.create(
            apply_fn=self.reward_critic.apply,
            params=self.reward_critic.init(reward_critic_key, env_state.next_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        self.cost_critic_state = TrainState.create(
            apply_fn=self.cost_critic.apply,
            params=self.cost_critic.init(cost_critic_key, env_state.next_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        # Last constraint estimate and mean episode length (kept when a rollout has no finished episode, as CPO),
        # plus the CaT running max of the constraint violation.
        self.p3o_state = {
            "cost_estimate": jnp.zeros((), dtype=jnp.float32),
            "episode_length_estimate": jnp.asarray(float(self.horizon), dtype=jnp.float32),
            "cat": cat.init_cat_state(),
        }

        if self.save_model:
            os.makedirs(self.save_path)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    def train(self):
        def jitable_train_function(key, parallel_seed_id):
            key, reset_key = jax.random.split(key, 2)
            reset_keys = jax.random.split(reset_key, self.nr_envs)
            env_state = self.env.reset(reset_keys, False)

            policy_state = self.policy_state
            reward_critic_state = self.reward_critic_state
            cost_critic_state = self.cost_critic_state
            p3o_state = self.p3o_state

            def get_constraint_cost(env_state):
                fallback_cost = env_state.terminated.astype(jnp.float32)
                return env_state.info.get("constraint/cost", fallback_cost).astype(jnp.float32)

            def calculate_reward_gae(critic_state, next_states, rewards, values, terminations):
                def compute_advantages(carry, t):
                    prev_advantage = carry[0]
                    advantage = delta[t] + self.gamma * self.gae_lambda * (1 - terminations[t]) * prev_advantage
                    return (advantage,), advantage

                next_values = self.reward_critic.apply(critic_state.params, next_states).squeeze(-1)
                delta = rewards + self.gamma * next_values * (1.0 - terminations) - values
                init_advantages = delta[-1]
                _, advantages = jax.lax.scan(compute_advantages, (init_advantages,), jnp.arange(self.nr_steps - 2, -1, -1))
                advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
                returns = advantages + values
                return advantages, returns

            def calculate_cost_gae(critic_state, next_states, costs, values, terminations):
                def compute_advantages(carry, t):
                    prev_advantage = carry[0]
                    advantage = delta[t] + self.cost_gamma * self.cost_gae_lambda * (1 - terminations[t]) * prev_advantage
                    return (advantage,), advantage

                next_values = self.cost_critic.apply(critic_state.params, next_states).squeeze(-1)
                delta = costs + self.cost_gamma * next_values * (1.0 - terminations) - values
                init_advantages = delta[-1]
                _, advantages = jax.lax.scan(compute_advantages, (init_advantages,), jnp.arange(self.nr_steps - 2, -1, -1))
                advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
                returns = advantages + values
                return advantages, returns

            def multi_learning_and_eval_save_iteration(multi_learning_and_eval_save_iteration_carry, multi_learning_iteration_step):
                policy_state, reward_critic_state, cost_critic_state, p3o_state, env_state, key = multi_learning_and_eval_save_iteration_carry

                def learning_iteration(learning_iteration_carry, learning_iteration_step):
                    policy_state, reward_critic_state, cost_critic_state, p3o_state, env_state, key = learning_iteration_carry

                    def single_rollout(single_rollout_carry, _):
                        policy_state, reward_critic_state, cost_critic_state, env_state, key = single_rollout_carry

                        key, subkey = jax.random.split(key)
                        observation = env_state.next_observation
                        action_mean, action_logstd = self.policy.apply(policy_state.params, observation)
                        action_std = jnp.exp(action_logstd)
                        action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                        log_prob = (-0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd).sum(1)
                        processed_action = self.get_processed_action(action)
                        reward_value = self.reward_critic.apply(reward_critic_state.params, observation).squeeze(-1)
                        cost_value = self.cost_critic.apply(cost_critic_state.params, observation).squeeze(-1)

                        env_state = self.env.step(env_state, processed_action)
                        cost = get_constraint_cost(env_state)
                        transition = (
                            observation,
                            env_state.actual_next_observation,
                            action,
                            env_state.reward,
                            cost,
                            reward_value,
                            cost_value,
                            env_state.terminated,
                            env_state.terminated | env_state.truncated,
                            log_prob,
                            env_state.info,
                        )

                        if self.render:
                            def render(env_state):
                                return self.env.render(env_state)

                            env_state = jax.experimental.io_callback(render, env_state, env_state)

                        return (policy_state, reward_critic_state, cost_critic_state, env_state, key), transition

                    single_rollout_carry, batch = jax.lax.scan(
                        single_rollout,
                        (policy_state, reward_critic_state, cost_critic_state, env_state, key),
                        None,
                        self.nr_steps,
                    )
                    policy_state, reward_critic_state, cost_critic_state, env_state, key = single_rollout_carry
                    states, next_states, actions, rewards, costs, reward_values, cost_values, terminations, dones, log_probs, infos = batch

                    if self.cat_enabled:
                        cat_constraints = cat.constraint_values(self.cat_source, infos, costs)
                        new_cat_state, cat_delta = cat.rollout_probabilities(
                            p3o_state["cat"], cat_constraints, self.cat_tau, self.cat_p_max, self.cat_min_p
                        )
                        gae_rewards, gae_terminations = cat.apply(rewards, terminations, cat_delta, self.cat_clip_reward_at_zero)
                    else:
                        new_cat_state = p3o_state["cat"]
                        gae_rewards, gae_terminations = rewards, terminations

                    reward_advantages, reward_returns = calculate_reward_gae(
                        reward_critic_state,
                        next_states,
                        gae_rewards,
                        reward_values,
                        gae_terminations,
                    )
                    cost_advantages, cost_returns = calculate_cost_gae(
                        cost_critic_state,
                        next_states,
                        costs,
                        cost_values,
                        terminations,
                    )

                    # Constraint value J_c and mean episode length T of this rollout (same estimators as CPO).
                    mean_rollout_cost = jnp.mean(jnp.sum(costs, axis=0))
                    measured_cost, nr_finished_episodes = lagrange.measure_cost(
                        self.cost_measure, costs, dones, p3o_state["cost_estimate"]
                    )
                    episode_length = jnp.where(
                        nr_finished_episodes > 0,
                        costs.size / jnp.maximum(nr_finished_episodes, 1.0),
                        p3o_state["episode_length_estimate"],
                    )
                    horizon = episode_length if self.cost_measure == "episode" else float(self.nr_steps)
                    cost_violation = measured_cost - self.cost_limit
                    constraint_offset = p3o_penalty.constraint_offset(
                        measured_cost, self.cost_limit, horizon, self.cost_gamma, self.constraint_normalization
                    )
                    p3o_state = {"cost_estimate": measured_cost, "episode_length_estimate": episode_length, "cat": new_cat_state}

                    def loss_fn(
                        policy_params,
                        reward_critic_params,
                        cost_critic_params,
                        state_b,
                        action_b,
                        log_prob_b,
                        reward_return_b,
                        reward_advantage_b,
                        cost_return_b,
                        cost_advantage_b,
                        offset_b,
                    ):
                        # Batched over the minibatch (no vmap): the P3O penalty is a relu of the minibatch mean.
                        action_mean, action_logstd = self.policy.apply(policy_params, state_b)
                        action_std = jnp.exp(action_logstd)
                        new_log_prob = -0.5 * ((action_b - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
                        new_log_prob = new_log_prob.sum(-1)
                        entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)

                        logratio = new_log_prob - log_prob_b
                        ratio = jnp.exp(logratio)
                        approx_kl_div = (ratio - 1) - logratio
                        clip_fraction = jnp.float32((jnp.abs(ratio - 1) > self.clip_range))

                        pg_loss1 = -reward_advantage_b * ratio
                        pg_loss2 = -reward_advantage_b * jnp.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
                        pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

                        penalty_loss, cost_surrogate_violation = p3o_penalty.penalty(
                            ratio, cost_advantage_b, offset_b, self.kappa, self.clip_range, self.clip_cost_surrogate
                        )
                        if not self.p3o_enabled:
                            penalty_loss = jnp.zeros_like(penalty_loss)

                        entropy_loss = jnp.mean(entropy.sum(-1))

                        new_reward_value = self.reward_critic.apply(reward_critic_params, state_b).squeeze(-1)
                        reward_critic_loss = jnp.mean(0.5 * (new_reward_value - reward_return_b) ** 2)

                        new_cost_value = self.cost_critic.apply(cost_critic_params, state_b).squeeze(-1)
                        cost_critic_loss = jnp.mean(0.5 * (new_cost_value - cost_return_b) ** 2)

                        loss = (
                            pg_loss
                            + penalty_loss
                            - self.entropy_coef * entropy_loss
                            + self.critic_coef * reward_critic_loss
                            + self.cost_critic_coef * cost_critic_loss
                        )

                        metrics = {
                            "loss/policy_gradient_loss": pg_loss,
                            "loss/p3o_penalty_loss": penalty_loss,
                            "loss/reward_critic_loss": reward_critic_loss,
                            "loss/cost_critic_loss": cost_critic_loss,
                            "loss/entropy_loss": entropy_loss,
                            "policy_ratio/approx_kl": jnp.mean(approx_kl_div),
                            "policy_ratio/clip_fraction": jnp.mean(clip_fraction),
                            "p3o/surrogate_violation": cost_surrogate_violation,
                            "p3o/penalty_active": jnp.float32(cost_surrogate_violation > 0.0),
                            "p3o/normalized_offset": offset_b,
                        }

                        return loss, (metrics)

                    batch_states = states.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)
                    batch_reward_advantages = reward_advantages.reshape(-1)
                    batch_reward_returns = reward_returns.reshape(-1)
                    batch_cost_advantages = cost_advantages.reshape(-1)
                    batch_cost_returns = cost_returns.reshape(-1)
                    batch_log_probs = log_probs.reshape(-1)

                    grad_loss_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True)

                    key, subkey = jax.random.split(key)
                    batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
                    batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
                    batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

                    def minibatch_update(carry, minibatch_indices):
                        policy_state, reward_critic_state, cost_critic_state = carry

                        minibatch_reward_advantages = batch_reward_advantages[minibatch_indices]
                        minibatch_reward_advantages = (
                            minibatch_reward_advantages - jnp.mean(minibatch_reward_advantages)
                        ) / (jnp.std(minibatch_reward_advantages) + 1e-8)

                        minibatch_cost_advantages, minibatch_offset = p3o_penalty.normalize_cost_advantages(
                            batch_cost_advantages[minibatch_indices], constraint_offset, self.cost_advantage_normalization
                        )

                        (loss, (metrics)), (policy_gradients, reward_critic_gradients, cost_critic_gradients) = grad_loss_fn(
                            policy_state.params,
                            reward_critic_state.params,
                            cost_critic_state.params,
                            batch_states[minibatch_indices],
                            batch_actions[minibatch_indices],
                            batch_log_probs[minibatch_indices],
                            batch_reward_returns[minibatch_indices],
                            minibatch_reward_advantages,
                            batch_cost_returns[minibatch_indices],
                            minibatch_cost_advantages,
                            minibatch_offset,
                        )

                        policy_state = policy_state.apply_gradients(grads=policy_gradients)
                        reward_critic_state = reward_critic_state.apply_gradients(grads=reward_critic_gradients)
                        cost_critic_state = cost_critic_state.apply_gradients(grads=cost_critic_gradients)

                        metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                        metrics["gradients/reward_critic_grad_norm"] = optax.global_norm(reward_critic_gradients)
                        metrics["gradients/cost_critic_grad_norm"] = optax.global_norm(cost_critic_gradients)

                        carry = (policy_state, reward_critic_state, cost_critic_state)

                        return carry, (metrics)

                    init_carry = (policy_state, reward_critic_state, cost_critic_state)
                    carry, (optimization_metrics) = jax.lax.scan(minibatch_update, init_carry, batch_indices)
                    policy_state, reward_critic_state, cost_critic_state = carry

                    # First minibatch of the first epoch: the penalty state at pi_k (before any update).
                    optimization_metrics["p3o/penalty_active_at_start"] = optimization_metrics["p3o/penalty_active"][0]
                    optimization_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
                    optimization_metrics["v_value/reward_explained_variance"] = 1 - jnp.var(reward_returns - reward_values) / (jnp.var(reward_returns) + 1e-8)
                    optimization_metrics["v_value/cost_explained_variance"] = 1 - jnp.var(cost_returns - cost_values) / (jnp.var(cost_returns) + 1e-8)
                    optimization_metrics["policy/std_dev"] = jnp.mean(jnp.exp(policy_state.params["params"]["policy_logstd"]))

                    constraint_metrics = {
                        "constraint/mean_step_cost": jnp.mean(costs),
                        "constraint/mean_rollout_cost": mean_rollout_cost,
                        "constraint/measured_cost": measured_cost,
                        "constraint/episode_cost_estimate": jnp.sum(costs) / jnp.maximum(nr_finished_episodes, 1.0),
                        "constraint/nr_finished_episodes": nr_finished_episodes,
                        "constraint/episode_length_estimate": episode_length,
                        "constraint/cost_limit": jnp.array(self.cost_limit),
                        "constraint/cost_violation": cost_violation,
                        "constraint/safe_time": jnp.mean(infos.get("constraint/safe_time", jnp.zeros_like(costs))),
                        "constraint/below_height_rate": jnp.mean(infos.get("constraint/below_height", jnp.zeros_like(costs))),
                        "p3o/constraint_offset": constraint_offset,
                        "p3o/kappa": jnp.array(self.kappa if self.p3o_enabled else 0.0),
                    }
                    if self.cat_enabled:
                        constraint_metrics.update({
                            "cat/termination_prob": jnp.mean(cat_delta),
                            "cat/termination_prob_when_violated": jnp.sum(cat_delta) / jnp.maximum(jnp.sum(cat_constraints > 0.0), 1.0),
                            "cat/violation_rate": jnp.mean(jnp.float32(cat_constraints > 0.0)),
                            "cat/running_max": new_cat_state["running_max"],
                            "cat/reward_scale": jnp.sum(gae_rewards) / (jnp.sum(rewards) + 1e-8),
                            "cat/effective_termination_rate": jnp.mean(gae_terminations),
                        })

                    combined_learning_iteration_step = (multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step + 1
                    steps_metrics = {
                        "steps/nr_env_steps": combined_learning_iteration_step * self.nr_steps * self.nr_envs,
                        "steps/nr_updates": combined_learning_iteration_step * self.nr_epochs * self.nr_minibatches,
                    }

                    combined_metrics = {**infos, **steps_metrics, **optimization_metrics, **constraint_metrics}
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

                    return (policy_state, reward_critic_state, cost_critic_state, p3o_state, env_state, key), None

                key, subkey = jax.random.split(key)
                learning_iteration_carry, _ = jax.lax.scan(
                    learning_iteration,
                    (policy_state, reward_critic_state, cost_critic_state, p3o_state, env_state, subkey),
                    jnp.arange(self.nr_updates_per_multi_learning_iteration),
                )
                policy_state, reward_critic_state, cost_critic_state, p3o_state, env_state, key = learning_iteration_carry

                if self.evaluation_active:
                    def single_eval_rollout(single_eval_rollout_carry, _):
                        policy_state, eval_env_state = single_eval_rollout_carry

                        action_mean, _ = self.policy.apply(policy_state.params, eval_env_state.next_observation)
                        action = action_mean
                        processed_action = self.get_processed_action(action)
                        eval_env_state = self.env.step(eval_env_state, processed_action)

                        return (policy_state, eval_env_state), None

                    key, reset_key = jax.random.split(key)
                    reset_keys = jax.random.split(reset_key, self.nr_envs)
                    eval_env_state = self.env.reset(reset_keys, True)
                    single_eval_rollout_carry, _ = jax.lax.scan(single_eval_rollout, (policy_state, eval_env_state), jnp.arange(self.horizon))
                    _, eval_env_state = single_eval_rollout_carry

                    eval_metrics = {
                        "eval/episode_return": jnp.mean(eval_env_state.info["rollout/episode_return"]),
                        "eval/episode_length": jnp.mean(eval_env_state.info["rollout/episode_length"]),
                        "eval/constraint_cost": jnp.mean(eval_env_state.info.get("constraint/episode_cost", jnp.zeros(self.nr_envs))),
                        "eval/safe_time": jnp.mean(eval_env_state.info.get("constraint/safe_time", jnp.zeros(self.nr_envs))),
                        "eval/below_height_rate": jnp.mean(eval_env_state.info.get("constraint/below_height", jnp.zeros(self.nr_envs))),
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

                if self.save_model:
                    def save_with_check(policy_state, reward_critic_state, cost_critic_state, p3o_state):
                        self.save(policy_state, reward_critic_state, cost_critic_state, p3o_state)
                    jax.debug.callback(save_with_check, policy_state, reward_critic_state, cost_critic_state, p3o_state)

                return (policy_state, reward_critic_state, cost_critic_state, p3o_state, env_state, key), None

            jax.lax.scan(
                multi_learning_and_eval_save_iteration,
                (policy_state, reward_critic_state, cost_critic_state, p3o_state, env_state, key),
                jnp.arange(self.nr_multi_learning_and_eval_save_iterations),
            )

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
        rlx_logger.info(f"| {name.ljust(30)}| {str(value).ljust(14)[:14]} |", flush=False)

    def start_logging(self, step):
        if self.track_console:
            rlx_logger.info("+" + "-" * 31 + "+" + "-" * 16 + "+", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")

    def end_logging(self):
        if self.track_console:
            rlx_logger.info("+" + "-" * 31 + "+" + "-" * 16 + "+")

    def save(self, policy_state, reward_critic_state, cost_critic_state, p3o_state):
        checkpoint = {
            "policy": policy_state,
            "critic": reward_critic_state,
            "cost_critic": cost_critic_state,
            "p3o_state": p3o_state,
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

    @classmethod
    def load(cls, config, env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(splitted_path[:-1]))
        checkpoint_file_name = splitted_path[-1]
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"

        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if key == "eval_sweep":  # eval settings come from the command line only
                continue
            if key not in config.algorithm or f"algorithm.{key}" in explicitly_set_algorithm_params:
                continue
            if isinstance(value, dict):  # p3o.*, cat.*: per sub-key, command-line settings win
                for sub_key, sub_value in value.items():
                    if f"algorithm.{key}.{sub_key}" not in explicitly_set_algorithm_params and sub_key in config.algorithm[key]:
                        config.algorithm[key][sub_key] = sub_value
            else:
                config.algorithm[key] = value
        model = cls(config, env, run_path, writer)

        target = {
            "policy": model.policy_state,
            "critic": model.reward_critic_state,
            "cost_critic": model.cost_critic_state,
            "p3o_state": model.p3o_state,
        }
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=orbax_utils.restore_args_from_target(target))
        model.p3o_state = checkpoint["p3o_state"]

        model.policy_state = checkpoint["policy"]
        model.reward_critic_state = checkpoint["critic"]
        model.cost_critic_state = checkpoint["cost_critic"]

        shutil.rmtree(checkpoint_dir)

        return model

    def test(self, episodes):
        if self.config.algorithm.get("eval_sweep", {}).get("enabled", False):
            from rl_x.algorithms.ncbf_ppo.flax_full_jit.eval_sweep import run_eval_sweep
            return run_eval_sweep(self)

        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

        @jax.jit
        def rollout(env_state, key):
            action_mean, _ = self.policy.apply(self.policy_state.params, env_state.next_observation)
            processed_action = self.get_processed_action(action_mean)
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
