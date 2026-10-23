"""Constrained Policy Optimization (Achiam et al., ICML 2017), full-jit JAX.

Rewritten in T07 (project plan F6): the previous cpo.py was clipped PPO + a Lagrange multiplier, i.e. the same
algorithm as ppo_lagrangian. That algorithm now lives in rl_x/algorithms/ppo_lagrangian (same defaults), so the
old behavior is `--algorithm.name=ppo_lagrangian.flax_full_jit`.

Per update:
  1. rollout of nr_steps x nr_envs (same env interface/batching as ppo / ppo_lagrangian),
  2. reward and cost GAE with the reward and cost critics,
  3. one CPO policy step on the full batch (cpo_core.cpo_policy_update: g, b, Fisher-vector products,
     conjugate gradients, dual solution / recovery step, backtracking line search on KL and the surrogates),
  4. reward and cost critics trained by MSE regression (nr_epochs x minibatches, Adam).

Constraint value: J_c is measured per rollout (cost_measure, see ppo_lagrangian/flax_full_jit/lagrange.py);
c = J_c - cost_limit, normalized to per-step units by default (cost_violation_normalization, see default_config.py).
"""
import os
import shutil
import json
from copy import deepcopy
import logging
import time

import tree
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.cpo.flax_full_jit.general_properties import GeneralProperties
from rl_x.algorithms.cpo.flax_full_jit.policy import get_policy
from rl_x.algorithms.cpo.flax_full_jit.critic import get_critic
from rl_x.algorithms.cpo.flax_full_jit.cpo_core import cpo_policy_update, gaussian_log_prob
from rl_x.algorithms.ppo_lagrangian.flax_full_jit.lagrange import measure_cost, COST_MEASURES

rlx_logger = logging.getLogger("rl_x")


class CPO:
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
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active
        self.target_kl = config.algorithm.target_kl
        self.cg_iterations = config.algorithm.cg_iterations
        self.cg_damping = config.algorithm.cg_damping
        self.line_search_steps = config.algorithm.line_search_steps
        self.line_search_decay = config.algorithm.line_search_decay
        self.fvp_sample_freq = max(int(config.algorithm.fvp_sample_freq), 1)
        self.standardize_reward_advantages = config.algorithm.standardize_reward_advantages
        self.cost_advantage_normalization = config.algorithm.cost_advantage_normalization
        self.cost_measure = config.algorithm.cost_measure
        self.cost_limit = config.algorithm.cost_limit
        self.cost_violation_normalization = config.algorithm.cost_violation_normalization
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

        if self.cost_measure not in COST_MEASURES:
            raise ValueError(f"Unknown cost_measure {self.cost_measure!r}, expected one of {COST_MEASURES}")
        if self.cost_violation_normalization not in ("per_step", "none"):
            raise ValueError(f"Unknown cost_violation_normalization {self.cost_violation_normalization!r}")
        if self.cost_advantage_normalization not in ("center", "standardize", "none"):
            raise ValueError(f"Unknown cost_advantage_normalization {self.cost_advantage_normalization!r}")

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

        # The policy is updated by the CPO step, not by an optimizer.
        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, env_state.next_observation),
            tx=optax.identity(),
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

        # Last constraint estimate and mean episode length (kept when a rollout has no finished episode).
        self.cpo_state = {
            "cost_estimate": jnp.zeros((), dtype=jnp.float32),
            "episode_length_estimate": jnp.asarray(float(self.horizon), dtype=jnp.float32),
        }

        if self.save_model:
            os.makedirs(self.save_path)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    def constraint_violation(self, costs, dones, cpo_state):
        """Returns (c, J_c, new_cpo_state, metrics) for one rollout. costs, dones: (nr_steps, nr_envs)."""
        measured_cost, nr_finished_episodes = measure_cost(self.cost_measure, costs, dones, cpo_state["cost_estimate"])
        episode_length = jnp.where(
            nr_finished_episodes > 0,
            costs.size / jnp.maximum(nr_finished_episodes, 1.0),
            cpo_state["episode_length_estimate"],
        )
        violation = measured_cost - self.cost_limit
        if self.cost_violation_normalization == "per_step":
            horizon = episode_length if self.cost_measure == "episode" else float(self.nr_steps)
            violation = violation / horizon
        new_cpo_state = {"cost_estimate": measured_cost, "episode_length_estimate": episode_length}
        metrics = {
            "constraint/measured_cost": measured_cost,
            "constraint/nr_finished_episodes": nr_finished_episodes,
            "constraint/episode_length_estimate": episode_length,
        }
        return violation, measured_cost, new_cpo_state, metrics

    def train(self):
        def jitable_train_function(key, parallel_seed_id):
            key, reset_key = jax.random.split(key, 2)
            reset_keys = jax.random.split(reset_key, self.nr_envs)
            env_state = self.env.reset(reset_keys, False)

            policy_state = self.policy_state
            reward_critic_state = self.reward_critic_state
            cost_critic_state = self.cost_critic_state
            cpo_state = self.cpo_state

            def get_constraint_cost(env_state):
                fallback_cost = env_state.terminated.astype(jnp.float32)
                return env_state.info.get("constraint/cost", fallback_cost).astype(jnp.float32)

            def calculate_gae(critic, critic_state, next_states, rewards, values, terminations, gamma, gae_lambda):
                def compute_advantages(carry, t):
                    prev_advantage = carry[0]
                    advantage = delta[t] + gamma * gae_lambda * (1 - terminations[t]) * prev_advantage
                    return (advantage,), advantage

                next_values = critic.apply(critic_state.params, next_states).squeeze(-1)
                delta = rewards + gamma * next_values * (1.0 - terminations) - values
                init_advantages = delta[-1]
                _, advantages = jax.lax.scan(compute_advantages, (init_advantages,), jnp.arange(self.nr_steps - 2, -1, -1))
                advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
                returns = advantages + values
                return advantages, returns

            def multi_learning_and_eval_save_iteration(multi_learning_and_eval_save_iteration_carry, multi_learning_iteration_step):
                policy_state, reward_critic_state, cost_critic_state, cpo_state, env_state, key = multi_learning_and_eval_save_iteration_carry

                def learning_iteration(learning_iteration_carry, learning_iteration_step):
                    policy_state, reward_critic_state, cost_critic_state, cpo_state, env_state, key = learning_iteration_carry

                    def single_rollout(single_rollout_carry, _):
                        policy_state, reward_critic_state, cost_critic_state, env_state, key = single_rollout_carry

                        key, subkey = jax.random.split(key)
                        observation = env_state.next_observation
                        action_mean, action_logstd = self.policy.apply(policy_state.params, observation)
                        action_std = jnp.exp(action_logstd)
                        action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                        log_prob = gaussian_log_prob(action_mean, action_logstd, action)
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

                    reward_advantages, reward_returns = calculate_gae(
                        self.reward_critic, reward_critic_state, next_states, rewards, reward_values, terminations,
                        self.gamma, self.gae_lambda,
                    )
                    cost_advantages, cost_returns = calculate_gae(
                        self.cost_critic, cost_critic_state, next_states, costs, cost_values, terminations,
                        self.cost_gamma, self.cost_gae_lambda,
                    )

                    cost_violation, measured_cost, cpo_state, constraint_metrics = self.constraint_violation(costs, dones, cpo_state)

                    batch_states = states.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)
                    batch_reward_advantages = reward_advantages.reshape(-1)
                    batch_reward_returns = reward_returns.reshape(-1)
                    batch_cost_advantages = cost_advantages.reshape(-1)
                    batch_cost_returns = cost_returns.reshape(-1)
                    batch_log_probs = log_probs.reshape(-1)

                    policy_reward_advantages = batch_reward_advantages
                    if self.standardize_reward_advantages:
                        policy_reward_advantages = (policy_reward_advantages - jnp.mean(policy_reward_advantages)) / (jnp.std(policy_reward_advantages) + 1e-8)
                    policy_cost_advantages = batch_cost_advantages
                    if self.cost_advantage_normalization in ("center", "standardize"):
                        policy_cost_advantages = policy_cost_advantages - jnp.mean(policy_cost_advantages)
                    if self.cost_advantage_normalization == "standardize":
                        policy_cost_advantages = policy_cost_advantages / (jnp.std(policy_cost_advantages) + 1e-8)

                    # CPO policy step on the full batch.
                    new_policy_params, cpo_metrics = cpo_policy_update(
                        self.policy.apply,
                        policy_state.params,
                        batch_states,
                        batch_actions,
                        batch_log_probs,
                        policy_reward_advantages,
                        policy_cost_advantages,
                        cost_violation,
                        target_kl=self.target_kl,
                        cg_iterations=self.cg_iterations,
                        cg_damping=self.cg_damping,
                        line_search_steps=self.line_search_steps,
                        line_search_decay=self.line_search_decay,
                        fvp_observations=batch_states[::self.fvp_sample_freq],
                    )
                    policy_state = policy_state.replace(params=new_policy_params)

                    # Critic regression.
                    def critic_loss_fn(reward_critic_params, cost_critic_params, state_b, reward_return_b, cost_return_b):
                        new_reward_value = self.reward_critic.apply(reward_critic_params, state_b).squeeze(-1)
                        reward_critic_loss = jnp.mean(0.5 * (new_reward_value - reward_return_b) ** 2)
                        new_cost_value = self.cost_critic.apply(cost_critic_params, state_b).squeeze(-1)
                        cost_critic_loss = jnp.mean(0.5 * (new_cost_value - cost_return_b) ** 2)
                        metrics = {
                            "loss/reward_critic_loss": reward_critic_loss,
                            "loss/cost_critic_loss": cost_critic_loss,
                        }
                        return reward_critic_loss + cost_critic_loss, metrics

                    grad_critic_loss_fn = jax.value_and_grad(critic_loss_fn, argnums=(0, 1), has_aux=True)

                    key, subkey = jax.random.split(key)
                    batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
                    batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
                    batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

                    def minibatch_update(carry, minibatch_indices):
                        reward_critic_state, cost_critic_state = carry
                        (_, metrics), (reward_critic_gradients, cost_critic_gradients) = grad_critic_loss_fn(
                            reward_critic_state.params,
                            cost_critic_state.params,
                            batch_states[minibatch_indices],
                            batch_reward_returns[minibatch_indices],
                            batch_cost_returns[minibatch_indices],
                        )
                        reward_critic_state = reward_critic_state.apply_gradients(grads=reward_critic_gradients)
                        cost_critic_state = cost_critic_state.apply_gradients(grads=cost_critic_gradients)
                        metrics["gradients/reward_critic_grad_norm"] = optax.global_norm(reward_critic_gradients)
                        metrics["gradients/cost_critic_grad_norm"] = optax.global_norm(cost_critic_gradients)
                        return (reward_critic_state, cost_critic_state), metrics

                    (reward_critic_state, cost_critic_state), optimization_metrics = jax.lax.scan(
                        minibatch_update, (reward_critic_state, cost_critic_state), batch_indices
                    )

                    optimization_metrics["lr/learning_rate"] = reward_critic_state.opt_state[1].hyperparams["learning_rate"]
                    optimization_metrics["v_value/reward_explained_variance"] = 1 - jnp.var(reward_returns - reward_values) / (jnp.var(reward_returns) + 1e-8)
                    optimization_metrics["v_value/cost_explained_variance"] = 1 - jnp.var(cost_returns - cost_values) / (jnp.var(cost_returns) + 1e-8)
                    optimization_metrics["policy/std_dev"] = jnp.mean(jnp.exp(policy_state.params["params"]["policy_logstd"]))

                    constraint_metrics = {
                        **constraint_metrics,
                        "constraint/mean_step_cost": jnp.mean(costs),
                        "constraint/mean_rollout_cost": jnp.mean(jnp.sum(costs, axis=0)),
                        "constraint/cost_limit": jnp.array(self.cost_limit),
                        "constraint/cost_violation": measured_cost - self.cost_limit,
                        "constraint/safe_time": jnp.mean(infos.get("constraint/safe_time", jnp.zeros_like(costs))),
                        "constraint/below_height_rate": jnp.mean(infos.get("constraint/below_height", jnp.zeros_like(costs))),
                    }

                    combined_learning_iteration_step = (multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step + 1
                    steps_metrics = {
                        "steps/nr_env_steps": combined_learning_iteration_step * self.nr_steps * self.nr_envs,
                        "steps/nr_updates": combined_learning_iteration_step,
                    }

                    combined_metrics = {**infos, **steps_metrics, **optimization_metrics, **constraint_metrics, **cpo_metrics}
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

                    return (policy_state, reward_critic_state, cost_critic_state, cpo_state, env_state, key), None

                key, subkey = jax.random.split(key)
                learning_iteration_carry, _ = jax.lax.scan(
                    learning_iteration,
                    (policy_state, reward_critic_state, cost_critic_state, cpo_state, env_state, subkey),
                    jnp.arange(self.nr_updates_per_multi_learning_iteration),
                )
                policy_state, reward_critic_state, cost_critic_state, cpo_state, env_state, key = learning_iteration_carry

                if self.evaluation_active:
                    def single_eval_rollout(single_eval_rollout_carry, _):
                        policy_state, eval_env_state = single_eval_rollout_carry

                        action_mean, _ = self.policy.apply(policy_state.params, eval_env_state.next_observation)
                        processed_action = self.get_processed_action(action_mean)
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
                    def save_with_check(policy_state, reward_critic_state, cost_critic_state, cpo_state):
                        self.save(policy_state, reward_critic_state, cost_critic_state, cpo_state)
                    jax.debug.callback(save_with_check, policy_state, reward_critic_state, cost_critic_state, cpo_state)

                return (policy_state, reward_critic_state, cost_critic_state, cpo_state, env_state, key), None

            jax.lax.scan(
                multi_learning_and_eval_save_iteration,
                (policy_state, reward_critic_state, cost_critic_state, cpo_state, env_state, key),
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

    def save(self, policy_state, reward_critic_state, cost_critic_state, cpo_state):
        checkpoint = {
            "policy": policy_state,
            "critic": reward_critic_state,
            "cost_critic": cost_critic_state,
            "cpo_state": cpo_state,
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
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = cls(config, env, run_path, writer)

        target = {
            "policy": model.policy_state,
            "critic": model.reward_critic_state,
            "cost_critic": model.cost_critic_state,
            "cpo_state": model.cpo_state,
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.policy_state = checkpoint["policy"]
        model.reward_critic_state = checkpoint["critic"]
        model.cost_critic_state = checkpoint["cost_critic"]
        model.cpo_state = checkpoint["cpo_state"]

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
