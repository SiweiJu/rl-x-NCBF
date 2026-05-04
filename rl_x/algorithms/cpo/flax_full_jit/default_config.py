from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.nr_parallel_seeds = 1
    config.total_timesteps = 2e9
    config.learning_rate = 4e-4
    config.anneal_learning_rate = True
    config.nr_steps = 128
    config.nr_epochs = 10
    config.minibatch_size = 32768
    config.gamma = 0.99
    config.gae_lambda = 0.9
    config.clip_range = 0.1
    config.entropy_coef = 0.0
    config.critic_coef = 1.0
    config.cost_critic_coef = 1.0
    config.max_grad_norm = 5.0
    config.std_dev = 1.0
    config.action_clipping_and_rescaling = False
    config.evaluation_and_save_frequency = 17301504  # -1 to disable
    config.evaluation_active = True

    # Constraint policy optimization settings. The cost comes from
    # env_state.info["constraint/cost"], with termination as a fallback.
    config.cost_gamma = 0.99
    config.cost_gae_lambda = 0.9
    config.cost_limit = 0.0
    config.cost_lambda_init = 0.0
    config.cost_lambda_lr = 0.05
    config.cost_lambda_max = 100.0
    config.normalize_cost_advantages = True
    config.scale_lagrangian_objective = True

    return config
