from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name
    
    config.device = "gpu"  # cpu, gpu
    config.total_timesteps = 1e9
    config.learning_rate = 3e-4
    config.anneal_learning_rate = False
    config.nr_steps = 2048
    config.nr_epochs = 10
    config.minibatch_size = 64
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clip_range = 0.2
    config.entropy_coef = 0.0
    config.critic_coef = 0.5
    config.max_grad_norm = 0.5
    config.std_dev = 1.0
    config.action_clipping_and_rescaling = True
    config.nr_hidden_units = 256
    config.evaluation_frequency = 204800  # -1 to disable
    config.evaluation_episodes = 10

    # ncbf config
    config.ncbf.type = 'FFNN'
    config.ncbf.H = 10  # prediction horizon
    config.ncbf.epoch = 10
    config.ncbf.minibatch_size = 64
    config.ncbf.lr = 1e-3
    config.ncbf.gamma_c = 0.0     # safety threshold
    config.ncbf.w_clf = 1.0    # weight for classification loss
    config.ncbf.w_cbf = 1.0    # weight for CBF loss
    config.ncbf.w_lip = 0.0    # weight for lipschitz loss
    config.ncbf.w_wd = 0.0     # weight for weight decay loss
    config.ncbf.lip_target = 0.0    # lipschitz target, set to 0.0 if just want small gradient
    config.ncbf.eta_cbf = 1.0 # class Kappa function parameter for CBF constraint
    return config
