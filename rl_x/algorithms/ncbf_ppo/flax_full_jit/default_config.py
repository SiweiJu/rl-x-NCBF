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
    config.max_grad_norm = 5.0
    config.std_dev = 1.0
    config.action_clipping_and_rescaling = False
    config.evaluation_and_save_frequency = 17301504  # -1 to disable
    config.evaluation_active = True

    # ncbf config
    config.ncbf = config_dict.ConfigDict()
    config.ncbf.type = 'FFNN'
    config.ncbf.H = 10  # prediction horizon
    config.ncbf.nr_minibatches = 10
    config.ncbf.minibatch_size = 512
    config.ncbf.lr = 1e-3
    config.ncbf.gamma_c = 0.0  # safety threshold
    config.ncbf.w_clf = 1.0  # weight for classification loss
    config.ncbf.w_cbf = 1.0  # weight for CBF loss
    config.ncbf.w_lip = 0.0  # weight for lipschitz loss
    config.ncbf.w_wd = 0.0  # weight for weight decay loss
    config.ncbf.lip_target = 0.0  # lipschitz target, set to 0.0 if just want small gradient
    config.ncbf.eta_cbf = 1.0  # class Kappa function parameter for CBF constraint

    config.ncbf.pretrain = config_dict.ConfigDict()
    config.ncbf.pretrain.nr_minibatches = 20
    config.ncbf.pretrain.nr_steps = 16384

    config.ncbf_buffer = config_dict.ConfigDict()
    config.ncbf_buffer.buffersize = int(1e6)

    return config
