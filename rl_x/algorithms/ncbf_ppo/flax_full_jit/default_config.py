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
    config.target_kl = 0.03  # <= 0 disables PPO update gating
    config.entropy_coef = 0.0
    config.critic_coef = 1.0
    config.max_grad_norm = 5.0
    config.std_dev = 1.0
    config.action_clipping_and_rescaling = False
    config.action_clip = 0.0
    config.hidden_layers = (512, 256, 128)
    config.evaluation_and_save_frequency = 17301504  # -1 to disable
    config.evaluation_active = True

    # ncbf config
    config.ncbf = config_dict.ConfigDict()
    config.ncbf.n_ensemble = 5  # number of networks to assemble for ncbf prediction
    config.ncbf.type = 'FFNN'
    config.ncbf.H = 25  # prediction horizon
    config.ncbf.lr = 1e-3
    config.ncbf.gamma_c = 0.0     # raw-logit safety threshold for full-jit BCE training
    config.ncbf.w_clf = 1.0    # weight for classification loss
    config.ncbf.w_cbf = 1.0    # weight for CBF loss
    config.ncbf.w_lip = 0.0    # weight for lipschitz loss
    config.ncbf.w_wd = 0.0     # weight for weight decay loss
    config.ncbf.L_max = 0.0    # lipschitz target, set to 0.0 if just want small gradient
    config.ncbf.eta_cbf = 1.0 # class Kappa function parameter for CBF constraint
    config.ncbf.loss_coef = 0.1
    config.ncbf.stop_encoder_gradient = False
    config.ncbf.use_safety_layer = False
    config.ncbf.nr_minibatches = 50
    config.ncbf.nr_hidden_units = 512
    config.ncbf.output_distribution = "deterministic"  # deterministic or logistic_normal
    config.ncbf.min_log_std = -5.0
    config.ncbf.max_log_std = 2.0
    config.ncbf.policy_loss_coef = 0.01  # weight for policy loss when training ncbf
    config.ncbf.coef_decay_lambda = 0.95  # decay lambda for ncbf loss coefficients < 1
    config.ncbf.action_clipping = False
    config.ncbf.lambda_slack = 1000.0  # weight for slack variable in safety layer QP
    config.ncbf.max_delta_u = 0.5  # max L2 action correction from the safety layer; <=0 disables

    config.ncbf.pretrain = config_dict.ConfigDict()
    config.ncbf.pretrain.nr_steps = 0    # note this params says pretrian ncbf with X normal policy steps (nr_steps of rollout x， nr_epochs of training, each with minibatches with minibatch size)
    config.ncbf.pretrain.nr_minibatches = 100

    config.ncbf_buffer = config_dict.ConfigDict()
    config.ncbf_buffer.pos_buffer_size = 1 # note this is in unit of nr_steps * nr_envs, with default params this is 128* 200 * 4096 = 104,857,600 transitions
    config.ncbf_buffer.neg_buffer_size = 1
    config.ncbf_buffer.neg_sampling_ratio = 0.5 # ratio of sampling from negative buffer

    config.next_step_predictor = config_dict.ConfigDict()
    config.next_step_predictor.lr = 1e-3
    config.next_step_predictor.nr_minibatches = 50
    config.next_step_predictor.pretrain_nr_steps = 0
    config.next_step_predictor.pretrain_nr_minibatches = 100
    config.next_step_predictor.history_encoder_type = 'FFNN'  # 'FFNN' or 'GRU'
    config.next_step_predictor.history_encoder_hidden_size = 64
    config.next_step_predictor.decoder_output_dim = None  # Will be set to observation_dim at runtime
    config.next_step_predictor.aux_loss_coef = 1.0

    if "booster" in algorithm_name.lower():
        apply_booster_defaults(config)
    return config


def apply_booster_defaults(config):
    config.total_timesteps = 700e6
    # config.learning_rate = 1e-5
    # config.anneal_learning_rate = False
    # config.nr_epochs = 20
    # config.minibatch_size = 409600
    # config.gamma = 0.995
    config.gae_lambda = 0.95
    config.clip_range = 0.2
    config.target_kl = 0.02
    config.entropy_coef = 0.005
    config.critic_coef = 0.5
    config.max_grad_norm = 1.0
    config.std_dev = 0.135
    config.action_clipping_and_rescaling = False
    config.action_clip = 1.0
    config.evaluation_active = False
    # config.evaluation_and_save_frequency = -1
    config.hidden_layers = (256, 128)
