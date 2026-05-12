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
    config.target_kl = 0.03
    config.adaptive_lr = False
    config.adaptive_lr_target_kl = 0.0  # <= 0 follows target_kl
    config.kl_margin = 2.0  # no LR change inside [target / margin, target * margin]
    config.kl_lr_scale = 1.25  # max multiplicative LR change per PPO update
    config.kl_lr_gain = 0.25  # log-space proportional gain outside the KL deadband
    config.lr_min = 1e-6
    config.lr_max = 0.0  # <= 0 uses the initial learning_rate as the max
    config.entropy_coef = 0.0
    config.critic_coef = 0.5
    config.max_grad_norm = 0.5
    config.std_dev = 1.0
    config.action_clipping_and_rescaling = False
    config.action_clip = 0.0
    config.nr_hidden_units = 256
    config.hidden_layers = (256, 128)
    config.use_history_latent_for_policy = False
    config.use_history_latent_for_value = False
    config.use_decoder_output_for_policy = False
    config.evaluation_frequency = 204800  # -1 to disable
    config.evaluation_episodes = 10

    # ncbf config
    config.ncbf = config_dict.ConfigDict()
    config.ncbf.type = 'FFNN'
    config.ncbf.H = 10  # prediction horizon
    config.ncbf.lr = 1e-3
    config.ncbf.w_clf = 1.0    # weight for classification loss
    config.ncbf.w_cbf = 1.0    # weight for CBF loss
    config.ncbf.w_lip = 0.0    # weight for lipschitz loss
    config.ncbf.w_wd = 0.0     # weight for weight decay loss
    config.ncbf.L_max = 0.0    # lipschitz target, set to 0.0 if just want small gradient
    config.ncbf.policy_loss_coef = 0.01  # weight for policy loss when training ncbf
    config.ncbf.use_safety_layer = False
    config.ncbf.nr_minibatches = 50
    config.ncbf.minibatch_size = 64
    config.ncbf.nr_hidden_units = 512
    config.ncbf.n_ensemble = 5  # number of networks to assemble for ncbf prediction
    config.ncbf.output_distribution = "deterministic"  # deterministic or logistic_normal
    config.ncbf.min_log_std = -5.0
    config.ncbf.max_log_std = 2.0
    config.ncbf.residual_mc_samples = 16

    # safety layer configs
    config.ncbf.eta_cbf = 1.0 # class Kappa function parameter for CBF constraint
    config.ncbf.gamma_c = 1.0     # safety threshold (>=1 is safe, <=0 is unsafe)
    config.ncbf.lambda_slack = 1000.0  # weight for slack variable in safety layer QP
    config.ncbf.max_delta_u = 0.0  # max L2 action correction from the safety layer; <=0 disables
    config.ncbf.safety_layer_projection = "soft_slack"  # soft_slack or hard_projection
    config.ncbf.safety_layer_min_grad_norm = 0.0  # <=0 disables low-gradient guard
    config.ncbf.post_check_actual_residual = False  # expensive safety-layer debug diagnostic
    config.ncbf.safety_layer_std_coeff_start = -2.0
    config.ncbf.safety_layer_std_coeff_final = 1.0

    config.ncbf.action_clipping = True
    config.ncbf.use_robust_safety_layer = False  # use robust safety layer that considers action noise

    config.ncbf_buffer = config_dict.ConfigDict()
    # unit of nr_steps * nr_envs
    config.ncbf_buffer.neg_buffer_size = 1
    config.ncbf_buffer.neg_sampling_ratio = 0.5 # ratio of sampling from negative buffer

    config.next_step_predictor = config_dict.ConfigDict()
    config.next_step_predictor.lr = 1e-3
    config.next_step_predictor.nr_minibatches = 50
    config.next_step_predictor.history_encoder_type = 'FFNN'  # 'FFNN' or 'GRU'
    config.next_step_predictor.history_encoder_hidden_size = 128
    config.next_step_predictor.decoder_output_dim = None  # Will be set to observation_dim at runtime


    config.action_noise_sampling_ratio = 0.0  # ratio of sampling actions for ncbf evaluation
    config.rollout_save_name = "rollouts"       #  default name for rollout saving
    if "booster" in algorithm_name.lower():
        apply_booster_defaults(config)
    return config


def apply_booster_defaults(config):
    config.total_timesteps = 700e6
    config.learning_rate = 1e-5
    config.anneal_learning_rate = False
    config.nr_steps = 50
    config.nr_epochs = 20
    config.minibatch_size = 409600
    config.gamma = 0.995
    config.gae_lambda = 0.95
    config.clip_range = 0.2
    config.entropy_coef = 0.005
    config.critic_coef = 0.5
    config.max_grad_norm = 1.0
    config.std_dev = 0.135
    config.action_clipping_and_rescaling = False
    config.action_clip = 1.0
    config.hidden_layers = (512, 256, 128)
