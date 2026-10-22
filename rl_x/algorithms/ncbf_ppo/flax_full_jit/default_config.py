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
    config.adaptive_lr = False
    config.adaptive_lr_target_kl = 0.0  # <= 0 follows target_kl
    config.kl_margin = 2.0  # no LR change inside [target / margin, target * margin]
    config.kl_lr_scale = 1.25  # max multiplicative LR change per PPO update
    config.kl_lr_gain = 0.25  # log-space proportional gain outside the KL deadband
    config.lr_min = 1e-6
    config.lr_max = 0.0  # <= 0 uses the initial learning_rate as the max
    config.entropy_coef = 0.0
    config.critic_coef = 1.0
    config.max_grad_norm = 5.0
    config.std_dev = 1.0
    config.action_clipping_and_rescaling = False
    config.action_clip = 0.0
    config.hidden_layers = (512, 256, 128)
    config.use_history_latent_for_policy = False
    config.use_history_latent_for_value = False
    config.use_decoder_output_for_policy = False
    config.evaluation_and_save_frequency = 17301504  # -1 to disable
    config.evaluation_active = True

    # ncbf config
    config.ncbf = config_dict.ConfigDict()
    config.ncbf.n_ensemble = 5  # number of networks to assemble for ncbf prediction
    config.ncbf.independent_ensemble = True  # False = legacy: train one network and repeat it n_ensemble times
    config.ncbf.ensemble_bootstrap_prob = 0.8  # per-member Bernoulli bootstrap mask on each sample; 1.0 disables
    config.ncbf.critic_action_index = "current"  # "current": train h(s_t, a_t); "previous": legacy h(s_t, a_{t-1})
    config.ncbf.type = 'FFNN'
    config.ncbf.H = 25  # prediction horizon
    config.ncbf.lr = 1e-3
    config.ncbf.gamma_c = 0.5     # safety threshold on the predicted safe probability (h is a sigmoid output)
    config.ncbf.gamma_c_scaled_by_eta = True  # paper form h_next >= (1-eta) h + eta*gamma_c; False = legacy "+ gamma_c"
    config.ncbf.label_discount = False  # weight negative labels by coef_decay_lambda ** steps_to_failure (paper eq. 2)
    config.ncbf.w_temporal = 0.0  # weight of the DT-CBF temporal consistency loss (paper eq. 4-6), on-policy samples only
    config.ncbf.temporal_mode = "adjacent"  # F2 fix: h(s_t,a_t) vs h(s_{t+1},a_{t+1}); "legacy_raw_vs_executed" = paper code (h(s_t,a_exec) vs h(s_t,a_raw), uses w_temporal; paper used w_cbf=1)
    config.ncbf.temporal_pairs = "safe_safe"  # "safe_safe" (paper eq. 4 on safe->safe pairs); "safe_safe_and_unsafe" = previous multistep term
    config.ncbf.terminal_critic_input = "pre_reset"  # F1 fix; "zeros" = paper code trained the episode-ending step on (0, 0)
    config.ncbf.first_step_prev_value = "gamma_c"  # F1 fix in the shield: first step of an episode uses h_prev = gamma_c; "input" = legacy h(reset input)
    config.ncbf.lipschitz_state_only = True  # Lipschitz penalty only on the state part of the input, not on the action
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
    config.ncbf.residual_mc_samples = 16
    config.ncbf.policy_loss_coef = 0.01  # weight for policy loss when training ncbf
    config.ncbf.coef_decay_lambda = 0.95  # decay lambda for ncbf loss coefficients < 1
    config.ncbf.action_clipping = False
    config.ncbf.lambda_slack = 1000.0  # weight for slack variable in safety layer QP
    config.ncbf.max_delta_u = 0.5  # max L2 action correction from the safety layer; <=0 disables
    config.ncbf.safety_layer_projection = "soft_slack"  # soft_slack or hard_projection
    config.ncbf.safety_layer_min_grad_norm = 0.0  # <=0 disables low-gradient guard
    config.ncbf.post_check_actual_residual = False  # expensive safety-layer debug diagnostic
    config.ncbf.safety_layer_std_coeff_start = -2.0  # beta at the start of training (negative = relax)
    config.ncbf.safety_layer_std_coeff_final = 1.0  # beta at the end of training; evaluation uses beta = 1
    config.ncbf.tighten_uncertainty = "total"  # sigma used in beta*sigma: none, total, epistemic, aleatoric
    config.ncbf.slack_uncertainty = "none"  # sigma scaling the QP slack (paper eq. 13): none (legacy 1/lambda), total, epistemic, aleatoric
    config.ncbf.anticipation_through_shield = False  # True: backprop the anticipation loss through the shield (paper); False: stop_gradient

    config.ncbf_buffer = config_dict.ConfigDict()
    # note this is in unit of nr_steps * nr_envs, with default params this is 128* 200 * 4096 = 104,857,600 transitions
    config.ncbf_buffer.neg_buffer_size = 1
    config.ncbf_buffer.neg_sampling_ratio = 0.5 # ratio of sampling from negative buffer

    config.next_step_predictor = config_dict.ConfigDict()
    config.next_step_predictor.lr = 1e-3
    config.next_step_predictor.nr_minibatches = 50
    config.next_step_predictor.history_encoder_type = 'FFNN'  # 'FFNN', 'GRU', or 'scan_conv' (circular conv over the obstacle range scan)
    config.next_step_predictor.scan_encoder_sectors = 8  # scan_conv: angular sectors kept after pooling over rays
    config.next_step_predictor.history_encoder_hidden_size = 64
    config.next_step_predictor.decoder_output_dim = None  # Will be set to observation_dim at runtime
    config.next_step_predictor.aux_loss_coef = 1.0

    # Counterfactual check of the shield (runner.mode=test): at decision points along a shielded rollout,
    # branch from the same state with the raw and the shielded action and compare the outcomes over H steps.
    config.counterfactual = config_dict.ConfigDict()
    config.counterfactual.enabled = False  # True: test() runs the counterfactual check instead of the batch evaluation
    config.counterfactual.nr_decision_points = 8  # per env
    config.counterfactual.interval = 50  # control steps of the shielded rollout between decision points
    config.counterfactual.horizon = 0  # steps to roll out after the branch; <= 0 uses ncbf.H
    config.counterfactual.continuation = "shielded"  # policy after the first step: "shielded" (the critic's semantics) or "raw"
    config.counterfactual.nr_repeats = 1  # rollouts per branch with different env randomness (shared by both branches)

    if "booster" in algorithm_name.lower():
        apply_booster_defaults(config)
    return config


def apply_booster_defaults(config):
    pass
    # config.total_timesteps = 700e6
    # config.learning_rate = 1e-5
    # config.anneal_learning_rate = False
    # config.nr_epochs = 20
    # config.minibatch_size = 409600
    # config.gamma = 0.995
    # config.gae_lambda = 0.95
    # config.clip_range = 0.2
    # config.target_kl = 0.02
    # config.entropy_coef = 0.005
    # config.critic_coef = 0.5
    # config.max_grad_norm = 1.0
    # config.std_dev = 0.135
    # config.action_clipping_and_rescaling = False
    # config.action_clip = 1.0
    # config.evaluation_active = False
    # config.hidden_layers = (512, 256, 128)
