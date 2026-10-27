from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    # PPO part: identical to ppo_lagrangian.flax_full_jit (paper scale for Go2 is set by the launcher).
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

    config.cost_gamma = 0.99
    config.cost_gae_lambda = 0.9
    config.cost_limit = 0.0
    # rollout: summed cost per nr_steps rollout; episode: cost per finished episode (failures per episode with
    # the Go2 cost), see ppo_lagrangian/flax_full_jit/lagrange.py. Same options as PPO-Lag; T08 used episode.
    config.cost_measure = "episode"

    # P3O (Zhang et al., IJCAI 2022), see penalty.py.
    config.p3o = config_dict.ConfigDict()
    config.p3o.enabled = True  # False: plain clipped PPO policy loss (cost critic still trained), e.g. PPO + CaT
    config.p3o.kappa = 20.0  # penalty factor; paper Table (all tasks) and OmniSafe P3O.yaml: 20
    config.p3o.clip_cost_surrogate = True  # paper eq. 7: max(r A_C, clip(r) A_C); False = OmniSafe (r A_C)
    config.p3o.constraint_normalization = "per_step"  # per_step | one_minus_gamma | none
    config.p3o.cost_advantage_normalization = "consistent"  # consistent (N-P3O) | standardize | center | none

    # CaT (Chane-Sane et al. 2024), see cat.py. Off by default.
    config.cat = config_dict.ConfigDict()
    config.cat.enabled = False
    config.cat.constraint_source = "collision"  # collision | penetration | cost
    config.cat.p_max = 1.0  # hard constraint (paper: contacts use 1.0)
    config.cat.min_p = 0.0
    config.cat.tau = 0.95  # EMA of the per-step max violation (reference code default)
    config.cat.clip_reward_at_zero = True  # reference code: reward = clip(r (1 - delta), 0)

    # Perturbation-sweep evaluation (runner.mode=test), same keys as ncbf_ppo; see
    # rl_x/algorithms/ncbf_ppo/flax_full_jit/eval_sweep.py and experiments/pncbf/eval/run_eval_sweep.py.
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.default_config import get_config as get_ncbf_ppo_config
    config.eval_sweep = get_ncbf_ppo_config("ncbf_ppo.flax_full_jit").eval_sweep

    return config
