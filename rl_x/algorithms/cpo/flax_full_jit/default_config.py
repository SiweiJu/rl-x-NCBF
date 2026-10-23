from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.nr_parallel_seeds = 1
    config.total_timesteps = 2e9
    config.nr_steps = 128
    config.gamma = 0.99
    config.gae_lambda = 0.9
    config.std_dev = 1.0
    config.action_clipping_and_rescaling = False
    config.evaluation_and_save_frequency = 17301504  # -1 to disable
    config.evaluation_active = True

    # Critics (reward and cost), trained by MSE regression with Adam after the policy step.
    config.learning_rate = 4e-4
    config.anneal_learning_rate = True
    config.nr_epochs = 10
    config.minibatch_size = 32768
    config.max_grad_norm = 5.0

    # Policy step: CPO trust region (Achiam et al. 2017), defaults from OmniSafe cpo.yaml.
    config.target_kl = 0.01  # trust region size delta
    config.cg_iterations = 15
    config.cg_damping = 0.1
    config.line_search_steps = 15
    config.line_search_decay = 0.8
    config.fvp_sample_freq = 1  # use every k-th sample for the Fisher-vector products
    config.standardize_reward_advantages = True
    config.cost_advantage_normalization = "center"  # center (OmniSafe), standardize, none

    # Constraint. The cost comes from env_state.info["constraint/cost"] (terminated as fallback);
    # on the locomotion envs it is 1 on the failure step and 0 otherwise.
    config.cost_gamma = 0.99
    config.cost_gae_lambda = 0.9
    # cost_measure: episode -> J_c = expected cost per episode (failures per episode), cost_limit in the same unit.
    #               rollout -> J_c = summed cost of one nr_steps rollout (the old ppo_lagrangian measure).
    config.cost_measure = "episode"
    config.cost_limit = 0.0
    # cost_violation_normalization: per_step -> c = (J_c - d) / T with T the mean episode length (or nr_steps for
    #   cost_measure=rollout), i.e. in the per-step units of the surrogate E[ratio A_c] (performance-difference
    #   lemma: Delta J_c ~= T * Delta E_t[ratio A_c]).
    #   none -> c = J_c - d as in OmniSafe (mixes episode units with the per-step surrogate).
    config.cost_violation_normalization = "per_step"

    # Perturbation-sweep evaluation (runner.mode=test), same keys as ncbf_ppo; see
    # rl_x/algorithms/ncbf_ppo/flax_full_jit/eval_sweep.py and experiments/pncbf/eval/run_eval_sweep.py.
    from rl_x.algorithms.ncbf_ppo.flax_full_jit.default_config import get_config as get_ncbf_ppo_config
    config.eval_sweep = get_ncbf_ppo_config("ncbf_ppo.flax_full_jit").eval_sweep

    return config
