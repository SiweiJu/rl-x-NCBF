from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.ncbf_ppo.flax_full_jit.default_config import get_config as get_flax_full_jit_config
from rl_x.algorithms.ncbf_ppo.flax_full_jit.general_properties import GeneralProperties
from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import PPO


def get_config(algorithm_name):
    return get_flax_full_jit_config(algorithm_name)


NCBF_PPO_FLAX_FULL_JIT_BOOSTER = extract_algorithm_name_from_file(__file__)
register_algorithm(NCBF_PPO_FLAX_FULL_JIT_BOOSTER, get_config, PPO, GeneralProperties)
