from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.ncbf_ppo.flax.default_config import get_config as get_flax_config
from rl_x.algorithms.ncbf_ppo.flax.general_properties import GeneralProperties
from rl_x.algorithms.ncbf_ppo.flax.ppo import PPO


def get_config(algorithm_name):
    return get_flax_config(algorithm_name)


NCBF_PPO_FLAX_BOOSTER = extract_algorithm_name_from_file(__file__)
register_algorithm(NCBF_PPO_FLAX_BOOSTER, get_config, PPO, GeneralProperties)
