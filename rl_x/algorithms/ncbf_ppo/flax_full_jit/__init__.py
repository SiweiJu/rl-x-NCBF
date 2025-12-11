from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.ncbf_ppo.flax_full_jit.ppo import PPO
from rl_x.algorithms.ncbf_ppo.flax_full_jit.default_config import get_config
from rl_x.algorithms.ncbf_ppo.flax_full_jit.general_properties import GeneralProperties


NCBF_PPO_FLAX_FULL_JIT = extract_algorithm_name_from_file(__file__)
register_algorithm(NCBF_PPO_FLAX_FULL_JIT, get_config, PPO, GeneralProperties)
