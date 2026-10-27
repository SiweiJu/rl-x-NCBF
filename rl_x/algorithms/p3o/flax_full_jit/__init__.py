from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.p3o.flax_full_jit.p3o import P3O
from rl_x.algorithms.p3o.flax_full_jit.default_config import get_config
from rl_x.algorithms.p3o.flax_full_jit.general_properties import GeneralProperties


P3O_FLAX_FULL_JIT = extract_algorithm_name_from_file(__file__)
register_algorithm(P3O_FLAX_FULL_JIT, get_config, P3O, GeneralProperties)
