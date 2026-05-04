from rl_x.algorithms.cpo.flax_full_jit.cpo import CPO
from rl_x.algorithms.ppo_lagrangian.flax_full_jit.general_properties import GeneralProperties


class PPOLagrangian(CPO):
    def general_properties():
        return GeneralProperties
