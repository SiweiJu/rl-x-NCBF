from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.perturbation_functions.default import DefaultDRPerturbation
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.perturbation_functions.none import NoneDRPerturbation
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.perturbation_functions.booster import BoosterDRPerturbation


def get_domain_randomization_perturbation_function(name, env, **kwargs):
    if name == "default":
        return DefaultDRPerturbation(env, **kwargs)
    elif name == "booster":
        return BoosterDRPerturbation(env, **kwargs)
    elif name == "none":
        return NoneDRPerturbation(env, **kwargs)
    else:
        raise NotImplementedError
