from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.domain_randomization.mujoco_model_functions.default import DefaultDRMuJoCoModel
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.domain_randomization.mujoco_model_functions.none import NoneDRMuJoCoModel
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.domain_randomization.mujoco_model_functions.booster import BoosterDRMuJoCoModel


def get_domain_randomization_mujoco_model_function(name, env, **kwargs):
    if name == "default":
        return DefaultDRMuJoCoModel(env, **kwargs)
    elif name == "booster":
        return BoosterDRMuJoCoModel(env, **kwargs)
    elif name == "none":
        return NoneDRMuJoCoModel(env, **kwargs)
    else:
        raise NotImplementedError
