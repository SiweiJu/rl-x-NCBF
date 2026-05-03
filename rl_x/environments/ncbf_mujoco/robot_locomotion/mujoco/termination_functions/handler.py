from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.termination_functions.below_height import BelowHeightTermination
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.termination_functions.booster import BoosterTermination


def get_termination_function(name, env, **kwargs):
    if name == "below_height":
        return BelowHeightTermination(env, **kwargs)
    elif name == "booster":
        return BoosterTermination(env, **kwargs)
    else:
        raise NotImplementedError
