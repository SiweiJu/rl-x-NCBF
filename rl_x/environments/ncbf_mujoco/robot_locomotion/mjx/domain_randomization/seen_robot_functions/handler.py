from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.seen_robot_functions.default import DefaultDRSeenRobotFunction
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.seen_robot_functions.none import NoneDRSeenRobotFunction
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.seen_robot_functions.booster import BoosterDRSeenRobotFunction


def get_domain_randomization_seen_robot_function(name, env, **kwargs):
    if name == "default":
        return DefaultDRSeenRobotFunction(env, **kwargs)
    elif name == "booster":
        return BoosterDRSeenRobotFunction(env, **kwargs)
    elif name == "none":
        return NoneDRSeenRobotFunction(env, **kwargs)
    else:
        raise NotImplementedError
