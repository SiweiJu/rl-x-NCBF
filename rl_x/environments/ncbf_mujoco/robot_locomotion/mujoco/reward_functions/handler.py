from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.reward_functions.default import DefaultReward
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.reward_functions.defaultG1 import DefaultG1Reward
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.reward_functions.G1ball import G1BallReward


def get_reward_function(name, env, **kwargs):
    if getattr(env, "use_ball_plate", False):
        name = "G1ball"

    if name == "default":
        return DefaultReward(env, **kwargs)
    elif name == "defaultG1":
        return DefaultG1Reward(env, **kwargs)
    elif name == "G1ball":
        return G1BallReward(env, **kwargs)
    else:
        raise NotImplementedError
