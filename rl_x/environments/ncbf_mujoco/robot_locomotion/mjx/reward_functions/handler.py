from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.reward_functions.default import DefaultReward
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.reward_functions.defaultG1 import DefaultG1Reward
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.reward_functions.G1ball import G1BallReward


def _humanoid_profile_from_name(name, env):
    if name in ("booster", "boosterball"):
        return "booster"
    if name in ("defaultG1", "G1ball"):
        return "g1"
    return env.env_config["reward"].get("profile", None)


def get_reward_function(name, env, **kwargs):
    if name == "default":
        return DefaultReward(env)
    elif name == "defaultG1":
        return DefaultG1Reward(env)
    elif name == "G1ball":
        return G1BallReward(env)
    else:
        raise NotImplementedError
