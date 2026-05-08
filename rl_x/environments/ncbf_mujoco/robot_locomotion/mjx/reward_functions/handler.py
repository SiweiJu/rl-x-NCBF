from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.reward_functions.default import DefaultReward
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.reward_functions.humanoid import HumanoidReward
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.reward_functions.humanoidball import HumanoidBallReward


def _humanoid_profile_from_name(name, env):
    if name in ("booster", "boosterball"):
        return "booster"
    if name in ("defaultG1", "G1ball"):
        return "g1"
    return env.env_config["reward"].get("profile", None)


def get_reward_function(name, env, **kwargs):
    if getattr(env, "use_ball_plate", False):
        name = "humanoidball"
    elif name == "default" and env.robot_config.get("short_name") == "g1":
        name = "humanoid"

    if name == "default":
        return DefaultReward(env, **kwargs)
    elif name == "defaultG1":
        return HumanoidReward(env, profile="g1", **kwargs)
    elif name == "G1ball":
        return HumanoidBallReward(env, profile="g1", **kwargs)
    elif name == "booster":
        return HumanoidReward(env, profile=env.env_config["reward"].get("profile", "g1"), **kwargs)
    elif name == "boosterball":
        return HumanoidBallReward(env, profile=env.env_config["reward"].get("profile", "g1"), **kwargs)
    elif name == "humanoid":
        return HumanoidReward(env, profile=_humanoid_profile_from_name(name, env), **kwargs)
    elif name == "humanoidball":
        return HumanoidBallReward(env, profile=_humanoid_profile_from_name(name, env), **kwargs)
    else:
        raise NotImplementedError
