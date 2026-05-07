from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.reward_functions.booster import BoosterReward
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.reward_functions.defaultG1 import DefaultG1Reward


def get_humanoid_reward_profile(env, profile=None):
    if profile is None:
        profile = env.env_config["reward"].get("profile", None)
    if profile is None:
        reward_type = env.env_config["reward"].get("type", "")
        if reward_type in ("booster", "boosterball"):
            profile = "booster"
        elif reward_type in ("defaultG1", "G1ball"):
            profile = "g1"
    if profile is None:
        profile = "booster" if env.robot_config.get("short_name") == "booster_t1" else "g1"
    if profile in ("g1", "unitree_g1", "defaultG1", "G1ball"):
        return "g1"
    if profile in ("booster_t1", "booster", "boosterball"):
        return "booster"
    raise ValueError(f"Unknown humanoid reward profile: {profile}")


class HumanoidReward:
    def __init__(self, env, profile=None):
        self.env = env
        self.profile = get_humanoid_reward_profile(env, profile)
        if self.profile == "booster":
            self._reward = BoosterReward(env)
        else:
            self._reward = DefaultG1Reward(env)

    def init(self, *args, **kwargs):
        return self._reward.init(*args, **kwargs)

    def calculate_joint_position_limits(self, *args, **kwargs):
        return self._reward.calculate_joint_position_limits(*args, **kwargs)

    def handle_model_change(self, *args, **kwargs):
        return self._reward.handle_model_change(*args, **kwargs)

    def setup(self, *args, **kwargs):
        return self._reward.setup(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self._reward.step(*args, **kwargs)

    def reward_and_info(self, *args, **kwargs):
        return self._reward.reward_and_info(*args, **kwargs)
