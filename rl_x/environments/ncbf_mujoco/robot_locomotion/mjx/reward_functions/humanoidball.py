from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.reward_functions.G1ball import G1BallReward
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.reward_functions.boosterball import BoosterBallReward
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.reward_functions.humanoid import get_humanoid_reward_profile


class HumanoidBallReward:
    def __init__(self, env, profile=None):
        self.env = env
        self.profile = get_humanoid_reward_profile(env, profile)
        if self.profile == "booster":
            self._reward = BoosterBallReward(env)
        else:
            self._reward = G1BallReward(env)

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
