import importlib
from pathlib import Path
import gymnasium as gym

from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.environment import LocomotionEnv
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.wrappers import RLXInfo, RecordEpisodeStatistics
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.async_vectorized_wrapper import AsyncVectorEnvWithSkipping
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.general_properties import GeneralProperties
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.default_config import apply_booster_defaults, apply_boosterball_defaults


def create_env(config):
    if config.environment.train_robot == "booster_t1" and not config.environment.get("booster_defaults_applied", False):
        apply_booster_defaults(config.environment)
    if (
        config.environment.train_robot == "booster_t1"
        and config.environment.ball_plate.enabled
        and not config.environment.get("boosterball_defaults_applied", False)
    ):
        apply_boosterball_defaults(config.environment)

    robot_config = importlib.import_module(f"rl_x.environments.ncbf_mujoco.robot_locomotion.robots.{config.environment.train_robot}.robot_config").robot_config
    robot_config["directory_path"] = Path(__file__).parent.parent / "robots" / config.environment.train_robot
    
    def make_env(seed):
        def thunk():
            env = LocomotionEnv(
                robot_config=robot_config,
                runner_mode=config.runner.mode,
                seed=seed,
                render=config.environment.render,
                env_config=config.environment,
                nr_envs=config.environment.nr_envs,
            )
            env = RecordEpisodeStatistics(env)
            env.action_space.seed(config.environment.seed)
            env.observation_space.seed(config.environment.seed)
            return env
        return thunk

    make_env_functions = [make_env(config.environment.seed + i) for i in range(config.environment.nr_envs)]
    if config.environment.nr_envs == 1:
        env = gym.vector.SyncVectorEnv(make_env_functions)
    else:
        env = AsyncVectorEnvWithSkipping(make_env_functions, config.environment.async_skip_percentage)
    env = RLXInfo(env)
    env.general_properties = GeneralProperties

    return env
