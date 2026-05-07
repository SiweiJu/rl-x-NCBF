import importlib
from pathlib import Path

from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.environment import LocomotionEnv
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.general_properties import GeneralProperties
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.default_config import apply_booster_defaults, apply_boosterball_defaults


def create_env(config):
    use_booster_defaults = config.environment.get("use_booster_defaults", True)
    if (
        config.environment.train_robot == "booster_t1"
        and use_booster_defaults
        and not config.environment.get("booster_defaults_applied", False)
    ):
        apply_booster_defaults(config.environment)
    if (
        config.environment.train_robot == "booster_t1"
        and use_booster_defaults
        and config.environment.ball_plate.enabled
        and not config.environment.get("boosterball_defaults_applied", False)
    ):
        apply_boosterball_defaults(config.environment)

    robot_config = importlib.import_module(f"rl_x.environments.ncbf_mujoco.robot_locomotion.robots.{config.environment.train_robot}.robot_config").robot_config
    robot_config["directory_path"] = Path(__file__).parent.parent / "robots" / config.environment.train_robot

    env = LocomotionEnv(
        robot_config=robot_config,
        runner_mode=config.runner.mode,
        render=config.environment.render,
        env_config=config.environment,
        nr_envs=config.environment.nr_envs,
    )

    env.general_properties = GeneralProperties

    return env
