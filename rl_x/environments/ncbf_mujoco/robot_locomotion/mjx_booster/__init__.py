from rl_x.environments.environment_manager import extract_environment_name_from_file, register_environment
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.create_env import create_env
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.default_config import get_config as get_mjx_config
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.general_properties import GeneralProperties


def get_config(environment_name):
    return get_mjx_config(environment_name)


ROBOT_LOCOMOTION_MJX_BOOSTER_ENV = extract_environment_name_from_file(__file__)
register_environment(ROBOT_LOCOMOTION_MJX_BOOSTER_ENV, get_config, create_env, GeneralProperties)
