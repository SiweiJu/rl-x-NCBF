from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.command_functions.random import RandomCommands
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.command_functions.random_trajectory import RandomTrajectoryCommands
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.command_functions.booster import BoosterCommands

def get_command_function(name, env, **kwargs):
    if name == "random":
        return RandomCommands(env, **kwargs)
    elif name == "random_trajectory":
        return RandomTrajectoryCommands(env, **kwargs)
    elif name == "booster":
        return BoosterCommands(env, **kwargs)
    else:
        raise NotImplementedError
