from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.command_functions.random import RandomCommands
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.command_functions.random_trajectory import RandomCommands as RandomTrajectoryCommands

def get_command_function(name, env, **kwargs):
    if name == "random":
        return RandomCommands(env, **kwargs)
    elif name == "random_trajectory":
        return RandomTrajectoryCommands(env, **kwargs)
    else:
        raise NotImplementedError
