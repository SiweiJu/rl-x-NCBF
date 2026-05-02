from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.termination_functions.below_height import BelowHeightTermination
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.termination_functions.G1ball import G1BallTermination


def get_termination_function(name, env, **kwargs):
    if getattr(env, "use_ball_plate", False):
        name = "G1ball"

    if name == "below_height":
        return BelowHeightTermination(env, **kwargs)
    elif name == "G1ball":
        return G1BallTermination(env, **kwargs)
    else:
        raise NotImplementedError
