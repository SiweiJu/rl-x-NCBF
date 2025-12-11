from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.control_functions.pd import PDControl


def get_control_function(name, env, **kwargs):
    if name == "pd":
        return PDControl(env, **kwargs)
    else:
        raise NotImplementedError
