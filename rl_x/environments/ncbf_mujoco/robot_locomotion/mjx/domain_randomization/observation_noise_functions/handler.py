from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.observation_noise_functions.default import DefaultDRObservationNoise
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.observation_noise_functions.none import NoneDRObservationNoise
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.observation_noise_functions.booster import BoosterDRObservationNoise
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.observation_noise_functions.ball import BallDRObservationNoise


def get_observation_noise_function(name, env, **kwargs):
    include_ball_plate_observations = getattr(env, "include_ball_plate_observations", False)
    if name == "default":
        if include_ball_plate_observations:
            return BallDRObservationNoise(env, base_type="default", **kwargs)
        return DefaultDRObservationNoise(env, **kwargs)
    elif name == "booster":
        if include_ball_plate_observations:
            return BallDRObservationNoise(env, base_type="booster", **kwargs)
        return BoosterDRObservationNoise(env, **kwargs)
    elif name == "ball":
        return BallDRObservationNoise(env, **kwargs)
    elif name == "none":
        return NoneDRObservationNoise(env, **kwargs)
    else:
        raise NotImplementedError
