import numpy as np

from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.domain_randomization.observation_noise_functions.booster import BoosterDRObservationNoise
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.domain_randomization.observation_noise_functions.default import DefaultDRObservationNoise


class BallDRObservationNoise:
    def __init__(self, env, base_type="booster"):
        self.env = env
        self.base_noise = self._get_base_noise(base_type)
        self.handles_normalization = getattr(self.base_noise, "handles_normalization", False)

        config = env.env_config["domain_randomization"]["observation_noise"]
        self.add_ball_plate_obs_noise = config.get("add_ball_plate_obs_noise", True)
        self.ball_plate_position_noise = config.get("ball_plate_position_noise", 0.015)
        self.ball_plate_velocity_noise = config.get("ball_plate_velocity_noise", 0.1)
        self.normalize_ball_plate_observations = config.get("normalize_ball_plate_observations", True)
        self.handles_ball_plate_normalization = self.handles_normalization and self.normalize_ball_plate_observations


    def _get_base_noise(self, base_type):
        if base_type == "default":
            return DefaultDRObservationNoise(self.env)
        if base_type == "booster":
            return BoosterDRObservationNoise(self.env)
        raise NotImplementedError(f"Unsupported ball observation noise base type: {base_type}")


    def init_attributes(self):
        self.base_noise.init_attributes()
        self.ball_plate_obs_idx = self.env.ball_plate_obs_idx
        if len(self.ball_plate_obs_idx) > 0:
            self.ball_plate_position_obs_idx = np.concatenate([
                self.ball_plate_obs_idx[:3],
                self.ball_plate_obs_idx[6:9],
            ]).astype(int)
            self.ball_plate_velocity_obs_idx = np.concatenate([
                self.ball_plate_obs_idx[3:6],
                self.ball_plate_obs_idx[9:],
            ]).astype(int)
        else:
            self.ball_plate_position_obs_idx = np.array([], dtype=int)
            self.ball_plate_velocity_obs_idx = np.array([], dtype=int)


    def modify_observation(self, observation):
        self.base_noise.modify_observation(observation)

        if not self.env.include_ball_plate_observations or len(self.ball_plate_obs_idx) == 0:
            return

        if self.add_ball_plate_obs_noise:
            observation[self.ball_plate_position_obs_idx] += self.env.np_rng.uniform(
                size=(len(self.ball_plate_position_obs_idx),),
                low=-self.ball_plate_position_noise,
                high=self.ball_plate_position_noise,
            )
            observation[self.ball_plate_velocity_obs_idx] += self.env.np_rng.uniform(
                size=(len(self.ball_plate_velocity_obs_idx),),
                low=-self.ball_plate_velocity_noise,
                high=self.ball_plate_velocity_noise,
            )

        if self.handles_ball_plate_normalization:
            self._normalize_ball_plate_observation(observation)


    def _normalize_ball_plate_observation(self, observation):
        ball_plate_plate_size = self.env.internal_state["ball_plate_plate_size"]
        position_scale = np.maximum(np.max(ball_plate_plate_size[:2]), 1e-6)
        observation[self.ball_plate_obs_idx[:3]] = np.clip(observation[self.ball_plate_obs_idx[:3]] / position_scale, -10.0, 10.0)
        observation[self.ball_plate_obs_idx[3:6]] = np.clip(observation[self.ball_plate_obs_idx[3:6]] / 5.0, -10.0, 10.0)
        observation[self.ball_plate_obs_idx[6:9]] = np.clip(observation[self.ball_plate_obs_idx[6:9]] / position_scale, -10.0, 10.0)
        observation[self.ball_plate_obs_idx[9:]] = np.clip(observation[self.ball_plate_obs_idx[9:]] / 5.0, -10.0, 10.0)
