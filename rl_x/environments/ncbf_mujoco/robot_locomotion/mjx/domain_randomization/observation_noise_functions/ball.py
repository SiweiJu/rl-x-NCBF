import jax
import jax.numpy as jnp

from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.observation_noise_functions.booster import BoosterDRObservationNoise
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.observation_noise_functions.default import DefaultDRObservationNoise


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
            self.ball_plate_position_obs_idx = jnp.concatenate([
                self.ball_plate_obs_idx[:3],
                self.ball_plate_obs_idx[6:9],
            ], dtype=int)
            self.ball_plate_velocity_obs_idx = jnp.concatenate([
                self.ball_plate_obs_idx[3:6],
                self.ball_plate_obs_idx[9:],
            ], dtype=int)
        else:
            self.ball_plate_position_obs_idx = jnp.array([], dtype=int)
            self.ball_plate_velocity_obs_idx = jnp.array([], dtype=int)


    def modify_observation(self, internal_state, observation, key):
        base_key, position_key, velocity_key = jax.random.split(key, 3)
        observation = self.base_noise.modify_observation(internal_state, observation, base_key)

        if not self.env.include_ball_plate_observations or len(self.ball_plate_obs_idx) == 0:
            return observation

        if self.add_ball_plate_obs_noise:
            observation = observation.at[self.ball_plate_position_obs_idx].add(
                jax.random.uniform(
                    position_key,
                    shape=(len(self.ball_plate_position_obs_idx),),
                    minval=-self.ball_plate_position_noise,
                    maxval=self.ball_plate_position_noise,
                )
            )
            observation = observation.at[self.ball_plate_velocity_obs_idx].add(
                jax.random.uniform(
                    velocity_key,
                    shape=(len(self.ball_plate_velocity_obs_idx),),
                    minval=-self.ball_plate_velocity_noise,
                    maxval=self.ball_plate_velocity_noise,
                )
            )

        if self.handles_ball_plate_normalization:
            observation = self._normalize_ball_plate_observation(internal_state, observation)

        return observation


    def _normalize_ball_plate_observation(self, internal_state, observation):
        ball_plate_plate_size = internal_state["ball_plate_plate_size"]
        position_scale = jnp.maximum(jnp.max(ball_plate_plate_size[:2]), 1e-6)
        observation = observation.at[self.ball_plate_obs_idx[:3]].set(jnp.clip(observation[self.ball_plate_obs_idx[:3]] / position_scale, -10.0, 10.0))
        observation = observation.at[self.ball_plate_obs_idx[3:6]].set(jnp.clip(observation[self.ball_plate_obs_idx[3:6]] / 5.0, -10.0, 10.0))
        observation = observation.at[self.ball_plate_obs_idx[6:9]].set(jnp.clip(observation[self.ball_plate_obs_idx[6:9]] / position_scale, -10.0, 10.0))
        observation = observation.at[self.ball_plate_obs_idx[9:]].set(jnp.clip(observation[self.ball_plate_obs_idx[9:]] / 5.0, -10.0, 10.0))
        return observation
