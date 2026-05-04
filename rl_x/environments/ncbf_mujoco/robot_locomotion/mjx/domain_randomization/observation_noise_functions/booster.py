import jax
import jax.numpy as jnp


class BoosterDRObservationNoise:
    handles_normalization = True

    def __init__(self, env):
        self.env = env
        config = env.env_config["domain_randomization"]["observation_noise"]
        self.add_joint_pos_noise = config.get("add_joint_pos_noise", True)
        self.joint_pos_noise_scale = config.get("joint_pos_noise_scale", 0.03)
        self.add_joint_vel_noise = config.get("add_joint_vel_noise", True)
        self.joint_vel_noise_scale = config.get("joint_vel_noise_scale", 0.3)
        self.add_gravity_noise = config.get("add_gravity_noise", True)
        self.gravity_noise_scale = config.get("gravity_noise_scale", 0.015)
        self.add_free_joint_lin_vel_noise = config.get("add_free_joint_lin_vel_noise", False)
        self.lin_vel_noise_scale = config.get("lin_vel_noise_scale", 0.1)
        self.add_free_joint_ang_vel_noise = config.get("add_free_joint_ang_vel_noise", True)
        self.ang_vel_noise_scale = config.get("ang_vel_noise_scale", 0.2)
        self.add_policy_ang_vel_noise = config.get("add_policy_ang_vel_noise", True)
        self.policy_ang_vel_noise_scale = config.get("policy_ang_vel_noise_scale", 0.2)
        self.normalize_observations = config.get("normalize_observations", True)
        self.norm_factors = config.get("norm_factors", {
            "joint_pos": 1.0,
            "joint_vel": 0.1,
            "gravity": 1.0,
            "lin_vel": 1.0,
            "ang_vel": 1.0,
        })


    def init_attributes(self):
        self.joint_positions_obs_idx = self.env.joint_positions_obs_idx
        self.joint_velocities_obs_idx = self.env.joint_velocities_obs_idx
        self.imu_linear_vel_obs_idx = self.env.imu_linear_vel_obs_idx
        self.imu_angular_vel_obs_idx = self.env.imu_angular_vel_obs_idx
        self.gravity_vector_obs_idx = self.env.gravity_vector_obs_idx


    def modify_observation(self, internal_state, observation, key):
        keys = jax.random.split(key, 5)
        if self.add_joint_pos_noise:
            observation = observation.at[self.joint_positions_obs_idx].add(
                jax.random.normal(keys[0], shape=(len(self.joint_positions_obs_idx),)) * self.joint_pos_noise_scale
            )
        if self.add_joint_vel_noise:
            observation = observation.at[self.joint_velocities_obs_idx].add(
                jax.random.normal(keys[1], shape=(len(self.joint_velocities_obs_idx),)) * self.joint_vel_noise_scale
            )
        if self.add_gravity_noise:
            observation = observation.at[self.gravity_vector_obs_idx].add(
                jax.random.normal(keys[2], shape=(len(self.gravity_vector_obs_idx),)) * self.gravity_noise_scale
            )
        if self.add_free_joint_lin_vel_noise:
            observation = observation.at[self.imu_linear_vel_obs_idx].add(
                jax.random.normal(keys[3], shape=(len(self.imu_linear_vel_obs_idx),)) * self.lin_vel_noise_scale
            )
        if self.add_free_joint_ang_vel_noise or self.add_policy_ang_vel_noise:
            observation = observation.at[self.imu_angular_vel_obs_idx].add(
                jax.random.normal(keys[4], shape=(len(self.imu_angular_vel_obs_idx),)) * self.ang_vel_noise_scale
            )

        if self.normalize_observations:
            observation = observation.at[self.joint_positions_obs_idx].multiply(self.norm_factors["joint_pos"])
            observation = observation.at[self.joint_velocities_obs_idx].multiply(self.norm_factors["joint_vel"])
            observation = observation.at[self.gravity_vector_obs_idx].multiply(self.norm_factors["gravity"])
            observation = observation.at[self.imu_linear_vel_obs_idx].multiply(self.norm_factors["lin_vel"])
            observation = observation.at[self.imu_angular_vel_obs_idx].multiply(self.norm_factors["ang_vel"])

        return observation
