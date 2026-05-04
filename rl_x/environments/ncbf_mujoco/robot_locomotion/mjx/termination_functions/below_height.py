import jax.numpy as jnp


class BelowHeightTermination:
    def __init__(self, env):
        self.env = env

        self.height_percentage_threshold = self.env.env_config["termination"]["height_percentage_threshold"]
        self.use_curriculum_scaling = self.env.env_config["termination"].get("use_curriculum_scaling", True)


    def should_terminate(self, internal_state):
        curriculum_scale = jnp.where(self.use_curriculum_scaling, 1 - internal_state["env_curriculum_coeff"], 1.0)
        below_height = internal_state["robot_imu_height_over_ground"] < (curriculum_scale * self.height_percentage_threshold * internal_state["robot_nominal_imu_height_over_ground"])

        return below_height
