import jax.numpy as jnp


class BelowHeightTermination:
    def __init__(self, env):
        self.env = env

        self.height_percentage_threshold = self.env.env_config["termination"]["height_percentage_threshold"]
        # Paper code (ncbf-q): the height threshold shrank with the env curriculum, (1 - coeff) * pct * nominal,
        # i.e. no height termination at full curriculum or in eval mode.
        self.height_curriculum_scaled = bool(self.env.env_config["termination"].get("height_curriculum_scaled", False))


    def should_terminate(self, internal_state):
        height_scale = (1.0 - internal_state["env_curriculum_coeff"]) if self.height_curriculum_scaled else 1.0
        below_height = internal_state["robot_imu_height_over_ground"] < (
            height_scale * self.height_percentage_threshold * internal_state["robot_nominal_imu_height_over_ground"]
        )

        body_roll = internal_state["imu_orientation_euler"][0]
        body_pitch = internal_state["imu_orientation_euler"][1]
        body_tilt_angle = jnp.sqrt(body_roll ** 2 + body_pitch ** 2)
        body_tilt = body_tilt_angle > internal_state["body_tilt_threshold"]

        return below_height | body_tilt
