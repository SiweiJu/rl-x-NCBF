import jax.numpy as jnp


class G1BallTermination:
    def __init__(self, env):
        self.env = env

        self.height_percentage_threshold = self.env.env_config["termination"]["height_percentage_threshold"]


    def should_terminate(self, internal_state):
        below_height = internal_state["robot_imu_height_over_ground"] < (
            self.height_percentage_threshold * internal_state["robot_nominal_imu_height_over_ground"]
        )

        body_roll = internal_state["imu_orientation_euler"][0]
        body_pitch = internal_state["imu_orientation_euler"][1]
        body_tilt_angle = jnp.sqrt(body_roll ** 2 + body_pitch ** 2)
        body_tilt = body_tilt_angle > internal_state["body_tilt_threshold"]

        ball_dropped = self.env.ball_plate_ball_has_dropped(internal_state)
        plate_dropped = self.env.ball_plate_plate_has_dropped(internal_state)

        return below_height | body_tilt | ball_dropped | plate_dropped
