import numpy as np


class G1BallTermination:
    def __init__(self, env):
        self.env = env

        self.height_percentage_threshold = self.env.env_config["termination"]["height_percentage_threshold"]


    def should_terminate(self):
        below_height = self.env.internal_state["robot_imu_height_over_ground"] < (
            self.height_percentage_threshold * self.env.internal_state["robot_nominal_imu_height_over_ground"]
        )

        body_roll = self.env.internal_state["imu_orientation_euler"][0]
        body_pitch = self.env.internal_state["imu_orientation_euler"][1]
        body_tilt_angle = np.sqrt(body_roll ** 2 + body_pitch ** 2)
        body_tilt = body_tilt_angle > self.env.internal_state["body_tilt_threshold"]

        ball_dropped = self.env.ball_plate_ball_has_dropped()
        plate_dropped = self.env.ball_plate_plate_has_dropped()

        if below_height:
            print("Termination: Below Height")

        if body_tilt:
            print("Termination: Tilt")

        if ball_dropped:
            print("Termination: Ball Dropped")

        if plate_dropped:
            print("Termination: Plate Dropped")

        return below_height or body_tilt or ball_dropped or plate_dropped
