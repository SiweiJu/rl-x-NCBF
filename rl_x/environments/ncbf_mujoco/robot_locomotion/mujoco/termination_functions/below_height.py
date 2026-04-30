import numpy as np

class BelowHeightTermination:
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

        print("body_tilt: ", body_tilt_angle)
        body_tilt = body_tilt_angle > self.env.internal_state["body_tilt_threshold"]

        if below_height:
            print("Termination: Below Height")

        if body_tilt:
            print("Termination: Tilt")
        return np.logical_or(below_height, body_tilt)
