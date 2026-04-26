class BelowHeightTermination:
    def __init__(self, env):
        self.env = env

        self.height_percentage_threshold = self.env.env_config["termination"]["height_percentage_threshold"]


    def should_terminate(self):
        below_height = self.env.internal_state["robot_imu_height_over_ground"] < ((1 - self.env.internal_state["env_curriculum_coeff"]) * self.height_percentage_threshold * self.env.internal_state["robot_nominal_imu_height_over_ground"])
        ball_dropped = self.env.use_ball_plate and self.env.ball_plate_has_dropped()

        if below_height:
            print("Termination: Below Height")
        if ball_dropped:
            print("Termination: Ball Dropped")
        return below_height | ball_dropped
