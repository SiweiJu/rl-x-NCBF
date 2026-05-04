class BoosterTermination:
    def __init__(self, env):
        self.env = env
        termination_config = self.env.env_config["termination"]
        self.min_height = termination_config.get("min_height", 0.3)
        self.max_height = termination_config.get("max_height", 1.0)


    def should_terminate(self, internal_state):
        height = internal_state["robot_imu_height_over_ground"]
        return (height < self.min_height) | (height > self.max_height)
