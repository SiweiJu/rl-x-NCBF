class NoneDRSeenRobotFunction:
    def __init__(self, env):
        self.env = env


    def init(self):
        self.env.internal_state["scaling_factor"] = self.env.scaling_factor
        self.env.internal_state["robot_nominal_qpos_height_over_ground"] = self.env.initial_qpos[2]
        self.env.internal_state["robot_nominal_imu_height_over_ground"] = self.env.initial_imu_height
        self.env.internal_state["nr_collisions_in_nominal"] = 0


    def sample(self):
        return
