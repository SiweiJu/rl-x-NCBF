class NoneDRSeenRobotFunction:
    def __init__(self, env):
        self.env = env


    def init(self, internal_state):
        internal_state["scaling_factor"] = self.env.scaling_factor
        internal_state["robot_nominal_qpos_height_over_ground"] = self.env.initial_qpos[2]
        internal_state["robot_nominal_imu_height_over_ground"] = self.env.initial_imu_height
        internal_state["nr_collisions_in_nominal"] = 0


    def sample(self, internal_state, mjx_model, data, should_randomize, key):
        return mjx_model, data
