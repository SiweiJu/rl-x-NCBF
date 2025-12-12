import numpy as np


class RandomTrajectoryCommands:
    def __init__(self, env):
        self.env = env
        self.max_velocity_per_m_factor = self.env.env_config["command"]["max_velocity_per_m_factor"]
        self.clip_max_velocity = self.env.env_config["command"]["clip_max_velocity"]
        self.zero_clip_threshold_percentage = self.env.env_config["command"]["zero_clip_threshold_percentage"]
        self.all_zero_chance = self.env.env_config["command"]["all_zero_chance"]
        self.single_zero_chance = self.env.env_config["command"]["single_zero_chance"]

        self.default_actuator_joint_keep_nominal = np.zeros(env.nr_actuator_joints, dtype=bool)
        self.default_actuator_joint_keep_nominal[env.robot_config["actuator_joints_to_stay_near_nominal"]] = 1.0
        self.default_actuator_joint_keep_nominal = np.array(self.default_actuator_joint_keep_nominal)

        self.trajectory_length = int(self.env.env_config["command"].get("trajectory_length_in_seconds", 10) / self.env.dt)
        self.min_steps_each_command = self.env.env_config["command"].get("min_steps_each_command", 10)

    def init(self):
        self.env.internal_state["actuator_joint_keep_nominal"] = self.default_actuator_joint_keep_nominal
        self._generate_random_trajectory()

    # generate a random velocity command at the start
    def _sample_single_command(self):
        """Sample one [vx, vy, yaw_vel] command with the same logic as get_next_command."""
        max_v = self.env.internal_state["max_command_velocity"]
        goal_velocities = self.env.np_rng.uniform(size=(3,), low=-max_v, high=max_v)

        # clip small values to zero
        thresh = self.zero_clip_threshold_percentage * max_v
        goal_velocities = np.where(np.abs(goal_velocities) < thresh, 0.0, goal_velocities)

        # sometimes make all zeros
        if self.env.np_rng.binomial(n=1, p=self.all_zero_chance):
            goal_velocities = np.zeros(3)

        # sometimes zero individual components
        mask_single_zero = self.env.np_rng.uniform(size=(3,)) < self.single_zero_chance
        goal_velocities = np.where(mask_single_zero, 0.0, goal_velocities)

        return goal_velocities

    def _generate_random_trajectory(self):
        """Generate a full random trajectory of length self.trajectory_length."""
        traj = np.zeros((self.trajectory_length, 3), dtype=float)
        for t in range(self.trajectory_length // self.min_steps_each_command):

            single_command = self._sample_single_command()
            start_idx = t * self.min_steps_each_command
            end_idx = start_idx + self.min_steps_each_command
            traj[start_idx:end_idx, :] = single_command

        self.command_trajectory = traj
        self.command_index = 0

    def get_next_command(self):
        goal_velocities = self.command_trajectory[self.command_index]
        self.command_index = (self.command_index + 1) % self.trajectory_length

        self.env.internal_state["goal_velocities"] = goal_velocities
        actuator_joint_keep_nominal = np.where(np.all(goal_velocities == 0.0), np.ones(self.env.nr_actuator_joints, dtype=bool), self.default_actuator_joint_keep_nominal)
        self.env.internal_state["actuator_joint_keep_nominal"] = actuator_joint_keep_nominal
