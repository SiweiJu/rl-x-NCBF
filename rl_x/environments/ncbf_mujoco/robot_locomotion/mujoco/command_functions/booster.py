import numpy as np


class BoosterCommands:
    observation_size = 6

    def __init__(self, env):
        self.env = env
        command_config = self.env.env_config["command"]
        self.max_x_vel = command_config.get("max_x_vel", 0.8)
        self.max_y_vel = command_config.get("max_y_vel", 0.5)
        self.max_yaw_vel = command_config.get("max_yaw_vel", 1.0)
        self.min_height = command_config.get("min_height", 0.67)
        self.max_height = command_config.get("max_height", 0.67)
        self.still_proportion = command_config.get("still_proportion", 0.2)
        self.gait_frequency_range = np.array(command_config.get("gait_frequency_range", [1.0, 1.5]))
        self.zero_clip_threshold_percentage = 0.0
        self.max_velocity_per_m_factor = command_config.get("max_velocity_per_m_factor", 2.0)
        self.clip_max_velocity = command_config.get("clip_max_velocity", 1.0)

        self.default_actuator_joint_keep_nominal = np.zeros(env.nr_actuator_joints, dtype=bool)
        self.default_actuator_joint_keep_nominal[env.robot_config["actuator_joints_to_stay_near_nominal"]] = 1.0


    def init(self):
        self.env.internal_state["actuator_joint_keep_nominal"] = self.default_actuator_joint_keep_nominal
        self.env.internal_state["goal_velocities"] = np.array([0.0, 0.0, 0.0])
        self.env.internal_state["goal_height"] = self.min_height
        self.env.internal_state["goal_gait_frequency"] = 0.0
        self.env.internal_state["booster_gait_process"] = 0.0


    def get_next_command(self):
        hold_still = self.env.np_rng.uniform() < self.still_proportion
        moving_scale = 1.0 - float(hold_still)
        goal_state = self.env.np_rng.uniform(
            low=np.array([
                -self.max_x_vel * moving_scale,
                -self.max_y_vel * moving_scale,
                -self.max_yaw_vel * moving_scale,
                self.min_height,
                self.gait_frequency_range[0] * moving_scale,
            ]),
            high=np.array([
                self.max_x_vel * moving_scale,
                self.max_y_vel * moving_scale,
                self.max_yaw_vel * moving_scale,
                self.max_height,
                self.gait_frequency_range[1] * moving_scale,
            ]),
        )

        goal_velocities = goal_state[:3]
        self.env.internal_state["goal_velocities"] = goal_velocities
        self.env.internal_state["goal_height"] = goal_state[3]
        self.env.internal_state["goal_gait_frequency"] = goal_state[4]
        self.env.internal_state["actuator_joint_keep_nominal"] = (
            np.ones(self.env.nr_actuator_joints, dtype=bool)
            if np.all(goal_velocities == 0.0)
            else self.default_actuator_joint_keep_nominal
        )


    def get_observation(self, internal_state):
        gait_process = internal_state["booster_gait_process"]
        gait_frequency = internal_state["goal_gait_frequency"]
        cos = np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
        sin = np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
        return np.concatenate([
            internal_state["goal_velocities"],
            np.array([internal_state["goal_height"], cos, sin]),
        ])
