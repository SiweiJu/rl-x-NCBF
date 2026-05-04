import numpy as np
import jax
import jax.numpy as jnp


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
        self.gait_frequency_range = jnp.array(command_config.get("gait_frequency_range", [1.0, 1.5]))
        self.zero_clip_threshold_percentage = 0.0
        self.max_velocity_per_m_factor = command_config.get("max_velocity_per_m_factor", 2.0)
        self.clip_max_velocity = command_config.get("clip_max_velocity", 1.0)

        self.default_actuator_joint_keep_nominal = np.zeros(env.nr_actuator_joints, dtype=bool)
        self.default_actuator_joint_keep_nominal[env.robot_config["actuator_joints_to_stay_near_nominal"]] = 1.0
        self.default_actuator_joint_keep_nominal = jnp.array(self.default_actuator_joint_keep_nominal)


    def init(self, internal_state):
        internal_state["actuator_joint_keep_nominal"] = self.default_actuator_joint_keep_nominal
        internal_state["goal_velocities"] = jnp.array([0.0, 0.0, 0.0])
        internal_state["goal_height"] = self.min_height
        internal_state["goal_gait_frequency"] = 0.0
        internal_state["booster_gait_process"] = 0.0


    def get_next_command(self, internal_state, should_sample_commands, subkey):
        still_key, goal_key = jax.random.split(subkey, 2)
        hold_still = jax.random.uniform(still_key) < self.still_proportion
        moving_scale = 1.0 - hold_still.astype(jnp.float32)

        goal_state = jax.random.uniform(
            goal_key,
            shape=(5,),
            minval=jnp.array([
                -self.max_x_vel * moving_scale,
                -self.max_y_vel * moving_scale,
                -self.max_yaw_vel * moving_scale,
                self.min_height,
                self.gait_frequency_range[0] * moving_scale,
            ]),
            maxval=jnp.array([
                self.max_x_vel * moving_scale,
                self.max_y_vel * moving_scale,
                self.max_yaw_vel * moving_scale,
                self.max_height,
                self.gait_frequency_range[1] * moving_scale,
            ]),
        )

        goal_velocities = goal_state[:3]
        internal_state["goal_velocities"] = jnp.where(should_sample_commands, goal_velocities, internal_state["goal_velocities"])
        internal_state["goal_height"] = jnp.where(should_sample_commands, goal_state[3], internal_state["goal_height"])
        internal_state["goal_gait_frequency"] = jnp.where(should_sample_commands, goal_state[4], internal_state["goal_gait_frequency"])

        actuator_joint_keep_nominal = jnp.where(
            jnp.all(goal_velocities == 0.0),
            jnp.ones(self.env.nr_actuator_joints, dtype=bool),
            self.default_actuator_joint_keep_nominal,
        )
        internal_state["actuator_joint_keep_nominal"] = jnp.where(
            should_sample_commands,
            actuator_joint_keep_nominal,
            internal_state["actuator_joint_keep_nominal"],
        )


    def get_observation(self, internal_state):
        gait_process = internal_state["booster_gait_process"]
        gait_frequency = internal_state["goal_gait_frequency"]
        cos = jnp.cos(2 * jnp.pi * gait_process) * (gait_frequency > 1.0e-8)
        sin = jnp.sin(2 * jnp.pi * gait_process) * (gait_frequency > 1.0e-8)
        return jnp.concatenate([
            internal_state["goal_velocities"],
            jnp.array([internal_state["goal_height"], cos, sin]),
        ])
