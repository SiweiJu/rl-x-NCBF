import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


class BoosterReward:
    def __init__(self, env):
        self.env = env
        config = env.env_config["reward"]

        self.survival = config.get("survival", 0.25) * env.dt
        self.tracking_w_exp_linvel_x = config.get("tracking_w_exp_linvel_x", 4.0)
        self.tracking_w_sum_linvel_x = config.get("tracking_w_sum_linvel_x", 1.5) * env.dt
        self.tracking_w_exp_linvel_y = config.get("tracking_w_exp_linvel_y", 4.0)
        self.tracking_w_sum_linvel_y = config.get("tracking_w_sum_linvel_y", 1.5) * env.dt
        self.tracking_w_exp_angvel = config.get("tracking_w_exp_angvel", 4.0)
        self.tracking_w_sum_angvel = config.get("tracking_w_sum_angvel", 1.0) * env.dt
        self.nominal_joint_pos_exp = config.get("tracking_nominal_joint_pos_exp", 4.0)
        self.nominal_joint_pos_coeff = config.get("tracking_nominal_joint_pos_coeff", 0.0) * env.dt
        self.joint_deviation_l1_coeff = config.get("joint_deviation_l1_coeff", -0.3) * env.dt
        self.base_height_coeff = config.get("base_height_coeff", -10.0) * env.dt
        self.orientation_coeff = config.get("orientation_coeff", -5.0) * env.dt
        self.joint_torque_coeff = config.get("joint_torque_coeff", -2.0e-4) * env.dt
        self.energy_coeff = config.get("energy_coeff", -2.0e-3) * env.dt
        self.z_vel_coeff = config.get("z_vel_coeff", -0.0) * env.dt
        self.roll_pitch_vel_coeff = config.get("roll_pitch_vel_coeff", -0.2) * env.dt
        self.joint_vel_coeff = config.get("joint_vel_coeff", -9.0e-4) * env.dt
        self.joint_acc_coeff = config.get("joint_acc_coeff", -1.0e-7) * env.dt
        self.root_acc_coeff = config.get("root_acc_coeff", -1.0e-4) * env.dt
        self.action_rate_coeff = config.get("action_rate_coeff", -1.0) * env.dt
        self.joint_position_limit_scale = config.get("joint_position_limit_scale", 0.98)
        self.joint_position_limit_coeff = config.get("joint_position_limit_coeff", -1.0) * env.dt
        self.feet_slip_coeff = config.get("feet_slip_coeff", -0.1) * env.dt
        self.feet_yaw_diff_coeff = config.get("feet_yaw_diff_coeff", -1.0) * env.dt
        self.feet_yaw_mean_coeff = config.get("feet_yaw_mean_coeff", -1.0) * env.dt
        self.feet_roll_coeff = config.get("feet_roll_coeff", -4.0) * env.dt
        self.feet_distance_coeff = config.get("feet_distance_coeff", 0.0) * env.dt
        self.feet_distance_target = config.get("feet_distance_target", 0.2)
        self.feet_swing_coeff = config.get("feet_swing_coeff", 6.0) * env.dt
        self.feet_swing_period = config.get("feet_swing_period", 0.2)
        self.air_time_max = config.get("air_time_max", 0.3)
        self.air_time_coeff = config.get("air_time_coeff", 0.0) * env.dt
        self.no_fly_coeff = config.get("no_fly_coeff", 0.0) * env.dt
        self.impact_threshold = config.get("impact_threshold", 150.0)
        self.impact_coeff = config.get("impact_coeff", 0.0) * env.dt

        self.limited_joints = np.array(env.initial_mj_model.jnt_limited, dtype=bool)
        self.limited_joint_ids = jnp.array(np.where(self.limited_joints)[0])
        self.limited_joints_qpos_id = jnp.array(env.initial_mj_model.jnt_qposadr[self.limited_joints])
        self.limited_joint_ranges = jnp.array(env.initial_mj_model.jnt_range[self.limited_joints])
        self.nominal_joint_qpos = env.initial_qpos
        nominal_joint_names = config.get("tracking_nominal_joint_pos_names", None)
        if nominal_joint_names is None:
            self.nominal_joint_qpos_id = self.limited_joints_qpos_id
        else:
            self.nominal_joint_qpos_id = jnp.array([
                env.initial_mj_model.joint(name).qposadr[0]
                for name in nominal_joint_names
            ])

        foot_geom_indices = np.array(env.foot_geom_indices)
        self.left_feet_in_feet = jnp.array(np.where(np.isin(foot_geom_indices, np.array(env.left_foot_geom_indices)))[0])
        self.right_feet_in_feet = jnp.array(np.where(np.isin(foot_geom_indices, np.array(env.right_foot_geom_indices)))[0])
        self.left_foot_body_ids = jnp.array([env.initial_mj_model.geom(int(geom_id)).bodyid[0] for geom_id in np.array(env.left_foot_geom_indices)])
        self.right_foot_body_ids = jnp.array([env.initial_mj_model.geom(int(geom_id)).bodyid[0] for geom_id in np.array(env.right_foot_geom_indices)])


    def init(self, internal_state, mjx_model):
        internal_state["joint_position_limits"] = self.calculate_joint_position_limits(mjx_model)
        self.setup(internal_state)


    def calculate_joint_position_limits(self, mjx_model):
        joint_ranges = mjx_model.jnt_range[self.limited_joint_ids]
        scale_factor = 0.5 * (1.0 - self.joint_position_limit_scale)
        range_diff = joint_ranges[:, 1] - joint_ranges[:, 0]
        lower = joint_ranges[:, 0] + scale_factor * range_diff
        upper = joint_ranges[:, 1] - scale_factor * range_diff
        return jnp.stack([lower, upper], axis=1)


    def handle_model_change(self, internal_state, mjx_model, should_change):
        internal_state["joint_position_limits"] = jnp.where(
            should_change,
            self.calculate_joint_position_limits(mjx_model),
            internal_state["joint_position_limits"],
        )


    def setup(self, internal_state):
        internal_state["feet_time_on_ground"] = jnp.zeros(self.env.nr_feet)
        internal_state["feet_time_in_air"] = jnp.zeros(self.env.nr_feet)
        internal_state["previous_feet_floor_contacts"] = jnp.ones(self.env.nr_feet, dtype=bool)
        internal_state["previous_actuator_joint_velocities"] = jnp.zeros(self.env.nr_actuator_joints)
        internal_state["previous_imu_linear_velocity"] = jnp.zeros(self.env.imu_linear_velocity_sensor_dim)
        internal_state["sum_tracking_performance_percentage"] = 0.0
        internal_state["booster_gait_process"] = 0.0
        internal_state["booster_last_qvel"] = jnp.zeros(self.env.initial_mj_model.nv)
        internal_state["booster_last_action"] = jnp.zeros(self.env.nr_actuator_joints)
        internal_state["booster_time_since_last_touchdown"] = jnp.zeros(2, dtype=jnp.float32)


    def step(self, data, internal_state):
        feet_floor_contacts = self.env.terrain_function.check_feet_floor_contact(data)
        internal_state["feet_time_on_ground"] = jnp.where(feet_floor_contacts, internal_state["feet_time_on_ground"] + self.env.dt, 0.0)
        internal_state["feet_time_in_air"] = jnp.where(feet_floor_contacts, 0.0, internal_state["feet_time_in_air"] + self.env.dt)
        internal_state["previous_feet_floor_contacts"] = feet_floor_contacts
        internal_state["previous_actuator_joint_velocities"] = data.qvel[self.env.actuator_joint_mask_qvel]
        internal_state["previous_imu_linear_velocity"] = data.sensordata[
            self.env.imu_linear_velocity_sensor_adr:self.env.imu_linear_velocity_sensor_adr + self.env.imu_linear_velocity_sensor_dim
        ]


    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


    def reward_and_info(self, data, mjx_model, internal_state, action, info):
        global_pos_root = data.qpos[:3]
        global_quat_root = data.qpos[3:7]
        global_rot = Rotation.from_quat(jnp.roll(global_quat_root, -1))
        global_vel_root = data.qvel[:6]
        local_vel_root_lin = global_rot.inv().apply(global_vel_root[:3])
        local_vel_root_ang = global_rot.inv().apply(global_vel_root[3:])

        goal_vel = internal_state["goal_velocities"]
        goal_height = internal_state["goal_height"]
        gait_frequency = internal_state["goal_gait_frequency"]

        survival_reward = self.survival
        tracking_reward_linvel_x = jnp.exp(-jnp.square(local_vel_root_lin[0] - goal_vel[0]) * self.tracking_w_exp_linvel_x) * self.tracking_w_sum_linvel_x
        tracking_reward_linvel_y = jnp.exp(-jnp.square(local_vel_root_lin[1] - goal_vel[1]) * self.tracking_w_exp_linvel_y) * self.tracking_w_sum_linvel_y
        tracking_reward_angvel = jnp.exp(-jnp.square(local_vel_root_ang[2] - goal_vel[2]) * self.tracking_w_exp_angvel) * self.tracking_w_sum_angvel

        base_height_reward = jnp.square(global_pos_root[2] - goal_height) * self.base_height_coeff
        projected_gravity = global_rot.inv().apply(jnp.array([0.0, 0.0, -1.0]))
        orientation_reward = jnp.sum(jnp.square(projected_gravity[:2])) * self.orientation_coeff

        joint_vel = data.qvel[self.env.actuator_joint_mask_qvel]
        joint_torque = data.qfrc_actuator[self.env.actuator_joint_mask_qvel]
        torque_reward = jnp.sum(jnp.square(joint_torque)) * self.joint_torque_coeff
        energy_reward = jnp.sum(jnp.clip(joint_vel * joint_torque, a_min=0.0)) * self.energy_coeff
        z_vel_reward = jnp.square(local_vel_root_lin[2]) * self.z_vel_coeff
        roll_pitch_vel_reward = jnp.sum(jnp.square(local_vel_root_ang[:2])) * self.roll_pitch_vel_coeff
        joint_vel_reward = jnp.sum(jnp.square(joint_vel)) * self.joint_vel_coeff
        acceleration_reward = jnp.sum(jnp.square((joint_vel - internal_state["booster_last_qvel"][self.env.actuator_joint_mask_qvel]) / self.env.dt)) * self.joint_acc_coeff
        root_acceleration_reward = jnp.sum(jnp.square((global_vel_root - internal_state["booster_last_qvel"][:6]) / self.env.dt)) * self.root_acc_coeff
        action_rate_reward = jnp.sum(jnp.square(action - internal_state["booster_last_action"])) * self.action_rate_coeff

        joint_positions = data.qpos[self.limited_joints_qpos_id]
        joint_limits = internal_state["joint_position_limits"]
        joint_position_limit_reward = (
            jnp.sum((joint_positions < joint_limits[:, 0]) | (joint_positions > joint_limits[:, 1])) *
            self.joint_position_limit_coeff
        )

        feet_contacts = self.env.terrain_function.check_feet_floor_contact(data)
        left_foot_on_ground = jnp.any(feet_contacts[self.left_feet_in_feet])
        right_foot_on_ground = jnp.any(feet_contacts[self.right_feet_in_feet])
        feet_on_ground = jnp.array([left_foot_on_ground, right_foot_on_ground])

        left_foot_vel = data.sensordata[self.env.left_foot_velocity_sensor_adr:self.env.left_foot_velocity_sensor_adr + 3]
        right_foot_vel = data.sensordata[self.env.right_foot_velocity_sensor_adr:self.env.right_foot_velocity_sensor_adr + 3]
        feet_slip_reward = (
            jnp.sum(jnp.square(left_foot_vel[:3] * feet_on_ground[0])) +
            jnp.sum(jnp.square(right_foot_vel[:3] * feet_on_ground[1]))
        ) * self.feet_slip_coeff

        left_foot_euler = Rotation.from_matrix(data.site_xmat[self.env.left_foot_site_id].reshape(3, 3)).as_euler("xyz")
        right_foot_euler = Rotation.from_matrix(data.site_xmat[self.env.right_foot_site_id].reshape(3, 3)).as_euler("xyz")
        left_foot_yaw = self._wrap_to_pi(left_foot_euler[2])
        right_foot_yaw = self._wrap_to_pi(right_foot_euler[2])
        feet_yaw_diff_reward = jnp.square(self._wrap_to_pi(left_foot_yaw - right_foot_yaw)) * self.feet_yaw_diff_coeff
        feet_yaw_mean = (left_foot_yaw * 0.5 + right_foot_yaw * 0.5) + jnp.pi * (jnp.abs(left_foot_yaw - right_foot_yaw) > jnp.pi)
        base_yaw = global_rot.as_euler("xyz")[2]
        feet_yaw_mean_reward = jnp.square(self._wrap_to_pi(base_yaw - feet_yaw_mean)) * self.feet_yaw_mean_coeff
        feet_roll_reward = (jnp.square(self._wrap_to_pi(left_foot_euler[0])) + jnp.square(self._wrap_to_pi(right_foot_euler[0]))) * self.feet_roll_coeff

        left_foot_pos = data.site_xpos[self.env.left_foot_site_id]
        right_foot_pos = data.site_xpos[self.env.right_foot_site_id]
        feet_distance = (
            jnp.cos(base_yaw) * (left_foot_pos[1] - right_foot_pos[1]) -
            jnp.sin(base_yaw) * (left_foot_pos[0] - right_foot_pos[0])
        )
        feet_distance_reward = jnp.clip(self.feet_distance_target - feet_distance, 0.0, 0.1) * self.feet_distance_coeff

        gait_process = jnp.fmod(internal_state["booster_gait_process"] + self.env.dt * gait_frequency, 1.0)
        left_swing = (jnp.abs(gait_process - 0.25) < 0.5 * self.feet_swing_period) & (gait_frequency > 1.0e-8)
        right_swing = (jnp.abs(gait_process - 0.75) < 0.5 * self.feet_swing_period) & (gait_frequency > 1.0e-8)
        feet_swing_reward = (
            (left_swing & ~feet_on_ground[0]).astype(jnp.float32) +
            (right_swing & ~feet_on_ground[1]).astype(jnp.float32)
        ) * self.feet_swing_coeff

        joint_qpos_reward = jnp.exp(
            -self.nominal_joint_pos_exp *
            jnp.sum(jnp.square(data.qpos[self.nominal_joint_qpos_id] - self.nominal_joint_qpos[self.nominal_joint_qpos_id]))
        ) * self.nominal_joint_pos_coeff
        joint_deviation_l1_penalty = (
            jnp.sum(jnp.abs(data.qpos[self.nominal_joint_qpos_id] - self.nominal_joint_qpos[self.nominal_joint_qpos_id])) *
            self.joint_deviation_l1_coeff
        )

        tslt = internal_state["booster_time_since_last_touchdown"]
        air_time_reward = 0.0
        for i in range(2):
            tslt_i, air_time_reward = jax.lax.cond(
                feet_on_ground[i],
                lambda: (0.0, air_time_reward + (tslt[i] - self.air_time_max) * (tslt[i] > 1e-6)),
                lambda: (tslt[i] + self.env.dt, air_time_reward),
            )
            tslt = tslt.at[i].set(tslt_i)
        air_time_reward *= self.air_time_coeff
        no_fly_reward = (jnp.logical_and(tslt[0] > 0.0, tslt[1] > 0.0) * 1.0) * self.no_fly_coeff

        left_foot_force_norm = jnp.linalg.norm(data.cfrc_ext[self.left_foot_body_ids, :3], axis=1)
        right_foot_force_norm = jnp.linalg.norm(data.cfrc_ext[self.right_foot_body_ids, :3], axis=1)
        impact_reward = (
            jnp.mean((left_foot_force_norm > self.impact_threshold) * 1.0 + (right_foot_force_norm > self.impact_threshold) * 1.0) *
            self.impact_coeff
        )

        tracking_reward = tracking_reward_linvel_x + tracking_reward_linvel_y + tracking_reward_angvel + joint_qpos_reward + feet_swing_reward
        penalty_rewards = (
            base_height_reward + orientation_reward + torque_reward + energy_reward + z_vel_reward +
            roll_pitch_vel_reward + joint_vel_reward + acceleration_reward + root_acceleration_reward +
            action_rate_reward + joint_position_limit_reward + feet_slip_reward + feet_yaw_diff_reward +
            feet_yaw_mean_reward + feet_roll_reward + feet_distance_reward + air_time_reward +
            no_fly_reward + impact_reward + joint_deviation_l1_penalty
        )
        reward = jnp.nan_to_num(survival_reward + tracking_reward + penalty_rewards, nan=0.0, posinf=0.0, neginf=0.0)

        internal_state["booster_gait_process"] = gait_process
        internal_state["booster_last_qvel"] = data.qvel
        internal_state["booster_last_action"] = action
        internal_state["booster_time_since_last_touchdown"] = tslt

        info["reward/survival"] = survival_reward
        info["reward/track_linvel_x"] = tracking_reward_linvel_x
        info["reward/track_linvel_y"] = tracking_reward_linvel_y
        info["reward/track_angvel"] = tracking_reward_angvel
        info["reward/joint_qpos"] = joint_qpos_reward
        info["reward/feet_swing"] = feet_swing_reward
        info["reward/base_height"] = base_height_reward
        info["reward/orientation"] = orientation_reward
        info["reward/torque"] = torque_reward
        info["reward/energy"] = energy_reward
        info["reward/z_vel"] = z_vel_reward
        info["reward/roll_pitch_vel"] = roll_pitch_vel_reward
        info["reward/joint_vel"] = joint_vel_reward
        info["reward/joint_acc"] = acceleration_reward
        info["reward/root_acc"] = root_acceleration_reward
        info["reward/action_rate"] = action_rate_reward
        info["reward/joint_position_limit"] = joint_position_limit_reward
        info["reward/feet_slip"] = feet_slip_reward
        info["reward/feet_yaw_diff"] = feet_yaw_diff_reward
        info["reward/feet_yaw_mean"] = feet_yaw_mean_reward
        info["reward/feet_roll"] = feet_roll_reward
        info["reward/feet_distance"] = feet_distance_reward
        info["reward/air_time"] = air_time_reward
        info["reward/no_fly"] = no_fly_reward
        info["reward/impact"] = impact_reward
        info["reward/joint_deviation_l1"] = joint_deviation_l1_penalty
        info["reward/tracking_total"] = tracking_reward
        info["reward/penalty_total"] = penalty_rewards
        info["reward/total"] = reward
        info["env_info/xy_vel_diff_abs"] = jnp.nan_to_num(jnp.mean(jnp.abs(goal_vel[:2] - local_vel_root_lin[:2])), nan=0.0)

        return reward
