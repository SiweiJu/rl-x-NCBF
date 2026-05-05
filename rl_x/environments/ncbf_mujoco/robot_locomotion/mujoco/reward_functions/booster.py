import numpy as np
from scipy.spatial.transform import Rotation


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
        self.limited_joints_qpos_id = env.initial_mj_model.jnt_qposadr[self.limited_joints]
        self.limited_joint_ranges = env.initial_mj_model.jnt_range[self.limited_joints]
        self.nominal_joint_qpos = env.initial_qpos
        nominal_joint_names = config.get("tracking_nominal_joint_pos_names", None)
        if nominal_joint_names is None:
            self.nominal_joint_qpos_id = self.limited_joints_qpos_id
        else:
            self.nominal_joint_qpos_id = np.array([
                env.initial_mj_model.joint(name).qposadr[0]
                for name in nominal_joint_names
            ])

        foot_geom_indices = np.array(env.foot_geom_indices)
        self.left_feet_in_feet = np.where(np.isin(foot_geom_indices, np.array(env.left_foot_geom_indices)))[0]
        self.right_feet_in_feet = np.where(np.isin(foot_geom_indices, np.array(env.right_foot_geom_indices)))[0]
        self.left_foot_body_ids = np.array([env.initial_mj_model.geom(int(geom_id)).bodyid[0] for geom_id in np.array(env.left_foot_geom_indices)])
        self.right_foot_body_ids = np.array([env.initial_mj_model.geom(int(geom_id)).bodyid[0] for geom_id in np.array(env.right_foot_geom_indices)])


    def init(self):
        self.env.internal_state["joint_position_limits"] = self.calculate_joint_position_limits()
        self.setup()


    def calculate_joint_position_limits(self):
        joint_ranges = self.env.internal_state["mj_model"].jnt_range[self.limited_joints]
        scale_factor = 0.5 * (1.0 - self.joint_position_limit_scale)
        range_diff = joint_ranges[:, 1] - joint_ranges[:, 0]
        lower = joint_ranges[:, 0] + scale_factor * range_diff
        upper = joint_ranges[:, 1] - scale_factor * range_diff
        return np.stack([lower, upper], axis=1)


    def handle_model_change(self):
        self.env.internal_state["joint_position_limits"] = self.calculate_joint_position_limits()


    def setup(self):
        self.env.internal_state["feet_time_on_ground"] = np.zeros(self.env.nr_feet)
        self.env.internal_state["feet_time_in_air"] = np.zeros(self.env.nr_feet)
        self.env.internal_state["previous_feet_floor_contacts"] = np.ones(self.env.nr_feet, dtype=bool)
        self.env.internal_state["previous_actuator_joint_velocities"] = np.zeros(self.env.nr_actuator_joints)
        self.env.internal_state["previous_imu_linear_velocity"] = np.zeros(self.env.imu_linear_velocity_sensor_dim)
        self.env.internal_state["sum_tracking_performance_percentage"] = 0.0
        self.env.internal_state["booster_gait_process"] = 0.0
        self.env.internal_state["booster_last_qvel"] = np.zeros(self.env.initial_mj_model.nv)
        self.env.internal_state["booster_last_action"] = np.zeros(self.env.nr_actuator_joints)
        self.env.internal_state["booster_time_since_last_touchdown"] = np.zeros(2, dtype=np.float32)


    def step(self):
        feet_floor_contacts = self.env.terrain_function.check_feet_floor_contact()
        self.env.internal_state["feet_time_on_ground"] = np.where(
            feet_floor_contacts,
            self.env.internal_state["feet_time_on_ground"] + self.env.dt,
            0.0,
        )
        self.env.internal_state["feet_time_in_air"] = np.where(
            feet_floor_contacts,
            0.0,
            self.env.internal_state["feet_time_in_air"] + self.env.dt,
        )
        self.env.internal_state["previous_feet_floor_contacts"] = feet_floor_contacts
        self.env.internal_state["previous_actuator_joint_velocities"] = self.env.internal_state["data"].qvel[self.env.actuator_joint_mask_qvel]
        self.env.internal_state["previous_imu_linear_velocity"] = self.env.internal_state["data"].sensordata[
            self.env.imu_linear_velocity_sensor_adr:self.env.imu_linear_velocity_sensor_adr + self.env.imu_linear_velocity_sensor_dim
        ]


    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


    @staticmethod
    def _rotation_from_site_xmat(site_xmat):
        xmat = site_xmat.reshape(3, 3)
        if np.linalg.det(xmat) <= 0.0:
            return Rotation.identity()
        return Rotation.from_matrix(xmat)


    def reward_and_info(self, action):
        data = self.env.internal_state["data"]
        state = self.env.internal_state

        global_pos_root = data.qpos[:3]
        global_quat_root = data.qpos[3:7]
        global_rot = Rotation.from_quat(global_quat_root[[1, 2, 3, 0]])
        global_vel_root = data.qvel[:6]
        local_vel_root_lin = global_rot.inv().apply(global_vel_root[:3])
        local_vel_root_ang = global_rot.inv().apply(global_vel_root[3:])

        goal_vel = state["goal_velocities"]
        goal_height = state["goal_height"]
        gait_frequency = state["goal_gait_frequency"]

        survival_reward = self.survival
        tracking_reward_linvel_x = np.exp(-np.square(local_vel_root_lin[0] - goal_vel[0]) * self.tracking_w_exp_linvel_x) * self.tracking_w_sum_linvel_x
        tracking_reward_linvel_y = np.exp(-np.square(local_vel_root_lin[1] - goal_vel[1]) * self.tracking_w_exp_linvel_y) * self.tracking_w_sum_linvel_y
        tracking_reward_angvel = np.exp(-np.square(local_vel_root_ang[2] - goal_vel[2]) * self.tracking_w_exp_angvel) * self.tracking_w_sum_angvel

        base_height_reward = np.square(global_pos_root[2] - goal_height) * self.base_height_coeff
        projected_gravity = global_rot.inv().apply(np.array([0.0, 0.0, -1.0]))
        orientation_reward = np.sum(np.square(projected_gravity[:2])) * self.orientation_coeff

        joint_vel = data.qvel[self.env.actuator_joint_mask_qvel]
        joint_torque = data.qfrc_actuator[self.env.actuator_joint_mask_qvel]
        torque_reward = np.sum(np.square(joint_torque)) * self.joint_torque_coeff
        energy_reward = np.sum(np.clip(joint_vel * joint_torque, a_min=0.0, a_max=None)) * self.energy_coeff
        z_vel_reward = np.square(local_vel_root_lin[2]) * self.z_vel_coeff
        roll_pitch_vel_reward = np.sum(np.square(local_vel_root_ang[:2])) * self.roll_pitch_vel_coeff
        joint_vel_reward = np.sum(np.square(joint_vel)) * self.joint_vel_coeff
        acceleration_reward = np.sum(np.square((joint_vel - state["booster_last_qvel"][self.env.actuator_joint_mask_qvel]) / self.env.dt)) * self.joint_acc_coeff
        root_acceleration_reward = np.sum(np.square((global_vel_root - state["booster_last_qvel"][:6]) / self.env.dt)) * self.root_acc_coeff
        action_rate_reward = np.sum(np.square(action - state["booster_last_action"])) * self.action_rate_coeff

        joint_positions = data.qpos[self.limited_joints_qpos_id]
        joint_limits = state["joint_position_limits"]
        joint_position_limit_reward = (
            np.sum((joint_positions < joint_limits[:, 0]) | (joint_positions > joint_limits[:, 1])) *
            self.joint_position_limit_coeff
        )

        feet_contacts = self.env.terrain_function.check_feet_floor_contact()
        left_foot_on_ground = np.any(feet_contacts[self.left_feet_in_feet])
        right_foot_on_ground = np.any(feet_contacts[self.right_feet_in_feet])
        feet_on_ground = np.array([left_foot_on_ground, right_foot_on_ground])

        left_foot_vel = data.sensordata[self.env.left_foot_velocity_sensor_adr:self.env.left_foot_velocity_sensor_adr + 3]
        right_foot_vel = data.sensordata[self.env.right_foot_velocity_sensor_adr:self.env.right_foot_velocity_sensor_adr + 3]
        feet_slip_reward = (
            np.sum(np.square(left_foot_vel[:3] * feet_on_ground[0])) +
            np.sum(np.square(right_foot_vel[:3] * feet_on_ground[1]))
        ) * self.feet_slip_coeff

        left_foot_euler = self._rotation_from_site_xmat(data.site_xmat[self.env.left_foot_site_id]).as_euler("xyz")
        right_foot_euler = self._rotation_from_site_xmat(data.site_xmat[self.env.right_foot_site_id]).as_euler("xyz")
        left_foot_yaw = self._wrap_to_pi(left_foot_euler[2])
        right_foot_yaw = self._wrap_to_pi(right_foot_euler[2])
        feet_yaw_diff_reward = np.square(self._wrap_to_pi(left_foot_yaw - right_foot_yaw)) * self.feet_yaw_diff_coeff
        feet_yaw_mean = (left_foot_yaw * 0.5 + right_foot_yaw * 0.5) + np.pi * (np.abs(left_foot_yaw - right_foot_yaw) > np.pi)
        base_yaw = global_rot.as_euler("xyz")[2]
        feet_yaw_mean_reward = np.square(self._wrap_to_pi(base_yaw - feet_yaw_mean)) * self.feet_yaw_mean_coeff
        feet_roll_reward = (np.square(self._wrap_to_pi(left_foot_euler[0])) + np.square(self._wrap_to_pi(right_foot_euler[0]))) * self.feet_roll_coeff

        left_foot_pos = data.site_xpos[self.env.left_foot_site_id]
        right_foot_pos = data.site_xpos[self.env.right_foot_site_id]
        feet_distance = (
            np.cos(base_yaw) * (left_foot_pos[1] - right_foot_pos[1]) -
            np.sin(base_yaw) * (left_foot_pos[0] - right_foot_pos[0])
        )
        feet_distance_reward = np.clip(self.feet_distance_target - feet_distance, 0.0, 0.1) * self.feet_distance_coeff

        gait_process = np.fmod(state["booster_gait_process"] + self.env.dt * gait_frequency, 1.0)
        left_swing = (np.abs(gait_process - 0.25) < 0.5 * self.feet_swing_period) and (gait_frequency > 1.0e-8)
        right_swing = (np.abs(gait_process - 0.75) < 0.5 * self.feet_swing_period) and (gait_frequency > 1.0e-8)
        feet_swing_reward = (
            np.float32(left_swing and not feet_on_ground[0]) +
            np.float32(right_swing and not feet_on_ground[1])
        ) * self.feet_swing_coeff

        joint_qpos_reward = np.exp(
            -self.nominal_joint_pos_exp *
            np.sum(np.square(data.qpos[self.nominal_joint_qpos_id] - self.nominal_joint_qpos[self.nominal_joint_qpos_id]))
        ) * self.nominal_joint_pos_coeff
        joint_deviation_l1_penalty = (
            np.sum(np.abs(data.qpos[self.nominal_joint_qpos_id] - self.nominal_joint_qpos[self.nominal_joint_qpos_id])) *
            self.joint_deviation_l1_coeff
        )

        tslt = state["booster_time_since_last_touchdown"].copy()
        air_time_reward = 0.0
        for i in range(2):
            if feet_on_ground[i]:
                if tslt[i] > 1e-6:
                    air_time_reward += tslt[i] - self.air_time_max
                tslt[i] = 0.0
            else:
                tslt[i] += self.env.dt
        air_time_reward *= self.air_time_coeff
        no_fly_reward = (np.logical_and(tslt[0] > 0.0, tslt[1] > 0.0) * 1.0) * self.no_fly_coeff

        left_foot_force_norm = np.linalg.norm(data.cfrc_ext[self.left_foot_body_ids, :3], axis=1)
        right_foot_force_norm = np.linalg.norm(data.cfrc_ext[self.right_foot_body_ids, :3], axis=1)
        impact_reward = (
            np.mean((left_foot_force_norm > self.impact_threshold) * 1.0 + (right_foot_force_norm > self.impact_threshold) * 1.0) *
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
        reward = np.nan_to_num(survival_reward + tracking_reward + penalty_rewards, nan=0.0, posinf=0.0, neginf=0.0)

        state["booster_gait_process"] = gait_process
        state["booster_last_qvel"] = data.qvel.copy()
        state["booster_last_action"] = action.copy()
        state["booster_time_since_last_touchdown"] = tslt

        state["info"]["reward/survival"] = survival_reward
        state["info"]["reward/track_linvel_x"] = tracking_reward_linvel_x
        state["info"]["reward/track_linvel_y"] = tracking_reward_linvel_y
        state["info"]["reward/track_angvel"] = tracking_reward_angvel
        state["info"]["reward/joint_qpos"] = joint_qpos_reward
        state["info"]["reward/feet_swing"] = feet_swing_reward
        state["info"]["reward/base_height"] = base_height_reward
        state["info"]["reward/orientation"] = orientation_reward
        state["info"]["reward/joint_torque"] = torque_reward
        state["info"]["reward/power_draw_penalty"] = energy_reward
        state["info"]["reward/z_velocity"] = z_vel_reward
        state["info"]["reward/roll_pitch_vel"] = roll_pitch_vel_reward
        state["info"]["reward/joint_velocity"] = joint_vel_reward
        state["info"]["reward/joint_acceleration"] = acceleration_reward
        state["info"]["reward/root_acceleration"] = root_acceleration_reward
        state["info"]["reward/action_rate"] = action_rate_reward
        state["info"]["reward/joint_position_limit"] = joint_position_limit_reward
        state["info"]["reward/feet_slip"] = feet_slip_reward
        state["info"]["reward/feet_yaw_diff"] = feet_yaw_diff_reward
        state["info"]["reward/feet_yaw_mean"] = feet_yaw_mean_reward
        state["info"]["reward/feet_roll"] = feet_roll_reward
        state["info"]["reward/feet_distance"] = feet_distance_reward
        state["info"]["reward/air_time"] = air_time_reward
        state["info"]["reward/no_fly"] = no_fly_reward
        state["info"]["reward/impact"] = impact_reward
        state["info"]["reward/joint_deviation_l1"] = joint_deviation_l1_penalty
        state["info"]["reward/tracking_total"] = tracking_reward
        state["info"]["reward/penalty_total"] = penalty_rewards
        state["info"]["reward/total"] = reward
        state["info"]["env_info/xy_vel_diff_abs"] = np.nan_to_num(np.mean(np.abs(goal_vel[:2] - local_vel_root_lin[:2])), nan=0.0)

        return reward
