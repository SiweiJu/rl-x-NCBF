import numpy as np

from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.reward_functions.default import DefaultReward


class DefaultG1Reward(DefaultReward):
    def __init__(self, env):
        super().__init__(env)

        reward_config = env.env_config["reward"]
        self.critical_initial_coeff = reward_config.get("critical_initial_coeff", 0.5)
        self.critical_final_coeff = reward_config.get("critical_final_coeff", 1.0)
        self.critical_fixed_coeff = reward_config.get("critical_fixed_coeff", -1.0)
        self.style_initial_coeff = reward_config.get("style_initial_coeff", 0.0)
        self.style_final_coeff = reward_config.get("style_final_coeff", 1.0)
        self.style_fixed_coeff = reward_config.get("style_fixed_coeff", -1.0)
        self.gait_initial_coeff = reward_config.get("gait_initial_coeff", 1.0)
        self.gait_final_coeff = reward_config.get("gait_final_coeff", 1.0)
        self.gait_fixed_coeff = reward_config.get("gait_fixed_coeff", 1.0)
        self.moving_command_threshold = reward_config.get("moving_command_threshold", 0.05)
        self.contact_count_coeff = reward_config.get("contact_count_coeff", 2.0) * env.dt
        self.foot_stance_time_coeff = reward_config.get("foot_stance_time_coeff", 1.0) * env.dt
        self.foot_stance_time_per_robot_size_m = reward_config.get("foot_stance_time_per_robot_size_m", 0.6)
        self.foot_clearance_coeff = reward_config.get("foot_clearance_coeff", 1.0) * env.dt
        self.foot_clearance_per_robot_size_m = reward_config.get("foot_clearance_per_robot_size_m", 0.08)
        self.foot_lift_bonus_coeff = reward_config.get("foot_lift_bonus_coeff", 1.0) * env.dt
        self.foot_lift_bonus_per_robot_size_m = reward_config.get("foot_lift_bonus_per_robot_size_m", 0.10)


    def _scheduled_coeff(self, fixed_coeff, initial_coeff, final_coeff, curriculum_progress):
        if fixed_coeff >= 0.0:
            return fixed_coeff
        return initial_coeff + (final_coeff - initial_coeff) * curriculum_progress


    def reward_and_info(self, action):
        curriculum_progress = self.env.internal_state["env_curriculum_coeff"]
        critical_coeff = self._scheduled_coeff(self.critical_fixed_coeff, self.critical_initial_coeff, self.critical_final_coeff, curriculum_progress)
        style_coeff = self._scheduled_coeff(self.style_fixed_coeff, self.style_initial_coeff, self.style_final_coeff, curriculum_progress)
        gait_coeff = self._scheduled_coeff(self.gait_fixed_coeff, self.gait_initial_coeff, self.gait_final_coeff, curriculum_progress)
        
        # Tracking velocity command reward
        current_imu_linear_velocity = self.env.internal_state["data"].sensordata[self.env.imu_linear_velocity_sensor_adr:self.env.imu_linear_velocity_sensor_adr + self.env.imu_linear_velocity_sensor_dim]
        desired_imu_linear_velocity_xy = self.env.internal_state["goal_velocities"][:2]
        xy_difference = desired_imu_linear_velocity_xy - current_imu_linear_velocity[:2]
        xy_velocity_difference_norm = np.sum(np.square(xy_difference))
        tracking_xy_velocity_command_reward = self.tracking_xy_velocity_command_coeff * np.exp(-xy_velocity_difference_norm / self.tracking_xy_temperature)

        # Tracking angular velocity command reward
        current_imu_angular_velocity = self.env.internal_state["data"].sensordata[self.env.imu_angular_velocity_sensor_adr:self.env.imu_angular_velocity_sensor_adr + self.env.imu_angular_velocity_sensor_dim]
        desired_imu_yaw_velocity = self.env.internal_state["goal_velocities"][2]
        yaw_velocity_difference_norm = np.square(current_imu_angular_velocity[2] - desired_imu_yaw_velocity)
        tracking_yaw_velocity_command_reward = self.tracking_yaw_velocity_command_coeff * np.exp(-yaw_velocity_difference_norm / self.tracking_yaw_temperature)

        # Alive reward
        alive_clipped_reward = critical_coeff * self.alive_clipped_coeff * 1.0
        alive_unclipped_reward = critical_coeff * self.alive_unclipped_coeff * 1.0

        # Critical penalties keep the robot safe and dynamically plausible.
        z_velocity_squared = current_imu_linear_velocity[2] ** 2
        z_velocity_reward = critical_coeff * self.z_velocity_coeff * -z_velocity_squared

        imu_acceleration_norm = np.mean(np.square((current_imu_linear_velocity - self.env.internal_state["previous_imu_linear_velocity"]) / self.env.dt))
        imu_acceleration_reward = critical_coeff * self.imu_acceleration_coeff * -imu_acceleration_norm

        current_imu_angular_velocity = self.env.internal_state["data"].sensordata[self.env.imu_angular_velocity_sensor_adr:self.env.imu_angular_velocity_sensor_adr + self.env.imu_angular_velocity_sensor_dim]
        angular_velocity_norm = np.sum(np.square(current_imu_angular_velocity[:2]))
        angular_velocity_reward = critical_coeff * self.roll_pitch_vel_coeff * -angular_velocity_norm

        roll_pitch_position_norm = np.sum(np.square(self.env.internal_state["imu_orientation_euler"][:2]))
        angular_position_reward = critical_coeff * self.roll_pitch_pos_coeff * -roll_pitch_position_norm

        joint_outside_limits = np.maximum(self.env.internal_state["data"].qpos[self.env.actuator_joint_mask_qpos] - self.env.internal_state["joint_position_limits"][:, 1], 0.0) + \
                               np.maximum(self.env.internal_state["joint_position_limits"][:, 0] - self.env.internal_state["data"].qpos[self.env.actuator_joint_mask_qpos], 0.0)
        joint_position_limit_reward = critical_coeff * self.joint_position_limit_coeff * -np.mean(joint_outside_limits)

        actuator_joint_velocity_limit = self.env.internal_state["actuator_joint_max_velocities"] * self.soft_actuator_joint_velocity_limit
        joint_velocity_outside_limits = np.maximum(np.abs(self.env.internal_state["data"].qvel[self.env.actuator_joint_mask_qvel]) - actuator_joint_velocity_limit, 0.0)
        joint_velocity_limit_reward = critical_coeff * self.actuator_joint_velocity_limit_coeff * -np.mean(joint_velocity_outside_limits)

        all_contact_relevant_geom_xpos = self.env.internal_state["data"].geom_xpos[self.env.reward_collision_sphere_geom_ids]
        all_contact_relevant_geom_sizes = self.env.internal_state["mj_model"].geom_size[self.env.reward_collision_sphere_geom_ids, 0]
        distance_between_geoms = np.linalg.norm(all_contact_relevant_geom_xpos[:, None] - all_contact_relevant_geom_xpos[None], axis=-1)
        contact_between_geoms = distance_between_geoms <= (all_contact_relevant_geom_sizes[:, None] + all_contact_relevant_geom_sizes[None])
        nr_collisions = (np.sum(contact_between_geoms) - len(self.env.reward_collision_sphere_geom_ids)) // 2
        nr_collisions = np.maximum(nr_collisions - self.env.internal_state["nr_collisions_in_nominal"], 0)
        collision_reward = critical_coeff * self.collision_coeff * -nr_collisions

        height_difference_squared = (self.env.internal_state["robot_imu_height_over_ground"] - self.env.internal_state["robot_nominal_imu_height_over_ground"]) ** 2
        base_height_reward = critical_coeff * self.base_height_coeff * -height_difference_squared

        feet_floor_contacts = self.env.terrain_function.check_feet_floor_contact()
        all_feet_off_ground_reward = critical_coeff * self.all_feet_off_ground_coeff * -float(np.all(~feet_floor_contacts))

        feet_global_linear_velocity_x = self.env.internal_state["data"].sensordata[self.env.feet_global_linear_velocity_sensor_adrs_start]
        feet_global_linear_velocity_y = self.env.internal_state["data"].sensordata[self.env.feet_global_linear_velocity_sensor_adrs_start + 1]
        feet_global_linear_velocity_xy_norm = np.square(feet_global_linear_velocity_x) + np.square(feet_global_linear_velocity_y)
        contact_filtered_feet_slip = np.mean(feet_floor_contacts * feet_global_linear_velocity_xy_norm)
        foot_slip_reward = critical_coeff * self.foot_slip_coeff * -contact_filtered_feet_slip

        feet_global_linear_velocity_z = self.env.internal_state["data"].sensordata[self.env.feet_global_linear_velocity_sensor_adrs_start + 2]
        squared_negative_z_velocity = np.mean(np.square(np.minimum(feet_global_linear_velocity_z, 0.0)))
        foot_z_velocity_reward = critical_coeff * self.foot_z_velocity_coeff * -squared_negative_z_velocity

        missing_lower_feet_contacts = self.env.terrain_function.check_flat_feet_floor_missing_contacts()
        contact_filtered_missing_lower_feet_contacts = np.mean(feet_floor_contacts * missing_lower_feet_contacts)
        foot_flat_contact_reward = critical_coeff * self.foot_flat_contact_coeff * -contact_filtered_missing_lower_feet_contacts

        # Style and efficiency penalties can ramp in later.
        actuator_joint_nominal_diff_norm = np.mean(np.square((self.env.internal_state["data"].qpos[self.env.actuator_joint_mask_qpos] * self.env.internal_state["actuator_joint_keep_nominal"]) - (self.env.internal_state["actuator_joint_nominal_positions"] * self.env.internal_state["actuator_joint_keep_nominal"])))
        actuator_joint_nominal_diff_reward = style_coeff * self.actuator_joint_nominal_diff_coeff * -actuator_joint_nominal_diff_norm

        velocity_norm = np.mean(np.square(self.env.internal_state["data"].qvel[self.env.actuator_joint_mask_qvel]))
        joint_velocity_reward = style_coeff * self.joint_velocity_coeff * -velocity_norm

        acceleration_norm = np.mean(np.square((self.env.internal_state["data"].qvel[self.env.actuator_joint_mask_qvel] - self.env.internal_state["previous_actuator_joint_velocities"]) / self.env.dt))
        acceleration_reward = style_coeff * self.joint_acceleration_coeff * -acceleration_norm

        torque_norm = np.mean(np.square(self.env.internal_state["data"].qfrc_actuator[self.env.actuator_joint_mask_qvel]))
        torque_reward = style_coeff * self.joint_torque_coeff * -torque_norm

        power_draw = np.mean(np.maximum(self.env.internal_state["data"].qfrc_actuator[self.env.actuator_joint_mask_qvel] * self.env.internal_state["data"].qvel[self.env.actuator_joint_mask_qvel], 0.0))
        power_draw_penalty_reward = style_coeff * self.power_draw_penalty_coeff * -power_draw

        action_rate_norm = np.mean(np.square(action - self.env.internal_state["last_action"]))
        action_rate_reward = style_coeff * self.action_rate_coeff * -action_rate_norm
        
        action_smoothness_norm = np.mean(np.square(action - 2 * self.env.internal_state["last_action"] + self.env.internal_state["second_last_action"]))
        action_smoothness_reward = style_coeff * self.action_smoothness_coeff * -action_smoothness_norm

        command_norm = np.linalg.norm(self.env.internal_state["goal_velocities"])
        is_moving_command = command_norm > self.moving_command_threshold

        feet_first_contact = feet_floor_contacts & (~self.env.internal_state["previous_feet_floor_contacts"])
        target_foot_air_time = self.foot_air_time_per_robot_size_m * self.env.internal_state["robot_dimensions_mean"]
        target_foot_air_time = is_moving_command * target_foot_air_time
        air_time_reward = np.mean(feet_first_contact * np.minimum(self.env.internal_state["feet_time_in_air"] - target_foot_air_time, 0.0))
        foot_air_time_reward = gait_coeff * self.foot_air_time_coeff * air_time_reward

        symmetry_air_violations = np.mean(np.where((~feet_floor_contacts[self.feet_symmetry_pairs[:, 0]]) & (~feet_floor_contacts[self.feet_symmetry_pairs[:, 1]]), 1, 0))
        symmetry_air_reward = gait_coeff * self.symmetry_air_coeff * -symmetry_air_violations

        nr_feet = float(self.env.nr_feet)
        nr_feet_in_contact = np.sum(feet_floor_contacts.astype(np.float32))
        moving_contact_target = float(max(self.env.nr_feet - 1, 1))
        target_nr_contacts = moving_contact_target if is_moving_command else nr_feet
        contact_count_error = (nr_feet_in_contact - target_nr_contacts) / nr_feet
        contact_count_reward = gait_coeff * self.contact_count_coeff * -np.square(contact_count_error)

        target_stance_time = self.foot_stance_time_per_robot_size_m * self.env.internal_state["robot_dimensions_mean"]
        long_stance_time = np.maximum(self.env.internal_state["feet_time_on_ground"] - target_stance_time, 0.0)
        foot_stance_time_reward = gait_coeff * self.foot_stance_time_coeff * -float(is_moving_command) * np.mean(long_stance_time)

        feet_xpos = self.env.internal_state["data"].geom_xpos[self.env.foot_geom_indices]
        feet_height_over_ground = feet_xpos[:, 2] - self.env.terrain_function.ground_height_at(feet_xpos[:, 0], feet_xpos[:, 1])
        target_foot_clearance = self.foot_clearance_per_robot_size_m * self.env.internal_state["robot_dimensions_mean"]
        foot_clearance_error = np.maximum(target_foot_clearance - feet_height_over_ground, 0.0)
        swing_feet = (~feet_floor_contacts).astype(np.float32)
        foot_clearance_reward = gait_coeff * self.foot_clearance_coeff * -float(is_moving_command) * np.mean(swing_feet * foot_clearance_error)

        target_foot_lift_bonus = self.foot_lift_bonus_per_robot_size_m * self.env.internal_state["robot_dimensions_mean"]
        foot_lift_fraction = np.clip(feet_height_over_ground / target_foot_lift_bonus, 0.0, 1.0)
        foot_lift_bonus_reward = gait_coeff * self.foot_lift_bonus_coeff * float(is_moving_command) * np.mean(swing_feet * foot_lift_fraction)

        tracking_reward = tracking_xy_velocity_command_reward + tracking_yaw_velocity_command_reward
        critical_penalty = z_velocity_reward + imu_acceleration_reward + angular_velocity_reward + angular_position_reward + \
                           joint_position_limit_reward + joint_velocity_limit_reward + collision_reward + base_height_reward + \
                           all_feet_off_ground_reward + foot_slip_reward + foot_z_velocity_reward + foot_flat_contact_reward
        style_penalty = actuator_joint_nominal_diff_reward + joint_velocity_reward + acceleration_reward + torque_reward + \
                        power_draw_penalty_reward + action_rate_reward + action_smoothness_reward
        gait_reward = foot_lift_bonus_reward
        gait_penalty = foot_air_time_reward + symmetry_air_reward + contact_count_reward + foot_stance_time_reward + foot_clearance_reward
        reward = tracking_reward + critical_penalty + style_penalty + gait_penalty + gait_reward + alive_clipped_reward
        reward = np.maximum(reward, 0.0) + alive_unclipped_reward
        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        self.env.internal_state["info"][f"reward/track_xy_vel_cmd"] = tracking_xy_velocity_command_reward
        self.env.internal_state["info"][f"reward/track_yaw_vel_cmd"] = tracking_yaw_velocity_command_reward
        self.env.internal_state["info"][f"reward/alive_clipped"] = alive_clipped_reward
        self.env.internal_state["info"][f"reward/alive_unclipped"] = alive_unclipped_reward
        self.env.internal_state["info"][f"reward/z_velocity"] = z_velocity_reward
        self.env.internal_state["info"][f"reward/imu_acceleration"] = imu_acceleration_reward
        self.env.internal_state["info"][f"reward/angular_velocity"] = angular_velocity_reward
        self.env.internal_state["info"][f"reward/angular_position"] = angular_position_reward
        self.env.internal_state["info"][f"reward/actuator_joint_nominal_diff"] = actuator_joint_nominal_diff_reward
        self.env.internal_state["info"][f"reward/joint_position_limit"] = joint_position_limit_reward
        self.env.internal_state["info"][f"reward/joint_velocity_limit"] = joint_velocity_limit_reward
        self.env.internal_state["info"][f"reward/joint_velocity"] = joint_velocity_reward
        self.env.internal_state["info"][f"reward/joint_acceleration"] = acceleration_reward
        self.env.internal_state["info"][f"reward/joint_torque"] = torque_reward
        self.env.internal_state["info"][f"reward/power_draw_penalty"] = power_draw_penalty_reward
        self.env.internal_state["info"][f"reward/action_rate"] = action_rate_reward
        self.env.internal_state["info"][f"reward/action_smoothness"] = action_smoothness_reward
        self.env.internal_state["info"][f"reward/collision"] = collision_reward
        self.env.internal_state["info"][f"reward/base_height"] = base_height_reward
        self.env.internal_state["info"][f"reward/foot_air_time"] = foot_air_time_reward
        self.env.internal_state["info"][f"reward/contact_count"] = contact_count_reward
        self.env.internal_state["info"][f"reward/foot_stance_time"] = foot_stance_time_reward
        self.env.internal_state["info"][f"reward/foot_clearance"] = foot_clearance_reward
        self.env.internal_state["info"][f"reward/foot_lift_bonus"] = foot_lift_bonus_reward
        self.env.internal_state["info"][f"reward/all_feet_off_ground"] = all_feet_off_ground_reward
        self.env.internal_state["info"][f"reward/symmetry_air"] = symmetry_air_reward
        self.env.internal_state["info"][f"reward/foot_slip"] = foot_slip_reward
        self.env.internal_state["info"][f"reward/foot_z_velocity"] = foot_z_velocity_reward
        self.env.internal_state["info"][f"reward/foot_flat_contact"] = foot_flat_contact_reward
        self.env.internal_state["info"][f"reward/critical_coeff"] = critical_coeff
        self.env.internal_state["info"][f"reward/style_coeff"] = style_coeff
        self.env.internal_state["info"][f"reward/gait_coeff"] = gait_coeff
        self.env.internal_state["info"][f"reward/critical_penalty_total"] = critical_penalty
        self.env.internal_state["info"][f"reward/style_penalty_total"] = style_penalty
        self.env.internal_state["info"][f"reward/gait_penalty_total"] = gait_penalty
        self.env.internal_state["info"][f"reward/gait_reward_total"] = gait_reward
        self.env.internal_state["info"][f"reward/total"] = reward
        self.env.internal_state["info"][f"env_info/xy_vel_diff_abs"] = np.nan_to_num(np.mean(np.minimum(np.abs(xy_difference), 2 * self.env.internal_state["max_command_velocity"])), nan=2 * self.env.internal_state["max_command_velocity"], posinf=2 * self.env.internal_state["max_command_velocity"], neginf=2 * self.env.internal_state["max_command_velocity"])

        return reward
