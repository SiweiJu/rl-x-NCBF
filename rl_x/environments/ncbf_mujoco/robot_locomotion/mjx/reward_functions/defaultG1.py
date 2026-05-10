import numpy as np
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.reward_functions.default import DefaultReward


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
        self.standing_command_threshold = reward_config.get("standing_command_threshold", self.moving_command_threshold)
        self.standing_leg_joint_velocity_coeff = reward_config.get("standing_leg_joint_velocity_coeff", 0.0) * env.dt
        self.below_height_penalty_coeff = reward_config.get("below_height_penalty_coeff", 0.0) * env.dt
        self.contact_count_coeff = reward_config.get("contact_count_coeff", 2.0) * env.dt
        self.foot_stance_time_coeff = reward_config.get("foot_stance_time_coeff", 1.0) * env.dt
        self.foot_stance_time_per_robot_size_m = reward_config.get("foot_stance_time_per_robot_size_m", 0.6)
        self.foot_clearance_coeff = reward_config.get("foot_clearance_coeff", 1.0) * env.dt
        self.foot_clearance_per_robot_size_m = reward_config.get("foot_clearance_per_robot_size_m", 0.08)
        self.foot_lift_bonus_coeff = reward_config.get("foot_lift_bonus_coeff", 1.0) * env.dt
        self.foot_lift_bonus_per_robot_size_m = reward_config.get("foot_lift_bonus_per_robot_size_m", 0.10)

        self.survival = reward_config.get("survival", 0.0) * env.dt
        self.tracking_w_exp_linvel_x = reward_config.get("tracking_w_exp_linvel_x", 4.0)
        self.tracking_w_sum_linvel_x = reward_config.get("tracking_w_sum_linvel_x", 0.0) * env.dt
        self.tracking_w_exp_linvel_y = reward_config.get("tracking_w_exp_linvel_y", 4.0)
        self.tracking_w_sum_linvel_y = reward_config.get("tracking_w_sum_linvel_y", 0.0) * env.dt
        self.tracking_w_exp_angvel = reward_config.get("tracking_w_exp_angvel", 4.0)
        self.tracking_w_sum_angvel = reward_config.get("tracking_w_sum_angvel", 0.0) * env.dt
        self.nominal_joint_pos_exp = reward_config.get("tracking_nominal_joint_pos_exp", 4.0)
        self.nominal_joint_pos_coeff = reward_config.get("tracking_nominal_joint_pos_coeff", 0.0) * env.dt
        self.joint_deviation_l1_coeff = reward_config.get("joint_deviation_l1_coeff", 0.0) * env.dt
        self.root_acc_coeff = reward_config.get("root_acc_coeff", 0.0) * env.dt
        self.feet_swing_coeff = reward_config.get("feet_swing_coeff", 0.0) * env.dt
        self.feet_swing_period = reward_config.get("feet_swing_period", 0.2)
        self.feet_yaw_diff_coeff = reward_config.get("feet_yaw_diff_coeff", 0.0) * env.dt
        self.feet_yaw_mean_coeff = reward_config.get("feet_yaw_mean_coeff", 0.0) * env.dt
        self.feet_roll_coeff = reward_config.get("feet_roll_coeff", 0.0) * env.dt
        self.feet_distance_target = reward_config.get("feet_distance_target", 0.2)
        self.feet_distance_coeff = reward_config.get("feet_distance_coeff", 0.0) * env.dt
        self.air_time_max = reward_config.get("air_time_max", 0.3)
        self.air_time_coeff = reward_config.get("air_time_coeff", 0.0) * env.dt
        self.no_fly_coeff = reward_config.get("no_fly_coeff", 0.0) * env.dt
        self.impact_threshold = reward_config.get("impact_threshold", 150.0)
        self.impact_coeff = reward_config.get("impact_coeff", 0.0) * env.dt

        self.nominal_joint_qpos = jnp.array(env.initial_qpos)
        nominal_joint_names = reward_config.get("tracking_nominal_joint_pos_names", None)
        if nominal_joint_names is None:
            self.nominal_joint_qpos_id = jnp.array(env.actuator_joint_mask_qpos)
        else:
            self.nominal_joint_qpos_id = jnp.array([
                env.initial_mj_model.joint(name).qposadr[0]
                for name in nominal_joint_names
            ])
        standing_leg_joint_names = reward_config.get("standing_leg_joint_velocity_names", None)
        if standing_leg_joint_names is None:
            standing_leg_keywords = reward_config.get("standing_leg_joint_velocity_name_keywords", ["hip", "knee", "ankle"])
            standing_leg_excluded_keywords = reward_config.get(
                "standing_leg_joint_velocity_excluded_name_keywords",
                ["shoulder", "elbow", "wrist", "waist", "head"],
            )
            standing_leg_joint_names = [
                joint_name
                for joint_name in env.actuator_joint_names
                if (
                    any(keyword in joint_name.lower() for keyword in standing_leg_keywords)
                    and not any(keyword in joint_name.lower() for keyword in standing_leg_excluded_keywords)
                )
            ]
        self.has_standing_leg_joint_velocity_reward = len(standing_leg_joint_names) > 0
        standing_leg_qvel_ids = [
            env.initial_mj_model.joint(joint_name).dofadr[0]
            for joint_name in standing_leg_joint_names
        ]
        if len(standing_leg_qvel_ids) == 0:
            standing_leg_qvel_ids = [int(env.actuator_joint_mask_qvel[0])]
        self.standing_leg_joint_qvel_id = jnp.array(standing_leg_qvel_ids)

        foot_geom_indices = np.array(env.foot_geom_indices)
        self.left_feet_in_feet = jnp.array(np.where(np.isin(foot_geom_indices, np.array(env.left_foot_geom_indices)))[0])
        self.right_feet_in_feet = jnp.array(np.where(np.isin(foot_geom_indices, np.array(env.right_foot_geom_indices)))[0])
        self.left_foot_body_ids = jnp.array([env.initial_mj_model.geom(int(geom_id)).bodyid[0] for geom_id in np.array(env.left_foot_geom_indices)])
        self.right_foot_body_ids = jnp.array([env.initial_mj_model.geom(int(geom_id)).bodyid[0] for geom_id in np.array(env.right_foot_geom_indices)])


    def _scheduled_coeff(self, fixed_coeff, initial_coeff, final_coeff, curriculum_progress):
        if fixed_coeff >= 0.0:
            return fixed_coeff
        return initial_coeff + (final_coeff - initial_coeff) * curriculum_progress

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

    def setup(self, internal_state):
        super().setup(internal_state)
        internal_state["humanoid_gait_process"] = 0.0
        internal_state["humanoid_last_qvel"] = jnp.zeros(self.env.initial_mj_model.nv)
        internal_state["humanoid_time_since_last_touchdown"] = jnp.zeros(2, dtype=jnp.float32)

    def step(self, data, internal_state):
        super().step(data, internal_state)
        internal_state["humanoid_last_qvel"] = data.qvel


    def extra_reward_terms(self, data, mjx_model, internal_state, action, info):
        return 0.0, 0.0, 0.0


    def reward_and_info(self, data, mjx_model, internal_state, action, info):
        curriculum_progress = internal_state["env_curriculum_coeff"]
        critical_coeff = self._scheduled_coeff(self.critical_fixed_coeff, self.critical_initial_coeff, self.critical_final_coeff, curriculum_progress)
        style_coeff = self._scheduled_coeff(self.style_fixed_coeff, self.style_initial_coeff, self.style_final_coeff, curriculum_progress)
        gait_coeff = self._scheduled_coeff(self.gait_fixed_coeff, self.gait_initial_coeff, self.gait_final_coeff, curriculum_progress)
        
        # Tracking velocity command reward
        current_imu_linear_velocity = data.sensordata[self.env.imu_linear_velocity_sensor_adr:self.env.imu_linear_velocity_sensor_adr + self.env.imu_linear_velocity_sensor_dim]
        desired_imu_linear_velocity_xy = internal_state["goal_velocities"][:2]
        command_norm = jnp.linalg.norm(internal_state["goal_velocities"])
        is_moving_command = command_norm > self.moving_command_threshold
        is_standing_command = command_norm <= self.standing_command_threshold
        moving_command_scale = is_moving_command.astype(jnp.float32)
        standing_command_scale = is_standing_command.astype(jnp.float32)
        xy_difference = desired_imu_linear_velocity_xy - current_imu_linear_velocity[:2]
        xy_velocity_difference_norm = jnp.sum(jnp.square(xy_difference))
        tracking_xy_velocity_command_reward = (
            self.tracking_xy_velocity_command_coeff
            * jnp.exp(-xy_velocity_difference_norm / self.tracking_xy_temperature)
            * moving_command_scale
        )
        standing_xy_velocity_command_reward = (
            self.tracking_xy_velocity_command_coeff
            * jnp.exp(-jnp.sum(jnp.square(current_imu_linear_velocity[:2])) / self.tracking_xy_temperature)
            * standing_command_scale
        )

        # Tracking angular velocity command reward
        current_imu_angular_velocity = data.sensordata[self.env.imu_angular_velocity_sensor_adr:self.env.imu_angular_velocity_sensor_adr + self.env.imu_angular_velocity_sensor_dim]
        desired_imu_yaw_velocity = internal_state["goal_velocities"][2]
        yaw_velocity_difference_norm = jnp.square(current_imu_angular_velocity[2] - desired_imu_yaw_velocity)
        tracking_yaw_velocity_command_reward = (
            self.tracking_yaw_velocity_command_coeff
            * jnp.exp(-yaw_velocity_difference_norm / self.tracking_yaw_temperature)
            * moving_command_scale
        )
        standing_yaw_velocity_command_reward = (
            self.tracking_yaw_velocity_command_coeff
            * jnp.exp(-jnp.square(current_imu_angular_velocity[2]) / self.tracking_yaw_temperature)
            * standing_command_scale
        )
        standing_command_reward = standing_xy_velocity_command_reward + standing_yaw_velocity_command_reward

        # Alive reward
        alive_clipped_reward = critical_coeff * self.alive_clipped_coeff * 1.0
        alive_unclipped_reward = critical_coeff * self.alive_unclipped_coeff * 1.0

        # Critical penalties keep the robot safe and dynamically plausible.
        z_velocity_squared = current_imu_linear_velocity[2] ** 2
        z_velocity_reward = critical_coeff * self.z_velocity_coeff * -z_velocity_squared

        imu_acceleration_norm = jnp.mean(jnp.square((current_imu_linear_velocity - internal_state["previous_imu_linear_velocity"]) / self.env.dt))
        imu_acceleration_reward = critical_coeff * self.imu_acceleration_coeff * -imu_acceleration_norm

        angular_velocity_norm = jnp.sum(jnp.square(current_imu_angular_velocity[:2]))
        angular_velocity_reward = critical_coeff * self.roll_pitch_vel_coeff * -angular_velocity_norm

        roll_pitch_position_norm = jnp.sum(jnp.square(internal_state["imu_orientation_euler"][:2]))
        angular_position_reward = critical_coeff * self.roll_pitch_pos_coeff * -roll_pitch_position_norm

        joint_positions = data.qpos[self.env.actuator_joint_mask_qpos]
        actuator_joint_position_limits = internal_state["joint_position_limits"][self.env.actuator_joint_mask_joints - 1]
        joint_outside_limits = jnp.maximum(joint_positions - actuator_joint_position_limits[:, 1], 0.0) + \
                               jnp.maximum(actuator_joint_position_limits[:, 0] - joint_positions, 0.0)
        joint_position_limit_reward = critical_coeff * self.joint_position_limit_coeff * -jnp.mean(joint_outside_limits)

        actuator_joint_velocity_limit = internal_state["actuator_joint_max_velocities"] * self.soft_actuator_joint_velocity_limit
        joint_velocity_outside_limits = jnp.maximum(jnp.abs(data.qvel[self.env.actuator_joint_mask_qvel]) - actuator_joint_velocity_limit, 0.0)
        joint_velocity_limit_reward = critical_coeff * self.actuator_joint_velocity_limit_coeff * -jnp.mean(joint_velocity_outside_limits)

        all_contact_relevant_geom_xpos = data.geom_xpos[self.env.reward_collision_sphere_geom_ids]
        all_contact_relevant_geom_sizes = mjx_model.geom_size[self.env.reward_collision_sphere_geom_ids, 0]
        distance_between_geoms = jnp.linalg.norm(all_contact_relevant_geom_xpos[:, None] - all_contact_relevant_geom_xpos[None], axis=-1)
        contact_between_geoms = distance_between_geoms <= (all_contact_relevant_geom_sizes[:, None] + all_contact_relevant_geom_sizes[None])
        nr_collisions = (jnp.sum(contact_between_geoms) - self.env.reward_collision_sphere_geom_ids.shape[0]) // 2
        nr_collisions = jnp.maximum(nr_collisions - internal_state["nr_collisions_in_nominal"], 0)
        collision_reward = critical_coeff * self.collision_coeff * -nr_collisions

        height_difference_squared = (internal_state["robot_imu_height_over_ground"] - internal_state["robot_nominal_imu_height_over_ground"]) ** 2
        base_height_reward = critical_coeff * self.base_height_coeff * -height_difference_squared
        below_height_threshold = (
            self.env.env_config["termination"].get("height_percentage_threshold", 0.8)
            * internal_state["robot_nominal_imu_height_over_ground"]
        )
        below_height_scale = (
            internal_state["robot_imu_height_over_ground"] < below_height_threshold
        ).astype(jnp.float32)
        below_height_penalty_reward = self.below_height_penalty_coeff * -below_height_scale

        feet_floor_contacts = self.env.terrain_function.check_feet_floor_contact(data)
        all_feet_off_ground_reward = critical_coeff * self.all_feet_off_ground_coeff * -jnp.all(~feet_floor_contacts).astype(jnp.float32)

        feet_global_linear_velocity_x = data.sensordata[self.env.feet_global_linear_velocity_sensor_adrs_start]
        feet_global_linear_velocity_y = data.sensordata[self.env.feet_global_linear_velocity_sensor_adrs_start + 1]
        feet_global_linear_velocity_xy_norm = jnp.square(feet_global_linear_velocity_x) + jnp.square(feet_global_linear_velocity_y)
        contact_filtered_feet_slip = jnp.mean(feet_floor_contacts * feet_global_linear_velocity_xy_norm)
        foot_slip_reward = critical_coeff * self.foot_slip_coeff * -contact_filtered_feet_slip

        feet_global_linear_velocity_z = data.sensordata[self.env.feet_global_linear_velocity_sensor_adrs_start + 2]
        squared_negative_z_velocity = jnp.mean(jnp.square(jnp.minimum(feet_global_linear_velocity_z, 0.0)))
        foot_z_velocity_reward = critical_coeff * self.foot_z_velocity_coeff * -squared_negative_z_velocity

        missing_lower_feet_contacts = self.env.terrain_function.check_flat_feet_floor_missing_contacts(data, mjx_model, internal_state)
        contact_filtered_missing_lower_feet_contacts = jnp.mean(feet_floor_contacts * missing_lower_feet_contacts)
        foot_flat_contact_reward = critical_coeff * self.foot_flat_contact_coeff * -contact_filtered_missing_lower_feet_contacts

        # Style and efficiency penalties can ramp in later.
        actuator_joint_nominal_diff_norm = jnp.mean(jnp.square((data.qpos[self.env.actuator_joint_mask_qpos] * internal_state["actuator_joint_keep_nominal"]) - (internal_state["actuator_joint_nominal_positions"] * internal_state["actuator_joint_keep_nominal"])))
        actuator_joint_nominal_diff_reward = style_coeff * self.actuator_joint_nominal_diff_coeff * -actuator_joint_nominal_diff_norm

        velocity_norm = jnp.mean(jnp.square(data.qvel[self.env.actuator_joint_mask_qvel]))
        joint_velocity_reward = style_coeff * self.joint_velocity_coeff * -velocity_norm

        acceleration_norm = jnp.mean(jnp.square((data.qvel[self.env.actuator_joint_mask_qvel] - internal_state["previous_actuator_joint_velocities"]) / self.env.dt))
        acceleration_reward = style_coeff * self.joint_acceleration_coeff * -acceleration_norm

        torque_norm = jnp.mean(jnp.square(data.qfrc_actuator[self.env.actuator_joint_mask_qvel]))
        torque_reward = style_coeff * self.joint_torque_coeff * -torque_norm

        power_draw = jnp.mean(jnp.maximum(data.qfrc_actuator[self.env.actuator_joint_mask_qvel] * data.qvel[self.env.actuator_joint_mask_qvel], 0.0))
        power_draw_penalty_reward = style_coeff * self.power_draw_penalty_coeff * -power_draw

        action_rate_norm = jnp.mean(jnp.square(action - internal_state["last_action"]))
        action_rate_reward = style_coeff * self.action_rate_coeff * -action_rate_norm
        
        action_smoothness_norm = jnp.mean(jnp.square(action - 2 * internal_state["last_action"] + internal_state["second_last_action"]))
        action_smoothness_reward = style_coeff * self.action_smoothness_coeff * -action_smoothness_norm

        standing_leg_joint_velocity_norm = jnp.mean(jnp.square(data.qvel[self.standing_leg_joint_qvel_id]))
        standing_leg_joint_velocity_reward = (
            self.standing_leg_joint_velocity_coeff
            * -standing_leg_joint_velocity_norm
            * standing_command_scale
            * jnp.asarray(self.has_standing_leg_joint_velocity_reward, dtype=jnp.float32)
        )

        feet_first_contact = feet_floor_contacts & (~internal_state["previous_feet_floor_contacts"])
        target_foot_air_time = self.foot_air_time_per_robot_size_m * internal_state["robot_dimensions_mean"]
        target_foot_air_time = moving_command_scale * target_foot_air_time
        air_time_reward = jnp.mean(feet_first_contact * jnp.minimum(internal_state["feet_time_in_air"] - target_foot_air_time, 0.0))
        foot_air_time_reward = gait_coeff * self.foot_air_time_coeff * moving_command_scale * air_time_reward

        symmetry_air_violations = jnp.mean(jnp.where((~feet_floor_contacts[self.feet_symmetry_pairs[:, 0]]) & (~feet_floor_contacts[self.feet_symmetry_pairs[:, 1]]), 1, 0))
        symmetry_air_reward = gait_coeff * self.symmetry_air_coeff * -symmetry_air_violations

        nr_feet = float(self.env.nr_feet)
        nr_feet_in_contact = jnp.sum(feet_floor_contacts.astype(jnp.float32))
        moving_contact_target = float(max(self.env.nr_feet - 1, 1))
        target_nr_contacts = jnp.where(is_moving_command, moving_contact_target, nr_feet)
        contact_count_error = (nr_feet_in_contact - target_nr_contacts) / nr_feet
        contact_count_reward = gait_coeff * self.contact_count_coeff * -jnp.square(contact_count_error)

        target_stance_time = self.foot_stance_time_per_robot_size_m * internal_state["robot_dimensions_mean"]
        long_stance_time = jnp.maximum(internal_state["feet_time_on_ground"] - target_stance_time, 0.0)
        foot_stance_time_reward = gait_coeff * self.foot_stance_time_coeff * -moving_command_scale * jnp.mean(long_stance_time)

        feet_height_over_ground = data.geom_xpos[self.env.foot_geom_indices, 2] - self.env.terrain_function.ground_height_at(
            internal_state,
            data.geom_xpos[self.env.foot_geom_indices, 0],
            data.geom_xpos[self.env.foot_geom_indices, 1],
        )
        target_foot_clearance = self.foot_clearance_per_robot_size_m * internal_state["robot_dimensions_mean"]
        foot_clearance_error = jnp.maximum(target_foot_clearance - feet_height_over_ground, 0.0)
        swing_feet = (~feet_floor_contacts).astype(jnp.float32)
        foot_clearance_reward = gait_coeff * self.foot_clearance_coeff * -moving_command_scale * jnp.mean(swing_feet * foot_clearance_error)

        target_foot_lift_bonus = self.foot_lift_bonus_per_robot_size_m * internal_state["robot_dimensions_mean"]
        foot_lift_fraction = jnp.clip(feet_height_over_ground / target_foot_lift_bonus, 0.0, 1.0)
        foot_lift_bonus_reward = gait_coeff * self.foot_lift_bonus_coeff * moving_command_scale * jnp.mean(swing_feet * foot_lift_fraction)

        survival_reward = self.survival
        tracking_reward_linvel_x = jnp.exp(-jnp.square(current_imu_linear_velocity[0] - desired_imu_linear_velocity_xy[0]) * self.tracking_w_exp_linvel_x) * self.tracking_w_sum_linvel_x * moving_command_scale
        tracking_reward_linvel_y = jnp.exp(-jnp.square(current_imu_linear_velocity[1] - desired_imu_linear_velocity_xy[1]) * self.tracking_w_exp_linvel_y) * self.tracking_w_sum_linvel_y * moving_command_scale
        tracking_reward_angvel = jnp.exp(-jnp.square(current_imu_angular_velocity[2] - desired_imu_yaw_velocity) * self.tracking_w_exp_angvel) * self.tracking_w_sum_angvel * moving_command_scale
        joint_qpos_reward = jnp.exp(
            -self.nominal_joint_pos_exp *
            jnp.sum(jnp.square(data.qpos[self.nominal_joint_qpos_id] - self.nominal_joint_qpos[self.nominal_joint_qpos_id]))
        ) * self.nominal_joint_pos_coeff
        joint_deviation_l1_penalty = (
            jnp.sum(jnp.abs(data.qpos[self.nominal_joint_qpos_id] - self.nominal_joint_qpos[self.nominal_joint_qpos_id])) *
            self.joint_deviation_l1_coeff
        )
        root_acceleration_reward = jnp.sum(jnp.square((data.qvel[:6] - internal_state["humanoid_last_qvel"][:6]) / self.env.dt)) * self.root_acc_coeff

        left_foot_on_ground = jnp.any(feet_floor_contacts[self.left_feet_in_feet])
        right_foot_on_ground = jnp.any(feet_floor_contacts[self.right_feet_in_feet])
        feet_on_ground = jnp.array([left_foot_on_ground, right_foot_on_ground])
        gait_frequency = internal_state.get("goal_gait_frequency", jnp.asarray(0.0))
        gait_process = jnp.fmod(internal_state["humanoid_gait_process"] + self.env.dt * gait_frequency, 1.0)
        left_swing = (jnp.abs(gait_process - 0.25) < 0.5 * self.feet_swing_period) & (gait_frequency > 1.0e-8) & is_moving_command
        right_swing = (jnp.abs(gait_process - 0.75) < 0.5 * self.feet_swing_period) & (gait_frequency > 1.0e-8) & is_moving_command
        feet_swing_reward = (
            (left_swing & ~feet_on_ground[0]).astype(jnp.float32) +
            (right_swing & ~feet_on_ground[1]).astype(jnp.float32)
        ) * self.feet_swing_coeff

        left_foot_euler = Rotation.from_matrix(data.site_xmat[self.env.left_foot_site_id].reshape(3, 3)).as_euler("xyz")
        right_foot_euler = Rotation.from_matrix(data.site_xmat[self.env.right_foot_site_id].reshape(3, 3)).as_euler("xyz")
        left_foot_yaw = self._wrap_to_pi(left_foot_euler[2])
        right_foot_yaw = self._wrap_to_pi(right_foot_euler[2])
        feet_yaw_diff_reward = jnp.square(self._wrap_to_pi(left_foot_yaw - right_foot_yaw)) * self.feet_yaw_diff_coeff
        feet_yaw_mean = (left_foot_yaw * 0.5 + right_foot_yaw * 0.5) + jnp.pi * (jnp.abs(left_foot_yaw - right_foot_yaw) > jnp.pi)
        base_yaw = self._wrap_to_pi(internal_state["imu_orientation_euler"][2])
        feet_yaw_mean_reward = jnp.square(self._wrap_to_pi(base_yaw - feet_yaw_mean)) * self.feet_yaw_mean_coeff
        feet_roll_reward = (jnp.square(self._wrap_to_pi(left_foot_euler[0])) + jnp.square(self._wrap_to_pi(right_foot_euler[0]))) * self.feet_roll_coeff

        left_foot_pos = data.site_xpos[self.env.left_foot_site_id]
        right_foot_pos = data.site_xpos[self.env.right_foot_site_id]
        feet_distance = (
            jnp.cos(base_yaw) * (left_foot_pos[1] - right_foot_pos[1]) -
            jnp.sin(base_yaw) * (left_foot_pos[0] - right_foot_pos[0])
        )
        feet_distance_reward = jnp.clip(self.feet_distance_target - feet_distance, 0.0, 0.1) * self.feet_distance_coeff

        tslt = internal_state["humanoid_time_since_last_touchdown"]
        touchdown_reward = jnp.where(feet_on_ground & (tslt > 1e-6), tslt - self.air_time_max, 0.0)
        air_time_reward = jnp.sum(touchdown_reward) * self.air_time_coeff * moving_command_scale
        tslt = jnp.where(feet_on_ground, 0.0, tslt + self.env.dt)
        no_fly_reward = (jnp.logical_and(tslt[0] > 0.0, tslt[1] > 0.0) * 1.0) * self.no_fly_coeff

        left_foot_force_norm = jnp.linalg.norm(data.cfrc_ext[self.left_foot_body_ids, :3], axis=1)
        right_foot_force_norm = jnp.linalg.norm(data.cfrc_ext[self.right_foot_body_ids, :3], axis=1)
        impact_reward = (
            jnp.mean((left_foot_force_norm > self.impact_threshold) * 1.0 + (right_foot_force_norm > self.impact_threshold) * 1.0) *
            self.impact_coeff
        )

        internal_state["humanoid_gait_process"] = gait_process
        internal_state["humanoid_time_since_last_touchdown"] = tslt

        extra_alive_reward, extra_positive_reward, extra_penalty = self.extra_reward_terms(data, mjx_model, internal_state, action, info)

        booster_tracking_reward = tracking_reward_linvel_x + tracking_reward_linvel_y + tracking_reward_angvel + joint_qpos_reward + feet_swing_reward
        booster_penalty = (
            root_acceleration_reward + feet_yaw_diff_reward + feet_yaw_mean_reward +
            feet_roll_reward + feet_distance_reward + air_time_reward + no_fly_reward +
            impact_reward + joint_deviation_l1_penalty
        )
        tracking_reward = (
            tracking_xy_velocity_command_reward +
            tracking_yaw_velocity_command_reward +
            standing_command_reward +
            booster_tracking_reward
        )
        critical_penalty = z_velocity_reward + imu_acceleration_reward + angular_velocity_reward + angular_position_reward + \
                           joint_position_limit_reward + joint_velocity_limit_reward + collision_reward + base_height_reward + \
                           all_feet_off_ground_reward + foot_slip_reward + foot_z_velocity_reward + foot_flat_contact_reward + \
                           below_height_penalty_reward
        style_penalty = actuator_joint_nominal_diff_reward + joint_velocity_reward + acceleration_reward + torque_reward + \
                        power_draw_penalty_reward + action_rate_reward + action_smoothness_reward
        gait_reward = foot_lift_bonus_reward
        gait_penalty = foot_air_time_reward + symmetry_air_reward + contact_count_reward + foot_stance_time_reward + foot_clearance_reward
        alive_total = alive_clipped_reward + alive_unclipped_reward + survival_reward + extra_alive_reward
        penalty_total = critical_penalty + style_penalty + standing_leg_joint_velocity_reward + gait_penalty + booster_penalty + extra_penalty
        pre_clip_total = (
            tracking_reward + penalty_total + gait_reward +
            extra_positive_reward + alive_clipped_reward + survival_reward + extra_alive_reward
        )
        reward = jnp.maximum(pre_clip_total, 0.0) + alive_unclipped_reward
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        info[f"reward/survival"] = survival_reward
        info[f"reward/track_linvel_x"] = tracking_reward_linvel_x
        info[f"reward/track_linvel_y"] = tracking_reward_linvel_y
        info[f"reward/track_angvel"] = tracking_reward_angvel
        info[f"reward/joint_qpos"] = joint_qpos_reward
        info[f"reward/feet_swing"] = feet_swing_reward
        info[f"reward/root_acceleration"] = root_acceleration_reward
        info[f"reward/feet_yaw_diff"] = feet_yaw_diff_reward
        info[f"reward/feet_yaw_mean"] = feet_yaw_mean_reward
        info[f"reward/feet_roll"] = feet_roll_reward
        info[f"reward/feet_distance"] = feet_distance_reward
        info[f"reward/air_time"] = air_time_reward
        info[f"reward/no_fly"] = no_fly_reward
        info[f"reward/impact"] = impact_reward
        info[f"reward/joint_deviation_l1"] = joint_deviation_l1_penalty
        info[f"reward/booster_tracking_total"] = booster_tracking_reward
        info[f"reward/booster_penalty_total"] = booster_penalty
        info[f"reward/track_xy_vel_cmd"] = tracking_xy_velocity_command_reward
        info[f"reward/track_yaw_vel_cmd"] = tracking_yaw_velocity_command_reward
        info[f"reward/standing_xy_vel_cmd"] = standing_xy_velocity_command_reward
        info[f"reward/standing_yaw_vel_cmd"] = standing_yaw_velocity_command_reward
        info[f"reward/standing_command"] = standing_command_reward
        info[f"reward/alive_clipped"] = alive_clipped_reward
        info[f"reward/alive_unclipped"] = alive_unclipped_reward
        info[f"reward/z_velocity"] = z_velocity_reward
        info[f"reward/imu_acceleration"] = imu_acceleration_reward
        info[f"reward/angular_velocity"] = angular_velocity_reward
        info[f"reward/angular_position"] = angular_position_reward
        info[f"reward/actuator_joint_nominal_diff"] = actuator_joint_nominal_diff_reward
        info[f"reward/joint_position_limit"] = joint_position_limit_reward
        info[f"reward/joint_velocity_limit"] = joint_velocity_limit_reward
        info[f"reward/joint_velocity"] = joint_velocity_reward
        info[f"reward/joint_acceleration"] = acceleration_reward
        info[f"reward/joint_torque"] = torque_reward
        info[f"reward/power_draw_penalty"] = power_draw_penalty_reward
        info[f"reward/action_rate"] = action_rate_reward
        info[f"reward/action_smoothness"] = action_smoothness_reward
        info[f"reward/standing_leg_joint_velocity"] = standing_leg_joint_velocity_reward
        info[f"reward/collision"] = collision_reward
        info[f"reward/base_height"] = base_height_reward
        info[f"reward/below_height_penalty"] = below_height_penalty_reward
        info[f"reward/foot_air_time"] = foot_air_time_reward
        info[f"reward/contact_count"] = contact_count_reward
        info[f"reward/foot_stance_time"] = foot_stance_time_reward
        info[f"reward/foot_clearance"] = foot_clearance_reward
        info[f"reward/foot_lift_bonus"] = foot_lift_bonus_reward
        info[f"reward/all_feet_off_ground"] = all_feet_off_ground_reward
        info[f"reward/symmetry_air"] = symmetry_air_reward
        info[f"reward/foot_slip"] = foot_slip_reward
        info[f"reward/foot_z_velocity"] = foot_z_velocity_reward
        info[f"reward/foot_flat_contact"] = foot_flat_contact_reward
        info[f"reward/critical_coeff"] = critical_coeff
        info[f"reward/style_coeff"] = style_coeff
        info[f"reward/gait_coeff"] = gait_coeff
        info[f"reward/tracking_total"] = tracking_reward
        info[f"reward/alive_total"] = alive_total
        info[f"reward/penalty_total"] = penalty_total
        info[f"reward/pre_clip_total"] = pre_clip_total
        info[f"reward/critical_penalty_total"] = critical_penalty
        info[f"reward/style_penalty_total"] = style_penalty
        info[f"reward/gait_penalty_total"] = gait_penalty
        info[f"reward/gait_reward_total"] = gait_reward
        info[f"reward/total"] = reward
        info[f"env_info/standing_leg_joint_velocity_norm"] = standing_leg_joint_velocity_norm
        info[f"env_info/is_moving_command"] = moving_command_scale
        info[f"env_info/is_standing_command"] = standing_command_scale
        info[f"env_info/below_height"] = below_height_scale
        info[f"env_info/xy_vel_diff_abs"] = jnp.nan_to_num(jnp.mean(jnp.minimum(jnp.abs(xy_difference), 2 * internal_state["max_command_velocity"])), nan=2 * internal_state["max_command_velocity"], posinf=2 * internal_state["max_command_velocity"], neginf=2 * internal_state["max_command_velocity"])

        return reward
