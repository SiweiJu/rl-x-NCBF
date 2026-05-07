from ml_collections import config_dict


def get_config(environment_name):
    config = {
        "name": environment_name,
        "nr_envs": 1,
        "nr_history_steps": 10,
        "seed": 1,
        "render": False,
        "train_robot": "unitree_go2",
        "use_booster_defaults": True,
        "ball_plate": {
            "enabled": False,
            "include_observations": True,
            "stand_after_drop": False,
            "post_drop_truncation_seconds": 3.0,
            "randomize_ball_plate_domain": True,
            "initial_joint_positions": {
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.2,
                "left_shoulder_yaw_joint": 0.0,
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": -0.2,
                "right_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": 0.0,
                "left_elbow_roll_joint": 0.0,
                "right_elbow_pitch_joint": 0.0,
                "right_elbow_roll_joint": 0.0,
            },
            "plate_home_pos": [0.45540412, 0.0, 1.02410541],
            "plate_home_quat": [1.0, 0.0, 0.0, 0.0],
            "plate_size": [0.16, 0.22, 0.008],
            "plate_size_scale_range": [0.9, 1.1],
            "randomize_plate_size": True,
            "plate_mass": 0.25,
            "plate_mass_range": [0.15, 0.4],
            "randomize_plate_mass": True,
            "ball_radius": 0.04,
            "ball_radius_range": [0.03, 0.055],
            "ball_mass": 0.08,
            "ball_mass_range": [0.04, 0.14],
            "randomize_ball_size_mass": True,
            "ball_home_pos": [0.28, 0.0, 0.689],
            "ball_start_offset": [0.0, 0.0, 0.04],
            "ball_start_offset_xy_range": [0.045, 0.045],
            "ball_initial_linear_velocity_xy_range": [0.12, 0.12],
            "ball_initial_angular_velocity_range": [2.5, 2.5, 2.5],
            "randomize_ball_initial_state": True,
            "left_fist_body": "left_elbow_roll_link",
            "right_fist_body": "right_elbow_roll_link",
            "left_fist_pos": [0.17, 0.0, 0.0],
            "right_fist_pos": [0.17, 0.0, 0.0],
            "use_capsule_hand": True,
            "fist_radius": 0.035,
            "fist_half_length": 0.075,
            "capsule_hand_mass": 0.2,
            "capsule_hand_mass_range": [0.12, 0.3],
            "capsule_hand_radius_scale_range": [0.9, 1.1],
            "capsule_hand_half_length_scale_range": [0.9, 1.1],
            "randomize_capsule_hand_mass": True,
            "randomize_capsule_hand_size": True,
            "capsule_hand_disable_forearm_guards": True,
            "capsule_hand_remove_visual_geoms": True,
            "capsule_hand_remove_reward_collision_geoms": True,
            "capsule_hand_removed_body_names": [
                "left_zero_link", "left_one_link", "left_two_link", "left_three_link", "left_four_link", "left_five_link", "left_six_link",
                "right_zero_link", "right_one_link", "right_two_link", "right_three_link", "right_four_link", "right_five_link", "right_six_link",
            ],
            "capsule_hand_removed_mesh_names": [
                "left_palm_link", "left_zero_link", "left_one_link", "left_two_link", "left_three_link", "left_four_link", "left_five_link", "left_six_link",
                "right_palm_link", "right_zero_link", "right_one_link", "right_two_link", "right_three_link", "right_four_link", "right_five_link", "right_six_link",
            ],
            "torso_contact_body": "torso_link",
            "torso_contact_pos": [0.0, 0.0, 0.2],
            "torso_contact_size": 0.11,
            "plate_torso_contact_friction": [1.5, 1.5, 0.005, 0.0001, 0.0001],
            "plate_torso_contact_dim": 3,
            "left_upper_arm_contact_body": "left_shoulder_yaw_link",
            "right_upper_arm_contact_body": "right_shoulder_yaw_link",
            "upper_arm_contact_fromto": [0.0, 0.0, -0.01, 0.0, 0.0, -0.085],
            "upper_arm_contact_radius": 0.045,
            "left_forearm_contact_body": "left_elbow_roll_link",
            "right_forearm_contact_body": "right_elbow_roll_link",
            "forearm_contact_fromto": [0.02, 0.0, 0.0, 0.16, 0.0, 0.0],
            "forearm_contact_radius": 0.05,
            "plate_arm_contact_friction": [1.5, 1.5, 0.005, 0.0001, 0.0001],
            "plate_arm_contact_dim": 3,
            "plate_ball_contact_friction": [4.0, 4.0, 0.001, 0.00001, 0.00001],
            "plate_ball_contact_dim": 3,
            "plate_ball_contact_solref": [0.005, 1.0],
            "plate_ball_contact_solimp": [0.99, 0.999, 0.0001, 0.5, 2.0],
            "plate_ball_contact_margin": 0.002,
            "plate_ball_contact_friction_tangential_range": [2.0, 6.0],
            "randomize_plate_ball_friction": True,
            "plate_support_contact_friction": [3.0, 3.0, 0.005, 0.0001, 0.0001],
            "plate_support_contact_dim": 6,
            "plate_support_contact_friction_tangential_range": [2.0, 5.0],
            "plate_robot_contact_friction_tangential_range": [1.0, 4.0],
            "plate_contact_timeconst_range": [0.003, 0.012],
            "randomize_plate_support_friction": True,
            "randomize_plate_robot_friction": True,
            "randomize_plate_contact_stiffness": True,
            "plate_support_clearance": 0.0,
            "plate_tilt_drop_threshold": 0.9,
            "drop_margin": 0.0,
            "drop_height": 0.08,
        },
        "control_type": "pd",
        "ncbf_use_policy_observations": False,
        "command": {
            "type": "random",
            "sampling_type": "step_probability_and_reset",
            "max_velocity_per_m_factor": 2.0,
            "clip_max_velocity": 1.0,
            "zero_clip_threshold_percentage": 0.1,
            "all_zero_chance": 0.04,
            "single_zero_chance": 0.005,
            "velocity_ratio": [1, 0.5, 0.2],
            # configs for test mode with type "random_trajectory"
            "trajectory_length_in_seconds": 10.0,

        },
        "env_curriculum_nr_levels": 100,
        "env_curriculum_level_success_episode_return": 8.0,
        "domain_randomization": {
            "sampling_type": "step_probability_and_reset",
            "action_delay": {
                "type": "default",
                "max_nr_delay_steps": 1,
                "mixed_chance": 0.05,
            },
            "initial_state": {
                "type": "random",
                "roll_angle_pi_factor": 0.05,
                "pitch_angle_pi_factor": 0.05,
                "yaw_angle_pi_factor": 1.0,
                "actuator_joint_position_offset_to_nominal": 0.01,
                "actuator_joint_nominal_position_factor": 0.5,
                "joint_velocity_max_factor": 0.5,
                "trunk_velocity_clip_mass_factor": 0.1,
                "trunk_velocity_clip_limit": 0.5,
            },
            "joint_dropout": {
                "type": "default",
                "dropout_open_chance": 0.001,
                "dropout_lock_chance": 0.001,
            },
            "mujoco_model": {
                "type": "default",
                "friction_tangential_factor": 1.0,
                "friction_torsional_factor": 1.0,
                "friction_rolling_factor": 1.0,
                "stiffness_factor": 0.5,
                "damping_factor": 0.6,
                "foot_solimp_factor": 0.8,
                "add_impratio": 1.0,
                "xy_gravity": 0.5,
                "z_gravity_factor": 0.1,
                "density_factor": 0.1,
                "viscosity_factor": 0.1,
            },
            "observation_noise": {
                "type": "default",
                "joint_position": 0.01,
                "joint_velocity": 1.5,
                "imu_angular_velocity": 0.2,
                "gravity_vector": 0.05,
                "exteroception": 0.03,
                "add_ball_plate_obs_noise": True,
                "ball_plate_position_noise": 0.015,
                "ball_plate_velocity_noise": 0.1,
                "normalize_ball_plate_observations": True,
            },
            "perturbation": {
                "sampling_type": "step_probability",
                "sampling_probability": 0.002,
                "type": "default",
                "trunk_velocity_clip_mass_factor": 0.1,
                "trunk_velocity_clip_limit": 1.0,
                "trunk_velocity_add_chance": 0.5,
                "max_joint_velocity": 0.5,
                "max_joint_position": 0.01,
            },
            "seen_robot": {
                "type": "default",
                "robot_size_scaling_factor": 0.0,
                "coupled_mass_inertia_factor": 0.1,
                "decoupled_mass_inertia_factor": 0.05,
                "add_com_displacement": 0.005,
                "add_inertia_orientation_rad": 0.01,
                "add_body_position": 0.0,
                "add_body_orientation_rad": 0.01,
                "add_imu_position": 0.05,
                "foot_size_factor": 0.05,
                "joint_axis_angle_rad": 0.01,
                "torque_limit_factor": 0.15,
                "add_actuator_joint_nominal_position": 0.01,
                "joint_velocity_max_factor": 0.15,
                "add_joint_range": 0.05,
                "joint_damping_factor": 0.3,
                "add_joint_damping": 0.003,
                "joint_armature_factor": 0.5,
                "add_joint_armature": 0.001,
                "joint_stiffness_factor": 0.1,
                "add_joint_stiffness": 0.1,
                "joint_friction_loss_factor": 1.0,
                "add_joint_friction_loss": 0.00001,
                "p_gain_factor": 0.1,
                "d_gain_factor": 0.1,
                "scaling_factor_factor": 0.1,
            },
            "unseen_robot": {
                "type": "default",
                "mass_inertia_factor": 0.25,
                "com_factor": 0.1,
                "body_position_factor": 0.01,
                "joint_damping_factor": 0.2,
                "joint_armature_factor": 0.2,
                "joint_stiffness_factor": 0.2,
                "joint_friction_loss_factor": 0.3,
                "p_gain_factor": 0.2,
                "d_gain_factor": 0.2,
                "position_offset": 0.03,
            },
        },
        "policy_exteroceptive_observation_type": "none",
        "critic_exteroceptive_observation_type": "height_over_ground",
        "reward": {
            "type": "default",
            "critical_initial_coeff": 0.5,
            "critical_final_coeff": 1.0,
            "critical_fixed_coeff": -1.0,
            "style_initial_coeff": 0.0,
            "style_final_coeff": 1.0,
            "style_fixed_coeff": -1.0,
            "gait_initial_coeff": 1.0,
            "gait_final_coeff": 1.0,
            "gait_fixed_coeff": 1.0,
            "moving_command_threshold": 0.05,
            "standing_command_threshold": 0.05,
            "standing_leg_joint_velocity_coeff": 0.2,
            "standing_leg_joint_velocity_name_keywords": ["hip", "knee", "ankle"],
            "standing_leg_joint_velocity_excluded_name_keywords": ["shoulder", "elbow", "wrist", "waist", "head"],
            "tracking_xy_velocity_command_coeff": 2.0,
            "tracking_xy_temperature": 0.25,
            "tracking_yaw_velocity_command_coeff": 1.0,
            "tracking_yaw_temperature": 0.25,
            "alive_clipped_coeff": 0.05,
            "alive_unclipped_coeff": 0.05,
            "z_velocity_coeff": 2.0,
            "imu_acceleration_coeff": 1e-4,
            "roll_pitch_vel_coeff": 0.05,
            "roll_pitch_pos_coeff": 10.0,
            "actuator_joint_nominal_diff_coeff": 20.0,
            "joint_position_limit_coeff": 40.0,
            "soft_joint_position_limit": 0.9,
            "actuator_joint_velocity_limit_coeff": 5.0,
            "soft_actuator_joint_velocity_limit": 0.9,
            "joint_velocity_coeff": 4e-4,
            "joint_acceleration_coeff": 5e-6,
            "joint_torque_coeff": 4e-4,
            "power_draw_penalty_coeff": 4e-4,
            "action_rate_coeff": 10.0,
            "action_smoothness_coeff": 0.1,
            "collision_coeff": 2.0,
            "base_height_coeff": 30.0,
            "foot_air_time_coeff": 3.0,
            "foot_air_time_per_robot_size_m": 0.4,
            "all_feet_off_ground_coeff": 2.0,
            "symmetry_air_coeff": 1.0,
            "contact_count_coeff": 6.0,
            "foot_stance_time_coeff": 2.0,
            "foot_stance_time_per_robot_size_m": 0.45,
            "foot_clearance_coeff": 25.0,
            "foot_clearance_per_robot_size_m": 0.10,
            "foot_lift_bonus_coeff": 1.0,
            "foot_lift_bonus_per_robot_size_m": 0.12,
            "foot_slip_coeff": 0.1,
            "foot_z_velocity_coeff": 0.2,
            "foot_flat_contact_coeff": 0.01,
            "ball_plate_centering_coeff": 2.0,
            "ball_plate_center_bonus_coeff": 2.0,
            "ball_plate_center_bonus_temperature": 0.01,
            "ball_plate_velocity_coeff": 0.1,
            "ball_plate_on_plate_coeff": 1.0,
            "ball_plate_alive_coeff": 0.2,
            "ball_plate_drop_penalty_coeff": 10.0,
        },
        "termination": {
            "type": "below_height",
            "height_percentage_threshold": 0.8,
            "body_tilt_threshold": 0.4,
            "terminate_on_ball_plate_drop": True,
        },
        "terrain": {
            "type": "hfield_diverse",
            "wave_fn_min": 0,
            "wave_fn_max": 2,
            "wave_height_max_per_m_factor": 0.3,
            "random_height_max_per_m_factor": 0.04,
            "block_probability": 0.5,
            "block_length_in_meters": 0.5,
            "block_height_max_per_m_factor": 0.1,
        },
        "add_goal_arrow": False,
        "timestep": 0.005,
        "episode_length_in_seconds": 20,
        "ncbf": {
            "H": 10,
        }
    }

    config = config_dict.ConfigDict(config)
    environment_name_lower = environment_name.lower()
    if "boosterball" in environment_name_lower or "booster_ball" in environment_name_lower:
        apply_boosterball_defaults(config)
    elif "booster" in environment_name_lower:
        apply_booster_defaults(config)
    return config


def apply_booster_defaults(config):
    config.booster_defaults_applied = True
    config.nr_envs = 8192
    config.train_robot = "booster_t1"
    config.command.type = "booster"
    config.command.sampling_type = "step_probability_and_reset"
    config.command.sampling_probability = 0.004
    config.command.max_x_vel = 0.8
    config.command.max_y_vel = 0.5
    config.command.max_yaw_vel = 1.0
    config.command.min_height = 0.67
    config.command.max_height = 0.67
    config.command.still_proportion = 0.2
    config.command.gait_frequency_range = [1.0, 1.5]

    config.domain_randomization.sampling_type = "step_probability_and_reset"
    config.domain_randomization.sampling_probability = 0.0
    config.domain_randomization.action_delay.type = "none"
    config.domain_randomization.initial_state.type = "default"
    config.domain_randomization.joint_dropout.type = "none"
    config.domain_randomization.unseen_robot.type = "none"

    config.domain_randomization.mujoco_model.type = "booster"
    config.domain_randomization.mujoco_model.randomize_gravity = True
    config.domain_randomization.mujoco_model.gravity_range = [9.51, 10.11]
    config.domain_randomization.mujoco_model.randomize_model_geom_friction_tangential = False
    config.domain_randomization.mujoco_model.model_geom_friction_tangential_range = [0.5, 1.5]
    config.domain_randomization.mujoco_model.randomize_model_geom_friction_torsional = False
    config.domain_randomization.mujoco_model.model_geom_friction_torsional_range = [0.1, 0.3]
    config.domain_randomization.mujoco_model.randomize_model_geom_friction_rolling = False
    config.domain_randomization.mujoco_model.model_geom_friction_rolling_range = [0.00008, 0.00012]
    config.domain_randomization.mujoco_model.randomize_floor_geom_friction_tangential = True
    config.domain_randomization.mujoco_model.floor_geom_friction_tangential_range = [0.5, 1.5]
    config.domain_randomization.mujoco_model.randomize_floor_geom_friction_torsional = True
    config.domain_randomization.mujoco_model.floor_geom_friction_torsional_range = [0.1, 0.3]
    config.domain_randomization.mujoco_model.randomize_floor_geom_friction_rolling = True
    config.domain_randomization.mujoco_model.floor_geom_friction_rolling_range = [0.00008, 0.00012]
    config.domain_randomization.mujoco_model.randomize_geom_damping = False
    config.domain_randomization.mujoco_model.geom_damping_range = [100.0, 500.0]
    config.domain_randomization.mujoco_model.randomize_geom_stiffness = False
    config.domain_randomization.mujoco_model.geom_stiffness_range = [100000.0, 300000.0]

    config.domain_randomization.seen_robot.type = "booster"
    config.domain_randomization.seen_robot.randomize_joint_damping = True
    config.domain_randomization.seen_robot.joint_damping_range = [0.005, 0.015]
    config.domain_randomization.seen_robot.randomize_joint_friction_loss = True
    config.domain_randomization.seen_robot.joint_friction_loss_range = [0.0, 0.5]
    config.domain_randomization.seen_robot.randomize_joint_armature = True
    config.domain_randomization.seen_robot.joint_armature_range = [0.007, 0.013]
    config.domain_randomization.seen_robot.randomize_com_displacement = True
    config.domain_randomization.seen_robot.com_displacement_range = [-0.05, 0.05]
    config.domain_randomization.seen_robot.randomize_link_mass = True
    config.domain_randomization.seen_robot.link_mass_multiplier_range = {
        "root_body": [0.8, 1.2],
        "other_bodies": [0.9, 1.1],
    }
    config.domain_randomization.seen_robot.add_p_gains_noise = True
    config.domain_randomization.seen_robot.add_d_gains_noise = True
    config.domain_randomization.seen_robot.p_gains_noise_scale = 0.15
    config.domain_randomization.seen_robot.d_gains_noise_scale = 0.15

    config.domain_randomization.observation_noise.type = "booster"
    config.domain_randomization.observation_noise.add_joint_pos_noise = True
    config.domain_randomization.observation_noise.joint_pos_noise_scale = 0.03
    config.domain_randomization.observation_noise.add_joint_vel_noise = True
    config.domain_randomization.observation_noise.joint_vel_noise_scale = 0.3
    config.domain_randomization.observation_noise.add_gravity_noise = True
    config.domain_randomization.observation_noise.gravity_noise_scale = 0.015
    config.domain_randomization.observation_noise.add_free_joint_lin_vel_noise = False
    config.domain_randomization.observation_noise.lin_vel_noise_scale = 0.1
    config.domain_randomization.observation_noise.add_free_joint_ang_vel_noise = True
    config.domain_randomization.observation_noise.ang_vel_noise_scale = 0.2
    config.domain_randomization.observation_noise.add_policy_ang_vel_noise = True
    config.domain_randomization.observation_noise.policy_ang_vel_noise_scale = 0.2
    config.domain_randomization.observation_noise.normalize_observations = True
    config.domain_randomization.observation_noise.norm_factors = {
        "joint_pos": 1.0,
        "joint_vel": 0.1,
        "gravity": 1.0,
        "lin_vel": 1.0,
        "ang_vel": 1.0,
    }

    config.domain_randomization.perturbation.type = "booster"
    config.domain_randomization.perturbation.sampling_type = "booster_step_probability_and_reset"
    config.domain_randomization.perturbation.sampling_probability = 0.004
    config.domain_randomization.perturbation.kick_robots = True
    config.domain_randomization.perturbation.kick_min_vel = 0.0
    config.domain_randomization.perturbation.kick_max_vel = 0.4
    config.domain_randomization.perturbation.kick_prob = 0.004
    config.domain_randomization.perturbation.kick_at_reset = True

    config.policy_exteroceptive_observation_type = "none"
    config.critic_exteroceptive_observation_type = "none"
    config.reward.update({
        "type": "humanoid",
        "profile": "booster",
        "critical_fixed_coeff": 1.0,
        "style_fixed_coeff": 1.0,
        "gait_fixed_coeff": 1.0,
        "tracking_xy_velocity_command_coeff": 0.0,
        "tracking_yaw_velocity_command_coeff": 0.0,
        "alive_clipped_coeff": 0.0,
        "alive_unclipped_coeff": 0.0,
        "survival": 0.25,
        "tracking_w_exp_linvel_x": 4.0,
        "tracking_w_sum_linvel_x": 1.5,
        "tracking_w_exp_linvel_y": 4.0,
        "tracking_w_sum_linvel_y": 1.5,
        "tracking_w_exp_angvel": 4.0,
        "tracking_w_sum_angvel": 1.0,
        "tracking_nominal_joint_pos_exp": 4.0,
        "tracking_nominal_joint_pos_coeff": 0.0,
        "tracking_nominal_joint_pos_names": [
            "AAHead_yaw", "Head_pitch",
            "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
            "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
            "Waist", "Right_Ankle_Roll", "Left_Ankle_Roll",
        ],
        "feet_swing_coeff": 6.0,
        "feet_swing_period": 0.2,
        "joint_deviation_l1_coeff": -0.3,
        "base_height_coeff": 10.0,
        "roll_pitch_pos_coeff": 5.0,
        "orientation_coeff": -5.0,
        "joint_torque_coeff": 2.0e-4,
        "power_draw_penalty_coeff": 2.0e-3,
        "energy_coeff": -2.0e-3,
        "z_velocity_coeff": 0.0,
        "z_vel_coeff": -0.0,
        "roll_pitch_vel_coeff": 0.2,
        "joint_velocity_coeff": 9.0e-4,
        "joint_vel_coeff": -9.0e-4,
        "joint_acceleration_coeff": 1.0e-7,
        "joint_acc_coeff": -1.0e-7,
        "root_acc_coeff": -1.0e-4,
        "action_rate_coeff": 1.0,
        "action_smoothness_coeff": 0.0,
        "standing_leg_joint_velocity_coeff": 0.2,
        "joint_position_limit_scale": 0.98,
        "soft_joint_position_limit": 0.98,
        "joint_position_limit_coeff": 1.0,
        "actuator_joint_velocity_limit_coeff": 0.0,
        "actuator_joint_nominal_diff_coeff": 0.0,
        "feet_slip_coeff": -0.1,
        "foot_slip_coeff": 0.1,
        "foot_air_time_coeff": 0.0,
        "symmetry_air_coeff": 0.0,
        "contact_count_coeff": 0.0,
        "foot_stance_time_coeff": 0.0,
        "foot_clearance_coeff": 0.0,
        "foot_lift_bonus_coeff": 0.0,
        "all_feet_off_ground_coeff": 0.0,
        "collision_coeff": 0.0,
        "foot_z_velocity_coeff": 0.0,
        "foot_flat_contact_coeff": 0.0,
        "feet_yaw_diff_coeff": -1.0,
        "feet_yaw_mean_coeff": -1.0,
        "feet_roll_coeff": -4.0,
        "feet_distance_target": 0.2,
        "feet_distance_coeff": 0.0,
        "air_time_max": 0.3,
        "air_time_coeff": 0.0,
        "no_fly_coeff": 0.0,
        "impact_threshold": 150.0,
        "impact_coeff": 0.0,
    })
    config.termination.type = "booster"
    config.termination.min_height = 0.3
    config.termination.max_height = 1.0
    config.terrain.type = "plane"


def apply_boosterball_defaults(config):
    if not config.get("booster_defaults_applied", False):
        apply_booster_defaults(config)

    config.boosterball_defaults_applied = True
    config.ball_plate.update({
        "enabled": True,
        "include_observations": True,
        "plate_home_pos": [0.355, 0.0, 0.853],
        "plate_home_quat": [1.0, 0.0, 0.0, 0.0],
        "plate_size": [0.18, 0.24, 0.008],
        "ball_home_pos": [0.355, 0.0, 0.901],
        "left_fist_body": "left_hand_link",
        "right_fist_body": "right_hand_link",
        "left_fist_pos": [0.0, 0.19, 0.0],
        "right_fist_pos": [0.0, -0.19, 0.0],
        "use_capsule_hand": False,
        "add_support_fist_geoms": False,
        "support_contact_geom_names": ["left_forearm_col", "right_forearm_col"],
        "torso_contact_geom_names": ["torso_col"],
        "preserve_contact_geom_names": ["left_forearm_col", "right_forearm_col", "torso_col"],
        "fist_radius": 0.035,
        "fist_half_length": 0.055,
        "torso_contact_body": "Trunk",
        "torso_contact_pos": [0.07, 0.0, 0.16],
        "torso_contact_size": 0.16,
        "left_upper_arm_contact_body": "AL3",
        "right_upper_arm_contact_body": "AR3",
        "left_upper_arm_contact_fromto": [0.0, -0.03, 0.0, 0.0, 0.17, 0.0],
        "right_upper_arm_contact_fromto": [0.0, 0.03, 0.0, 0.0, -0.17, 0.0],
        "left_forearm_contact_body": "left_hand_link",
        "right_forearm_contact_body": "right_hand_link",
        "left_forearm_contact_fromto": [0.0, 0.0, 0.0, 0.0, 0.22, 0.0],
        "right_forearm_contact_fromto": [0.0, 0.0, 0.0, 0.0, -0.22, 0.0],
        "upper_arm_contact_radius": 0.055,
        "forearm_contact_radius": 0.04,
        "plate_torso_contact_friction": [1.5, 1.5, 0.005, 0.0001, 0.0001],
        "plate_arm_contact_friction": [1.5, 1.5, 0.005, 0.0001, 0.0001],
        "plate_support_contact_friction": [3.0, 3.0, 0.005, 0.0001, 0.0001],
        "plate_support_contact_dim": 6,
        "plate_support_clearance": 0.0,
        "plate_robot_contact_solref": [0.005, 1.0],
        "plate_robot_contact_solimp": [0.99, 0.999, 0.0001, 0.5, 2.0],
        "plate_robot_contact_margin": 0.001,
    })
    config.ball_plate.initial_joint_positions = config_dict.ConfigDict({
        "Left_Shoulder_Pitch": -1.05,
        "Left_Shoulder_Roll": -1.1,
        "Left_Elbow_Pitch": 1.0,
        "Left_Elbow_Yaw": -1.4,
        "Right_Shoulder_Pitch": -1.05,
        "Right_Shoulder_Roll": 1.1,
        "Right_Elbow_Pitch": 1.0,
        "Right_Elbow_Yaw": 1.4,
    })
    config.reward.update({
        "type": "humanoidball",
        "profile": "booster",
        "ball_plate_centering_coeff": 2.0,
        "ball_plate_center_bonus_coeff": 2.0,
        "ball_plate_center_bonus_temperature": 0.01,
        "ball_plate_velocity_coeff": 0.1,
        "ball_plate_on_plate_coeff": 1.0,
        "ball_plate_alive_coeff": 0.2,
        "ball_plate_drop_penalty_coeff": 10.0,
    })
    config.domain_randomization.observation_noise.type = "ball"
    config.termination.type = "boosterball"
