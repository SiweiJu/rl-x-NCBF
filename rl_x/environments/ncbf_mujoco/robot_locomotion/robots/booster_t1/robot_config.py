robot_config = {
    "short_name": "booster_t1",
    "root_body_name": "Trunk",
    "root_free_joint_name": "root",
    "left_foot_site_name": "left_foot",
    "right_foot_site_name": "right_foot",
    "left_foot_geom_names": ["left_foot_1_col", "left_foot_2_col"],
    "right_foot_geom_names": ["right_foot_1_col", "right_foot_2_col"],
    "left_foot_velocity_sensor_name": "left_foot_global_linvel",
    "right_foot_velocity_sensor_name": "right_foot_global_linvel",

    "actuator_joint_names": [
        "AAHead_yaw",
        "Head_pitch",
        "Left_Shoulder_Pitch",
        "Left_Shoulder_Roll",
        "Left_Elbow_Pitch",
        "Left_Elbow_Yaw",
        "Right_Shoulder_Pitch",
        "Right_Shoulder_Roll",
        "Right_Elbow_Pitch",
        "Right_Elbow_Yaw",
        "Waist",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Knee_Pitch",
        "Right_Ankle_Pitch",
        "Right_Ankle_Roll",
    ],

    # The source Booster training config does not impose explicit joint velocity
    # clipping. Use a high limit so RL-X's safety clipping does not change that.
    "actuator_joint_max_velocities": [100.0] * 23,

    "scaling_factor": 1.0,

    "actuator_joints_to_stay_near_nominal": [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 22
    ],
}
