robot_config = {
    "short_name": "go2",
    "left_foot_site_name": "FL_foot",
    "right_foot_site_name": "FR_foot",
    "left_foot_geom_names": ["FL_foot", "RL_foot"],
    "right_foot_geom_names": ["FR_foot", "RR_foot"],
    "left_foot_velocity_sensor_name": "FL_foot_global_linear_velocity",
    "right_foot_velocity_sensor_name": "FR_foot_global_linear_velocity",

    "actuator_joint_max_velocities": [
        30.1, 30.1, 15.7,
        30.1, 30.1, 15.7,
        30.1, 30.1, 15.7,
        30.1, 30.1, 15.7
    ],

    "scaling_factor": 0.3,

    "actuator_joints_to_stay_near_nominal": []
}
