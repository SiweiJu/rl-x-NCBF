robot_config = {
    "short_name": "g1",
    "left_foot_site_name": "left_foot",
    "right_foot_site_name": "right_foot",
    "left_foot_geom_names": ["left_foot"],
    "right_foot_geom_names": ["right_foot"],
    "left_foot_velocity_sensor_name": "left_foot_global_linear_velocity",
    "right_foot_velocity_sensor_name": "right_foot_global_linear_velocity",

    "actuator_joint_max_velocities": [
        32.0, 32.0, 32.0, 20.0, 53.0, 53.0,
        32.0, 32.0, 32.0, 20.0, 53.0, 53.0,
        32.0,
        53.0, 53.0, 53.0, 53.0, 53.0,
        53.0, 53.0, 53.0, 53.0, 53.0
    ],

    "scaling_factor": 0.5,

    "actuator_joints_to_stay_near_nominal": [5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
}
