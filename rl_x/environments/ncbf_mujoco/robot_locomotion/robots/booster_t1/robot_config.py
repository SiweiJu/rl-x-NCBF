import math


_BOOSTER_T1_ACTUATOR_JOINT_NAMES = [
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
]

_BOOSTER_T1_ACTUATOR_MODELS = [
    "DM4310",
    "DM4310",
    "E4310",
    "E4310",
    "E4310",
    "E4310",
    "E4310",
    "E4310",
    "E4310",
    "E4310",
    "E6408",
    "E8112",
    "E6408",
    "E6408",
    "E8116",
    "T1AnkleE4315Parallel",
    "T1AnkleE4315Parallel",
    "E8112",
    "E6408",
    "E6408",
    "E8116",
    "T1AnkleE4315Parallel",
    "T1AnkleE4315Parallel",
]

_BOOSTER_T1_ACTUATOR_MODEL_PARAMS = {
    "DM4310": {
        "effort_limit": 7.0,
        "velocity_limit": 12.57,
        "knee_point_velocity": 41.89,
        "armature": 0.0018,
    },
    "E4310": {
        "effort_limit": 38.3,
        "velocity_limit": 17.59,
        "knee_point_velocity": 7.85,
        "armature": 0.0282528,
    },
    "E6408": {
        "effort_limit": 68.0,
        "velocity_limit": 14.66,
        "knee_point_velocity": 1.88,
        "armature": 0.0478125,
    },
    "E8112": {
        "effort_limit": 96.0,
        "velocity_limit": 16.76,
        "knee_point_velocity": 7.54,
        "armature": 0.0523908,
    },
    "E8116": {
        "effort_limit": 130.0,
        "velocity_limit": 14.66,
        "knee_point_velocity": 6.28,
        "armature": 0.0636012,
    },
    # BoosterTrain's T1 ankle parallel wrapper keeps E4315 torque/speed and
    # doubles the reflected armature for both pitch and roll serial indices.
    "T1AnkleE4315Parallel": {
        "effort_limit": 76.0,
        "velocity_limit": 12.57,
        "knee_point_velocity": 2.62,
        "armature": 2.0 * 0.0339552,
    },
}

_BOOSTER_T1_NATURAL_FREQUENCY_HZ = 10.0
_BOOSTER_T1_DAMPING_RATIO = 2.0
_BOOSTER_T1_OMEGA = 2.0 * math.pi * _BOOSTER_T1_NATURAL_FREQUENCY_HZ


def _booster_t1_actuator_value(name):
    return [
        _BOOSTER_T1_ACTUATOR_MODEL_PARAMS[model_name][name]
        for model_name in _BOOSTER_T1_ACTUATOR_MODELS
    ]


BOOSTER_T1_ACTUATOR_JOINT_EFFORT_LIMITS = _booster_t1_actuator_value("effort_limit")
BOOSTER_T1_ACTUATOR_JOINT_VELOCITY_LIMITS = _booster_t1_actuator_value("velocity_limit")
BOOSTER_T1_ACTUATOR_JOINT_KNEE_POINT_VELOCITIES = _booster_t1_actuator_value("knee_point_velocity")
BOOSTER_T1_ACTUATOR_JOINT_ARMATURES = _booster_t1_actuator_value("armature")
BOOSTER_T1_ACTUATOR_JOINT_STIFFNESS = [
    armature * _BOOSTER_T1_OMEGA ** 2
    for armature in BOOSTER_T1_ACTUATOR_JOINT_ARMATURES
]
BOOSTER_T1_ACTUATOR_JOINT_DAMPING = [
    2.0 * _BOOSTER_T1_DAMPING_RATIO * armature * _BOOSTER_T1_OMEGA
    for armature in BOOSTER_T1_ACTUATOR_JOINT_ARMATURES
]
BOOSTER_T1_ACTION_SCALE = [
    0.25 * effort / stiffness
    for effort, stiffness in zip(
        BOOSTER_T1_ACTUATOR_JOINT_EFFORT_LIMITS,
        BOOSTER_T1_ACTUATOR_JOINT_STIFFNESS,
    )
]


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

    "actuator_joint_names": _BOOSTER_T1_ACTUATOR_JOINT_NAMES,

    "actuator_joint_max_velocities": BOOSTER_T1_ACTUATOR_JOINT_VELOCITY_LIMITS,
    "actuator_joint_effort_limits": BOOSTER_T1_ACTUATOR_JOINT_EFFORT_LIMITS,
    "actuator_joint_velocity_limits": BOOSTER_T1_ACTUATOR_JOINT_VELOCITY_LIMITS,
    "actuator_joint_knee_point_velocities": BOOSTER_T1_ACTUATOR_JOINT_KNEE_POINT_VELOCITIES,
    "actuator_joint_armatures": BOOSTER_T1_ACTUATOR_JOINT_ARMATURES,
    "actuator_joint_stiffness": BOOSTER_T1_ACTUATOR_JOINT_STIFFNESS,
    "actuator_joint_damping": BOOSTER_T1_ACTUATOR_JOINT_DAMPING,

    "use_torque_pd_control": True,
    "scaling_factor": BOOSTER_T1_ACTION_SCALE,

    "actuator_joints_to_stay_near_nominal": [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 22
    ],
}
