from __future__ import annotations

import numpy as np


JOINT_NAMES = [
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


DEFAULT_QPOS = np.array([
    0.0, 0.0,
    0.2, -1.35, 0.0, -0.5,
    0.2, 1.35, 0.0, 0.5,
    0.0,
    -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,
], dtype=np.float32)


STIFFNESS = np.array([
    20.0, 20.0,
    20.0, 20.0, 20.0, 20.0,
    20.0, 20.0, 20.0, 20.0,
    200.0,
    200.0, 200.0, 200.0, 200.0, 50.0, 50.0,
    200.0, 200.0, 200.0, 200.0, 50.0, 50.0,
], dtype=np.float32)


DAMPING = np.array([
    0.2, 0.2,
    0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, 0.5,
    5.0,
    5.0, 5.0, 5.0, 5.0, 3.0, 3.0,
    5.0, 5.0, 5.0, 5.0, 3.0, 3.0,
], dtype=np.float32)


TORQUE_LIMIT = np.array([
    7.0, 7.0,
    10.0, 10.0, 10.0, 10.0,
    10.0, 10.0, 10.0, 10.0,
    30.0,
    60.0, 25.0, 30.0, 60.0, 24.0, 15.0,
    60.0, 25.0, 30.0, 60.0, 24.0, 15.0,
], dtype=np.float32)


JOINT_LIMIT_LOW = np.array([
    -1.57, -0.35,
    -3.31, -1.74, -2.27, -2.27,
    -3.31, -1.57, -2.27, -2.27,
    -1.57,
    -1.8, -0.2, -1.0, 0.0, -0.87, -0.44,
    -1.8, -1.57, -1.0, 0.0, -0.87, -0.44,
], dtype=np.float32)


JOINT_LIMIT_HIGH = np.array([
    1.57, 1.22,
    1.22, 1.57, 2.27, 2.27,
    1.22, 1.74, 2.27, 2.27,
    1.57,
    1.57, 1.57, 1.0, 2.34, 0.35, 0.44,
    1.57, 0.2, 1.0, 2.34, 0.35, 0.44,
], dtype=np.float32)


PREPARE_QPOS = np.array([
    0.0, 0.0,
    0.25, -1.4, 0.0, -0.5,
    0.25, 1.4, 0.0, 0.5,
    0.0,
    -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,
    -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,
], dtype=np.float32)

