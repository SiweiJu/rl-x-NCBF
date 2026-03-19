import numpy as np
import mujoco
from mujoco import viewer

XML_PATH = "box.xml"  # <- change this

OBSTACLE_BODY = "obstacle"
# sample region in world XY
X_RANGE = (-1.0, 1.0)
Y_RANGE = ( 0.5, 2.0)

# box half-height (must match size z in XML)
BOX_HALF_HEIGHT = 0.20

rng = np.random.default_rng(0)

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# precompute mocap index for the obstacle body
bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBSTACLE_BODY)
mid = model.body_mocapid[bid]
assert mid >= 0, f"Body '{OBSTACLE_BODY}' is not mocap. Did you set mocap='true'?"

def randomize_obstacle():
    x = rng.uniform(*X_RANGE)
    y = rng.uniform(*Y_RANGE)
    z = BOX_HALF_HEIGHT  # sit on plane z=0
    data.mocap_pos[mid] = np.array([x, y, z], dtype=np.float64)
    data.mocap_quat[mid] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

def reset_episode():
    mujoco.mj_resetData(model, data)
    randomize_obstacle()
    mujoco.mj_forward(model, data)

reset_episode()

with viewer.launch_passive(model, data) as v:
    steps = 0
    while v.is_running():
        # Example: re-randomize every 2 seconds
        if steps % int(2.0 / model.opt.timestep) == 0:
            reset_episode()

        mujoco.mj_step(model, data)
        v.sync()
        steps += 1