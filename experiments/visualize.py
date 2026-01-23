import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path

# === Paths ===
ROLLOUT_PATHS = [
    "rollouts/episode_rollout.npz",  # main robot
    "rollouts/episode_rollout1.npz"   # ghost robot
]

XML_PATH = "rl_x/environments/ncbf_mujoco/robot_locomotion/robots/unitree_go2/data/two_bots.xml"

# === Load rollouts ===
rollouts = [np.load(path) for path in ROLLOUT_PATHS]
qpos_trajs = [r["qpos"] for r in rollouts]
qvel_trajs = [r["qvel"] for r in rollouts]

# === Load model and create data struct ===
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Load the terrain heights
terrain_bumps = np.load("rollouts/terrain_data.npy")
print(f"Terrain data shape: {terrain_bumps.shape}") # Should be (80, 80) or 6400 total

# Inject heights into the model
model.hfield_data[:] = terrain_bumps.flatten()

# Compute DOF offset for ghost robot
n_dof_main = qpos_trajs[0].shape[1]  # number of joints in main robot

# Initialize forward kinematics
mujoco.mj_forward(model, data)

# Use model constants to be 100% safe
nq = model.nq // 2  # Should be 19
nv = model.nv // 2  # Should be 18

# PRE-LOOP: Make ghost transparent
ghost_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_ghost")
for i in range(model.ngeom):
    if model.geom_bodyid[i] >= ghost_body_id: # Simple way: ghost is defined after main
        model.geom_rgba[i, 3] = 0.3

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set camera to track the main robot
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
    viewer.cam.fixedcamid = cam_id
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    n_steps = max([len(traj) for traj in qpos_trajs])

    for t in range(n_steps):
        # 1. Main Robot
        idx0 = min(t, len(qpos_trajs[0]) - 1)
        data.qpos[0:nq] = qpos_trajs[0][idx0]
        data.qvel[0:nv] = qvel_trajs[0][idx0]

        # --- Ghost Robot ---
        idx1 = min(t, len(qpos_trajs[1]) - 1)
        data.qpos[nq : 2*nq] = qpos_trajs[1][idx1]
        data.qvel[nv : 2*nv] = qvel_trajs[1][idx1]

        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.001)

        # 3. Propagate changes to physics
        try:
            mujoco.mj_forward(model, data)
        except Exception as e:
            print(f"Physics error at step {t}: {e}")
            break

        viewer.sync()
        time.sleep(0.02)

        if not viewer.is_running():
            break
