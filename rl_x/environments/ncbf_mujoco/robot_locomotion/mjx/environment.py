from copy import deepcopy
from pathlib import Path
from functools import partial
import mujoco
from mujoco import mjx
from dm_control import mjcf
import pygame
import numpy as np
from scipy.spatial.transform import Rotation as Rotation_NP
from jax.scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp


from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.state import State
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.box_space import BoxSpace
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.viewer import MujocoViewer
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.control_functions.handler import get_control_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.command_functions.handler import get_command_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.obstacle_scan import (
    OBSTACLE_MOTIONS, OBSTACLE_SHAPES, PRIMITIVE_BOX, PRIMITIVE_CIRCLE, PRIMITIVE_NONE,
    primitive_range_scan, primitive_signed_distance, primitives_in_world)
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.initial_state_functions.handler import get_initial_state_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.sampling_functions.handler import get_sampling_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.reward_functions.handler import get_reward_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.termination_functions.handler import get_termination_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.action_delay_functions.handler import get_domain_randomization_action_delay_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.mujoco_model_functions.handler import get_domain_randomization_mujoco_model_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.seen_robot_functions.handler import get_domain_randomization_seen_robot_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.unseen_robot_functions.handler import get_domain_randomization_unseen_robot_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.perturbation_functions.handler import get_domain_randomization_perturbation_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.observation_noise_functions.handler import get_observation_noise_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.domain_randomization.joint_dropout_functions.handler import get_joint_dropout_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.exteroceptive_observation_functions.handler import get_exteroceptive_observation_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.terrain_functions.handler import get_terrain_function


class LocomotionEnv:
    def __init__(self, robot_config, runner_mode, render, env_config, nr_envs):
        
        self.robot_config = robot_config
        self.runner_mode = runner_mode
        self.should_render = render
        self.env_config = env_config
        self.add_goal_arrow = env_config["add_goal_arrow"]
        self.ball_plate_config = env_config.get("ball_plate", {})
        self.use_ball_plate = bool(self.ball_plate_config.get("enabled", False))
        self.stand_after_ball_plate_drop = self.use_ball_plate and bool(self.ball_plate_config.get("stand_after_drop"))
        self.ball_plate_post_drop_truncation_seconds = float(self.ball_plate_config.get("post_drop_truncation_seconds"))
        self.use_capsule_hand = self.use_ball_plate and bool(self.ball_plate_config.get("use_capsule_hand"))
        self.include_ball_plate_observations = self.use_ball_plate and bool(self.ball_plate_config.get("include_observations"))
        self.record_transition_physics_state = bool(env_config.get("record_transition_physics_state", False))
        # F13 (default off, it changes observations): run mjx.forward after the reset writes and after the post-physics
        # DR / shock writes, so site poses and sensors (IMU) match qpos/qvel. Off = legacy: the first observation of an
        # episode and the observation right after a shock carry stale IMU values.
        self.forward_after_state_writes = bool(env_config.get("forward_after_state_writes", False))
        # T11: record the world-model state x^- / x^+ (wm_state.extract_state), the DR vector xi after the DR step,
        # the delayed (applied) action and the realized action delay in info["wm/..."] (project plan 4.2-4.4).
        self.record_wm_state = bool(env_config.get("record_wm_state", False))
        self.obstacle_config = env_config.get("obstacles", {})
        self.use_obstacles = bool(self.obstacle_config.get("enabled", False))
        self.nr_obstacles = int(self.obstacle_config.get("nr_obstacles", 0)) if self.use_obstacles else 0
        # Which obstacle information each network sees: "state" (privileged relative position, velocity
        # and radius), "scan" (2D range scan around the base), or "none".
        self.obstacle_policy_observation = self.obstacle_config.get("policy_observation", "state") if self.use_obstacles else "none"
        self.obstacle_ncbf_observation = self.obstacle_config.get("ncbf_observation", "policy") if self.use_obstacles else "none"
        self.obstacle_value_observation = self.obstacle_config.get("value_observation", "policy") if self.use_obstacles else "none"
        if self.obstacle_ncbf_observation == "policy":
            self.obstacle_ncbf_observation = self.obstacle_policy_observation
        if self.obstacle_value_observation == "policy":
            self.obstacle_value_observation = self.obstacle_policy_observation
        for name, value, allowed in (
            ("policy_observation", self.obstacle_policy_observation, ("state", "scan", "none")),
            ("ncbf_observation", self.obstacle_ncbf_observation, ("state", "scan", "none")),
            ("value_observation", self.obstacle_value_observation, ("state", "scan", "state_and_scan", "none")),
        ):
            if value not in allowed:
                raise ValueError(f"environment.obstacles.{name} must be one of {allowed} (or 'policy'), got {value!r}")
        obstacle_observation_users = (self.obstacle_policy_observation, self.obstacle_ncbf_observation, self.obstacle_value_observation)
        self.use_obstacle_state_observation = any("state" in user for user in obstacle_observation_users)
        self.use_obstacle_scan_observation = any("scan" in user for user in obstacle_observation_users)
        self.nr_obstacle_observations = 5 * self.nr_obstacles if self.use_obstacle_state_observation else 0  # per obstacle: relative xy, relative velocity xy, radius
        scan_config = self.obstacle_config.get("scan", {})
        self.obstacle_scan_nr_rays = int(scan_config.get("nr_rays", 64)) if self.use_obstacle_scan_observation else 0
        self.obstacle_scan_max_range = float(scan_config.get("max_range", 5.0))
        fov = jnp.deg2rad(float(scan_config.get("fov_degrees", 360.0)))
        nr_rays = max(self.obstacle_scan_nr_rays, 1)
        # Ray angles in the yaw frame, evenly spaced over the field of view and centred on the forward axis.
        self.obstacle_scan_angles = -fov / 2.0 + (jnp.arange(nr_rays) + 0.5) * fov / nr_rays
        self.obstacle_scan_buffer_length = int(scan_config.get("latency_steps_range", (0, 0))[1]) + 1

        def normalized_weights(weights, names, default, what):
            weights = dict(default if weights is None else weights)
            unknown = set(weights) - set(names)
            if unknown:
                raise ValueError(f"environment.obstacles.{what}.weights: unknown entries {sorted(unknown)}, expected {names}")
            values = np.array([float(weights.get(name, 0.0)) for name in names])
            if np.any(values < 0.0) or values.sum() <= 0.0:
                raise ValueError(f"environment.obstacles.{what}.weights must be non-negative with a positive sum")
            return jnp.asarray(values / values.sum())

        self.obstacle_shape_probs = normalized_weights(self.obstacle_config.get("shapes", {}).get("weights"), OBSTACLE_SHAPES, {"circle": 1.0}, "shapes")
        self.obstacle_motion_probs = normalized_weights(self.obstacle_config.get("motions", {}).get("weights"), OBSTACLE_MOTIONS, {"constant": 1.0}, "motions")
        self.nr_envs = nr_envs
        self.nr_history_steps = env_config["nr_history_steps"]
        safety_config = env_config.get("safety", {})
        self.safe_time_limit_seconds = safety_config.get("safe_time_limit_seconds")
        self.root_body_name = robot_config.get("root_body_name", "trunk")

        xml_path = (self.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
        xml_handle = mjcf.from_path(xml_path)
        if self.use_capsule_hand:
            self._remove_capsule_hand_visual_geoms_from_xml(xml_handle)

        # Remove all unnecessary assets, materials, meshes and geoms during training
        # This removes all geoms besides feet and floor, if the contacts for other geoms should be enabled this needs to be changed
        # Also if you want to render the training, the lines can be commented out
        for texture in xml_handle.asset.find_all("texture"):
            texture.remove()
        for material in xml_handle.asset.find_all("material"):
            material.remove()
        for mesh in xml_handle.asset.find_all("mesh"):
            mesh.remove()
        ball_plate_preserved_geom_names = set()
        if self.use_ball_plate:
            ball_plate_preserved_geom_names.update(self.ball_plate_config.get("preserve_contact_geom_names", []))
            ball_plate_preserved_geom_names.update(self.ball_plate_config.get("support_contact_geom_names", []))
            ball_plate_preserved_geom_names.update(self.ball_plate_config.get("torso_contact_geom_names", []))

        for geom in xml_handle.find_all("geom"):
            is_foot_geom = geom.name and "foot" in geom.name
            is_floor_geom = geom.name == "floor"
            is_reward_collision_sphere_geom = geom.dclass and geom.dclass.dclass == "reward_collision_sphere"
            is_ball_plate_contact_geom = geom.name in ball_plate_preserved_geom_names
            if not is_foot_geom and not is_floor_geom and not is_reward_collision_sphere_geom and not is_ball_plate_contact_geom:
                geom.remove()
            if is_floor_geom:
                geom.material = ""

        if "hfield" in env_config["terrain"]["type"]:
            xml_handle.asset.insert("hfield", 0, name="empty_hfield", file="default_hfield_80.png", size="4 4 30.0 0.125")
            floor = xml_handle.find("geom", "floor")
            floor.type = "hfield"
            floor.hfield = "empty_hfield"

        if self.use_ball_plate:
            self._make_ball_plate_contact_geoms_mjx_compatible(xml_handle)
            self._add_ball_plate_to_xml(xml_handle)
        
        if self.should_render and self.add_goal_arrow:
            trunk = xml_handle.find("body", self.root_body_name)
            trunk.add("body", name="dir_arrow", pos="0 0 0.15")
            dir_vec = xml_handle.find("body", "dir_arrow")
            dir_vec.add("site", name="dir_arrow_ball", type="sphere", size=".02", pos="-.1 0 0")
            dir_vec.add("site", name="dir_arrow", type="cylinder", size=".01", fromto="0 0 -.1 0 0 .1")
        
        self.initial_mj_model = mujoco.MjModel.from_xml_string(xml=xml_handle.to_xml_string(), assets=xml_handle.get_assets())
        self.initial_mj_model.opt.timestep = env_config["timestep"]
        self.data = mujoco.MjData(self.initial_mj_model)
        self.initial_mjx_model = mjx.put_model(self.initial_mj_model)
        self.mjx_data = mjx.make_data(self.initial_mjx_model)
        self.mjx_data = mjx.forward(self.initial_mjx_model, self.mjx_data)  # Necessary because of error with toddlerbot
        self.c_model = deepcopy(self.initial_mj_model)
        self.c_data = mujoco.MjData(self.c_model)
        self.c_data.qpos = self.initial_mj_model.keyframe("home").qpos
        mujoco.mj_forward(self.c_model, self.c_data)
        
        self.imu_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.trunk_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, self.root_body_name)
        self.actuator_joint_max_velocities = jnp.array(robot_config["actuator_joint_max_velocities"])
        self.initial_qpos = jnp.array(self.initial_mj_model.keyframe("home").qpos)
        self.initial_imu_orientation_rotation_inverse = Rotation.from_matrix(self.c_data.site_xmat[self.imu_site_id].reshape(3, 3)).inv()
        self.initial_imu_height = self.c_data.site_xpos[self.imu_site_id, 2]
        self.actuator_joint_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, actuator_trnid[0]) for actuator_trnid in self.initial_mj_model.actuator_trnid]
        self.actuator_joint_mask_joints = jnp.array([self.initial_mj_model.joint(joint_name).id for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qpos = jnp.array([self.initial_mj_model.joint(joint_name).qposadr[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qvel = jnp.array([self.initial_mj_model.joint(joint_name).dofadr[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_qposadr_to_nominal_index = {
            int(qposadr): nominal_index
            for nominal_index, qposadr in enumerate(np.asarray(self.actuator_joint_mask_qpos))
        }
        self.nr_actuator_joints = len(self.actuator_joint_names)
        self.nr_joints = self.initial_mj_model.njnt

        imu_angular_velocity_sensor_id = self.initial_mj_model.sensor("imu_angular_velocity").id
        self.imu_angular_velocity_sensor_adr = self.initial_mj_model.sensor_adr[imu_angular_velocity_sensor_id]
        self.imu_angular_velocity_sensor_dim = self.initial_mj_model.sensor_dim[imu_angular_velocity_sensor_id]
        imu_linear_velocity_sensor_id = self.initial_mj_model.sensor("imu_linear_velocity").id
        self.imu_linear_velocity_sensor_adr = self.initial_mj_model.sensor_adr[imu_linear_velocity_sensor_id]
        self.imu_linear_velocity_sensor_dim = self.initial_mj_model.sensor_dim[imu_linear_velocity_sensor_id]

        geom_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) for geom_id in range(self.initial_mj_model.ngeom)]
        self.feet_names = [geom_name for geom_name in geom_names if geom_name and "foot" in geom_name]
        self.foot_geom_indices = jnp.array([mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, foot_name) for foot_name in self.feet_names])
        self.nr_feet = len(self.feet_names)

        feet_xpos = self.c_data.geom_xpos[self.foot_geom_indices]
        x_pos, y_pos, z_pos = feet_xpos[:, 0], feet_xpos[:, 1], feet_xpos[:, 2]
        abs_y_feet_xpos = np.array([x_pos, jnp.abs(y_pos), z_pos]).T
        distances_between_abs_y_feet = np.linalg.norm(abs_y_feet_xpos[:, None] - abs_y_feet_xpos[None], axis=-1)
        min_dist_indices = np.argmin(distances_between_abs_y_feet + np.eye(len(abs_y_feet_xpos)) * 1000, axis=1)
        feet_symmetry_set = set([(min(i, min_dist_indices[i]), max(i, min_dist_indices[i])) for i in range(len(min_dist_indices)) if min_dist_indices[min_dist_indices[i]] == i])
        self.feet_symmetry_pairs = jnp.array([list(pair) for pair in feet_symmetry_set])
        self.body_ids_of_feet = jnp.array([self.initial_mj_model.geom(geom_id).bodyid[0] for geom_id in self.foot_geom_indices])
        all_feet_are_sphere = jnp.all(self.initial_mjx_model.geom_type[self.foot_geom_indices] == 2)
        all_feet_are_capsule = jnp.all(self.initial_mjx_model.geom_type[self.foot_geom_indices] == 3)
        all_feet_are_box = jnp.all(self.initial_mjx_model.geom_type[self.foot_geom_indices] == 6)
        if not bool(all_feet_are_sphere | all_feet_are_capsule | all_feet_are_box):
            raise ValueError("Foot geoms are not all of type sphere, capsule or box.")
        self.foot_type = "sphere" if all_feet_are_sphere else ("capsule" if all_feet_are_capsule else "box")
        self.foot_type_int = 0 if self.foot_type == "sphere" else (2 if self.foot_type == "capsule" else 1)

        feet_global_linear_velocity_sensor_ids = [self.initial_mj_model.sensor(f"{foot_name}_global_linear_velocity").id for foot_name in self.feet_names]
        self.feet_global_linear_velocity_sensor_adrs_start = jnp.array([self.initial_mj_model.sensor_adr[sensor_id] for sensor_id in feet_global_linear_velocity_sensor_ids])
        self.left_foot_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, robot_config.get("left_foot_site_name", "left_foot"))
        self.right_foot_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, robot_config.get("right_foot_site_name", "right_foot"))
        self.left_foot_geom_indices = jnp.array([
            mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in robot_config.get("left_foot_geom_names", [name for name in self.feet_names if "left" in name])
        ])
        self.right_foot_geom_indices = jnp.array([
            mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in robot_config.get("right_foot_geom_names", [name for name in self.feet_names if "right" in name])
        ])
        left_foot_velocity_sensor_id = self.initial_mj_model.sensor(robot_config.get("left_foot_velocity_sensor_name", "left_foot_global_linvel")).id
        right_foot_velocity_sensor_id = self.initial_mj_model.sensor(robot_config.get("right_foot_velocity_sensor_name", "right_foot_global_linvel")).id
        self.left_foot_velocity_sensor_adr = self.initial_mj_model.sensor_adr[left_foot_velocity_sensor_id]
        self.right_foot_velocity_sensor_adr = self.initial_mj_model.sensor_adr[right_foot_velocity_sensor_id]

        body_to_parentid = jnp.array([self.initial_mj_model.body(body_id).parentid[0] for body_id in range(self.initial_mj_model.nbody)])
        body_to_children_count = jnp.array([jnp.sum(body_to_parentid == body_id) for body_id in range(self.initial_mj_model.nbody)])
        self.body_ids_of_actuator_joints = jnp.array([self.initial_mj_model.joint(joint_name).bodyid[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_nr_direct_child_actuator_joints = body_to_children_count[self.body_ids_of_actuator_joints]

        self.floor_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        if self.use_ball_plate:
            self.ball_plate_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "ball_plate_center")
            self.ball_plate_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "ball_plate")
            self.ball_plate_ball_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "plate_ball")
            self.ball_plate_ball_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "plate_ball_geom")
            self.ball_plate_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ball_plate_geom")
            self.ball_plate_qposadr = self.initial_mj_model.joint("ball_plate_freejoint").qposadr[0]
            self.ball_plate_qveladr = self.initial_mj_model.joint("ball_plate_freejoint").dofadr[0]
            self.ball_plate_ball_qposadr = self.initial_mj_model.joint("plate_ball_freejoint").qposadr[0]
            self.ball_plate_ball_qveladr = self.initial_mj_model.joint("plate_ball_freejoint").dofadr[0]
            self.ball_plate_plate_size = jnp.array(self.ball_plate_config["plate_size"], dtype=jnp.float32)
            self.ball_plate_plate_mass = float(self.ball_plate_config["plate_mass"])
            self.ball_plate_plate_mass_range = jnp.array(
                self.ball_plate_config.get("plate_mass_range", [self.ball_plate_plate_mass, self.ball_plate_plate_mass]),
                dtype=jnp.float32,
            )
            self.ball_plate_plate_size_scale_range = jnp.array(
                self.ball_plate_config.get("plate_size_scale_range", [1.0, 1.0]),
                dtype=jnp.float32,
            )
            self.ball_plate_ball_radius = float(self.ball_plate_config["ball_radius"])
            self.ball_plate_ball_mass = float(self.ball_plate_config["ball_mass"])
            self.ball_plate_ball_radius_range = jnp.array(
                self.ball_plate_config.get("ball_radius_range", [self.ball_plate_ball_radius, self.ball_plate_ball_radius]),
                dtype=jnp.float32,
            )
            self.ball_plate_ball_mass_range = jnp.array(
                self.ball_plate_config.get("ball_mass_range", [self.ball_plate_ball_mass, self.ball_plate_ball_mass]),
                dtype=jnp.float32,
            )
            self.ball_plate_ball_start_offset = jnp.array(self.ball_plate_config["ball_start_offset"], dtype=jnp.float32)
            self.ball_plate_ball_start_offset_xy_range = jnp.array(
                self.ball_plate_config.get("ball_start_offset_xy_range", [0.0, 0.0]),
                dtype=jnp.float32,
            )
            self.ball_plate_ball_initial_linear_velocity_xy_range = jnp.array(
                self.ball_plate_config.get("ball_initial_linear_velocity_xy_range", [0.0, 0.0]),
                dtype=jnp.float32,
            )
            self.ball_plate_ball_initial_angular_velocity_range = jnp.array(
                self.ball_plate_config.get("ball_initial_angular_velocity_range", [0.0, 0.0, 0.0]),
                dtype=jnp.float32,
            )
            self.ball_plate_ball_pair_ids = jnp.array(self._get_ball_plate_pair_ids(["plate_ball_geom"]), dtype=int)
            self.ball_plate_support_pair_ids = jnp.array(self._get_ball_plate_pair_ids(self._get_ball_plate_support_contact_geom_names()), dtype=int)
            self.ball_plate_robot_pair_ids = jnp.array(self._get_ball_plate_pair_ids(self._get_ball_plate_robot_contact_geom_names()), dtype=int)
            all_ball_plate_pair_ids = sorted(set(np.concatenate([
                np.asarray(self.ball_plate_ball_pair_ids, dtype=int),
                np.asarray(self.ball_plate_support_pair_ids, dtype=int),
                np.asarray(self.ball_plate_robot_pair_ids, dtype=int),
            ]).tolist()))
            self.ball_plate_all_pair_ids = jnp.array(all_ball_plate_pair_ids, dtype=int)
            self.ball_plate_nominal_pair_friction = jnp.array(self.initial_mj_model.pair_friction, dtype=jnp.float32)
            self.ball_plate_nominal_pair_solref = jnp.array(self.initial_mj_model.pair_solref, dtype=jnp.float32)
            self.ball_plate_nominal_plate_ball_friction = (
                float(self.initial_mj_model.pair_friction[int(np.asarray(self.ball_plate_ball_pair_ids)[0]), 0])
                if len(self.ball_plate_ball_pair_ids) > 0
                else float(self.ball_plate_config.get("plate_ball_contact_friction", [4.0])[0])
            )
            self.ball_plate_nominal_support_friction = float(self.ball_plate_config.get("plate_support_contact_friction", [3.0])[0])
            self.ball_plate_nominal_robot_friction = float(self.ball_plate_config.get("plate_arm_contact_friction", [1.5])[0])
            self.ball_plate_nominal_contact_timeconst = (
                float(self.initial_mj_model.pair_solref[int(np.asarray(self.ball_plate_all_pair_ids)[0]), 0])
                if len(self.ball_plate_all_pair_ids) > 0
                else float(self.ball_plate_config.get("plate_ball_contact_solref", [0.005])[0])
            )
            self.ball_plate_plate_ball_friction_tangential_range = jnp.array(
                self.ball_plate_config.get("plate_ball_contact_friction_tangential_range", [self.ball_plate_nominal_plate_ball_friction] * 2),
                dtype=jnp.float32,
            )
            self.ball_plate_plate_support_friction_tangential_range = jnp.array(
                self.ball_plate_config.get("plate_support_contact_friction_tangential_range", [3.0, 3.0]),
                dtype=jnp.float32,
            )
            self.ball_plate_plate_robot_friction_tangential_range = jnp.array(
                self.ball_plate_config.get("plate_robot_contact_friction_tangential_range", [1.5, 1.5]),
                dtype=jnp.float32,
            )
            self.ball_plate_contact_timeconst_range = jnp.array(
                self.ball_plate_config.get("plate_contact_timeconst_range", [0.005, 0.005]),
                dtype=jnp.float32,
            )
            self.ball_plate_drop_margin = float(self.ball_plate_config["drop_margin"])
            self.ball_plate_drop_height = float(self.ball_plate_config["drop_height"])
            self.ball_plate_plate_support_clearance = float(self.ball_plate_config["plate_support_clearance"])
            self.ball_plate_tilt_drop_threshold = float(self.ball_plate_config["plate_tilt_drop_threshold"])
            self.ball_plate_left_fist_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, self.ball_plate_config["left_fist_body"])
            self.ball_plate_right_fist_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, self.ball_plate_config["right_fist_body"])
            self.ball_plate_left_fist_pos = jnp.array(self.ball_plate_config["left_fist_pos"], dtype=jnp.float32)
            self.ball_plate_right_fist_pos = jnp.array(self.ball_plate_config["right_fist_pos"], dtype=jnp.float32)
            self.ball_plate_capsule_hand_radius = float(self.ball_plate_config["fist_radius"])
            self.ball_plate_capsule_hand_half_length = float(self.ball_plate_config["fist_half_length"])
            self.ball_plate_capsule_hand_mass = float(self.ball_plate_config.get("capsule_hand_mass", 0.0))
            capsule_hand_radius_range = self.ball_plate_config.get("capsule_hand_radius_range")
            if capsule_hand_radius_range is None:
                capsule_hand_radius_range = np.asarray(
                    self.ball_plate_config.get("capsule_hand_radius_scale_range", [1.0, 1.0]),
                    dtype=float,
                ) * self.ball_plate_capsule_hand_radius
            capsule_hand_half_length_range = self.ball_plate_config.get("capsule_hand_half_length_range")
            if capsule_hand_half_length_range is None:
                capsule_hand_half_length_range = np.asarray(
                    self.ball_plate_config.get("capsule_hand_half_length_scale_range", [1.0, 1.0]),
                    dtype=float,
                ) * self.ball_plate_capsule_hand_half_length
            self.ball_plate_capsule_hand_radius_range = jnp.array(
                capsule_hand_radius_range,
                dtype=jnp.float32,
            )
            self.ball_plate_capsule_hand_half_length_range = jnp.array(
                capsule_hand_half_length_range,
                dtype=jnp.float32,
            )
            self.ball_plate_capsule_hand_mass_range = jnp.array(
                self.ball_plate_config.get("capsule_hand_mass_range", [self.ball_plate_capsule_hand_mass, self.ball_plate_capsule_hand_mass]),
                dtype=jnp.float32,
            )
            if self.use_capsule_hand:
                self.ball_plate_capsule_hand_geom_ids = jnp.array([
                    mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_plate_support_fist"),
                    mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_plate_support_fist"),
                ], dtype=int)
                self.ball_plate_capsule_hand_body_ids = jnp.array([
                    mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_capsule_hand"),
                    mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_capsule_hand"),
                ], dtype=int)
                first_capsule_hand_body_id = int(np.asarray(self.ball_plate_capsule_hand_body_ids)[0])
                self.ball_plate_capsule_hand_nominal_body_inertia = jnp.array(
                    self.initial_mj_model.body_inertia[first_capsule_hand_body_id],
                    dtype=jnp.float32,
                )
            else:
                self.ball_plate_capsule_hand_geom_ids = jnp.array([], dtype=int)
                self.ball_plate_capsule_hand_body_ids = jnp.array([], dtype=int)
                self.ball_plate_capsule_hand_nominal_body_inertia = jnp.zeros(3, dtype=jnp.float32)
            self.nr_ball_plate_observations = 12
        else:
            self.nr_ball_plate_observations = 0

        self.reward_collision_sphere_geom_ids = jnp.array([geom.id for geom in [self.initial_mj_model.geom(geom_id) for geom_id in range(self.initial_mj_model.ngeom)] if geom.group[0] == 5], dtype=jnp.int32)

        self.has_equality_constraints = len(self.initial_mj_model.eq_data) > 0

        self.robot_dimensions_mean = 0.5  # This can be calculated smartly...

        self.env_curriculum_nr_levels = env_config["env_curriculum_nr_levels"]
        self.env_curriculum_level_success_episode_return = env_config["env_curriculum_level_success_episode_return"]

        self.control_function = get_control_function(env_config["control_type"], self)
        self.control_frequency_hz = self.control_function.control_frequency_hz
        self.nr_substeps = int(round(1 / self.control_frequency_hz / env_config["timestep"]))
        self.dt = env_config["timestep"] * self.nr_substeps
        self.horizon = int(round(env_config["episode_length_in_seconds"] * self.control_frequency_hz))
        self.command_function = get_command_function(env_config["command"]["type"], self)
        self.command_observation_size = getattr(self.command_function, "observation_size", 3)
        self.command_sampling_function = get_sampling_function(
            env_config["command"]["sampling_type"],
            self,
            probability=env_config["command"].get("sampling_probability", 0.002),
        )
        self.initial_state_function = get_initial_state_function(env_config["domain_randomization"]["initial_state"]["type"], self)
        self.reward_function = get_reward_function(env_config["reward"]["type"], self)
        self.termination_function = get_termination_function(env_config["termination"]["type"], self)
        self.policy_exteroceptive_observation_function = get_exteroceptive_observation_function(env_config["policy_exteroceptive_observation_type"], self)
        self.critic_exteroceptive_observation_function = get_exteroceptive_observation_function(env_config["critic_exteroceptive_observation_type"], self)
        self.terrain_function = get_terrain_function(env_config["terrain"]["type"], self)
        self.domain_randomization_sampling_function = get_sampling_function(
            env_config["domain_randomization"]["sampling_type"],
            self,
            probability=env_config["domain_randomization"].get("sampling_probability", 0.002),
        )
        self.domain_randomization_action_delay_function = get_domain_randomization_action_delay_function(env_config["domain_randomization"]["action_delay"]["type"], self)
        self.domain_randomization_mujoco_model_function = get_domain_randomization_mujoco_model_function(env_config["domain_randomization"]["mujoco_model"]["type"], self)
        self.domain_randomization_seen_robot_function = get_domain_randomization_seen_robot_function(env_config["domain_randomization"]["seen_robot"]["type"], self)
        self.domain_randomization_unseen_robot_function = get_domain_randomization_unseen_robot_function(env_config["domain_randomization"]["unseen_robot"]["type"], self)
        self.domain_randomization_perturbation_function = get_domain_randomization_perturbation_function(env_config["domain_randomization"]["perturbation"]["type"], self)
        self.domain_randomization_perturbation_sampling_function = get_sampling_function(
            env_config["domain_randomization"]["perturbation"]["sampling_type"],
            self,
            probability=env_config["domain_randomization"]["perturbation"].get("sampling_probability", 0.002),
        )
        self.observation_noise_function = get_observation_noise_function(env_config["domain_randomization"]["observation_noise"]["type"], self)
        self.joint_dropout_function = get_joint_dropout_function(env_config["domain_randomization"]["joint_dropout"]["type"], self)
        
        action_space_size = self.nr_actuator_joints
        actuator_joint_limit_positions = self.initial_mjx_model.jnt_range[self.actuator_joint_mask_joints]
        actuator_joint_nominal_positions = self.initial_qpos[self.actuator_joint_mask_qpos]
        scaling_factor = robot_config["scaling_factor"]
        actuator_joint_limit_positions_normalized = (actuator_joint_limit_positions - actuator_joint_nominal_positions[:, None]) / scaling_factor

        self.single_action_space = BoxSpace(low=actuator_joint_limit_positions_normalized[:, 0], high=actuator_joint_limit_positions_normalized[:, 1], shape=(action_space_size,), dtype=jnp.float32)

        self.single_observation_space = self.get_observation_space()

        self.observation_noise_function.init_attributes()

        if self.should_render:
            self.viewer = MujocoViewer(self.initial_mj_model, self.dt)

            self.dir_arrow_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "dir_arrow")
            self.uses_hfield = self.initial_mj_model.hfield_data.shape[0] != 0
            self.light_xdir = self.c_data.light_xdir
            self.light_xpos = self.c_data.light_xpos

            pygame.init()
            pygame.joystick.init()
            self.joystick_present = False
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.joystick_present = True
        del self.c_model, self.c_data


    def _remove_capsule_hand_visual_geoms_from_xml(self, xml_handle):
        if not self.ball_plate_config.get("capsule_hand_remove_visual_geoms", True):
            return

        def get_mesh_name(geom):
            mesh = getattr(geom, "mesh", None)
            if mesh is None:
                return None
            name = getattr(mesh, "name", None)
            if name is not None:
                return str(name)
            return str(mesh)

        removed_mesh_names = set(self.ball_plate_config.get("capsule_hand_removed_mesh_names", []))
        if removed_mesh_names:
            for geom in list(xml_handle.find_all("geom")):
                mesh_name = get_mesh_name(geom)
                if mesh_name in removed_mesh_names:
                    geom.remove()

        remove_reward_collision_geoms = self.ball_plate_config.get("capsule_hand_remove_reward_collision_geoms", True)
        for body_name in self.ball_plate_config.get("capsule_hand_removed_body_names", []):
            body = xml_handle.find("body", body_name)
            if body is None:
                continue
            body_tree = [body, *list(body.find_all("body"))]
            geoms = []
            seen_geom_ids = set()
            for body_part in body_tree:
                for geom in list(body_part.find_all("geom")):
                    geom_id = id(geom)
                    if geom_id not in seen_geom_ids:
                        geoms.append(geom)
                        seen_geom_ids.add(geom_id)
            for geom in geoms:
                is_reward_collision_sphere = geom.dclass and geom.dclass.dclass == "reward_collision_sphere"
                if remove_reward_collision_geoms or not is_reward_collision_sphere:
                    geom.remove()


    def _make_ball_plate_contact_geoms_mjx_compatible(self, xml_handle):
        for geom_name in self.ball_plate_config.get("mjx_capsule_contact_geom_names", []):
            geom = xml_handle.find("geom", geom_name)
            if geom is not None:
                geom.type = "capsule"


    def _add_ball_plate_to_xml(self, xml_handle):
        plate_size = np.array(self.ball_plate_config["plate_size"], dtype=float)
        plate_pos = np.array(self.ball_plate_config["plate_home_pos"], dtype=float)
        plate_quat = np.array(self.ball_plate_config["plate_home_quat"], dtype=float)
        ball_radius = float(self.ball_plate_config["ball_radius"])
        fist_radius = float(self.ball_plate_config["fist_radius"])
        fist_half_length = float(self.ball_plate_config["fist_half_length"])
        plate_ball_pair_friction = " ".join(map(str, self.ball_plate_config.get("plate_ball_contact_friction", [1.0, 1.0, 0.005, 0.0001, 0.0001])))
        plate_ball_pair_dim = str(self.ball_plate_config.get("plate_ball_contact_dim", 3))
        plate_ball_pair_solref = " ".join(map(str, self.ball_plate_config.get("plate_ball_contact_solref", [0.02, 1.0])))
        plate_ball_pair_solimp = " ".join(map(str, self.ball_plate_config.get("plate_ball_contact_solimp", [0.9, 0.95, 0.001, 0.5, 2.0])))
        plate_ball_pair_margin = str(self.ball_plate_config.get("plate_ball_contact_margin", 0.0))
        plate_support_pair_friction = " ".join(map(str, self.ball_plate_config.get("plate_support_contact_friction", [1.0, 1.0, 0.005, 0.0001, 0.0001])))
        plate_support_pair_dim = str(self.ball_plate_config.get("plate_support_contact_dim", 3))
        plate_torso_pair_friction = " ".join(map(str, self.ball_plate_config.get("plate_torso_contact_friction", [1.0, 1.0, 0.005, 0.0001, 0.0001])))
        plate_torso_pair_dim = str(self.ball_plate_config.get("plate_torso_contact_dim", 3))
        plate_arm_pair_friction = " ".join(map(str, self.ball_plate_config.get("plate_arm_contact_friction", [1.0, 1.0, 0.005, 0.0001, 0.0001])))
        plate_arm_pair_dim = str(self.ball_plate_config.get("plate_arm_contact_dim", 3))
        arm_self_pair_friction = " ".join(map(str, self.ball_plate_config.get("arm_self_contact_friction", self.ball_plate_config.get("plate_arm_contact_friction", [1.0, 1.0, 0.005, 0.0001, 0.0001]))))
        arm_self_pair_dim = str(self.ball_plate_config.get("arm_self_contact_dim", self.ball_plate_config.get("plate_arm_contact_dim", 3)))
        robot_contact_pair_kwargs = {}
        if "plate_robot_contact_solref" in self.ball_plate_config:
            robot_contact_pair_kwargs["solref"] = " ".join(map(str, self.ball_plate_config["plate_robot_contact_solref"]))
        if "plate_robot_contact_solimp" in self.ball_plate_config:
            robot_contact_pair_kwargs["solimp"] = " ".join(map(str, self.ball_plate_config["plate_robot_contact_solimp"]))
        if "plate_robot_contact_margin" in self.ball_plate_config:
            robot_contact_pair_kwargs["margin"] = str(self.ball_plate_config["plate_robot_contact_margin"])
        arm_self_contact_pair_kwargs = {}
        if "arm_self_contact_solref" in self.ball_plate_config:
            arm_self_contact_pair_kwargs["solref"] = " ".join(map(str, self.ball_plate_config["arm_self_contact_solref"]))
        if "arm_self_contact_solimp" in self.ball_plate_config:
            arm_self_contact_pair_kwargs["solimp"] = " ".join(map(str, self.ball_plate_config["arm_self_contact_solimp"]))
        if "arm_self_contact_margin" in self.ball_plate_config:
            arm_self_contact_pair_kwargs["margin"] = str(self.ball_plate_config["arm_self_contact_margin"])

        left_fist = xml_handle.find("body", self.ball_plate_config["left_fist_body"])
        right_fist = xml_handle.find("body", self.ball_plate_config["right_fist_body"])
        torso_contact_body = xml_handle.find("body", self.ball_plate_config["torso_contact_body"])
        left_upper_arm_contact_body = xml_handle.find("body", self.ball_plate_config["left_upper_arm_contact_body"])
        right_upper_arm_contact_body = xml_handle.find("body", self.ball_plate_config["right_upper_arm_contact_body"])
        left_forearm_contact_body = xml_handle.find("body", self.ball_plate_config["left_forearm_contact_body"])
        right_forearm_contact_body = xml_handle.find("body", self.ball_plate_config["right_forearm_contact_body"])
        left_fist_pos = np.array(self.ball_plate_config["left_fist_pos"], dtype=float)
        right_fist_pos = np.array(self.ball_plate_config["right_fist_pos"], dtype=float)
        torso_contact_pos = np.array(self.ball_plate_config["torso_contact_pos"], dtype=float)
        torso_contact_size = float(self.ball_plate_config["torso_contact_size"])
        left_upper_arm_contact_fromto = np.array(
            self.ball_plate_config.get("left_upper_arm_contact_fromto", self.ball_plate_config["upper_arm_contact_fromto"]),
            dtype=float,
        )
        right_upper_arm_contact_fromto = np.array(
            self.ball_plate_config.get("right_upper_arm_contact_fromto", self.ball_plate_config["upper_arm_contact_fromto"]),
            dtype=float,
        )
        upper_arm_contact_radius = float(self.ball_plate_config["upper_arm_contact_radius"])
        left_forearm_contact_fromto = np.array(
            self.ball_plate_config.get("left_forearm_contact_fromto", self.ball_plate_config["forearm_contact_fromto"]),
            dtype=float,
        )
        right_forearm_contact_fromto = np.array(
            self.ball_plate_config.get("right_forearm_contact_fromto", self.ball_plate_config["forearm_contact_fromto"]),
            dtype=float,
        )
        forearm_contact_radius = float(self.ball_plate_config["forearm_contact_radius"])
        add_torso_plate_guard = self.ball_plate_config.get("add_torso_plate_guard", True)
        add_upper_arm_plate_guards = self.ball_plate_config.get("add_upper_arm_plate_guards", True)
        add_forearm_plate_guards = self.ball_plate_config.get("add_forearm_plate_guards")
        if add_forearm_plate_guards is None:
            add_forearm_plate_guards = not (
                self.use_capsule_hand and self.ball_plate_config.get("capsule_hand_disable_forearm_guards", True)
            )
        add_support_fist_geoms = self.ball_plate_config.get("add_support_fist_geoms", True)
        if add_support_fist_geoms:
            if self.use_capsule_hand:
                capsule_hand_mass = float(self.ball_plate_config.get("capsule_hand_mass", 0.0))
                left_capsule_hand = left_fist.add("body", name="left_capsule_hand", pos=" ".join(map(str, left_fist_pos)))
                right_capsule_hand = right_fist.add("body", name="right_capsule_hand", pos=" ".join(map(str, right_fist_pos)))
                left_capsule_hand.add("geom", name="left_plate_support_fist", type="capsule", size=str(fist_radius), fromto=f"{-fist_half_length} 0.0 0.0 {fist_half_length} 0.0 0.0", rgba="0.68 0.68 0.68 1", mass=str(capsule_hand_mass), contype="0", conaffinity="0")
                right_capsule_hand.add("geom", name="right_plate_support_fist", type="capsule", size=str(fist_radius), fromto=f"{-fist_half_length} 0.0 0.0 {fist_half_length} 0.0 0.0", rgba="0.68 0.68 0.68 1", mass=str(capsule_hand_mass), contype="0", conaffinity="0")
            else:
                left_fist.add("geom", name="left_plate_support_fist", type="capsule", size=str(fist_radius), fromto=f"{left_fist_pos[0] - fist_half_length} {left_fist_pos[1]} {left_fist_pos[2]} {left_fist_pos[0] + fist_half_length} {left_fist_pos[1]} {left_fist_pos[2]}", rgba="0.68 0.68 0.68 1", contype="0", conaffinity="0")
                right_fist.add("geom", name="right_plate_support_fist", type="capsule", size=str(fist_radius), fromto=f"{right_fist_pos[0] - fist_half_length} {right_fist_pos[1]} {right_fist_pos[2]} {right_fist_pos[0] + fist_half_length} {right_fist_pos[1]} {right_fist_pos[2]}", rgba="0.68 0.68 0.68 1", contype="0", conaffinity="0")
        if add_torso_plate_guard:
            torso_contact_body.add("geom", name="torso_plate_guard", type="sphere", pos=" ".join(map(str, torso_contact_pos)), size=str(torso_contact_size), rgba="0.2 0.2 0.2 0", contype="0", conaffinity="0")
        if add_upper_arm_plate_guards:
            left_upper_arm_contact_body.add("geom", name="left_upper_arm_plate_guard", type="capsule", size=str(upper_arm_contact_radius), fromto=" ".join(map(str, left_upper_arm_contact_fromto)), rgba="0.2 0.2 0.2 0", contype="0", conaffinity="0")
            right_upper_arm_contact_body.add("geom", name="right_upper_arm_plate_guard", type="capsule", size=str(upper_arm_contact_radius), fromto=" ".join(map(str, right_upper_arm_contact_fromto)), rgba="0.2 0.2 0.2 0", contype="0", conaffinity="0")
        if add_forearm_plate_guards:
            left_forearm_contact_body.add("geom", name="left_forearm_plate_guard", type="capsule", size=str(forearm_contact_radius), fromto=" ".join(map(str, left_forearm_contact_fromto)), rgba="0.2 0.2 0.2 0", contype="0", conaffinity="0")
            right_forearm_contact_body.add("geom", name="right_forearm_plate_guard", type="capsule", size=str(forearm_contact_radius), fromto=" ".join(map(str, right_forearm_contact_fromto)), rgba="0.2 0.2 0.2 0", contype="0", conaffinity="0")

        plate = xml_handle.worldbody.add("body", name="ball_plate", pos=" ".join(map(str, plate_pos)), quat=" ".join(map(str, plate_quat)))
        plate.add("freejoint", name="ball_plate_freejoint")
        plate.add(
            "geom",
            name="ball_plate_geom",
            type="box",
            size=" ".join(map(str, plate_size)),
            mass=str(self.ball_plate_config["plate_mass"]),
            rgba="0.15 0.16 0.18 1",
            contype="0",
            conaffinity="0",
            friction="0.55 0.002 0.00001",
        )
        plate.add("site", name="ball_plate_center", pos=f"0 0 {plate_size[2]}")

        initial_ball_pos = np.array(self.ball_plate_config["ball_home_pos"], dtype=float)
        ball = xml_handle.worldbody.add("body", name="plate_ball", pos=" ".join(map(str, initial_ball_pos)))
        ball.add("freejoint", name="plate_ball_freejoint")
        ball.add(
            "geom",
            name="plate_ball_geom",
            type="sphere",
            size=str(ball_radius),
            mass=str(self.ball_plate_config["ball_mass"]),
            rgba="0.9 0.12 0.08 1",
            contype="0",
            conaffinity="0",
            friction="0.55 0.002 0.00001",
        )

        xml_handle.contact.add(
            "pair",
            geom1="ball_plate_geom",
            geom2="plate_ball_geom",
            condim=plate_ball_pair_dim,
            friction=plate_ball_pair_friction,
            solref=plate_ball_pair_solref,
            solimp=plate_ball_pair_solimp,
            margin=plate_ball_pair_margin,
        )
        support_contact_geom_names = self.ball_plate_config.get("support_contact_geom_names")
        if support_contact_geom_names is None:
            support_contact_geom_names = ["left_plate_support_fist", "right_plate_support_fist"] if add_support_fist_geoms else []
        for support_geom_name in support_contact_geom_names:
            xml_handle.contact.add("pair", geom1=support_geom_name, geom2="ball_plate_geom", condim=plate_support_pair_dim, friction=plate_support_pair_friction, **robot_contact_pair_kwargs)
        for torso_geom_name in self._get_ball_plate_torso_contact_geom_names():
            xml_handle.contact.add("pair", geom1=torso_geom_name, geom2="ball_plate_geom", condim=plate_torso_pair_dim, friction=plate_torso_pair_friction, **robot_contact_pair_kwargs)
        for arm_geom_name in self._get_ball_plate_arm_contact_geom_names():
            xml_handle.contact.add("pair", geom1=arm_geom_name, geom2="ball_plate_geom", condim=plate_arm_pair_dim, friction=plate_arm_pair_friction, **robot_contact_pair_kwargs)
        for geom1_name, geom2_name in self.ball_plate_config.get("arm_self_contact_geom_pairs", []):
            xml_handle.contact.add("pair", geom1=geom1_name, geom2=geom2_name, condim=arm_self_pair_dim, friction=arm_self_pair_friction, **arm_self_contact_pair_kwargs)
        xml_handle.contact.add("pair", geom1="floor", geom2="ball_plate_geom")
        xml_handle.contact.add("pair", geom1="floor", geom2="plate_ball_geom")

        self._append_ball_plate_to_home_key(xml_handle, plate_pos, plate_quat, initial_ball_pos)
        self._apply_ball_plate_initial_joint_positions_to_home_key(xml_handle)


    def _get_ball_plate_support_contact_geom_names(self):
        support_contact_geom_names = self.ball_plate_config.get("support_contact_geom_names")
        if support_contact_geom_names is not None:
            return list(support_contact_geom_names)

        if self.ball_plate_config.get("add_support_fist_geoms", True):
            return ["left_plate_support_fist", "right_plate_support_fist"]
        return []


    def _get_ball_plate_robot_contact_geom_names(self):
        return [
            *self._get_ball_plate_torso_contact_geom_names(),
            *self._get_ball_plate_arm_contact_geom_names(),
        ]


    def _get_ball_plate_torso_contact_geom_names(self):
        torso_contact_geom_names = []
        if self.ball_plate_config.get("add_torso_plate_guard", True):
            torso_contact_geom_names.append("torso_plate_guard")
        torso_contact_geom_names.extend(self.ball_plate_config.get("torso_contact_geom_names", []))
        return torso_contact_geom_names


    def _get_ball_plate_arm_contact_geom_names(self):
        arm_contact_geom_names = []
        support_contact_geom_names = set(self._get_ball_plate_support_contact_geom_names())
        if self.ball_plate_config.get("add_upper_arm_plate_guards", True):
            arm_contact_geom_names.extend([
                "left_upper_arm_plate_guard",
                "right_upper_arm_plate_guard",
            ])
        add_forearm_plate_guards = self.ball_plate_config.get("add_forearm_plate_guards")
        if add_forearm_plate_guards is None:
            add_forearm_plate_guards = not (
                self.use_capsule_hand and self.ball_plate_config.get("capsule_hand_disable_forearm_guards", True)
            )
        if add_forearm_plate_guards:
            arm_contact_geom_names.extend([
                "left_forearm_plate_guard",
                "right_forearm_plate_guard",
            ])
        arm_contact_geom_names.extend(self.ball_plate_config.get("arm_contact_geom_names", []))
        return [geom_name for geom_name in arm_contact_geom_names if geom_name not in support_contact_geom_names]

    def _get_ball_plate_pair_ids(self, other_geom_names):
        other_geom_names = set(other_geom_names)
        pair_ids = []
        for pair_id in range(self.initial_mj_model.npair):
            geom1_id = int(self.initial_mj_model.pair_geom1[pair_id])
            geom2_id = int(self.initial_mj_model.pair_geom2[pair_id])
            geom1_name = mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
            if (
                (geom1_name == "ball_plate_geom" and geom2_name in other_geom_names)
                or (geom2_name == "ball_plate_geom" and geom1_name in other_geom_names)
            ):
                pair_ids.append(pair_id)
        return pair_ids


    def _append_ball_plate_to_home_key(self, xml_handle, initial_plate_pos, initial_plate_quat, initial_ball_pos):
        home_key = xml_handle.find("key", "home")
        if home_key is None or home_key.qpos is None:
            return

        home_qpos = home_key.qpos
        if isinstance(home_qpos, str):
            home_qpos = np.fromstring(home_qpos, sep=" ")
        else:
            home_qpos = np.asarray(home_qpos, dtype=float)
        home_key.qpos = np.concatenate([home_qpos, [*initial_plate_pos, *initial_plate_quat, *initial_ball_pos, 1.0, 0.0, 0.0, 0.0]])


    def _apply_ball_plate_initial_joint_positions_to_home_key(self, xml_handle):
        initial_joint_positions = self.ball_plate_config.get("initial_joint_positions", {})
        if not initial_joint_positions:
            return

        home_key = xml_handle.find("key", "home")
        if home_key is None or home_key.qpos is None:
            return

        home_qpos = home_key.qpos
        if isinstance(home_qpos, str):
            home_qpos = np.fromstring(home_qpos, sep=" ")
        else:
            home_qpos = np.asarray(home_qpos, dtype=float)

        temp_model = mujoco.MjModel.from_xml_string(xml=xml_handle.to_xml_string(), assets=xml_handle.get_assets())
        for joint_name, joint_position in initial_joint_positions.items():
            joint_id = mujoco.mj_name2id(temp_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Unknown ball_plate initial joint: {joint_name}")

            qposadr = temp_model.jnt_qposadr[joint_id]
            if temp_model.jnt_limited[joint_id]:
                joint_position = np.clip(joint_position, temp_model.jnt_range[joint_id, 0], temp_model.jnt_range[joint_id, 1])
            home_qpos[qposadr] = joint_position

        home_key.qpos = home_qpos


    def _apply_ball_plate_initial_joint_positions_to_state(self, qpos, qvel=None, internal_state=None):
        initial_joint_positions = self.ball_plate_config.get("initial_joint_positions", {})
        if not initial_joint_positions:
            return qpos, qvel

        model = self.initial_mj_model

        for joint_name, joint_position in initial_joint_positions.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Unknown ball_plate initial joint: {joint_name}")

            if model.jnt_limited[joint_id]:
                joint_position = np.clip(joint_position, model.jnt_range[joint_id, 0], model.jnt_range[joint_id, 1])

            qposadr = model.jnt_qposadr[joint_id]
            qpos = qpos.at[qposadr].set(joint_position)
            nominal_index = self.actuator_joint_qposadr_to_nominal_index.get(int(qposadr))
            if internal_state is not None and nominal_index is not None:
                nominal_positions = internal_state["actuator_joint_nominal_positions"].at[nominal_index].set(joint_position)
                internal_state["actuator_joint_nominal_positions"] = nominal_positions
            if qvel is not None:
                dofadr = model.jnt_dofadr[joint_id]
                qvel = qvel.at[dofadr].set(0.0)

        return qpos, qvel


    def _apply_ball_plate_post_drop_command(self, internal_state):
        if not self.stand_after_ball_plate_drop:
            return

        post_drop = internal_state["ball_plate_drop_latched"]
        internal_state["goal_velocities"] = jnp.where(
            post_drop,
            jnp.zeros_like(internal_state["goal_velocities"]),
            internal_state["goal_velocities"],
        )
        internal_state["actuator_joint_keep_nominal"] = jnp.where(
            post_drop,
            jnp.ones_like(internal_state["actuator_joint_keep_nominal"], dtype=bool),
            internal_state["actuator_joint_keep_nominal"],
        )
        if "goal_gait_frequency" in internal_state:
            internal_state["goal_gait_frequency"] = jnp.where(
                post_drop,
                jnp.asarray(0.0, dtype=jnp.asarray(internal_state["goal_gait_frequency"]).dtype),
                internal_state["goal_gait_frequency"],
            )


    def _update_ball_plate_post_drop_state(self, internal_state):
        if not self.stand_after_ball_plate_drop:
            internal_state["ball_plate_newly_dropped"] = False
            internal_state["ball_plate_post_drop_truncated"] = False
            return jnp.asarray(False)

        dropped = internal_state["ball_plate_dropped"]
        was_latched = internal_state["ball_plate_drop_latched"]
        drop_latched = was_latched | dropped
        newly_dropped = dropped & (~was_latched)
        time_since_drop = jnp.where(
            drop_latched,
            internal_state["ball_plate_time_since_drop"] + self.dt,
            0.0,
        )
        post_drop_truncated = drop_latched & (
            time_since_drop >= self.ball_plate_post_drop_truncation_seconds
        )

        internal_state["ball_plate_drop_latched"] = drop_latched
        internal_state["ball_plate_newly_dropped"] = newly_dropped
        internal_state["ball_plate_time_since_drop"] = time_since_drop
        internal_state["ball_plate_post_drop_truncated"] = post_drop_truncated
        return post_drop_truncated

    
    def _sample_obstacle_shapes(self, key, nr):
        """Primitives of nr new obstacles, drawn from shapes.weights (see OBSTACLE_SHAPES)."""
        cfg = self.obstacle_config
        shape_cfg = cfg.get("shapes", {})
        keys = jax.random.split(key, 9)
        shape = jax.random.choice(keys[0], len(OBSTACLE_SHAPES), (nr,), p=self.obstacle_shape_probs)
        zeros = jnp.zeros((nr,))

        def uniform(k, value_range, shape=(nr,)):
            return jax.random.uniform(k, shape, minval=value_range[0], maxval=value_range[1])

        def slots(*columns):  # list of P per-obstacle arrays -> [nr, P]
            return jnp.stack(columns, axis=-1)

        # circle
        radius = uniform(keys[1], cfg["radius_range"])
        circle = {
            "kind": slots(jnp.full((nr,), PRIMITIVE_CIRCLE), jnp.zeros((nr,), int), jnp.zeros((nr,), int)),
            "offset": jnp.zeros((nr, 3, 2)),
            "half": jnp.stack([slots(radius, zeros, zeros), jnp.zeros((nr, 3))], axis=-1),
            "angle": jnp.zeros((nr, 3)),
        }
        # box
        half_length = uniform(keys[2], shape_cfg.get("box_half_length_range", (0.15, 0.5)))
        half_width = uniform(keys[3], shape_cfg.get("box_half_width_range", (0.1, 0.3)))
        box = {
            "kind": slots(jnp.full((nr,), PRIMITIVE_BOX), jnp.zeros((nr,), int), jnp.zeros((nr,), int)),
            "offset": jnp.zeros((nr, 3, 2)),
            "half": jnp.stack([slots(half_length, zeros, zeros), slots(half_width, zeros, zeros)], axis=-1),
            "angle": jnp.zeros((nr, 3)),
        }
        # chain of 2-3 overlapping circles along the body axis (a person with a bag, a cart)
        count_range = shape_cfg.get("circles_count_range", (2, 3))
        count = jax.random.randint(keys[4], (nr,), count_range[0], count_range[1] + 1)
        radii = uniform(keys[5], cfg["radius_range"], (nr, 3))
        spacing = 1.6 * jnp.max(radii, axis=-1)
        three = count >= 3
        x_offsets = jnp.where(three[:, None], jnp.stack([-spacing, zeros, spacing], axis=-1),
                              jnp.stack([-0.5 * spacing, 0.5 * spacing, zeros], axis=-1))
        circles = {
            "kind": slots(jnp.full((nr,), PRIMITIVE_CIRCLE), jnp.full((nr,), PRIMITIVE_CIRCLE),
                          jnp.where(three, PRIMITIVE_CIRCLE, PRIMITIVE_NONE)),
            "offset": jnp.stack([x_offsets, jnp.zeros((nr, 3))], axis=-1),
            "half": jnp.stack([radii, jnp.zeros((nr, 3))], axis=-1),
            "angle": jnp.zeros((nr, 3)),
        }
        # L shape: two boxes meeting at a right angle, centred on their bounding box
        arm_a = uniform(keys[6], shape_cfg.get("l_shape_arm_half_length_range", (0.25, 0.5)))
        arm_b = uniform(keys[7], shape_cfg.get("l_shape_arm_half_length_range", (0.25, 0.5)))
        arm_width = uniform(keys[8], shape_cfg.get("l_shape_arm_half_width_range", (0.05, 0.15)))
        centre_x, centre_y = 0.5 * (arm_a - arm_width), 0.5 * (arm_b - arm_width)
        l_shape = {
            "kind": slots(jnp.full((nr,), PRIMITIVE_BOX), jnp.full((nr,), PRIMITIVE_BOX), jnp.zeros((nr,), int)),
            "offset": jnp.stack([slots(arm_a - arm_width - centre_x, -centre_x, zeros),
                                 slots(-centre_y, arm_b - arm_width - centre_y, zeros)], axis=-1),
            "half": jnp.stack([slots(arm_a, arm_width, zeros), slots(arm_width, arm_b, zeros)], axis=-1),
            "angle": jnp.zeros((nr, 3)),
        }

        options = {"circle": circle, "box": box, "circles": circles, "l_shape": l_shape}
        primitives = {}
        for field in ("kind", "offset", "half", "angle"):
            value = options[OBSTACLE_SHAPES[0]][field]
            for index, name in enumerate(OBSTACLE_SHAPES[1:], start=1):
                selected = (shape == index).reshape((nr,) + (1,) * (value.ndim - 1))
                value = jnp.where(selected, options[name][field], value)
            primitives[field] = value
        primitives["kind"] = primitives["kind"].astype(jnp.int32)
        extent = jnp.where(primitives["kind"] == PRIMITIVE_BOX, jnp.linalg.norm(primitives["half"], axis=-1),
                           jnp.where(primitives["kind"] == PRIMITIVE_CIRCLE, primitives["half"][..., 0], 0.0))
        primitives["bounding_radius"] = jnp.max(jnp.linalg.norm(primitives["offset"], axis=-1) + extent, axis=-1)
        primitives["shape"] = shape.astype(jnp.int32)
        return primitives


    def _obstacle_speed(self, internal_state, age):
        """Speed of each obstacle at a given age, following its motion type."""
        motion = internal_state["obstacle_motion"]
        speed = internal_state["obstacle_speed"]
        moving_phase = jnp.mod(age + internal_state["obstacle_phase"], internal_state["obstacle_period"])
        stop_and_go = jnp.where(moving_phase < internal_state["obstacle_duty"] * internal_state["obstacle_period"], speed, 0.0)
        change = internal_state["obstacle_target_speed"] - speed
        accelerating = speed + jnp.clip(change, -internal_state["obstacle_accel"] * age, internal_state["obstacle_accel"] * age)
        speed = jnp.where(motion == OBSTACLE_MOTIONS.index("stop_and_go"), stop_and_go, speed)
        return jnp.where(motion == OBSTACLE_MOTIONS.index("accelerating"), accelerating, speed)


    def _sample_obstacles(self, robot_xy, robot_vel_xy, internal_state, key, respawn_mask):
        """Respawn the obstacles selected by respawn_mask on a ring around the robot, heading at it."""
        cfg = self.obstacle_config
        motion_cfg = cfg.get("motions", {})
        (distance_key, angle_key, aim_key, speed_key, shape_key, heading_key, motion_key,
         turn_key, turn_sign_key, accel_key, target_speed_key, period_key, duty_key, phase_key) = jax.random.split(key, 14)
        nr = self.nr_obstacles

        distance = jax.random.uniform(distance_key, (nr,), minval=cfg["spawn_distance_range"][0], maxval=cfg["spawn_distance_range"][1])
        angle = jax.random.uniform(angle_key, (nr,), minval=0.0, maxval=2.0 * jnp.pi)
        spawn_pos = robot_xy[None, :] + distance[:, None] * jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1)

        min_speed, max_speed = cfg["speed_range"]
        if cfg.get("speed_curriculum", True):
            max_speed = min_speed + internal_state["env_curriculum_coeff"] * (max_speed - min_speed)
        speed = min_speed + jax.random.uniform(speed_key, (nr,)) * (max_speed - min_speed)
        target_speed = min_speed + jax.random.uniform(target_speed_key, (nr,)) * (max_speed - min_speed)
        shapes = self._sample_obstacle_shapes(shape_key, nr)

        # Aim at the robot's current position, or with aim_lead at where it will be on arrival if it keeps its velocity.
        arrival_time = distance / jnp.maximum(speed, 1e-3)
        target = robot_xy[None, :]
        if cfg.get("aim_lead", False):
            target = target + robot_vel_xy[None, :] * arrival_time[:, None]
        to_target = target - spawn_pos
        to_target = to_target / (jnp.linalg.norm(to_target, axis=-1, keepdims=True) + 1e-6)
        lateral = jnp.stack([-to_target[:, 1], to_target[:, 0]], axis=-1)
        aim_offset = jax.random.uniform(aim_key, (nr,), minval=-cfg["aim_offset"], maxval=cfg["aim_offset"])
        heading = target + lateral * aim_offset[:, None] - spawn_pos
        move_dir = jnp.arctan2(heading[:, 1], heading[:, 0])

        motion = jax.random.choice(motion_key, len(OBSTACLE_MOTIONS), (nr,), p=self.obstacle_motion_probs).astype(jnp.int32)
        turn_range = motion_cfg.get("turn_rate_range", (0.3, 1.5))
        turn_rate = jax.random.uniform(turn_key, (nr,), minval=turn_range[0], maxval=turn_range[1])
        turn_rate = turn_rate * jnp.where(jax.random.bernoulli(turn_sign_key, 0.5, (nr,)), 1.0, -1.0)
        turn_rate = jnp.where(motion == OBSTACLE_MOTIONS.index("turning"), turn_rate, 0.0)
        accel_range = motion_cfg.get("acceleration_range", (0.5, 2.0))
        accel = jax.random.uniform(accel_key, (nr,), minval=accel_range[0], maxval=accel_range[1])
        accel = jnp.where(motion == OBSTACLE_MOTIONS.index("accelerating"), accel, 0.0)
        period_range = motion_cfg.get("stop_and_go_period_range", (1.0, 3.0))
        period = jax.random.uniform(period_key, (nr,), minval=period_range[0], maxval=period_range[1])
        duty_range = motion_cfg.get("stop_and_go_duty_range", (0.3, 0.7))
        duty = jax.random.uniform(duty_key, (nr,), minval=duty_range[0], maxval=duty_range[1])
        duty = jnp.where(motion == OBSTACLE_MOTIONS.index("stop_and_go"), duty, 1.0)
        phase = jax.random.uniform(phase_key, (nr,)) * period

        # Live long enough to pass the robot, so slow obstacles also arrive.
        effective_speed = jnp.where(motion == OBSTACLE_MOTIONS.index("stop_and_go"), speed * duty, speed)
        effective_speed = jnp.where(motion == OBSTACLE_MOTIONS.index("accelerating"), jnp.minimum(speed, target_speed), effective_speed)
        lifetime = distance / jnp.maximum(effective_speed, 0.3) + cfg.get("lifetime_margin_seconds", 2.0)
        lifetime = jnp.minimum(lifetime, cfg["max_lifetime_seconds"])

        new_values = {
            "obstacle_pos": spawn_pos,
            "obstacle_heading": jax.random.uniform(heading_key, (nr,), minval=0.0, maxval=2.0 * jnp.pi),
            "obstacle_radius": shapes["bounding_radius"],
            "obstacle_shape": shapes["shape"],
            "obstacle_prim_kind": shapes["kind"],
            "obstacle_prim_offset": shapes["offset"],
            "obstacle_prim_half": shapes["half"],
            "obstacle_prim_angle": shapes["angle"],
            "obstacle_motion": motion,
            "obstacle_speed": speed,
            "obstacle_move_dir": move_dir,
            "obstacle_turn_rate": turn_rate,
            "obstacle_accel": accel,
            "obstacle_target_speed": target_speed,
            "obstacle_period": period,
            "obstacle_duty": duty,
            "obstacle_phase": phase,
            "obstacle_age": jnp.zeros((nr,)),
            "obstacle_lifetime": lifetime,
        }
        for name, value in new_values.items():
            old = internal_state[name]
            mask = respawn_mask.reshape((nr,) + (1,) * (old.ndim - 1))
            internal_state[name] = jnp.where(mask, value.astype(old.dtype), old)
        initial_speed = self._obstacle_speed(internal_state, internal_state["obstacle_age"])
        initial_vel = initial_speed[:, None] * jnp.stack([jnp.cos(internal_state["obstacle_move_dir"]), jnp.sin(internal_state["obstacle_move_dir"])], axis=-1)
        internal_state["obstacle_vel"] = jnp.where(respawn_mask[:, None], initial_vel, internal_state["obstacle_vel"])


    def _obstacle_primitives(self, internal_state):
        return primitives_in_world(
            internal_state["obstacle_pos"], internal_state["obstacle_heading"], internal_state["obstacle_prim_offset"],
            internal_state["obstacle_prim_half"], internal_state["obstacle_prim_angle"], internal_state["obstacle_prim_kind"])


    def _update_obstacle_collision(self, robot_xy, internal_state):
        signed_distance = primitive_signed_distance(robot_xy, *self._obstacle_primitives(internal_state))
        clearance = jnp.min(signed_distance.reshape(self.nr_obstacles, -1), axis=-1) - self.obstacle_config["robot_radius"]
        internal_state["obstacle_min_clearance"] = jnp.min(clearance)
        internal_state["obstacle_collision"] = jnp.any(clearance < 0.0)


    def _reset_obstacles(self, data, internal_state, key):
        if not self.use_obstacles:
            return
        spawn_key, noise_key, dropout_key, latency_key = jax.random.split(key, 4)
        robot_xy = data.qpos[:2]
        self._sample_obstacles(robot_xy, data.qvel[:2], internal_state, spawn_key, jnp.ones(self.nr_obstacles, dtype=bool))
        self._update_obstacle_collision(robot_xy, internal_state)
        # Sensor parameters drawn once per episode
        scan_cfg = self.obstacle_config.get("scan", {})
        noise_range = scan_cfg.get("noise_range", (0.0, 0.0))
        dropout_range = scan_cfg.get("dropout_prob_range", (0.0, 0.0))
        latency_range = scan_cfg.get("latency_steps_range", (0, 0))
        internal_state["obstacle_scan_noise"] = jax.random.uniform(noise_key, (), minval=noise_range[0], maxval=noise_range[1])
        internal_state["obstacle_scan_dropout"] = jax.random.uniform(dropout_key, (), minval=dropout_range[0], maxval=dropout_range[1])
        internal_state["obstacle_scan_latency"] = jax.random.randint(latency_key, (), latency_range[0], latency_range[1] + 1).astype(jnp.int32)
        internal_state["obstacle_scan_buffer_valid"] = jnp.asarray(False)


    def _step_obstacles(self, data, internal_state, key):
        if not self.use_obstacles:
            return
        cfg = self.obstacle_config
        motion_cfg = cfg.get("motions", {})
        respawn_key, walk_key = jax.random.split(key)
        robot_xy = data.qpos[:2]
        robot_vel_xy = data.qvel[:2]

        # Random-walk obstacles change their turn rate; turning and random-walk obstacles rotate with their path.
        random_walk = internal_state["obstacle_motion"] == OBSTACLE_MOTIONS.index("random_walk")
        max_turn = motion_cfg.get("random_walk_max_turn_rate", 2.0)
        walked_turn_rate = jnp.clip(
            internal_state["obstacle_turn_rate"]
            + motion_cfg.get("random_walk_turn_noise", 2.0) * jnp.sqrt(self.dt) * jax.random.normal(walk_key, (self.nr_obstacles,)),
            -max_turn, max_turn)
        internal_state["obstacle_turn_rate"] = jnp.where(random_walk, walked_turn_rate, internal_state["obstacle_turn_rate"])
        internal_state["obstacle_move_dir"] = internal_state["obstacle_move_dir"] + internal_state["obstacle_turn_rate"] * self.dt
        internal_state["obstacle_heading"] = internal_state["obstacle_heading"] + internal_state["obstacle_turn_rate"] * self.dt

        internal_state["obstacle_age"] = internal_state["obstacle_age"] + self.dt
        speed = self._obstacle_speed(internal_state, internal_state["obstacle_age"])
        move_dir = internal_state["obstacle_move_dir"]
        internal_state["obstacle_vel"] = speed[:, None] * jnp.stack([jnp.cos(move_dir), jnp.sin(move_dir)], axis=-1)
        internal_state["obstacle_pos"] = internal_state["obstacle_pos"] + internal_state["obstacle_vel"] * self.dt
        # Collision is checked before respawning so an obstacle cannot disappear on the step it hits.
        self._update_obstacle_collision(robot_xy, internal_state)

        relative_pos = internal_state["obstacle_pos"] - robot_xy[None, :]
        relative_vel = internal_state["obstacle_vel"] - robot_vel_xy[None, :]
        receding = jnp.sum(relative_pos * relative_vel, axis=-1) > 0.0
        far = jnp.linalg.norm(relative_pos, axis=-1) > cfg["despawn_distance"]
        too_old = internal_state["obstacle_age"] > internal_state["obstacle_lifetime"]
        self._sample_obstacles(robot_xy, robot_vel_xy, internal_state, respawn_key, (receding & far) | too_old)


    def get_obstacle_observation(self, data, internal_state, key):
        """Privileged block (if any network uses it), then the range-scan block (if any network uses it)."""
        state_key, scan_key = jax.random.split(key)
        return jnp.concatenate([
            self.get_obstacle_state_observation(data, internal_state, state_key),
            self.get_obstacle_scan_observation(data, internal_state, scan_key),
        ])


    def get_obstacle_state_observation(self, data, internal_state, key):
        """Per obstacle: position and velocity relative to the base in the yaw frame, and radius (normalized)."""
        if not self.use_obstacle_state_observation:
            return jnp.zeros(0)
        yaw = internal_state["imu_orientation_euler"][2]
        cos_yaw, sin_yaw = jnp.cos(yaw), jnp.sin(yaw)

        def to_yaw_frame(v):
            return jnp.stack([cos_yaw * v[..., 0] + sin_yaw * v[..., 1], -sin_yaw * v[..., 0] + cos_yaw * v[..., 1]], axis=-1)

        relative_pos = to_yaw_frame(internal_state["obstacle_pos"] - data.qpos[:2][None, :])
        relative_vel = to_yaw_frame(internal_state["obstacle_vel"] - data.qvel[:2][None, :])
        noise_scale = self.obstacle_config.get("observation_noise", 0.0) * internal_state["env_curriculum_coeff"]
        relative_pos = relative_pos + jax.random.uniform(key, relative_pos.shape, minval=-1.0, maxval=1.0) * noise_scale
        per_obstacle = jnp.concatenate([
            relative_pos / 5.0,
            relative_vel / 5.0,
            internal_state["obstacle_radius"][:, None] / 0.5,
        ], axis=-1)
        return per_obstacle.reshape(-1)


    def get_obstacle_scan_observation(self, data, internal_state, key):
        """Planar range scan from the base in the yaw frame, as proximity 1 - range / max_range (0 = nothing in range).

        Per-episode noise level, dropout and latency (drawn in _reset_obstacles) and per-step angle jitter and
        spurious returns are all scaled by the env curriculum coefficient.
        """
        if not self.use_obstacle_scan_observation:
            return jnp.zeros(0)
        scan_cfg = self.obstacle_config.get("scan", {})
        noise_key, dropout_key, jitter_key, spurious_key, spurious_range_key = jax.random.split(key, 5)
        coeff = internal_state["env_curriculum_coeff"]
        nr_rays = self.obstacle_scan_nr_rays
        max_range = self.obstacle_scan_max_range

        jitter = scan_cfg.get("angle_jitter", 0.0) * coeff
        ray_angles = self.obstacle_scan_angles + jax.random.uniform(jitter_key, (nr_rays,), minval=-1.0, maxval=1.0) * jitter
        ranges = primitive_range_scan(data.qpos[:2], internal_state["imu_orientation_euler"][2], ray_angles,
                                      *self._obstacle_primitives(internal_state), max_range)
        ranges = ranges + jax.random.uniform(noise_key, (nr_rays,), minval=-1.0, maxval=1.0) * internal_state["obstacle_scan_noise"] * coeff
        spurious = jax.random.uniform(spurious_key, (nr_rays,)) < scan_cfg.get("spurious_prob", 0.0) * coeff
        spurious_range = jax.random.uniform(spurious_range_key, (nr_rays,), minval=0.3, maxval=max_range)
        ranges = jnp.where(spurious, jnp.minimum(ranges, spurious_range), ranges)
        dropped = jax.random.uniform(dropout_key, (nr_rays,)) < internal_state["obstacle_scan_dropout"] * coeff
        ranges = jnp.where(dropped, max_range, ranges)
        proximity = jnp.clip(1.0 - ranges / max_range, 0.0, 1.0)

        # Latency: keep the last scans, newest first, and return the delayed one.
        buffer = internal_state["obstacle_scan_buffer"]
        buffer = jnp.where(internal_state["obstacle_scan_buffer_valid"],
                           jnp.concatenate([proximity[None, :], buffer[:-1]], axis=0),
                           jnp.tile(proximity[None, :], (buffer.shape[0], 1)))
        internal_state["obstacle_scan_buffer"] = buffer
        internal_state["obstacle_scan_buffer_valid"] = jnp.asarray(True)
        delay = jnp.clip(jnp.round(internal_state["obstacle_scan_latency"] * coeff), 0, buffer.shape[0] - 1).astype(jnp.int32)
        return buffer[delay]


    def render(self, state):
        mjx_model = state.mjx_model
        mj_model = self.viewer.model
        for field in mjx.Model.fields():
            if field.type in [jax.Array, np.ndarray]:
                field_name = field.name
                if field.name in ["mesh_conver", "dof_hasfrictionloss", "tendon_hasfrictionloss", "_sizes"]:
                    continue
                if field_name in "geom_rbound_hfield":
                    field_name = "geom_rbound"
                mjx_value = getattr(mjx_model, field_name)
                mj_value = getattr(mj_model, field_name)
                if mjx_value.shape != mj_value.shape:
                    mjx_value = mjx_value.reshape(mj_value.shape)
                setattr(mj_model, field_name, mjx_value)
        if self.uses_hfield and state.info_episode_store["episode_step"] == 1:
            mujoco.mjr_uploadHField(mj_model, self.viewer.context, 0)

        env_id = 0
        data = mjx.get_data(mj_model, state.data)[env_id]

        data.light_xdir = self.light_xdir
        data.light_xpos = self.light_xpos

        if self.runner_mode == "test":
            explicit_velocity_commands = False
            if self.joystick_present:
                pygame.event.pump()
                goal_x_velocity = -self.joystick.get_axis(1)
                goal_y_velocity = -self.joystick.get_axis(0)
                goal_yaw_velocity = -self.joystick.get_axis(3)
                explicit_velocity_commands = True
            elif Path("commands.txt").is_file():
                with open("commands.txt", "r") as f:
                    commands = f.readlines()
                if len(commands) == 3:
                    goal_x_velocity = float(commands[0])
                    goal_y_velocity = float(commands[1])
                    goal_yaw_velocity = float(commands[2])
                    explicit_velocity_commands = True
            if explicit_velocity_commands:
                goal_velocities = jnp.array([goal_x_velocity, goal_y_velocity, goal_yaw_velocity])
                goal_velocities = jnp.where(jnp.abs(goal_velocities) < (self.command_function.zero_clip_threshold_percentage * state.internal_state["max_command_velocity"]), 0.0, goal_velocities)
                goal_velocities = jnp.clip(goal_velocities, -state.internal_state["max_command_velocity"], state.internal_state["max_command_velocity"])
                state.internal_state["goal_velocities"] = jnp.tile(goal_velocities, (self.nr_envs, 1))
                actuator_joint_keep_nominal = jnp.where(jnp.all(goal_velocities == 0.0), jnp.ones(self.nr_actuator_joints, dtype=bool), self.command_function.default_actuator_joint_keep_nominal)
                state.internal_state["actuator_joint_keep_nominal"] = jnp.tile(actuator_joint_keep_nominal, (self.nr_envs, 1))

        if self.stand_after_ball_plate_drop:
            post_drop = state.internal_state["ball_plate_drop_latched"]
            state.internal_state["goal_velocities"] = jnp.where(
                post_drop[:, None],
                jnp.zeros_like(state.internal_state["goal_velocities"]),
                state.internal_state["goal_velocities"],
            )
            state.internal_state["actuator_joint_keep_nominal"] = jnp.where(
                post_drop[:, None],
                jnp.ones_like(state.internal_state["actuator_joint_keep_nominal"], dtype=bool),
                state.internal_state["actuator_joint_keep_nominal"],
            )
            if "goal_gait_frequency" in state.internal_state:
                state.internal_state["goal_gait_frequency"] = jnp.where(
                    post_drop,
                    jnp.zeros_like(state.internal_state["goal_gait_frequency"]),
                    state.internal_state["goal_gait_frequency"],
                )

        if self.add_goal_arrow:
            goal_velocities = state.internal_state["goal_velocities"][env_id]
            trunk_rotation = state.internal_state["imu_orientation_euler"][env_id][2]
            desired_angle = trunk_rotation + np.arctan2(goal_velocities[1], goal_velocities[0])
            rot_mat = Rotation_NP.from_euler('xyz', (np.array([np.pi/2, 0, np.pi/2 + desired_angle]))).as_matrix()
            data.site("dir_arrow").xmat = rot_mat.reshape((9,))
            magnitude = np.sqrt(np.sum(np.square([goal_velocities[0], goal_velocities[1]])))
            mj_model.site_size[self.dir_arrow_id, 1] = magnitude * 0.1
            arrow_offset = -(0.1 - (magnitude * 0.1))
            data.site("dir_arrow").xpos += [arrow_offset * np.sin(np.pi/2 + desired_angle), -arrow_offset * np.cos(np.pi/2 + desired_angle), 0]
            data.site("dir_arrow_ball").xpos = data.body("dir_arrow").xpos + [-0.1 * np.sin(np.pi/2 + desired_angle), 0.1 * np.cos(np.pi/2 + desired_angle), 0]
        
        self.viewer.render(data)

        return state
    

    @partial(jax.vmap, in_axes=(None, 0, None))
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        mjx_model = self.initial_mjx_model
        data = self.mjx_data

        next_observation = jnp.zeros(self.single_observation_space.shape, dtype=jnp.float32)
        reward = 0.0
        terminated = False
        truncated = False
        last_action = jnp.zeros(self.nr_actuator_joints)
        history_stack = jnp.tile(next_observation[None, :], (self.nr_history_steps, 1))

        last_observation = next_observation
        internal_state = {
            "in_eval_mode": eval_mode,
            "env_curriculum_coeff": jnp.where(eval_mode, 1.0, 0.0),
            "env_curriculum_levels_in_a_row": 0.0,
            "actuator_joint_nominal_positions": self.initial_qpos[self.actuator_joint_mask_qpos],
            "actuator_joint_max_velocities": self.actuator_joint_max_velocities,
            "goal_velocities": jnp.array([0.0, 0.0, 0.0]),
            "imu_orientation_rotation": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]),
            "imu_orientation_rotation_inverse": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]).inv(),
            "imu_orientation_euler": jnp.array([0.0, 0.0, 0.0]),
            "last_action": jnp.zeros(self.nr_actuator_joints),
            "second_last_action": jnp.zeros(self.nr_actuator_joints),
            "last_state": last_observation,
            "joint_dropout_mask": jnp.ones(self.nr_actuator_joints, dtype=bool),
            "robot_dimensions_mean": self.robot_dimensions_mean,
            "body_tilt_threshold": self.env_config["termination"]["body_tilt_threshold"],
            "max_command_velocity": jnp.minimum(self.robot_dimensions_mean * self.command_function.max_velocity_per_m_factor, self.command_function.clip_max_velocity),
            "ball_plate_ball_radius": self.ball_plate_ball_radius if self.use_ball_plate else 0.0,
            "ball_plate_ball_mass": self.ball_plate_ball_mass if self.use_ball_plate else 0.0,
            "ball_plate_plate_size": self.ball_plate_plate_size if self.use_ball_plate else jnp.zeros(3),
            "ball_plate_plate_mass": self.ball_plate_plate_mass if self.use_ball_plate else 0.0,
            "ball_plate_plate_ball_friction_tangential": self.ball_plate_nominal_plate_ball_friction if self.use_ball_plate else 0.0,
            "ball_plate_plate_support_friction_tangential": self.ball_plate_nominal_support_friction if self.use_ball_plate else 0.0,
            "ball_plate_plate_robot_friction_tangential": self.ball_plate_nominal_robot_friction if self.use_ball_plate else 0.0,
            "ball_plate_contact_timeconst": self.ball_plate_nominal_contact_timeconst if self.use_ball_plate else 0.0,
            "ball_plate_capsule_hand_radius": self.ball_plate_capsule_hand_radius if self.use_ball_plate else 0.0,
            "ball_plate_capsule_hand_half_length": self.ball_plate_capsule_hand_half_length if self.use_ball_plate else 0.0,
            "ball_plate_capsule_hand_mass": self.ball_plate_capsule_hand_mass if self.use_ball_plate else 0.0,
            "ball_plate_ball_dropped": False,
            "ball_plate_plate_dropped": False,
            "ball_plate_dropped": False,
            "ball_plate_drop_latched": False,
            "ball_plate_newly_dropped": False,
            "ball_plate_time_since_drop": 0.0,
            "ball_plate_post_drop_truncated": False,
            "nr_collisions_in_nominal": 0,
            "history_stack": history_stack,
            "obstacle_pos": jnp.zeros((self.nr_obstacles, 2)),
            "obstacle_vel": jnp.zeros((self.nr_obstacles, 2)),
            "obstacle_radius": jnp.zeros(self.nr_obstacles),
            "obstacle_age": jnp.zeros(self.nr_obstacles),
            "obstacle_lifetime": jnp.zeros(self.nr_obstacles),
            "obstacle_heading": jnp.zeros(self.nr_obstacles),
            "obstacle_shape": jnp.zeros(self.nr_obstacles, dtype=jnp.int32),
            "obstacle_prim_kind": jnp.zeros((self.nr_obstacles, 3), dtype=jnp.int32),
            "obstacle_prim_offset": jnp.zeros((self.nr_obstacles, 3, 2)),
            "obstacle_prim_half": jnp.zeros((self.nr_obstacles, 3, 2)),
            "obstacle_prim_angle": jnp.zeros((self.nr_obstacles, 3)),
            "obstacle_motion": jnp.zeros(self.nr_obstacles, dtype=jnp.int32),
            "obstacle_speed": jnp.zeros(self.nr_obstacles),
            "obstacle_move_dir": jnp.zeros(self.nr_obstacles),
            "obstacle_turn_rate": jnp.zeros(self.nr_obstacles),
            "obstacle_accel": jnp.zeros(self.nr_obstacles),
            "obstacle_target_speed": jnp.zeros(self.nr_obstacles),
            "obstacle_period": jnp.ones(self.nr_obstacles),
            "obstacle_duty": jnp.ones(self.nr_obstacles),
            "obstacle_phase": jnp.zeros(self.nr_obstacles),
            "obstacle_collision": False,
            "obstacle_min_clearance": 0.0,
            "obstacle_scan_noise": 0.0,
            "obstacle_scan_dropout": 0.0,
            "obstacle_scan_latency": jnp.asarray(0, dtype=jnp.int32),
            "obstacle_scan_buffer": jnp.zeros((self.obstacle_scan_buffer_length, self.obstacle_scan_nr_rays)),
            "obstacle_scan_buffer_valid": jnp.asarray(False),
        }
        self.command_function.init(internal_state)
        self.reward_function.init(internal_state, mjx_model)
        self.terrain_function.init(internal_state)
        self.joint_dropout_function.init(internal_state)
        self.domain_randomization_action_delay_function.init(internal_state)
        self.domain_randomization_seen_robot_function.init(internal_state)
        self.domain_randomization_unseen_robot_function.init(internal_state)

        info = {}
        self.reward_function.reward_and_info(data, mjx_model, internal_state, jnp.zeros(self.nr_actuator_joints), info)
        info["rollout/episode_return"] = reward
        info["rollout/episode_length"] = 0
        info["env_curriculum/coefficient"] = internal_state["env_curriculum_coeff"]
        info["constraint/cost"] = 0.0
        info["constraint/terminated"] = False
        self._init_transition_record_info(info)
        info["constraint/safe_time"] = 0.0
        info["constraint/episode_cost"] = 0.0
        info_episode_store = {
            "episode_return": reward,
            "episode_step": 0,
            "episode_total_xy_velocity_diff_abs": 0.0,
            "episode_safe_time": 0.0,
            "episode_cost": 0.0,
        }

        state = State(mjx_model, data, next_observation, next_observation, reward, terminated, truncated, info, info_episode_store, internal_state, key, last_action, last_observation, history_stack)
        
        return self._reset(state)


    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.jit, static_argnums=(0,))
    def _vmap_reset(self, state):
        return self._reset(state)


    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        key, initial_state_key, terrain_key, domain_randomization_key, ball_plate_key, command_key, observation_key, obstacle_key = jax.random.split(state.key, 8)
        state = state.replace(key=key)

        mjx_model = self.terrain_function.sample(state.mjx_model, state.internal_state, terrain_key)

        data = self.mjx_data
        qpos, qvel = self.initial_state_function.setup(mjx_model, state.internal_state, initial_state_key)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(self.nr_actuator_joints))
        # data = mjx.forward(self.initial_mjx_model, data)

        last_action = jnp.zeros(self.nr_actuator_joints)
        last_state = jnp.zeros(self.single_observation_space.shape)

        new_state = state
        new_internal_state = dict(new_state.internal_state)

        episode_success = new_state.info_episode_store["episode_return"] >= self.env_curriculum_level_success_episode_return
        new_internal_state["env_curriculum_levels_in_a_row"] = jnp.where(episode_success,
            jnp.where(new_state.internal_state["env_curriculum_levels_in_a_row"] >= 0,
                new_state.internal_state["env_curriculum_levels_in_a_row"] + 1,
                1
            ),
            jnp.where(new_state.internal_state["env_curriculum_levels_in_a_row"] < 0,
                new_state.internal_state["env_curriculum_levels_in_a_row"] - 1,
                -1
            )
        )
        new_internal_state["env_curriculum_coeff"] =  jnp.clip(
            new_internal_state["env_curriculum_coeff"] +
            new_internal_state["env_curriculum_levels_in_a_row"] / self.env_curriculum_nr_levels, 0.0, 1.0)
        new_internal_state["env_curriculum_coeff"] = jnp.where(
            new_internal_state["in_eval_mode"], 1.0, new_internal_state["env_curriculum_coeff"])
        new_internal_state["imu_orientation_rotation"] = Rotation.from_matrix(data.site_xmat[self.imu_site_id].reshape(3, 3))
        new_internal_state["imu_orientation_rotation_inverse"] = new_internal_state["imu_orientation_rotation"].inv()
        new_internal_state["imu_orientation_euler"] = new_internal_state["imu_orientation_rotation"].as_euler("xyz")
        new_internal_state["last_action"] = last_action
        new_internal_state["second_last_action"] = jnp.zeros(self.nr_actuator_joints)

        self.reward_function.setup(new_internal_state)
        self.domain_randomization_action_delay_function.setup(new_internal_state)
        data, mjx_model = self.handle_domain_randomization(new_internal_state, mjx_model, data, domain_randomization_key, is_episode_start=True)
        data, mjx_model = self._reset_ball_plate_state(data, mjx_model, new_internal_state, ball_plate_key)
        if self.forward_after_state_writes:  # F13
            data = mjx.forward(mjx_model, data)
        new_internal_state["imu_orientation_rotation"] = Rotation.from_matrix(data.site_xmat[self.imu_site_id].reshape(3, 3))
        new_internal_state["imu_orientation_rotation_inverse"] = new_internal_state["imu_orientation_rotation"].inv()
        new_internal_state["imu_orientation_euler"] = new_internal_state["imu_orientation_rotation"].as_euler("xyz")
        should_sample_commands = self.command_sampling_function.setup(command_key)
        self.command_function.get_next_command(new_internal_state, should_sample_commands, command_key)
        self._apply_ball_plate_post_drop_command(new_internal_state)
        self._reset_obstacles(data, new_internal_state, obstacle_key)

        next_observation = self.get_observation(data, mjx_model, new_internal_state, observation_key, jnp.zeros(self.nr_actuator_joints))
        new_info = dict(new_state.info)
        if self.use_obstacles:
            new_info["env_info/obstacle_collision"] = jnp.asarray(new_internal_state["obstacle_collision"], dtype=jnp.float32)
            new_info["env_info/obstacle_min_clearance"] = jnp.asarray(new_internal_state["obstacle_min_clearance"], dtype=jnp.float32)
        if self.use_ball_plate:
            ball_plate_metrics = self.get_ball_plate_metrics(data, new_internal_state)
            new_info[f"env_info/ball_plate_radial_distance"] = ball_plate_metrics["radial_distance"]
            new_info[f"env_info/ball_plate_on_plate"] = ball_plate_metrics["on_plate"].astype(jnp.float32)
            new_info[f"env_info/ball_plate_ball_dropped"] = ball_plate_metrics["ball_dropped"].astype(jnp.float32)
            new_info[f"env_info/ball_plate_plate_dropped"] = ball_plate_metrics["plate_dropped"].astype(jnp.float32)
            new_info[f"env_info/ball_plate_dropped"] = ball_plate_metrics["dropped"].astype(jnp.float32)
            new_info[f"env_info/ball_plate_drop_latched"] = jnp.asarray(new_internal_state["ball_plate_drop_latched"], dtype=jnp.float32)
            new_info[f"env_info/ball_plate_time_since_drop"] = new_internal_state["ball_plate_time_since_drop"]
            new_info[f"env_info/ball_plate_post_drop_truncated"] = jnp.asarray(new_internal_state["ball_plate_post_drop_truncated"], dtype=jnp.float32)
        last_state = next_observation
        new_internal_state["last_state"] = last_state
        reward = 0.0
        terminated = False
        truncated = False
        info_episode_store = {
            "episode_return": reward,
            "episode_step": 0,
            "episode_total_xy_velocity_diff_abs": 0.0,
            "episode_safe_time": 0.0,
            "episode_cost": 0.0,
        }

        history_stack = jnp.tile(next_observation[None, :], (self.nr_history_steps, 1))
        new_internal_state["history_stack"] = history_stack

        # Reset everything besides parts of the internal_state, info and the key
        new_state = new_state.replace(
            mjx_model=mjx_model,
            data=data,
            next_observation=next_observation,
            actual_next_observation=next_observation,
            reward=reward,
            terminated=terminated, truncated=truncated,
            info=new_info,
            info_episode_store=info_episode_store,
            last_state=last_state,
            last_action=last_action,
            history_stack=history_stack,
            internal_state=new_internal_state,
        )

        return new_state


    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self._step(state, action)


    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        key, action_delay_key, domain_randomization_key, command_sampling_key, command_key, observation_key, terrain_key, obstacle_key = jax.random.split(state.key, 8)
        state = state.replace(key=key)

        chosen_action = action[:self.nr_actuator_joints]
        delayed_action = self.domain_randomization_action_delay_function.delay_action(chosen_action, state.internal_state, action_delay_key)
        if self.record_wm_state:
            from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_state
            wm_delay_steps = wm_state.realized_delay_steps(self, state.internal_state, action_delay_key)

        target_joint_positions = self.control_function.process_action(delayed_action, state.internal_state)

        data, _ = jax.lax.scan(
            f=lambda data, _: (mjx.step(state.mjx_model, data.replace(ctrl=target_joint_positions)), None),
            init=state.data,
            xs=(),
            length=self.nr_substeps
        )
        max_qvel = 100 * jnp.ones(self.initial_mj_model.nv)
        max_qvel = max_qvel.at[self.actuator_joint_mask_qvel].set(state.internal_state["actuator_joint_max_velocities"])
        data = data.replace(qvel=jnp.clip(data.qvel, -max_qvel, max_qvel))

        state.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(data.site_xmat[self.imu_site_id].reshape(3, 3))
        state.internal_state["imu_orientation_rotation_inverse"] = state.internal_state["imu_orientation_rotation"].inv()
        state.internal_state["imu_orientation_euler"] = state.internal_state["imu_orientation_rotation"].as_euler("xyz")

        # x^-_{t+1}: physics state after the physics substeps and the qvel clip, before any shock / DR resampling.
        pre_shock_qpos, pre_shock_qvel = data.qpos, data.qvel
        if self.record_wm_state:
            wm_x_minus = wm_state.extract_state(self, data, state.internal_state, chosen_action)
        data, mjx_model, dr_resampled, perturbed = self.handle_domain_randomization(
            state.internal_state, state.mjx_model, data, domain_randomization_key, return_flags=True)
        if self.forward_after_state_writes:  # F13: sensors / site poses after the shock and DR writes
            data = mjx.forward(mjx_model, data)
            state.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(data.site_xmat[self.imu_site_id].reshape(3, 3))
            state.internal_state["imu_orientation_rotation_inverse"] = state.internal_state["imu_orientation_rotation"].inv()
            state.internal_state["imu_orientation_euler"] = state.internal_state["imu_orientation_rotation"].as_euler("xyz")
        state = state.replace(data=data, mjx_model=mjx_model)

        self.terrain_function.pre_step(data, state.internal_state)

        reward = self.reward_function.reward_and_info(data, mjx_model, state.internal_state, chosen_action, state.info)

        post_drop_truncated = self._update_ball_plate_post_drop_state(state.internal_state)

        should_sample_commands = self.command_sampling_function.step(command_sampling_key)
        self.command_function.get_next_command(state.internal_state, should_sample_commands, command_key)
        self._apply_ball_plate_post_drop_command(state.internal_state)
        self._step_obstacles(data, state.internal_state, obstacle_key)

        next_observation = self.get_observation(data, mjx_model, state.internal_state, observation_key, chosen_action)
        constraint_terminated = self.termination_function.should_terminate(state.internal_state)
        if self.use_obstacles:
            if self.obstacle_config.get("terminate_on_collision", True):
                constraint_terminated = constraint_terminated | state.internal_state["obstacle_collision"]
            reward = reward - self.obstacle_config.get("collision_penalty", 0.0) * state.internal_state["obstacle_collision"].astype(jnp.float32)
        qvel_limit_terminated = jnp.any(jnp.abs(data.qvel[:3]) == 100.0)
        terminated = constraint_terminated | qvel_limit_terminated
        truncated = (state.info_episode_store["episode_step"] >= (self.horizon - 1)) | post_drop_truncated
        done = terminated | truncated

        data = self.terrain_function.post_step(data, mjx_model, state.internal_state, terrain_key)
        self.reward_function.step(data, state.internal_state)

        previous_obs = state.internal_state["last_state"]
        last_action = chosen_action
        history_stack = state.internal_state["history_stack"]
        new_history_stack = jnp.roll(history_stack, -1, axis=0).at[-1].set(next_observation)

        new_internal_state = dict(state.internal_state)
        new_internal_state["second_last_action"] = state.internal_state["last_action"]
        new_internal_state["last_action"] = chosen_action
        new_internal_state["last_state"] = next_observation
        new_internal_state["history_stack"] = new_history_stack

        new_info_episode_store = dict(state.info_episode_store)
        new_info_episode_store["episode_step"] += 1
        new_info_episode_store["episode_return"] += reward
        new_info_episode_store["episode_total_xy_velocity_diff_abs"] = state.info["env_info/xy_vel_diff_abs"]
        new_info_episode_store["episode_safe_time"] += jnp.where(constraint_terminated, 0.0, self.dt)

        safe_time_limit = jnp.array(self.safe_time_limit_seconds, dtype=jnp.float32)
        safe_time_shortfall = jnp.where(
            safe_time_limit > 0.0,
            jnp.maximum(safe_time_limit - new_info_episode_store["episode_safe_time"], 0.0) / (safe_time_limit + 1e-8),
            1.0,
        )
        constraint_cost = constraint_terminated.astype(jnp.float32) * safe_time_shortfall
        new_info_episode_store["episode_cost"] += constraint_cost

        new_info = dict(state.info)
        new_info["rollout/episode_return"] = jnp.where(done, new_info_episode_store["episode_return"],
                                                         state.info["rollout/episode_return"])
        new_info["rollout/episode_length"] = jnp.where(done, new_info_episode_store["episode_step"],
                                                         state.info["rollout/episode_length"])
        new_info["env_curriculum/coefficient"] = state.internal_state["env_curriculum_coeff"]
        new_info["constraint/cost"] = constraint_cost
        new_info["constraint/terminated"] = constraint_terminated
        new_info["constraint/safe_time"] = new_info_episode_store["episode_safe_time"]
        new_info["constraint/episode_cost"] = new_info_episode_store["episode_cost"]
        new_info["shock/perturbed"] = jnp.asarray(perturbed, dtype=jnp.float32)
        new_info["shock/dr_resampled"] = jnp.asarray(dr_resampled, dtype=jnp.float32)
        if self.record_transition_physics_state:
            # x^+_{t+1} is the state after the shock / DR step (before a possible episode reset).
            new_info["transition/x_minus_qpos"] = pre_shock_qpos
            new_info["transition/x_minus_qvel"] = pre_shock_qvel
            new_info["transition/x_plus_qpos"] = data.qpos
            new_info["transition/x_plus_qvel"] = data.qvel
        if self.record_wm_state:
            new_info["wm/x_minus"] = wm_x_minus
            new_info["wm/x_plus"] = wm_state.extract_state(self, data, state.internal_state, chosen_action)
            new_info["wm/xi_next"] = wm_state.extract_xi(self, mjx_model, state.internal_state)
            new_info["wm/u_applied"] = jnp.asarray(delayed_action, dtype=jnp.float32)
            new_info["wm/delay_steps"] = jnp.asarray(wm_delay_steps, dtype=jnp.int32)
        if self.use_obstacles:
            new_info["env_info/obstacle_collision"] = jnp.asarray(state.internal_state["obstacle_collision"], dtype=jnp.float32)
            new_info["env_info/obstacle_min_clearance"] = jnp.asarray(state.internal_state["obstacle_min_clearance"], dtype=jnp.float32)
        if self.use_ball_plate:
            new_info["env_info/ball_plate_drop_latched"] = jnp.asarray(state.internal_state["ball_plate_drop_latched"], dtype=jnp.float32)
            new_info["env_info/ball_plate_time_since_drop"] = state.internal_state["ball_plate_time_since_drop"]
            new_info["env_info/ball_plate_post_drop_truncated"] = jnp.asarray(state.internal_state["ball_plate_post_drop_truncated"], dtype=jnp.float32)


        state = state.replace(internal_state=new_internal_state, info=new_info, info_episode_store=new_info_episode_store)

        def when_done(_):
            start_state = self._reset(state)
            start_state = start_state.replace(
                actual_next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=state.info,
            )
            return start_state
        def when_not_done(_):
            return state.replace(data=data, next_observation=next_observation, actual_next_observation=next_observation,
                                 reward=reward, terminated=terminated, truncated=truncated,
                                 last_state=previous_obs, last_action=last_action, history_stack=new_history_stack)
        state = jax.lax.cond(done, when_done, when_not_done, None)

        return state


    def get_observation(self, data, mjx_model, internal_state, key, action):
        # feet ground contact appears twice in the observation, one normalized for critic, one originaly for forward dynamics model
        feet_ground_contact = self.terrain_function.check_feet_floor_contact(data)

        # omit unactuated joints from observation
        qpos = jnp.concatenate([data.qpos[:7], data.qpos[self.actuator_joint_mask_qpos]], axis=0)
        qvel = jnp.concatenate([data.qvel[:6], data.qvel[self.actuator_joint_mask_qvel]], axis=0)

        robot_height = internal_state["robot_imu_height_over_ground"]
        if self.env_config["termination"]["type"] == "booster":
            robot_height_safe = (
                (robot_height >= self.env_config["termination"]["min_height"]) &
                (robot_height <= self.env_config["termination"]["max_height"])
            ).astype(jnp.float32)
        else:
            robot_height_threshold = self.env_config["termination"]["height_percentage_threshold"] * internal_state["robot_nominal_imu_height_over_ground"]
            robot_height_safe = (robot_height >= robot_height_threshold).astype(jnp.float32)

        body_roll = internal_state["imu_orientation_euler"][0]
        body_pitch = internal_state["imu_orientation_euler"][1]
        body_tilt = jnp.sqrt(body_roll ** 2 + body_pitch ** 2)
        body_tilt_safe = jnp.where(
            self.env_config["termination"]["type"] == "booster",
            1.0,
            (body_tilt <= internal_state["body_tilt_threshold"]).astype(jnp.float32),
        )

        command_observation = self.command_function.get_observation(internal_state) \
            if hasattr(self.command_function, "get_observation") else internal_state["goal_velocities"]

        if self.use_ball_plate:
            ball_plate_metrics = self.get_ball_plate_metrics(data, internal_state)
            internal_state["ball_plate_ball_dropped"] = ball_plate_metrics["ball_dropped"]
            internal_state["ball_plate_plate_dropped"] = ball_plate_metrics["plate_dropped"]
            internal_state["ball_plate_dropped"] = ball_plate_metrics["dropped"]
            ball_not_falling = (~ball_plate_metrics["ball_dropped"]).astype(jnp.float32)
            plate_not_falling = (~ball_plate_metrics["plate_dropped"]).astype(jnp.float32)
            ball_plate_has_dropped = (
                internal_state["ball_plate_drop_latched"] | ball_plate_metrics["dropped"]
            ).astype(jnp.float32)
            ball_plate_safety_observation = jnp.array([ball_not_falling, plate_not_falling, ball_plate_has_dropped])
        else:
            ball_plate_safety_observation = jnp.array([], dtype=jnp.float32)

        key, obstacle_noise_key = jax.random.split(key)
        if self.use_obstacles:
            obstacle_safety_observation = jnp.array([(~internal_state["obstacle_collision"]).astype(jnp.float32)])
        else:
            obstacle_safety_observation = jnp.array([], dtype=jnp.float32)

        observation = jnp.concatenate([
            data.qpos[self.actuator_joint_mask_qpos],
            data.qvel[self.actuator_joint_mask_qvel],
            action,
            feet_ground_contact,
            internal_state["feet_time_on_ground"],
            internal_state["feet_time_in_air"],
            data.sensordata[self.imu_linear_velocity_sensor_adr:self.imu_linear_velocity_sensor_adr + self.imu_linear_velocity_sensor_dim],
            data.sensordata[self.imu_angular_velocity_sensor_adr:self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim],
            command_observation,
            internal_state["imu_orientation_rotation_inverse"].apply(jnp.array([0.0, 0.0, -1.0])),
            self.get_ball_plate_observation(data, internal_state),
            self.get_obstacle_observation(data, internal_state, obstacle_noise_key),
            jnp.array([self.policy_exteroceptive_observation_function.get_exteroceptive_observation(data, mjx_model, internal_state)]).reshape(-1),
            jnp.array([self.critic_exteroceptive_observation_function.get_exteroceptive_observation(data, mjx_model, internal_state)]).reshape(-1),
            qpos, # qpos all not normalized, base pose can be dummy for deployment
            qvel, # qvel all not normalized
            feet_ground_contact,
            jnp.array([robot_height_safe]),
            jnp.array([body_tilt_safe]),
            ball_plate_safety_observation,
            obstacle_safety_observation,
        ])

        # Add noise
        observation = self.observation_noise_function.modify_observation(internal_state, observation, key)

        if not getattr(self.observation_noise_function, "handles_normalization", False):
            # Normalize and clip
            observation = observation.at[self.joint_positions_obs_idx].set((observation[self.joint_positions_obs_idx] - internal_state["actuator_joint_nominal_positions"]) / 3.14)
            observation = observation.at[self.joint_velocities_obs_idx].set(observation[self.joint_velocities_obs_idx] / 100.0)
            observation = observation.at[self.joint_previous_actions_obs_idx].set(observation[self.joint_previous_actions_obs_idx] / 10.0)
            observation = observation.at[self.feet_ground_contact_obs_idx].set((observation[self.feet_ground_contact_obs_idx] / 0.5) - 1.0)
            observation = observation.at[self.feet_time_on_ground_obs_idx].set(jnp.clip((observation[self.feet_time_on_ground_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0))
            observation = observation.at[self.feet_time_in_air_obs_idx].set(jnp.clip((observation[self.feet_time_in_air_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0))
            observation = observation.at[self.imu_linear_vel_obs_idx].set(jnp.clip(observation[self.imu_linear_vel_obs_idx] / 10.0, -1.0, 1.0))
            observation = observation.at[self.imu_angular_vel_obs_idx].set(jnp.clip(observation[self.imu_angular_vel_obs_idx] / 50.0, -1.0, 1.0))
            if len(self.ball_plate_obs_idx) > 0:
                ball_plate_plate_size = internal_state["ball_plate_plate_size"]
                observation = observation.at[self.ball_plate_obs_idx[:3]].set(jnp.clip(observation[self.ball_plate_obs_idx[:3]] / jnp.maximum(jnp.max(ball_plate_plate_size[:2]), 1e-6), -10.0, 10.0))
                observation = observation.at[self.ball_plate_obs_idx[3:6]].set(jnp.clip(observation[self.ball_plate_obs_idx[3:6]] / 5.0, -10.0, 10.0))
                observation = observation.at[self.ball_plate_obs_idx[6:9]].set(jnp.clip(observation[self.ball_plate_obs_idx[6:9]] / jnp.maximum(jnp.max(ball_plate_plate_size[:2]), 1e-6), -10.0, 10.0))
                observation = observation.at[self.ball_plate_obs_idx[9:]].set(jnp.clip(observation[self.ball_plate_obs_idx[9:]] / 5.0, -10.0, 10.0))
            if len(self.policy_exteroception_obs_idx) > 0:
                observation = observation.at[self.policy_exteroception_obs_idx].set(jnp.clip((observation[self.policy_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0))
            if len(self.critic_exteroception_obs_idx) > 0:
                observation = observation.at[self.critic_exteroception_obs_idx].set(jnp.clip((observation[self.critic_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0))

        observation = jnp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = jnp.clip(observation, -10.0, 10.0)

        return observation


    def _reset_ball_plate_state(self, data, mjx_model, internal_state, key):
        if not self.use_ball_plate:
            return data, mjx_model

        ball_plate_params = self._sample_ball_plate_domain_randomization(key, internal_state)
        ball_radius = ball_plate_params["ball_radius"]
        ball_mass = ball_plate_params["ball_mass"]
        plate_size = ball_plate_params["plate_size"]
        plate_mass = ball_plate_params["plate_mass"]
        mjx_model = self._apply_ball_plate_domain_randomization(mjx_model, ball_plate_params)

        data = mjx.forward(mjx_model, data)
        plate_pos, plate_quat = self._get_ball_plate_supported_pose(
            data,
            plate_size,
            ball_plate_params["capsule_hand_radius"],
            ball_plate_params["capsule_hand_half_length"],
        )
        qpos = data.qpos.at[self.ball_plate_qposadr:self.ball_plate_qposadr + 3].set(plate_pos)
        qpos = qpos.at[self.ball_plate_qposadr + 3:self.ball_plate_qposadr + 7].set(plate_quat)
        qvel = data.qvel.at[self.ball_plate_qveladr:self.ball_plate_qveladr + 6].set(0.0)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(mjx_model, data)

        plate_xmat = data.site_xmat[self.ball_plate_site_id].reshape(3, 3)
        ball_start_offset = ball_plate_params["ball_start_offset"].at[2].set(ball_radius)
        ball_pos = data.site_xpos[self.ball_plate_site_id] + plate_xmat @ ball_start_offset

        qpos = data.qpos.at[self.ball_plate_ball_qposadr:self.ball_plate_ball_qposadr + 3].set(ball_pos)
        qpos = qpos.at[self.ball_plate_ball_qposadr + 3:self.ball_plate_ball_qposadr + 7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
        ball_qvel = jnp.zeros(6, dtype=jnp.float32)
        ball_qvel = ball_qvel.at[:2].set(ball_plate_params["ball_initial_linear_velocity_xy"])
        ball_qvel = ball_qvel.at[3:].set(ball_plate_params["ball_initial_angular_velocity"])
        qvel = data.qvel.at[self.ball_plate_ball_qveladr:self.ball_plate_ball_qveladr + 6].set(ball_qvel)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(mjx_model, data)

        internal_state["ball_plate_ball_radius"] = ball_radius
        internal_state["ball_plate_ball_mass"] = ball_mass
        internal_state["ball_plate_plate_size"] = plate_size
        internal_state["ball_plate_plate_mass"] = plate_mass
        internal_state["ball_plate_plate_ball_friction_tangential"] = ball_plate_params["plate_ball_friction_tangential"]
        internal_state["ball_plate_plate_support_friction_tangential"] = ball_plate_params["plate_support_friction_tangential"]
        internal_state["ball_plate_plate_robot_friction_tangential"] = ball_plate_params["plate_robot_friction_tangential"]
        internal_state["ball_plate_contact_timeconst"] = ball_plate_params["contact_timeconst"]
        internal_state["ball_plate_capsule_hand_radius"] = ball_plate_params["capsule_hand_radius"]
        internal_state["ball_plate_capsule_hand_half_length"] = ball_plate_params["capsule_hand_half_length"]
        internal_state["ball_plate_capsule_hand_mass"] = ball_plate_params["capsule_hand_mass"]
        internal_state["ball_plate_ball_dropped"] = False
        internal_state["ball_plate_plate_dropped"] = False
        internal_state["ball_plate_dropped"] = False
        internal_state["ball_plate_drop_latched"] = False
        internal_state["ball_plate_newly_dropped"] = False
        internal_state["ball_plate_time_since_drop"] = 0.0
        internal_state["ball_plate_post_drop_truncated"] = False
        return data, mjx_model


    def _get_ball_plate_support_points(self, data):
        left_body_xmat = data.xmat[self.ball_plate_left_fist_body_id].reshape(3, 3)
        right_body_xmat = data.xmat[self.ball_plate_right_fist_body_id].reshape(3, 3)
        left_support = data.xpos[self.ball_plate_left_fist_body_id] + left_body_xmat @ self.ball_plate_left_fist_pos
        right_support = data.xpos[self.ball_plate_right_fist_body_id] + right_body_xmat @ self.ball_plate_right_fist_pos
        return left_support, right_support


    def _get_ball_plate_support_surface_height(self, data, capsule_radius=None, capsule_half_length=None):
        left_support, right_support = self._get_ball_plate_support_points(data)
        fist_radius = float(self.ball_plate_config["fist_radius"]) if capsule_radius is None else capsule_radius
        if not self.ball_plate_config.get("add_support_fist_geoms", True):
            return jnp.maximum(left_support[2], right_support[2]) + fist_radius

        left_body_xmat = data.xmat[self.ball_plate_left_fist_body_id].reshape(3, 3)
        right_body_xmat = data.xmat[self.ball_plate_right_fist_body_id].reshape(3, 3)
        fist_half_length = float(self.ball_plate_config["fist_half_length"]) if capsule_half_length is None else capsule_half_length
        left_capsule_axis = left_body_xmat[:, 0]
        right_capsule_axis = right_body_xmat[:, 0]
        left_endpoint_heights = jnp.array([
            left_support[2] - left_capsule_axis[2] * fist_half_length,
            left_support[2] + left_capsule_axis[2] * fist_half_length,
        ])
        right_endpoint_heights = jnp.array([
            right_support[2] - right_capsule_axis[2] * fist_half_length,
            right_support[2] + right_capsule_axis[2] * fist_half_length,
        ])
        capsule_axis_height = jnp.maximum(jnp.max(left_endpoint_heights), jnp.max(right_endpoint_heights))
        return capsule_axis_height + fist_radius


    def _get_ball_plate_supported_pose(self, data, plate_size=None, capsule_radius=None, capsule_half_length=None):
        if plate_size is None:
            plate_size = self.ball_plate_plate_size
        left_support, right_support = self._get_ball_plate_support_points(data)
        support_center = 0.5 * (left_support + right_support)
        support_surface_height = self._get_ball_plate_support_surface_height(data, capsule_radius, capsule_half_length)
        plate_pos = support_center.at[2].set(
            support_surface_height
            + plate_size[2]
            + self.ball_plate_plate_support_clearance
        )
        return plate_pos, jnp.asarray(self.ball_plate_config["plate_home_quat"], dtype=jnp.float32)


    def _sample_curriculum_scalar(self, key, nominal, value_range, curriculum_coeff, randomize):
        if not randomize:
            return jnp.asarray(nominal, dtype=jnp.float32)
        sampled = jax.random.uniform(key, minval=value_range[0], maxval=value_range[1])
        return jnp.asarray(nominal, dtype=jnp.float32) + curriculum_coeff * (sampled - jnp.asarray(nominal, dtype=jnp.float32))


    def _sample_ball_plate_domain_randomization(self, key, internal_state):
        curriculum_coeff = internal_state["env_curriculum_coeff"]
        if not self.ball_plate_config.get("randomize_ball_plate_domain", True):
            curriculum_coeff = jnp.asarray(0.0, dtype=jnp.float32)

        keys = jax.random.split(key, 14)
        ball_radius = self._sample_curriculum_scalar(
            keys[0],
            self.ball_plate_ball_radius,
            self.ball_plate_ball_radius_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_ball_size_mass", True),
        )
        ball_mass = self._sample_curriculum_scalar(
            keys[1],
            self.ball_plate_ball_mass,
            self.ball_plate_ball_mass_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_ball_size_mass", True),
        )
        plate_mass = self._sample_curriculum_scalar(
            keys[2],
            self.ball_plate_plate_mass,
            self.ball_plate_plate_mass_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_plate_mass", True),
        )

        plate_size_scale = self._sample_curriculum_scalar(
            keys[3],
            1.0,
            self.ball_plate_plate_size_scale_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_plate_size", True),
        )
        plate_size = self.ball_plate_plate_size.at[:2].set(self.ball_plate_plate_size[:2] * plate_size_scale)

        if self.ball_plate_config.get("randomize_ball_initial_state", True):
            ball_start_offset_xy = curriculum_coeff * jax.random.uniform(
                keys[4],
                minval=-self.ball_plate_ball_start_offset_xy_range,
                maxval=self.ball_plate_ball_start_offset_xy_range,
                shape=(2,),
            )
            ball_initial_linear_velocity_xy = curriculum_coeff * jax.random.uniform(
                keys[5],
                minval=-self.ball_plate_ball_initial_linear_velocity_xy_range,
                maxval=self.ball_plate_ball_initial_linear_velocity_xy_range,
                shape=(2,),
            )
            ball_initial_angular_velocity = curriculum_coeff * jax.random.uniform(
                keys[6],
                minval=-self.ball_plate_ball_initial_angular_velocity_range,
                maxval=self.ball_plate_ball_initial_angular_velocity_range,
                shape=(3,),
            )
        else:
            ball_start_offset_xy = jnp.zeros(2, dtype=jnp.float32)
            ball_initial_linear_velocity_xy = jnp.zeros(2, dtype=jnp.float32)
            ball_initial_angular_velocity = jnp.zeros(3, dtype=jnp.float32)
        ball_start_offset = self.ball_plate_ball_start_offset.at[:2].add(ball_start_offset_xy)

        plate_ball_friction_tangential = self._sample_curriculum_scalar(
            keys[7],
            self.ball_plate_nominal_plate_ball_friction,
            self.ball_plate_plate_ball_friction_tangential_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_plate_ball_friction", True),
        )
        plate_support_friction_tangential = self._sample_curriculum_scalar(
            keys[8],
            self.ball_plate_nominal_support_friction,
            self.ball_plate_plate_support_friction_tangential_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_plate_support_friction", True),
        )
        plate_robot_friction_tangential = self._sample_curriculum_scalar(
            keys[9],
            self.ball_plate_nominal_robot_friction,
            self.ball_plate_plate_robot_friction_tangential_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_plate_robot_friction", True),
        )
        contact_timeconst = self._sample_curriculum_scalar(
            keys[10],
            self.ball_plate_nominal_contact_timeconst,
            self.ball_plate_contact_timeconst_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_plate_contact_stiffness", True),
        )
        capsule_hand_radius = self._sample_curriculum_scalar(
            keys[11],
            self.ball_plate_capsule_hand_radius,
            self.ball_plate_capsule_hand_radius_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_capsule_hand_size", True),
        )
        capsule_hand_half_length = self._sample_curriculum_scalar(
            keys[12],
            self.ball_plate_capsule_hand_half_length,
            self.ball_plate_capsule_hand_half_length_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_capsule_hand_size", True),
        )
        capsule_hand_mass = self._sample_curriculum_scalar(
            keys[13],
            self.ball_plate_capsule_hand_mass,
            self.ball_plate_capsule_hand_mass_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_capsule_hand_mass", True),
        )

        return {
            "ball_radius": ball_radius,
            "ball_mass": ball_mass,
            "plate_size": plate_size,
            "plate_mass": plate_mass,
            "ball_start_offset": ball_start_offset,
            "ball_initial_linear_velocity_xy": ball_initial_linear_velocity_xy,
            "ball_initial_angular_velocity": ball_initial_angular_velocity,
            "plate_ball_friction_tangential": plate_ball_friction_tangential,
            "plate_support_friction_tangential": plate_support_friction_tangential,
            "plate_robot_friction_tangential": plate_robot_friction_tangential,
            "contact_timeconst": contact_timeconst,
            "capsule_hand_radius": capsule_hand_radius,
            "capsule_hand_half_length": capsule_hand_half_length,
            "capsule_hand_mass": capsule_hand_mass,
        }


    def _set_pair_tangential_friction(self, pair_friction, pair_ids, value):
        if len(pair_ids) == 0:
            return pair_friction
        pair_friction = pair_friction.at[pair_ids, 0].set(value)
        pair_friction = pair_friction.at[pair_ids, 1].set(value)
        return pair_friction


    def _get_capsule_hand_inertia(self, radius, half_length, mass):
        nominal_size = self.ball_plate_capsule_hand_radius + self.ball_plate_capsule_hand_half_length
        size_scale = (radius + half_length) / jnp.maximum(nominal_size, 1e-6)
        mass_scale = mass / jnp.maximum(self.ball_plate_capsule_hand_mass, 1e-6)
        return jnp.maximum(self.ball_plate_capsule_hand_nominal_body_inertia * mass_scale * size_scale ** 2, 1e-7)


    def _apply_ball_plate_domain_randomization(self, mjx_model, params):
        ball_radius = params["ball_radius"]
        ball_mass = params["ball_mass"]
        plate_size = params["plate_size"]
        plate_mass = params["plate_mass"]
        capsule_hand_radius = params["capsule_hand_radius"]
        capsule_hand_half_length = params["capsule_hand_half_length"]
        capsule_hand_mass = params["capsule_hand_mass"]

        geom_size = mjx_model.geom_size.at[self.ball_plate_ball_geom_id, 0].set(ball_radius)
        geom_size = geom_size.at[self.ball_plate_geom_id].set(plate_size)
        geom_rbound = mjx_model.geom_rbound.at[self.ball_plate_ball_geom_id].set(ball_radius)
        geom_rbound = geom_rbound.at[self.ball_plate_geom_id].set(jnp.linalg.norm(plate_size))
        site_pos = mjx_model.site_pos.at[self.ball_plate_site_id, 2].set(plate_size[2])

        body_mass = mjx_model.body_mass.at[self.ball_plate_ball_body_id].set(ball_mass)
        body_mass = body_mass.at[self.ball_plate_body_id].set(plate_mass)
        ball_inertia = jnp.full(3, 0.4 * ball_mass * ball_radius ** 2)
        plate_inertia = (plate_mass / 3.0) * jnp.array([
            plate_size[1] ** 2 + plate_size[2] ** 2,
            plate_size[0] ** 2 + plate_size[2] ** 2,
            plate_size[0] ** 2 + plate_size[1] ** 2,
        ])
        body_inertia = mjx_model.body_inertia.at[self.ball_plate_ball_body_id].set(ball_inertia)
        body_inertia = body_inertia.at[self.ball_plate_body_id].set(plate_inertia)
        if self.use_capsule_hand:
            nr_capsule_hands = len(self.ball_plate_capsule_hand_geom_ids)
            capsule_geom_size = jnp.broadcast_to(
                jnp.array([capsule_hand_radius, capsule_hand_half_length, 0.0], dtype=jnp.float32),
                (nr_capsule_hands, 3),
            )
            geom_size = geom_size.at[self.ball_plate_capsule_hand_geom_ids].set(capsule_geom_size)
            geom_rbound = geom_rbound.at[self.ball_plate_capsule_hand_geom_ids].set(capsule_hand_radius + capsule_hand_half_length)
            capsule_inertia = self._get_capsule_hand_inertia(capsule_hand_radius, capsule_hand_half_length, capsule_hand_mass)
            body_mass = body_mass.at[self.ball_plate_capsule_hand_body_ids].set(jnp.full((nr_capsule_hands,), capsule_hand_mass))
            body_inertia = body_inertia.at[self.ball_plate_capsule_hand_body_ids].set(jnp.broadcast_to(capsule_inertia, (nr_capsule_hands, 3)))

        pair_friction = mjx_model.pair_friction
        pair_friction = self._set_pair_tangential_friction(pair_friction, self.ball_plate_ball_pair_ids, params["plate_ball_friction_tangential"])
        pair_friction = self._set_pair_tangential_friction(pair_friction, self.ball_plate_support_pair_ids, params["plate_support_friction_tangential"])
        pair_friction = self._set_pair_tangential_friction(pair_friction, self.ball_plate_robot_pair_ids, params["plate_robot_friction_tangential"])

        pair_solref = mjx_model.pair_solref
        if len(self.ball_plate_all_pair_ids) > 0:
            pair_solref = pair_solref.at[self.ball_plate_all_pair_ids, 0].set(params["contact_timeconst"])

        return mjx_model.tree_replace(
            {
                "geom_size": geom_size,
                "geom_rbound": geom_rbound,
                "site_pos": site_pos,
                "body_mass": body_mass,
                "body_inertia": body_inertia,
                "pair_friction": pair_friction,
                "pair_solref": pair_solref,
            }
        )


    def get_ball_plate_metrics(self, data, internal_state):
        if not self.use_ball_plate:
            return {
                "plate_support_relative_position": jnp.zeros(3),
                "plate_velocity": jnp.zeros(3),
                "ball_dropped": False,
                "plate_dropped": False,
                "relative_position": jnp.zeros(3),
                "relative_velocity": jnp.zeros(3),
                "radial_distance": 0.0,
                "on_plate": False,
                "dropped": False,
            }

        plate_pos = data.site_xpos[self.ball_plate_site_id]
        plate_xmat = data.site_xmat[self.ball_plate_site_id].reshape(3, 3)
        ball_pos = data.xpos[self.ball_plate_ball_body_id]
        left_support, right_support = self._get_ball_plate_support_points(data)
        support_center = 0.5 * (left_support + right_support)
        support_height = jnp.maximum(left_support[2], right_support[2])
        relative_position = plate_xmat.T @ (ball_pos - plate_pos)
        relative_velocity = plate_xmat.T @ data.qvel[self.ball_plate_ball_qveladr:self.ball_plate_ball_qveladr + 3]
        plate_support_relative_position = plate_pos - support_center
        plate_velocity = data.qvel[self.ball_plate_qveladr:self.ball_plate_qveladr + 3]
        plate_up = plate_xmat[:, 2]

        plate_size = internal_state["ball_plate_plate_size"]
        outside_x = jnp.abs(relative_position[0]) > (plate_size[0] + self.ball_plate_drop_margin)
        outside_y = jnp.abs(relative_position[1]) > (plate_size[1] + self.ball_plate_drop_margin)
        below_plate = relative_position[2] < -self.ball_plate_drop_height
        near_plate_height = relative_position[2] <= (internal_state["ball_plate_ball_radius"] * 3.0)
        plate_below_support = data.xpos[self.ball_plate_body_id, 2] < (support_height - self.ball_plate_drop_height)
        plate_tilted = plate_up[2] < self.ball_plate_tilt_drop_threshold
        ball_dropped = outside_x | outside_y | below_plate
        plate_dropped = plate_below_support | plate_tilted
        dropped = ball_dropped | plate_dropped
        on_plate = (~outside_x) & (~outside_y) & (~below_plate) & near_plate_height

        return {
            "plate_support_relative_position": plate_support_relative_position,
            "plate_velocity": plate_velocity,
            "ball_dropped": ball_dropped,
            "plate_dropped": plate_dropped,
            "relative_position": relative_position,
            "relative_velocity": relative_velocity,
            "radial_distance": jnp.linalg.norm(relative_position[:2]),
            "on_plate": on_plate,
            "dropped": dropped,
        }


    def get_ball_plate_observation(self, data, internal_state):
        if not self.use_ball_plate:
            return jnp.zeros(0)

        metrics = self.get_ball_plate_metrics(data, internal_state)
        return jnp.concatenate([
            metrics["plate_support_relative_position"],
            metrics["plate_velocity"],
            metrics["relative_position"],
            metrics["relative_velocity"],
        ])


    def ball_plate_has_dropped(self, internal_state):
        return internal_state["ball_plate_dropped"] if self.use_ball_plate else False


    def ball_plate_ball_has_dropped(self, internal_state):
        return internal_state["ball_plate_ball_dropped"] if self.use_ball_plate else False


    def ball_plate_plate_has_dropped(self, internal_state):
        return internal_state["ball_plate_plate_dropped"] if self.use_ball_plate else False
    

    def _init_transition_record_info(self, info):
        info["shock/perturbed"] = jnp.asarray(0.0, dtype=jnp.float32)
        info["shock/dr_resampled"] = jnp.asarray(0.0, dtype=jnp.float32)
        if self.record_transition_physics_state:
            nq, nv = self.initial_mj_model.nq, self.initial_mj_model.nv
            info["transition/x_minus_qpos"] = jnp.zeros(nq)
            info["transition/x_minus_qvel"] = jnp.zeros(nv)
            info["transition/x_plus_qpos"] = jnp.zeros(nq)
            info["transition/x_plus_qvel"] = jnp.zeros(nv)
        if self.record_wm_state:
            from rl_x.algorithms.ncbf_ppo.flax_full_jit import wm_state
            info["wm/x_minus"] = jnp.zeros(wm_state.state_layout(self).dim, dtype=jnp.float32)
            info["wm/x_plus"] = jnp.zeros(wm_state.state_layout(self).dim, dtype=jnp.float32)
            info["wm/xi_next"] = jnp.zeros(wm_state.xi_layout(self).dim, dtype=jnp.float32)
            info["wm/u_applied"] = jnp.zeros(self.nr_actuator_joints, dtype=jnp.float32)
            info["wm/delay_steps"] = jnp.asarray(0, dtype=jnp.int32)


    def handle_domain_randomization(self, internal_state, mjx_model, data, key, is_episode_start=False, return_flags=False):
        domain_sampling_key, domain_perturbation_sampling_key, seen_robot_key, unseen_robot_key, mujoco_model_key, action_delay_key, joint_dropout_key, perturbation_key = jax.random.split(key, 8)

        should_randomize_domain_episode_start = self.domain_randomization_sampling_function.setup(domain_sampling_key)
        should_randomize_domain_perturbation_episode_start = self.domain_randomization_perturbation_sampling_function.setup(domain_perturbation_sampling_key, internal_state["env_curriculum_coeff"])
        should_randomize_domain_step = self.domain_randomization_sampling_function.step(domain_sampling_key)
        should_randomize_domain_perturbation_step = self.domain_randomization_perturbation_sampling_function.step(domain_perturbation_sampling_key, internal_state["env_curriculum_coeff"])
        should_randomize_domain = jnp.where(is_episode_start, should_randomize_domain_episode_start | internal_state["in_eval_mode"], should_randomize_domain_step)
        should_randomize_domain_perturbation = jnp.where(is_episode_start, should_randomize_domain_perturbation_episode_start, should_randomize_domain_perturbation_step)

        self.domain_randomization_unseen_robot_function.sample(internal_state, should_randomize_domain, unseen_robot_key)
        mjx_model, data = self.domain_randomization_seen_robot_function.sample(internal_state, mjx_model, data, should_randomize_domain, seen_robot_key)
        mjx_model = self.domain_randomization_mujoco_model_function.sample(internal_state, mjx_model, should_randomize_domain, mujoco_model_key)
        self.domain_randomization_action_delay_function.sample(internal_state, should_randomize_domain, action_delay_key)
        mjx_model = self.joint_dropout_function.sample(internal_state, mjx_model, should_randomize_domain, joint_dropout_key)
        self.reward_function.handle_model_change(internal_state, mjx_model, should_randomize_domain)
        if self.use_ball_plate:
            mjx_model = self._apply_ball_plate_domain_randomization(mjx_model, {
                "ball_radius": internal_state["ball_plate_ball_radius"],
                "ball_mass": internal_state["ball_plate_ball_mass"],
                "plate_size": internal_state["ball_plate_plate_size"],
                "plate_mass": internal_state["ball_plate_plate_mass"],
                "plate_ball_friction_tangential": internal_state["ball_plate_plate_ball_friction_tangential"],
                "plate_support_friction_tangential": internal_state["ball_plate_plate_support_friction_tangential"],
                "plate_robot_friction_tangential": internal_state["ball_plate_plate_robot_friction_tangential"],
                "contact_timeconst": internal_state["ball_plate_contact_timeconst"],
                "capsule_hand_radius": internal_state["ball_plate_capsule_hand_radius"],
                "capsule_hand_half_length": internal_state["ball_plate_capsule_hand_half_length"],
                "capsule_hand_mass": internal_state["ball_plate_capsule_hand_mass"],
            })

        data = self.domain_randomization_perturbation_function.sample(internal_state, mjx_model, data, should_randomize_domain_perturbation, perturbation_key)

        if return_flags:
            return data, mjx_model, should_randomize_domain, should_randomize_domain_perturbation
        return data, mjx_model
    

    def get_observation_space(self):
        current_observation_idx = 0

        self.joint_positions_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_actuator_joints)])
        current_observation_idx += self.nr_actuator_joints
        self.joint_velocities_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_actuator_joints)])
        current_observation_idx += self.nr_actuator_joints
        self.joint_previous_actions_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_actuator_joints)])
        current_observation_idx += self.nr_actuator_joints
        self.feet_ground_contact_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_feet)])
        current_observation_idx += self.nr_feet
        self.feet_time_on_ground_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_feet)])
        current_observation_idx += self.nr_feet
        self.feet_time_in_air_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_feet)])
        current_observation_idx += self.nr_feet
        self.imu_linear_vel_obs_idx = jnp.array([current_observation_idx + i for i in range(self.imu_linear_velocity_sensor_dim)])
        current_observation_idx += self.imu_linear_velocity_sensor_dim
        self.imu_angular_vel_obs_idx = jnp.array([current_observation_idx + i for i in range(self.imu_angular_velocity_sensor_dim)])
        current_observation_idx += self.imu_angular_velocity_sensor_dim
        self.goal_velocities_obs_idx = jnp.array([current_observation_idx + i for i in range(self.command_observation_size)])
        current_observation_idx += self.command_observation_size
        self.gravity_vector_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3
        self.ball_plate_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_ball_plate_observations)])
        current_observation_idx += self.nr_ball_plate_observations
        self.obstacle_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_obstacle_observations)], dtype=int)
        current_observation_idx += self.nr_obstacle_observations
        self.obstacle_scan_obs_idx = jnp.array([current_observation_idx + i for i in range(self.obstacle_scan_nr_rays)], dtype=int)
        current_observation_idx += self.obstacle_scan_nr_rays
        self.policy_exteroception_obs_idx = jnp.array([current_observation_idx + i for i in range(self.policy_exteroceptive_observation_function.nr_exteroceptive_observations)])
        current_observation_idx += self.policy_exteroceptive_observation_function.nr_exteroceptive_observations
        self.critic_exteroception_obs_idx = jnp.array([current_observation_idx + i for i in range(self.critic_exteroceptive_observation_function.nr_exteroceptive_observations)])
        current_observation_idx += self.critic_exteroceptive_observation_function.nr_exteroceptive_observations

        self.qpos_observation_idx = jnp.array([current_observation_idx + i for i in range(self.nr_actuator_joints + 7)])
        current_observation_idx += self.nr_actuator_joints + 7

        self.qvel_observation_idx = jnp.array([current_observation_idx + i for i in range(self.nr_actuator_joints + 6)])
        current_observation_idx += self.nr_actuator_joints + 6

        # contact is limited to feet right now
        self.contact_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_feet)])
        current_observation_idx += self.nr_feet

        self.body_height_obs_idx = jnp.array([current_observation_idx])
        current_observation_idx += 1

        self.body_tilt_obs_idx = jnp.array([current_observation_idx])
        current_observation_idx += 1

        if self.use_ball_plate:
            self.ball_not_falling_obs_idx = jnp.array([current_observation_idx], dtype=int)
            current_observation_idx += 1

            self.plate_not_falling_obs_idx = jnp.array([current_observation_idx], dtype=int)
            current_observation_idx += 1

            self.ball_plate_dropped_obs_idx = jnp.array([current_observation_idx], dtype=int)
            current_observation_idx += 1
        else:
            self.ball_not_falling_obs_idx = jnp.array([], dtype=int)
            self.plate_not_falling_obs_idx = jnp.array([], dtype=int)
            self.ball_plate_dropped_obs_idx = jnp.array([], dtype=int)

        if self.use_obstacles:
            self.obstacle_no_collision_obs_idx = jnp.array([current_observation_idx], dtype=int)
            current_observation_idx += 1
        else:
            self.obstacle_no_collision_obs_idx = jnp.array([], dtype=int)
        def obstacle_idx_for(observation_type):
            blocks = []
            if "state" in observation_type:
                blocks.append(self.obstacle_obs_idx)
            if "scan" in observation_type:
                blocks.append(self.obstacle_scan_obs_idx)
            return jnp.concatenate(blocks, dtype=int) if blocks else jnp.array([], dtype=int)

        obstacle_policy_obs_idx = obstacle_idx_for(self.obstacle_policy_observation)
        obstacle_ncbf_obs_idx = obstacle_idx_for(self.obstacle_ncbf_observation)
        obstacle_value_obs_idx = obstacle_idx_for(self.obstacle_value_observation)

        ball_plate_policy_obs_idx = self.ball_plate_obs_idx if self.include_ball_plate_observations else jnp.array([], dtype=int)
        ball_plate_policy_state_obs_idx = jnp.concatenate([
            ball_plate_policy_obs_idx,
            self.ball_plate_dropped_obs_idx,
        ], dtype=int) if self.use_ball_plate else jnp.array([], dtype=int)

        def policy_indices_with(obstacle_idx):
            return jnp.concatenate([
                self.joint_positions_obs_idx,
                self.joint_velocities_obs_idx,
                self.joint_previous_actions_obs_idx,
                self.imu_angular_vel_obs_idx,
                self.goal_velocities_obs_idx,
                self.gravity_vector_obs_idx,
                ball_plate_policy_state_obs_idx,
                obstacle_idx,
                self.policy_exteroception_obs_idx,
            ], dtype=int)

        self.policy_observation_indices = policy_indices_with(obstacle_policy_obs_idx)


        self.critic_observation_indices = jnp.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.feet_ground_contact_obs_idx,
            self.feet_time_on_ground_obs_idx,
            self.feet_time_in_air_obs_idx,
            self.imu_linear_vel_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.goal_velocities_obs_idx,
            self.gravity_vector_obs_idx,
            ball_plate_policy_state_obs_idx,
            obstacle_value_obs_idx,
            self.critic_exteroception_obs_idx,
        ], dtype=int)

        self.dynamics_observation_indices = jnp.concatenate([
            self.qpos_observation_idx,
            self.qvel_observation_idx], dtype=int)

        if self.env_config["ncbf_use_policy_observations"]:
            # this is for the state and action model used in ncbf, which uses the same obs as the policy
            # (with the obstacle part chosen by obstacles.ncbf_observation)
            self.ncbf_observation_indices = policy_indices_with(obstacle_ncbf_obs_idx)
            # The history encoder feeds the safety critic, so it sees the same observations
            self.history_observation_indices = self.ncbf_observation_indices
        else:
            actuator_qpos_idx = self.qpos_observation_idx[7:]
            actuator_qvel_idx = self.qvel_observation_idx[6:]
            self.ncbf_observation_indices = jnp.concatenate([actuator_qpos_idx, actuator_qvel_idx], dtype=int)
            self.history_observation_indices = self.policy_observation_indices

            # note all obs here is not normalized or clipped to pass into the forward step function for dynamics models
        # TODO get rid of htis after updating safety layer function based on q
        self.ncbf_obs_in_dynamics_state_idx = jnp.concatenate([
            self.actuator_joint_mask_qpos,
            self.actuator_joint_mask_qvel + self.initial_mj_model.nq
        ], dtype=int
        )
        self.act_in_ncbf_obs_idx = jnp.where(jnp.isin(self.ncbf_observation_indices, self.joint_previous_actions_obs_idx))[0]

        # Optionally also predict the next obstacle observation of the safety critic, so the history
        # encoder has a direct signal to encode obstacle motion
        obstacle_prediction_idx = obstacle_ncbf_obs_idx if self.obstacle_config.get("predict_next_observation", False) else jnp.array([], dtype=int)
        self.next_state_indices = jnp.concatenate([
            self.qvel_observation_idx[:3],
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.ball_plate_obs_idx,
            obstacle_prediction_idx,
        ], dtype=int
        )

        # target value for the predictor
        ncbf_target_indices = [
            self.body_height_obs_idx,
            self.body_tilt_obs_idx,
        ]
        ncbf_target_names = ["height", "tilt"]
        if self.use_ball_plate:
            ncbf_target_indices.extend([
                self.ball_not_falling_obs_idx,
                self.plate_not_falling_obs_idx,
            ])
            ncbf_target_names.extend(["ball_drop", "plate_drop"])
        if self.use_obstacles:
            ncbf_target_indices.append(self.obstacle_no_collision_obs_idx)
            ncbf_target_names.append("collision")
        self.ncbf_target_indices = jnp.concatenate(ncbf_target_indices, dtype=int)
        self.ncbf_target_names = ncbf_target_names
        return BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(current_observation_idx,), dtype=jnp.float32)


    def close(self):
        if self.should_render:
            self.viewer.close()
            pygame.quit()
