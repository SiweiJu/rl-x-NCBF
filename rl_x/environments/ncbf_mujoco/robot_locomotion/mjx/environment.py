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
        self.include_ball_plate_observations = self.use_ball_plate and bool(self.ball_plate_config.get("include_observations", True))
        self.nr_envs = nr_envs
        self.nr_history_steps = env_config["nr_history_steps"]
        safety_config = env_config.get("safety", {})
        self.safe_time_limit_seconds = safety_config.get("safe_time_limit_seconds", 0.0)
        self.root_body_name = robot_config.get("root_body_name", "trunk")

        xml_path = (self.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
        xml_handle = mjcf.from_path(xml_path)

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
            self.nr_ball_plate_observations = 12 if self.include_ball_plate_observations else 0
        else:
            self.nr_ball_plate_observations = 0

        self.reward_collision_sphere_geom_ids = jnp.array([geom.id for geom in [self.initial_mj_model.geom(geom_id) for geom_id in range(self.initial_mj_model.ngeom)] if geom.group[0] == 5])

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
        robot_contact_pair_kwargs = {}
        if "plate_robot_contact_solref" in self.ball_plate_config:
            robot_contact_pair_kwargs["solref"] = " ".join(map(str, self.ball_plate_config["plate_robot_contact_solref"]))
        if "plate_robot_contact_solimp" in self.ball_plate_config:
            robot_contact_pair_kwargs["solimp"] = " ".join(map(str, self.ball_plate_config["plate_robot_contact_solimp"]))
        if "plate_robot_contact_margin" in self.ball_plate_config:
            robot_contact_pair_kwargs["margin"] = str(self.ball_plate_config["plate_robot_contact_margin"])

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
        add_support_fist_geoms = self.ball_plate_config.get("add_support_fist_geoms", True)
        if add_support_fist_geoms:
            left_fist.add("geom", name="left_plate_support_fist", type="capsule", size=str(fist_radius), fromto=f"{left_fist_pos[0] - fist_half_length} {left_fist_pos[1]} {left_fist_pos[2]} {left_fist_pos[0] + fist_half_length} {left_fist_pos[1]} {left_fist_pos[2]}", rgba="0.68 0.68 0.68 1", contype="0", conaffinity="0")
            right_fist.add("geom", name="right_plate_support_fist", type="capsule", size=str(fist_radius), fromto=f"{right_fist_pos[0] - fist_half_length} {right_fist_pos[1]} {right_fist_pos[2]} {right_fist_pos[0] + fist_half_length} {right_fist_pos[1]} {right_fist_pos[2]}", rgba="0.68 0.68 0.68 1", contype="0", conaffinity="0")
        torso_contact_body.add("geom", name="torso_plate_guard", type="sphere", pos=" ".join(map(str, torso_contact_pos)), size=str(torso_contact_size), rgba="0.2 0.2 0.2 0", contype="0", conaffinity="0")
        left_upper_arm_contact_body.add("geom", name="left_upper_arm_plate_guard", type="capsule", size=str(upper_arm_contact_radius), fromto=" ".join(map(str, left_upper_arm_contact_fromto)), rgba="0.2 0.2 0.2 0", contype="0", conaffinity="0")
        right_upper_arm_contact_body.add("geom", name="right_upper_arm_plate_guard", type="capsule", size=str(upper_arm_contact_radius), fromto=" ".join(map(str, right_upper_arm_contact_fromto)), rgba="0.2 0.2 0.2 0", contype="0", conaffinity="0")
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
        for torso_geom_name in ["torso_plate_guard", *self.ball_plate_config.get("torso_contact_geom_names", [])]:
            xml_handle.contact.add("pair", geom1=torso_geom_name, geom2="ball_plate_geom", condim=plate_torso_pair_dim, friction=plate_torso_pair_friction, **robot_contact_pair_kwargs)
        for arm_geom_name in [
            "left_upper_arm_plate_guard",
            "right_upper_arm_plate_guard",
            "left_forearm_plate_guard",
            "right_forearm_plate_guard",
            *self.ball_plate_config.get("arm_contact_geom_names", []),
        ]:
            xml_handle.contact.add("pair", geom1=arm_geom_name, geom2="ball_plate_geom", condim=plate_arm_pair_dim, friction=plate_arm_pair_friction, **robot_contact_pair_kwargs)
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
            "torso_plate_guard",
            *self.ball_plate_config.get("torso_contact_geom_names", []),
            "left_upper_arm_plate_guard",
            "right_upper_arm_plate_guard",
            "left_forearm_plate_guard",
            "right_forearm_plate_guard",
            *self.ball_plate_config.get("arm_contact_geom_names", []),
        ]


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
            "ball_plate_ball_dropped": False,
            "ball_plate_plate_dropped": False,
            "ball_plate_dropped": False,
            "nr_collisions_in_nominal": 0,
            "history_stack": history_stack,
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
        key, initial_state_key, terrain_key, domain_randomization_key, ball_plate_key, command_key, observation_key = jax.random.split(state.key, 7)
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
        should_sample_commands = self.command_sampling_function.setup(command_key)
        self.command_function.get_next_command(new_internal_state, should_sample_commands, command_key)

        next_observation = self.get_observation(data, mjx_model, new_internal_state, observation_key, jnp.zeros(self.nr_actuator_joints))
        new_info = dict(new_state.info)
        if self.use_ball_plate:
            ball_plate_metrics = self.get_ball_plate_metrics(data, new_internal_state)
            new_info[f"env_info/ball_plate_radial_distance"] = ball_plate_metrics["radial_distance"]
            new_info[f"env_info/ball_plate_on_plate"] = ball_plate_metrics["on_plate"].astype(jnp.float32)
            new_info[f"env_info/ball_plate_ball_dropped"] = ball_plate_metrics["ball_dropped"].astype(jnp.float32)
            new_info[f"env_info/ball_plate_plate_dropped"] = ball_plate_metrics["plate_dropped"].astype(jnp.float32)
            new_info[f"env_info/ball_plate_dropped"] = ball_plate_metrics["dropped"].astype(jnp.float32)
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
        key, action_delay_key, domain_randomization_key, command_sampling_key, command_key, observation_key, terrain_key = jax.random.split(state.key, 7)
        state = state.replace(key=key)

        chosen_action = action[:self.nr_actuator_joints]
        delayed_action = self.domain_randomization_action_delay_function.delay_action(chosen_action, state.internal_state, action_delay_key)

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

        data, mjx_model = self.handle_domain_randomization(state.internal_state, state.mjx_model, data, domain_randomization_key)
        state = state.replace(data=data, mjx_model=mjx_model)

        self.terrain_function.pre_step(data, state.internal_state)

        reward = self.reward_function.reward_and_info(data, mjx_model, state.internal_state, chosen_action, state.info)

        should_sample_commands = self.command_sampling_function.step(command_sampling_key)
        self.command_function.get_next_command(state.internal_state, should_sample_commands, command_key)

        next_observation = self.get_observation(data, mjx_model, state.internal_state, observation_key, chosen_action)
        constraint_terminated = self.termination_function.should_terminate(state.internal_state)
        qvel_limit_terminated = jnp.any(jnp.abs(data.qvel[:3]) == 100.0)
        terminated = constraint_terminated | qvel_limit_terminated
        truncated = state.info_episode_store["episode_step"] >= (self.horizon - 1)
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
            ball_plate_safety_observation = jnp.array([ball_not_falling, plate_not_falling])
        else:
            ball_plate_safety_observation = jnp.array([], dtype=jnp.float32)

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
            jnp.array([self.policy_exteroceptive_observation_function.get_exteroceptive_observation(data, mjx_model, internal_state)]).reshape(-1),
            jnp.array([self.critic_exteroceptive_observation_function.get_exteroceptive_observation(data, mjx_model, internal_state)]).reshape(-1),
            qpos, # qpos all not normalized, base pose can be dummy for deployment
            qvel, # qvel all not normalized
            feet_ground_contact,
            jnp.array([robot_height_safe]),
            jnp.array([body_tilt_safe]),
            ball_plate_safety_observation,
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
        plate_pos, plate_quat = self._get_ball_plate_supported_pose(data, plate_size)
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
        internal_state["ball_plate_ball_dropped"] = False
        internal_state["ball_plate_plate_dropped"] = False
        internal_state["ball_plate_dropped"] = False
        return data, mjx_model


    def _get_ball_plate_support_points(self, data):
        left_body_xmat = data.xmat[self.ball_plate_left_fist_body_id].reshape(3, 3)
        right_body_xmat = data.xmat[self.ball_plate_right_fist_body_id].reshape(3, 3)
        left_support = data.xpos[self.ball_plate_left_fist_body_id] + left_body_xmat @ self.ball_plate_left_fist_pos
        right_support = data.xpos[self.ball_plate_right_fist_body_id] + right_body_xmat @ self.ball_plate_right_fist_pos
        return left_support, right_support


    def _get_ball_plate_supported_pose(self, data, plate_size=None):
        if plate_size is None:
            plate_size = self.ball_plate_plate_size
        left_support, right_support = self._get_ball_plate_support_points(data)
        support_center = 0.5 * (left_support + right_support)
        support_height = jnp.maximum(left_support[2], right_support[2])
        plate_pos = support_center.at[2].set(
            support_height
            + float(self.ball_plate_config["fist_radius"])
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

        keys = jax.random.split(key, 11)
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
        }


    def _set_pair_tangential_friction(self, pair_friction, pair_ids, value):
        if len(pair_ids) == 0:
            return pair_friction
        pair_friction = pair_friction.at[pair_ids, 0].set(value)
        pair_friction = pair_friction.at[pair_ids, 1].set(value)
        return pair_friction


    def _apply_ball_plate_domain_randomization(self, mjx_model, params):
        ball_radius = params["ball_radius"]
        ball_mass = params["ball_mass"]
        plate_size = params["plate_size"]
        plate_mass = params["plate_mass"]

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
        if not self.include_ball_plate_observations:
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
    

    def handle_domain_randomization(self, internal_state, mjx_model, data, key, is_episode_start=False):
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
            })

        data = self.domain_randomization_perturbation_function.sample(internal_state, mjx_model, data, should_randomize_domain_perturbation, perturbation_key)

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
        else:
            self.ball_not_falling_obs_idx = jnp.array([], dtype=int)
            self.plate_not_falling_obs_idx = jnp.array([], dtype=int)

        self.policy_observation_indices = jnp.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.goal_velocities_obs_idx,
            self.gravity_vector_obs_idx,
            self.ball_plate_obs_idx,
            self.policy_exteroception_obs_idx,
        ], dtype=int)


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
            self.ball_plate_obs_idx,
            self.critic_exteroception_obs_idx,
        ], dtype=int)

        self.dynamics_observation_indices = jnp.concatenate([
            self.qpos_observation_idx,
            self.qvel_observation_idx], dtype=int)

        if self.env_config["ncbf_use_policy_observations"]:
            # this is for the state and action model used in ncbf, which uses the same obs as the policy
            self.ncbf_observation_indices = self.policy_observation_indices
        else:
            actuator_qpos_idx = self.qpos_observation_idx[7:]
            actuator_qvel_idx = self.qvel_observation_idx[6:]
            self.ncbf_observation_indices = jnp.concatenate([actuator_qpos_idx, actuator_qvel_idx], dtype=int)

            # note all obs here is not normalized or clipped to pass into the forward step function for dynamics models
        # TODO get rid of htis after updating safety layer function based on q
        self.ncbf_obs_in_dynamics_state_idx = jnp.concatenate([
            self.actuator_joint_mask_qpos,
            self.actuator_joint_mask_qvel + self.initial_mj_model.nq
        ], dtype=int
        )
        self.act_in_ncbf_obs_idx = jnp.where(jnp.isin(self.ncbf_observation_indices, self.joint_previous_actions_obs_idx))[0]

        self.next_state_indices = jnp.concatenate([
            self.qvel_observation_idx[:3],
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
        ], dtype=int
        )

        # target value for the predictor
        ncbf_target_indices = [
            self.body_height_obs_idx,
            self.body_tilt_obs_idx,
        ]
        if self.use_ball_plate:
            ncbf_target_indices.extend([
                self.ball_not_falling_obs_idx,
                self.plate_not_falling_obs_idx,
            ])
        self.ncbf_target_indices = jnp.concatenate(ncbf_target_indices, dtype=int)
        return BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(current_observation_idx,), dtype=jnp.float32)


    def close(self):
        if self.should_render:
            self.viewer.close()
            pygame.quit()
