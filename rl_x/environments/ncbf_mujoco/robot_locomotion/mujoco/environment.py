import copy
from copy import deepcopy
from pathlib import Path
import gymnasium as gym
import mujoco
from dm_control import mjcf
import pygame
import numpy as np
from scipy.spatial.transform import Rotation


from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.viewer import MujocoViewer
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.control_functions.handler import get_control_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.command_functions.handler import get_command_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.domain_randomization.initial_state_functions.handler import get_initial_state_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.sampling_functions.handler import get_sampling_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.reward_functions.handler import get_reward_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.termination_functions.handler import get_termination_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.domain_randomization.action_delay_functions.handler import get_domain_randomization_action_delay_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.domain_randomization.mujoco_model_functions.handler import get_domain_randomization_mujoco_model_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.domain_randomization.seen_robot_functions.handler import get_domain_randomization_seen_robot_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.domain_randomization.unseen_robot_functions.handler import get_domain_randomization_unseen_robot_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.domain_randomization.perturbation_functions.handler import get_domain_randomization_perturbation_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.domain_randomization.observation_noise_functions.handler import get_observation_noise_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.domain_randomization.joint_dropout_functions.handler import get_joint_dropout_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.exteroceptive_observation_functions.handler import get_exteroceptive_observation_function
from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.terrain_functions.handler import get_terrain_function


class LocomotionEnv(gym.Env):
    def __init__(self, robot_config, runner_mode, seed, render, env_config, nr_envs):
        
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
        self.nr_envs = nr_envs
        self.nr_history_steps = env_config["nr_history_steps"]
        self.root_body_name = robot_config.get("root_body_name", "trunk")

        self.np_rng = np.random.default_rng(seed)

        xml_path = (self.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
        xml_handle = mjcf.from_path(xml_path)
        if self.use_capsule_hand:
            self._remove_capsule_hand_visual_geoms_from_xml(xml_handle)

        # Set the MuJoCo solver iterations, the XML uses very low values by default for MJX
        xml_handle.option.iterations = 100
        xml_handle.option.ls_iterations = 50
        xml_handle.option.flag.eulerdamp = "enable"

        if "hfield" in env_config["terrain"]["type"]:
            xml_handle.asset.insert("hfield", 0, name="empty_hfield", file="default_hfield_80.png", size="4 4 30.0 0.125")
            floor = xml_handle.find("geom", "floor")
            floor.type = "hfield"
            floor.hfield = "empty_hfield"
        
        if self.should_render and self.add_goal_arrow:
            trunk = xml_handle.find("body", self.root_body_name)
            dir_base = trunk.add("body", name="dir_arrow_base", pos="0 0 0.2")
            dir_base.add("geom", name="dir_arrow_ball", type="sphere", size=".035", pos="0 0 0",
                         dclass="visual", rgba="0.1 0.45 1.0 1")
            dir_base.add("geom", name="dir_arrow", type="cylinder", size=".012 .001", pos="0 0 0",
                         dclass="visual", rgba="0.1 0.45 1.0 1")
            dir_base.add("geom", name="yaw_arrow", type="cylinder", size=".018 .001", pos="0 0 0",
                         dclass="visual", rgba="1.0 0.55 0.05 1")
            dir_base.add("geom", name="yaw_arrow_ball", type="sphere", size=".045", pos="0 0 0",
                         dclass="visual", rgba="1.0 0.55 0.05 1")

        if self.should_render:
            # add the safety light
            trunk = xml_handle.find("body", self.root_body_name)
            trunk.add("geom", name="safety_light", pos="0 0 1", type="sphere", dclass="visual", size="0.05", rgba="1 0 0 1", group="0")
            # light_vec = xml_handle.find("body", "safety_light_body")
            # light_vec.add("site", name="safety_light", type="sphere", size="0.2", pos="0 0 0.6", rgba="0 1 0 1")
            #
            # trunk = xml_handle.worldbody
            # trunk.add("site", name="safety_light", type="sphere", size="0.5", pos="0 0 1.5", rgba="1 0 0 1")

        if self.use_ball_plate:
            self._add_ball_plate_to_xml(xml_handle)

        self.initial_mj_model = mujoco.MjModel.from_xml_string(xml=xml_handle.to_xml_string(), assets=xml_handle.get_assets())
        self.initial_mj_model.opt.timestep = env_config["timestep"]
        self.data = mujoco.MjData(self.initial_mj_model)
        self.c_model = deepcopy(self.initial_mj_model)
        self.c_data = mujoco.MjData(self.c_model)
        self.c_data.qpos = self.initial_mj_model.keyframe("home").qpos
        mujoco.mj_forward(self.c_model, self.c_data)
        
        self.imu_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.trunk_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, self.root_body_name)
        self.actuator_joint_max_velocities = np.array(robot_config["actuator_joint_max_velocities"])
        self.initial_qpos = np.array(self.initial_mj_model.keyframe("home").qpos)
        self.initial_imu_orientation_rotation_inverse = Rotation.from_matrix(self.c_data.site_xmat[self.imu_site_id].reshape(3, 3)).inv()
        self.initial_imu_height = self.c_data.site_xpos[self.imu_site_id, 2]
        self.actuator_joint_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, actuator_trnid[0]) for actuator_trnid in self.initial_mj_model.actuator_trnid]
        self.actuator_joint_mask_joints = np.array([self.initial_mj_model.joint(joint_name).id for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qpos = np.array([self.initial_mj_model.joint(joint_name).qposadr[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qvel = np.array([self.initial_mj_model.joint(joint_name).dofadr[0] for joint_name in self.actuator_joint_names])
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
        self.foot_geom_indices = np.array([mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, foot_name) for foot_name in self.feet_names])
        self.nr_feet = len(self.feet_names)

        feet_xpos = self.c_data.geom_xpos[self.foot_geom_indices]
        x_pos, y_pos, z_pos = feet_xpos[:, 0], feet_xpos[:, 1], feet_xpos[:, 2]
        abs_y_feet_xpos = np.array([x_pos, np.abs(y_pos), z_pos]).T
        distances_between_abs_y_feet = np.linalg.norm(abs_y_feet_xpos[:, None] - abs_y_feet_xpos[None], axis=-1)
        min_dist_indices = np.argmin(distances_between_abs_y_feet + np.eye(len(abs_y_feet_xpos)) * 1000, axis=1)
        feet_symmetry_set = set([(min(i, min_dist_indices[i]), max(i, min_dist_indices[i])) for i in range(len(min_dist_indices)) if min_dist_indices[min_dist_indices[i]] == i])
        self.feet_symmetry_pairs = np.array([list(pair) for pair in feet_symmetry_set])
        self.body_ids_of_feet = np.array([self.initial_mj_model.geom(geom_id).bodyid[0] for geom_id in self.foot_geom_indices])
        all_feet_are_sphere = np.all(self.initial_mj_model.geom_type[self.foot_geom_indices] == 2)
        all_feet_are_capsule = np.all(self.initial_mj_model.geom_type[self.foot_geom_indices] == 3)
        all_feet_are_box = np.all(self.initial_mj_model.geom_type[self.foot_geom_indices] == 6)
        if not bool(all_feet_are_sphere | all_feet_are_capsule | all_feet_are_box):
            raise ValueError("Foot geoms are not all of type sphere, capsule or box.")
        self.foot_type = "sphere" if all_feet_are_sphere else ("capsule" if all_feet_are_capsule else "box")
        self.foot_type_int = 0 if self.foot_type == "sphere" else (2 if self.foot_type == "capsule" else 1)

        feet_global_linear_velocity_sensor_ids = [self.initial_mj_model.sensor(f"{foot_name}_global_linear_velocity").id for foot_name in self.feet_names]
        self.feet_global_linear_velocity_sensor_adrs_start = np.array([self.initial_mj_model.sensor_adr[sensor_id] for sensor_id in feet_global_linear_velocity_sensor_ids])
        self.left_foot_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, robot_config.get("left_foot_site_name", "left_foot"))
        self.right_foot_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, robot_config.get("right_foot_site_name", "right_foot"))
        self.left_foot_geom_indices = np.array([
            mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in robot_config.get("left_foot_geom_names", [name for name in self.feet_names if "left" in name])
        ])
        self.right_foot_geom_indices = np.array([
            mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in robot_config.get("right_foot_geom_names", [name for name in self.feet_names if "right" in name])
        ])
        left_foot_velocity_sensor_id = self.initial_mj_model.sensor(robot_config.get("left_foot_velocity_sensor_name", "left_foot_global_linvel")).id
        right_foot_velocity_sensor_id = self.initial_mj_model.sensor(robot_config.get("right_foot_velocity_sensor_name", "right_foot_global_linvel")).id
        self.left_foot_velocity_sensor_adr = self.initial_mj_model.sensor_adr[left_foot_velocity_sensor_id]
        self.right_foot_velocity_sensor_adr = self.initial_mj_model.sensor_adr[right_foot_velocity_sensor_id]

        body_to_parentid = np.array([self.initial_mj_model.body(body_id).parentid[0] for body_id in range(self.initial_mj_model.nbody)])
        body_to_children_count = np.array([np.sum(body_to_parentid == body_id) for body_id in range(self.initial_mj_model.nbody)])
        self.body_ids_of_actuator_joints = np.array([self.initial_mj_model.joint(joint_name).bodyid[0] for joint_name in self.actuator_joint_names])
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
            self.ball_plate_plate_size = np.array(self.ball_plate_config["plate_size"], dtype=float)
            self.ball_plate_plate_mass = float(self.ball_plate_config["plate_mass"])
            self.ball_plate_plate_mass_range = np.array(
                self.ball_plate_config.get("plate_mass_range", [self.ball_plate_plate_mass, self.ball_plate_plate_mass]),
                dtype=float,
            )
            self.ball_plate_plate_size_scale_range = np.array(
                self.ball_plate_config.get("plate_size_scale_range", [1.0, 1.0]),
                dtype=float,
            )
            self.ball_plate_ball_radius = float(self.ball_plate_config["ball_radius"])
            self.ball_plate_ball_mass = float(self.ball_plate_config["ball_mass"])
            self.ball_plate_ball_radius_range = np.array(
                self.ball_plate_config.get("ball_radius_range", [self.ball_plate_ball_radius, self.ball_plate_ball_radius]),
                dtype=float,
            )
            self.ball_plate_ball_mass_range = np.array(
                self.ball_plate_config.get("ball_mass_range", [self.ball_plate_ball_mass, self.ball_plate_ball_mass]),
                dtype=float,
            )
            self.ball_plate_ball_start_offset = np.array(self.ball_plate_config["ball_start_offset"], dtype=float)
            self.ball_plate_ball_start_offset_xy_range = np.array(
                self.ball_plate_config.get("ball_start_offset_xy_range", [0.0, 0.0]),
                dtype=float,
            )
            self.ball_plate_ball_initial_linear_velocity_xy_range = np.array(
                self.ball_plate_config.get("ball_initial_linear_velocity_xy_range", [0.0, 0.0]),
                dtype=float,
            )
            self.ball_plate_ball_initial_angular_velocity_range = np.array(
                self.ball_plate_config.get("ball_initial_angular_velocity_range", [0.0, 0.0, 0.0]),
                dtype=float,
            )
            self.ball_plate_ball_pair_ids = np.array(self._get_ball_plate_pair_ids(["plate_ball_geom"]), dtype=int)
            self.ball_plate_support_pair_ids = np.array(self._get_ball_plate_pair_ids(self._get_ball_plate_support_contact_geom_names()), dtype=int)
            self.ball_plate_robot_pair_ids = np.array(self._get_ball_plate_pair_ids(self._get_ball_plate_robot_contact_geom_names()), dtype=int)
            self.ball_plate_all_pair_ids = np.array(sorted(set(np.concatenate([
                self.ball_plate_ball_pair_ids,
                self.ball_plate_support_pair_ids,
                self.ball_plate_robot_pair_ids,
            ]).tolist())), dtype=int)
            self.ball_plate_nominal_pair_friction = np.array(self.initial_mj_model.pair_friction, dtype=float)
            self.ball_plate_nominal_pair_solref = np.array(self.initial_mj_model.pair_solref, dtype=float)
            self.ball_plate_nominal_plate_ball_friction = (
                float(self.initial_mj_model.pair_friction[self.ball_plate_ball_pair_ids[0], 0])
                if len(self.ball_plate_ball_pair_ids) > 0
                else float(self.ball_plate_config.get("plate_ball_contact_friction", [4.0])[0])
            )
            self.ball_plate_nominal_support_friction = float(self.ball_plate_config.get("plate_support_contact_friction", [3.0])[0])
            self.ball_plate_nominal_robot_friction = float(self.ball_plate_config.get("plate_arm_contact_friction", [1.5])[0])
            self.ball_plate_nominal_contact_timeconst = (
                float(self.initial_mj_model.pair_solref[self.ball_plate_all_pair_ids[0], 0])
                if len(self.ball_plate_all_pair_ids) > 0
                else float(self.ball_plate_config.get("plate_ball_contact_solref", [0.005])[0])
            )
            self.ball_plate_plate_ball_friction_tangential_range = np.array(
                self.ball_plate_config.get("plate_ball_contact_friction_tangential_range", [self.ball_plate_nominal_plate_ball_friction] * 2),
                dtype=float,
            )
            self.ball_plate_plate_support_friction_tangential_range = np.array(
                self.ball_plate_config.get("plate_support_contact_friction_tangential_range", [3.0, 3.0]),
                dtype=float,
            )
            self.ball_plate_plate_robot_friction_tangential_range = np.array(
                self.ball_plate_config.get("plate_robot_contact_friction_tangential_range", [1.5, 1.5]),
                dtype=float,
            )
            self.ball_plate_contact_timeconst_range = np.array(
                self.ball_plate_config.get("plate_contact_timeconst_range", [0.005, 0.005]),
                dtype=float,
            )
            self.ball_plate_drop_margin = float(self.ball_plate_config["drop_margin"])
            self.ball_plate_drop_height = float(self.ball_plate_config["drop_height"])
            self.ball_plate_plate_support_clearance = float(self.ball_plate_config["plate_support_clearance"])
            self.ball_plate_tilt_drop_threshold = float(self.ball_plate_config["plate_tilt_drop_threshold"])
            self.ball_plate_left_fist_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, self.ball_plate_config["left_fist_body"])
            self.ball_plate_right_fist_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, self.ball_plate_config["right_fist_body"])
            self.ball_plate_left_fist_pos = np.array(self.ball_plate_config["left_fist_pos"], dtype=float)
            self.ball_plate_right_fist_pos = np.array(self.ball_plate_config["right_fist_pos"], dtype=float)
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
            self.ball_plate_capsule_hand_radius_range = np.array(
                capsule_hand_radius_range,
                dtype=float,
            )
            self.ball_plate_capsule_hand_half_length_range = np.array(
                capsule_hand_half_length_range,
                dtype=float,
            )
            self.ball_plate_capsule_hand_mass_range = np.array(
                self.ball_plate_config.get("capsule_hand_mass_range", [self.ball_plate_capsule_hand_mass, self.ball_plate_capsule_hand_mass]),
                dtype=float,
            )
            if self.use_capsule_hand:
                self.ball_plate_capsule_hand_geom_ids = np.array([
                    mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_plate_support_fist"),
                    mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_plate_support_fist"),
                ], dtype=int)
                self.ball_plate_capsule_hand_body_ids = np.array([
                    mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_capsule_hand"),
                    mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_capsule_hand"),
                ], dtype=int)
                self.ball_plate_capsule_hand_nominal_body_inertia = np.array(
                    self.initial_mj_model.body_inertia[self.ball_plate_capsule_hand_body_ids[0]],
                    dtype=float,
                )
            else:
                self.ball_plate_capsule_hand_geom_ids = np.array([], dtype=int)
                self.ball_plate_capsule_hand_body_ids = np.array([], dtype=int)
                self.ball_plate_capsule_hand_nominal_body_inertia = np.zeros(3)
            self.nr_ball_plate_observations = 12
        else:
            self.nr_ball_plate_observations = 0

        self.reward_collision_sphere_geom_ids = np.array([geom.id for geom in [self.initial_mj_model.geom(geom_id) for geom_id in range(self.initial_mj_model.ngeom)] if geom.group[0] == 5], dtype=int)
        
        self.reward_collision_sphere_geoms_and_feet_geoms_ids = np.concatenate((self.reward_collision_sphere_geom_ids, self.foot_geom_indices))
        self.dim_geom_ids = self.reward_collision_sphere_geoms_and_feet_geoms_ids - 1

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
        self.command_function_type = env_config["command"]["type"]
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
        self.domain_randomization_perturbation_sampling_function = get_sampling_function(env_config["domain_randomization"]["perturbation"]["sampling_type"], self, probability=env_config["domain_randomization"]["perturbation"]["sampling_probability"])
        self.observation_noise_function = get_observation_noise_function(env_config["domain_randomization"]["observation_noise"]["type"], self)
        self.joint_dropout_function = get_joint_dropout_function(env_config["domain_randomization"]["joint_dropout"]["type"], self)


        action_space_size = self.nr_actuator_joints
        # action_space_low = -np.ones(action_space_size) * np.inf
        # action_space_high = np.ones(action_space_size) * np.inf

        actuator_limit_positions = self.initial_mj_model.jnt_range[self.actuator_joint_mask_joints]
        scaling_factor = robot_config["scaling_factor"]
        actuator_nominal_positions = self.initial_qpos[self.actuator_joint_mask_qpos]

        # normalize action space:
        actuator_limit_positions_normalized = (actuator_limit_positions - actuator_nominal_positions[:, np.newaxis]) / scaling_factor
        self.action_space = gym.spaces.Box(low=actuator_limit_positions_normalized[:, 0], high=actuator_limit_positions_normalized[:, 1], shape=(action_space_size,), dtype=np.float32)

        self.observation_space = self.get_observation_space()

        self.observation_noise_function.init_attributes()

        eval_mode = True
        self.max_curriculum_level = 0.0
        self.internal_state = {
            "mj_model": deepcopy(self.initial_mj_model),
            "data": mujoco.MjData(self.initial_mj_model),
            "in_eval_mode": eval_mode,
            "env_curriculum_coeff": np.where(eval_mode, self.max_curriculum_level, 0.0),
            "env_curriculum_levels_in_a_row": 0.0,
            "actuator_joint_nominal_positions": self.initial_qpos[self.actuator_joint_mask_qpos],
            "actuator_joint_max_velocities": self.actuator_joint_max_velocities,
            "goal_velocities": np.array([0.0, 0.0, 0.0]),
            "imu_orientation_rotation": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]),
            "imu_orientation_rotation_inverse": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]).inv(),
            "imu_orientation_euler": np.array([0.0, 0.0, 0.0]),
            "last_action": np.zeros(self.nr_actuator_joints),
            "second_last_action": np.zeros(self.nr_actuator_joints),
            "joint_dropout_mask": np.ones(self.nr_actuator_joints, dtype=bool),
            "robot_dimensions_mean": self.robot_dimensions_mean,
            "body_tilt_threshold": env_config["termination"]["body_tilt_threshold"],
            "max_command_velocity": np.minimum(self.robot_dimensions_mean * self.command_function.max_velocity_per_m_factor, self.command_function.clip_max_velocity),
            "ball_plate_ball_radius": self.ball_plate_ball_radius if self.use_ball_plate else 0.0,
            "ball_plate_ball_mass": self.ball_plate_ball_mass if self.use_ball_plate else 0.0,
            "ball_plate_plate_size": self.ball_plate_plate_size if self.use_ball_plate else np.zeros(3),
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
            "info": {
                "rollout/episode_return": 0.0,
                "rollout/episode_length": 0,
                "env_curriculum/coefficient": np.where(eval_mode, self.max_curriculum_level, 0.0),
            },
            "info_episode_store": {
                "episode_return": 0.0,
                "episode_step": 0,
                "episode_total_xy_velocity_diff_abs": 0.0,
            },
        }
        self.command_function.init()
        self.reward_function.init()
        self.terrain_function.init()
        self.joint_dropout_function.init()
        self.domain_randomization_action_delay_function.init()
        self.domain_randomization_seen_robot_function.init()
        self.domain_randomization_unseen_robot_function.init()
        self.reward_function.reward_and_info(np.zeros(self.nr_actuator_joints))

        if self.should_render:
            self.viewer = MujocoViewer(self.initial_mj_model, self.dt)

            if self.add_goal_arrow:
                self.dir_arrow_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "dir_arrow")
                self.dir_arrow_ball_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "dir_arrow_ball")
                self.yaw_arrow_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "yaw_arrow")
                self.yaw_arrow_ball_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "yaw_arrow_ball")
            self.uses_hfield = self.initial_mj_model.hfield_data.shape[0] != 0
            self.light_xdir = self.c_data.light_xdir
            self.light_xpos = self.c_data.light_xpos

            # cache site id once
            self.safety_light_site_id = mujoco.mj_name2id(
                self.initial_mj_model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "safety_light",
            )

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


    def _apply_ball_plate_initial_joint_positions_to_state(self, qpos, qvel=None):
        initial_joint_positions = self.ball_plate_config.get("initial_joint_positions", {})
        if not initial_joint_positions:
            return

        model = self.internal_state["mj_model"]
        qposadr_to_nominal_index = {
            int(qposadr): nominal_index
            for nominal_index, qposadr in enumerate(self.actuator_joint_mask_qpos)
        }

        for joint_name, joint_position in initial_joint_positions.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Unknown ball_plate initial joint: {joint_name}")

            if model.jnt_limited[joint_id]:
                joint_position = np.clip(joint_position, model.jnt_range[joint_id, 0], model.jnt_range[joint_id, 1])

            qposadr = model.jnt_qposadr[joint_id]
            qpos[qposadr] = joint_position
            nominal_index = qposadr_to_nominal_index.get(int(qposadr))
            if nominal_index is not None:
                self.internal_state["actuator_joint_nominal_positions"][nominal_index] = joint_position
            if qvel is not None:
                dofadr = model.jnt_dofadr[joint_id]
                qvel[dofadr] = 0.0


    def _apply_ball_plate_post_drop_command(self):
        if not self.stand_after_ball_plate_drop:
            return

        post_drop = bool(self.internal_state["ball_plate_drop_latched"])
        if not post_drop:
            return

        self.internal_state["goal_velocities"] = np.zeros_like(self.internal_state["goal_velocities"])
        self.internal_state["actuator_joint_keep_nominal"] = np.ones(self.nr_actuator_joints, dtype=bool)
        if "goal_gait_frequency" in self.internal_state:
            self.internal_state["goal_gait_frequency"] = 0.0


    def _update_ball_plate_post_drop_state(self):
        if not self.stand_after_ball_plate_drop:
            self.internal_state["ball_plate_newly_dropped"] = False
            self.internal_state["ball_plate_post_drop_truncated"] = False
            return False

        dropped = bool(self.internal_state["ball_plate_dropped"])
        was_latched = bool(self.internal_state["ball_plate_drop_latched"])
        drop_latched = was_latched or dropped
        newly_dropped = dropped and not was_latched
        time_since_drop = self.internal_state["ball_plate_time_since_drop"] + self.dt if drop_latched else 0.0
        post_drop_truncated = drop_latched and (
            time_since_drop >= self.ball_plate_post_drop_truncation_seconds
        )

        self.internal_state["ball_plate_drop_latched"] = drop_latched
        self.internal_state["ball_plate_newly_dropped"] = newly_dropped
        self.internal_state["ball_plate_time_since_drop"] = time_since_drop
        self.internal_state["ball_plate_post_drop_truncated"] = post_drop_truncated
        return post_drop_truncated


    @staticmethod
    def _xmat_with_z_axis(z_axis):
        x_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = np.cross(z_axis, x_axis)
        y_axis_norm = np.linalg.norm(y_axis)
        if y_axis_norm < 1e-8:
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            y_axis /= y_axis_norm
        x_axis = np.cross(y_axis, z_axis)
        return np.column_stack((x_axis, y_axis, z_axis))


    def _update_goal_arrow(self):
        data = self.internal_state["data"]
        model = self.viewer.model
        goal_velocities = np.asarray(self.internal_state["goal_velocities"], dtype=np.float64)
        goal_xy_local = goal_velocities[:2]
        goal_speed = np.linalg.norm(goal_xy_local)
        arrow_base_pos = data.body(self.root_body_name).xpos + np.array([0.0, 0.0, 0.5], dtype=np.float64)
        trunk_yaw = self.internal_state["imu_orientation_euler"][2]
        cos_yaw = np.cos(trunk_yaw)
        sin_yaw = np.sin(trunk_yaw)
        yaw_offset_dir = np.array([-sin_yaw, cos_yaw, 0.0], dtype=np.float64)

        data.geom_xpos[self.dir_arrow_ball_geom_id] = arrow_base_pos

        if goal_speed < 1e-6:
            model.geom_size[self.dir_arrow_geom_id, 1] = 1e-6
            data.geom_xpos[self.dir_arrow_geom_id] = arrow_base_pos
            model.geom_rgba[self.dir_arrow_geom_id, 3] = 0.0
            model.geom_rgba[self.dir_arrow_ball_geom_id, 3] = 0.25
        else:
            goal_xy_world = np.array([
                cos_yaw * goal_xy_local[0] - sin_yaw * goal_xy_local[1],
                sin_yaw * goal_xy_local[0] + cos_yaw * goal_xy_local[1],
            ], dtype=np.float64)
            goal_xy_world /= np.linalg.norm(goal_xy_world)
            arrow_dir = np.array([goal_xy_world[0], goal_xy_world[1], 0.0], dtype=np.float64)
            yaw_offset_dir = np.array([-arrow_dir[1], arrow_dir[0], 0.0], dtype=np.float64)

            max_command_velocity = max(float(self.internal_state["max_command_velocity"]), 1e-6)
            normalized_speed = min(goal_speed / max_command_velocity, 1.0)
            arrow_half_length = 0.05 + 0.20 * normalized_speed

            model.geom_size[self.dir_arrow_geom_id, 1] = arrow_half_length
            model.geom_rgba[self.dir_arrow_geom_id, 3] = 1.0
            model.geom_rgba[self.dir_arrow_ball_geom_id, 3] = 1.0
            data.geom_xmat[self.dir_arrow_geom_id] = self._xmat_with_z_axis(arrow_dir).reshape((9,))
            data.geom_xpos[self.dir_arrow_geom_id] = arrow_base_pos + arrow_dir * arrow_half_length

        goal_yaw_velocity = goal_velocities[2]
        yaw_base_pos = arrow_base_pos + 0.22 * yaw_offset_dir

        if np.abs(goal_yaw_velocity) < 1e-6:
            data.geom_xpos[self.yaw_arrow_geom_id] = yaw_base_pos
            data.geom_xpos[self.yaw_arrow_ball_geom_id] = yaw_base_pos
            model.geom_size[self.yaw_arrow_geom_id, 1] = 1e-6
            model.geom_rgba[self.yaw_arrow_geom_id, 3] = 0.0
            model.geom_rgba[self.yaw_arrow_ball_geom_id, 3] = 0.25
            return

        max_command_velocity = max(float(self.internal_state["max_command_velocity"]), 1e-6)
        yaw_velocity_scale = max_command_velocity
        if hasattr(self.command_function, "velocity_ratio"):
            yaw_velocity_scale *= max(float(self.command_function.velocity_ratio[2]), 1e-6)
        normalized_yaw_speed = min(np.abs(goal_yaw_velocity) / max(yaw_velocity_scale, 1e-6), 1.0)
        yaw_dir = np.array([0.0, 0.0, np.sign(goal_yaw_velocity)], dtype=np.float64)
        yaw_length = 0.08 + 0.42 * normalized_yaw_speed
        yaw_half_length = 0.5 * yaw_length

        model.geom_size[self.yaw_arrow_geom_id, 1] = yaw_half_length
        model.geom_rgba[self.yaw_arrow_geom_id, 3] = 1.0
        model.geom_rgba[self.yaw_arrow_ball_geom_id, 3] = 1.0
        data.geom_xmat[self.yaw_arrow_geom_id] = self._xmat_with_z_axis(yaw_dir).reshape((9,))
        data.geom_xpos[self.yaw_arrow_geom_id] = yaw_base_pos + yaw_dir * yaw_half_length
        data.geom_xpos[self.yaw_arrow_ball_geom_id] = yaw_base_pos + yaw_dir * yaw_length


    def render(self):
        if self.uses_hfield and self.internal_state["info_episode_store"]["episode_step"] == 1:
            mujoco.mjr_uploadHField(self.internal_state["mj_model"], self.viewer.context, 0)
        
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
                goal_velocities = np.array([goal_x_velocity, goal_y_velocity, goal_yaw_velocity])
                goal_velocities = np.where(np.abs(goal_velocities) < (self.command_function.zero_clip_threshold_percentage * self.internal_state["max_command_velocity"]), 0.0, goal_velocities)
                self.internal_state["goal_velocities"] = np.clip(goal_velocities, -self.internal_state["max_command_velocity"], self.internal_state["max_command_velocity"])
                actuator_keep_nominal_commands = np.where(np.all(goal_velocities == 0.0), np.ones(self.nr_actuator_joints, dtype=bool), self.command_function.default_actuator_joint_keep_nominal)
                self.internal_state["actuator_joint_keep_nominal"] = actuator_keep_nominal_commands

        self._apply_ball_plate_post_drop_command()

        if self.add_goal_arrow:
            self._update_goal_arrow()

        # add safety light
        safety = self.internal_state["safe_prediction"]
        if safety < -0.05:
            safety_color = np.array([1.0, 0.0, 0.0, 1.0])  # red
        elif safety < 0.05:
            safety_color = np.array([1.0, 1.0, 0.0, 1.0])  # yellow
        else:
            safety_color = np.array([0.0, 1.0, 0.0, 1.0])  # green

        # write RGBA into the model array
        self.initial_mj_model.geom_rgba[self.safety_light_site_id] = safety_color

        self._update_prediction_visualization_overlay()
        self.viewer.render(self.internal_state["data"])


    def reset(self, seed=None):
        self.terrain_function.sample()

        qpos, qvel = self.initial_state_function.setup()

        self.internal_state["data"] = mujoco.MjData(self.internal_state["mj_model"])
        self.internal_state["data"].qpos = qpos
        self.internal_state["data"].qvel = qvel
        self.internal_state["data"].ctrl = np.zeros(self.nr_actuator_joints)
        mujoco.mj_forward(self.internal_state["mj_model"], self.internal_state["data"])

        episode_success = self.internal_state["info_episode_store"]["episode_return"] >= self.env_curriculum_level_success_episode_return
        self.internal_state["env_curriculum_levels_in_a_row"] = np.where(episode_success,
            np.where(self.internal_state["env_curriculum_levels_in_a_row"] >= 0,
                self.internal_state["env_curriculum_levels_in_a_row"] + 1,
                1
            ),
            np.where(self.internal_state["env_curriculum_levels_in_a_row"] < 0,
                self.internal_state["env_curriculum_levels_in_a_row"] - 1,
                -1
            )
        )
        self.internal_state["env_curriculum_coeff"] =  np.clip(self.internal_state["env_curriculum_coeff"] + self.internal_state["env_curriculum_levels_in_a_row"] / self.env_curriculum_nr_levels, 0.0, 1.0)
        self.internal_state["env_curriculum_coeff"] = np.where(self.internal_state["in_eval_mode"], self.max_curriculum_level, self.internal_state["env_curriculum_coeff"])
        
        self.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(self.internal_state["data"].site_xmat[self.imu_site_id].reshape(3, 3))
        self.internal_state["imu_orientation_rotation_inverse"] = self.internal_state["imu_orientation_rotation"].inv()
        self.internal_state["imu_orientation_euler"] = self.internal_state["imu_orientation_rotation"].as_euler("xyz")
        self.internal_state["last_action"] = np.zeros(self.nr_actuator_joints)
        self.internal_state["last_state"] = np.zeros(self.observation_space.shape)
        self.internal_state["info"]["last_action"] = self.internal_state["last_action"].copy()

        self.internal_state["second_last_action"] = np.zeros(self.nr_actuator_joints)
        self.reward_function.setup()
        self.domain_randomization_action_delay_function.setup()
        self.handle_domain_randomization(is_episode_start=True)

        if self.use_ball_plate:
            qpos = self.internal_state["data"].qpos.copy()
            qvel = self.internal_state["data"].qvel.copy()
            self._reset_ball_plate_state(qpos, qvel)
            self.internal_state["data"] = mujoco.MjData(self.internal_state["mj_model"])
            self.internal_state["data"].qpos = qpos
            self.internal_state["data"].qvel = qvel
            self.internal_state["data"].ctrl = np.zeros(self.nr_actuator_joints)
            mujoco.mj_forward(self.internal_state["mj_model"], self.internal_state["data"])
            self.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(self.internal_state["data"].site_xmat[self.imu_site_id].reshape(3, 3))
            self.internal_state["imu_orientation_rotation_inverse"] = self.internal_state["imu_orientation_rotation"].inv()
            self.internal_state["imu_orientation_euler"] = self.internal_state["imu_orientation_rotation"].as_euler("xyz")

        should_sample_commands = self.command_sampling_function.setup()
        if should_sample_commands:
            self.command_function.get_next_command()
        self._apply_ball_plate_post_drop_command()

        next_observation = self.get_observation(np.zeros(self.nr_actuator_joints))
        if self.use_ball_plate:
            ball_plate_metrics = self.get_ball_plate_metrics()
            self.internal_state["info"][f"env_info/ball_plate_radial_distance"] = ball_plate_metrics["radial_distance"]
            self.internal_state["info"][f"env_info/ball_plate_on_plate"] = float(ball_plate_metrics["on_plate"])
            self.internal_state["info"][f"env_info/ball_plate_ball_dropped"] = float(ball_plate_metrics["ball_dropped"])
            self.internal_state["info"][f"env_info/ball_plate_plate_dropped"] = float(ball_plate_metrics["plate_dropped"])
            self.internal_state["info"][f"env_info/ball_plate_dropped"] = float(ball_plate_metrics["dropped"])
            self.internal_state["info"][f"env_info/ball_plate_drop_latched"] = float(self.internal_state["ball_plate_drop_latched"])
            self.internal_state["info"][f"env_info/ball_plate_time_since_drop"] = self.internal_state["ball_plate_time_since_drop"]
            self.internal_state["info"][f"env_info/ball_plate_post_drop_truncated"] = float(self.internal_state["ball_plate_post_drop_truncated"])
        self.internal_state["last_state"] = next_observation.copy()
        self.internal_state["info"]["last_state"] = self.internal_state["last_state"].copy()

        history_stack = np.tile(next_observation[None, :], (self.nr_history_steps, 1))
        self.internal_state["history_stack"] = history_stack

        self.internal_state["info_episode_store"] = {
            "episode_return": 0.0,
            "episode_step": 0,
            "episode_total_xy_velocity_diff_abs": 0.0,
        }
        self.internal_state["safe_prediction"] = 1.0

        self.internal_state["info"]["history_stack"] = history_stack
        return next_observation, self.internal_state["info"]


    def step(self, action):
        chosen_action = action[:self.nr_actuator_joints]
        delayed_action = self.domain_randomization_action_delay_function.delay_action(chosen_action)

        target_joint_positions = self.control_function.process_action(delayed_action)

        self.internal_state["data"].ctrl = target_joint_positions
        mujoco.mj_step(self.internal_state["mj_model"], self.internal_state["data"], self.nr_substeps)

        # for debugging
        # copy data to avoid modifying it in-place
        # data_copy = mujoco.MjData(self.internal_state["mj_model"])
        # mujoco.mj_copyData(data_copy, self.initial_mj_model, self.internal_state["data"])
        # data_copy1 = copy.deepcopy(self.internal_state["data"])
        # x_next_pred_without_model = self.system_dynamics(data_copy1, target_joint_positions)
        # data_copy2 = copy.deepcopy(self.internal_state["data"])
        # x_next_pred_with_model = self.system_dynamics_with_model(self.internal_state["mj_model"], data_copy2, target_joint_positions)
        #
        # x_next_pred_with_qpos = self.system_dynamics_with_only_qpos_and_qvel(self.internal_state["data"].qpos, self.internal_state["data"].qvel, target_joint_positions)
        # self.internal_state["data"].ctrl = target_joint_positions
        # mujoco.mj_step(self.internal_state["mj_model"], self.internal_state["data"], self.nr_substeps)

        # x_next_true = np.concatenate([self.internal_state["data"].qpos[self.actuator_joint_mask_qpos], self.internal_state["data"].qvel[self.actuator_joint_mask_qvel]], axis=0)
        # print("State diff with model:", np.linalg.norm(x_next_pred_with_model - x_next_true),
        #       "State diff without model:", np.linalg.norm(x_next_pred_without_model - x_next_true),
        #       "State diff with only qpos and qvel:", np.linalg.norm(x_next_pred_with_qpos - x_next_true))

        max_qvel = 100 * np.ones(self.initial_mj_model.nv)
        max_qvel[self.actuator_joint_mask_qvel] = self.internal_state["actuator_joint_max_velocities"]
        self.internal_state["data"].qvel = np.clip(self.internal_state["data"].qvel, -max_qvel, max_qvel)

        self.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(self.internal_state["data"].site_xmat[self.imu_site_id].reshape(3, 3))
        self.internal_state["imu_orientation_rotation_inverse"] = self.internal_state["imu_orientation_rotation"].inv()
        self.internal_state["imu_orientation_euler"] = self.internal_state["imu_orientation_rotation"].as_euler("xyz")

        self.handle_domain_randomization(is_episode_start=False)

        self.terrain_function.pre_step()

        reward = self.reward_function.reward_and_info(chosen_action)

        post_drop_truncated = self._update_ball_plate_post_drop_state()

        should_sample_commands = self.command_sampling_function.step()
        if should_sample_commands or self.command_function_type == "random_trajectory":
            self.command_function.get_next_command()
        self._apply_ball_plate_post_drop_command()

        next_observation = self.get_observation(chosen_action)
        terminated = self.termination_function.should_terminate() | np.any(np.abs(self.internal_state["data"].qvel[:3]) == 100.0)
        truncated = (self.internal_state["info_episode_store"]["episode_step"] >= (self.horizon - 1)) or post_drop_truncated
        done = terminated | truncated

        last_state = self.internal_state["last_state"].copy()
        last_action = self.internal_state["last_action"].copy()

        self.terrain_function.post_step()
        self.reward_function.step()

        history_stack = self.internal_state["history_stack"]
        history_stack_new = np.roll(history_stack, -1, axis=0)
        history_stack_new[-1, :] = next_observation
        self.internal_state["history_stack"] = history_stack_new
        self.internal_state["second_last_action"] = self.internal_state["last_action"].copy()
        self.internal_state["last_action"] = chosen_action.copy()
        self.internal_state["last_state"] = next_observation.copy()
        self.internal_state["info_episode_store"]["episode_step"] += 1
        self.internal_state["info_episode_store"]["episode_return"] += reward
        self.internal_state["info_episode_store"]["episode_total_xy_velocity_diff_abs"] += self.internal_state["info"]["env_info/xy_vel_diff_abs"]
        self.internal_state["info"]["rollout/episode_return"] = np.where(done, self.internal_state["info_episode_store"]["episode_return"], self.internal_state["info"]["rollout/episode_return"])
        self.internal_state["info"]["rollout/episode_length"] = np.where(done, self.internal_state["info_episode_store"]["episode_step"], self.internal_state["info"]["rollout/episode_length"])
        self.internal_state["info"]["env_curriculum/coefficient"] = self.internal_state["env_curriculum_coeff"]
        if self.use_ball_plate:
            self.internal_state["info"]["env_info/ball_plate_drop_latched"] = float(self.internal_state["ball_plate_drop_latched"])
            self.internal_state["info"]["env_info/ball_plate_time_since_drop"] = self.internal_state["ball_plate_time_since_drop"]
            self.internal_state["info"]["env_info/ball_plate_post_drop_truncated"] = float(self.internal_state["ball_plate_post_drop_truncated"])
        self.internal_state["info"]["last_state"] = last_state
        self.internal_state["info"]["last_action"] = last_action
        if self.should_render:
            self.render()
        self.internal_state["info"]["history_stack"] = history_stack_new

        return next_observation, reward, terminated, truncated, self.internal_state["info"]


    def get_observation(self, action):
        qpos = np.concatenate([self.internal_state["data"].qpos[:7], self.internal_state["data"].qpos[self.actuator_joint_mask_qpos]], axis=0)
        qvel = np.concatenate([self.internal_state["data"].qvel[:6], self.internal_state["data"].qvel[self.actuator_joint_mask_qvel]], axis=0)

        feet_ground_contact = self.terrain_function.check_feet_floor_contact()

        robot_height = self.internal_state["robot_imu_height_over_ground"]
        if self.env_config["termination"]["type"] == "booster":
            robot_height_safe = np.float32(
                (robot_height >= self.env_config["termination"]["min_height"]) and
                (robot_height <= self.env_config["termination"]["max_height"])
            )
        else:
            robot_height_threshold = self.env_config["termination"]["height_percentage_threshold"] * self.internal_state["robot_nominal_imu_height_over_ground"]
            robot_height_safe = np.float32(robot_height >= robot_height_threshold)

        body_roll = self.internal_state["imu_orientation_euler"][0]
        body_pitch = self.internal_state["imu_orientation_euler"][1]
        body_tilt = np.sqrt(body_roll ** 2 + body_pitch ** 2)
        body_tilt_safe = np.float32(True if self.env_config["termination"]["type"] == "booster" else body_tilt <= self.internal_state["body_tilt_threshold"])

        command_observation = self.command_function.get_observation(self.internal_state) \
            if hasattr(self.command_function, "get_observation") else self.internal_state["goal_velocities"]

        if self.use_ball_plate:
            ball_plate_metrics = self.get_ball_plate_metrics()
            ball_not_falling = float(not bool(ball_plate_metrics["ball_dropped"]))
            plate_not_falling = float(not bool(ball_plate_metrics["plate_dropped"]))
            ball_plate_has_dropped = float(bool(self.internal_state["ball_plate_drop_latched"] or ball_plate_metrics["dropped"]))
            ball_plate_safety_observation = np.array([ball_not_falling, plate_not_falling, ball_plate_has_dropped], dtype=np.float32)
        else:
            ball_plate_safety_observation = np.array([], dtype=np.float32)

        observation = np.concatenate([
            self.internal_state["data"].qpos[self.actuator_joint_mask_qpos],
            self.internal_state["data"].qvel[self.actuator_joint_mask_qvel],
            action.squeeze(),
            feet_ground_contact,
            self.internal_state["feet_time_on_ground"],
            self.internal_state["feet_time_in_air"],
            self.internal_state["data"].sensordata[self.imu_linear_velocity_sensor_adr:self.imu_linear_velocity_sensor_adr + self.imu_linear_velocity_sensor_dim],
            self.internal_state["data"].sensordata[self.imu_angular_velocity_sensor_adr:self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim],
            command_observation,
            self.internal_state["imu_orientation_rotation_inverse"].apply(np.array([0.0, 0.0, -1.0])),
            self.get_ball_plate_observation(),
            np.array([self.policy_exteroceptive_observation_function.get_exteroceptive_observation()]).reshape(-1),
            np.array([self.critic_exteroceptive_observation_function.get_exteroceptive_observation()]).reshape(-1),
            qpos,    # qpos all
            qvel,    # qvel all
            feet_ground_contact,
            np.array([robot_height_safe]),
            np.array([body_tilt_safe]),
            ball_plate_safety_observation,
        ])

        # Add noise
        self.observation_noise_function.modify_observation(observation)

        if not getattr(self.observation_noise_function, "handles_normalization", False):
            # Normalize and clip
            observation[self.joint_positions_obs_idx] = (observation[self.joint_positions_obs_idx] - self.internal_state["actuator_joint_nominal_positions"]) / 3.14
            observation[self.joint_velocities_obs_idx] /= 100.0
            observation[self.joint_previous_actions_obs_idx] /= 10.0
            observation[self.feet_ground_contact_obs_idx] = (observation[self.feet_ground_contact_obs_idx] / 0.5) - 1.0
            observation[self.feet_time_on_ground_obs_idx] = np.clip((observation[self.feet_time_on_ground_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0)
            observation[self.feet_time_in_air_obs_idx] = np.clip((observation[self.feet_time_in_air_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0)
            observation[self.imu_linear_vel_obs_idx] = np.clip(observation[self.imu_linear_vel_obs_idx] / 10.0, -1.0, 1.0)
            observation[self.imu_angular_vel_obs_idx] = np.clip(observation[self.imu_angular_vel_obs_idx] / 50.0, -1.0, 1.0)
            if len(self.ball_plate_obs_idx) > 0:
                ball_plate_plate_size = self.internal_state["ball_plate_plate_size"]
                observation[self.ball_plate_obs_idx[:3]] = np.clip(observation[self.ball_plate_obs_idx[:3]] / np.maximum(np.max(ball_plate_plate_size[:2]), 1e-6), -10.0, 10.0)
                observation[self.ball_plate_obs_idx[3:6]] = np.clip(observation[self.ball_plate_obs_idx[3:6]] / 5.0, -10.0, 10.0)
                observation[self.ball_plate_obs_idx[6:9]] = np.clip(observation[self.ball_plate_obs_idx[6:9]] / np.maximum(np.max(ball_plate_plate_size[:2]), 1e-6), -10.0, 10.0)
                observation[self.ball_plate_obs_idx[9:]] = np.clip(observation[self.ball_plate_obs_idx[9:]] / 5.0, -10.0, 10.0)
            if len(self.policy_exteroception_obs_idx) > 0:
                observation[self.policy_exteroception_obs_idx] = np.clip((observation[self.policy_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0)
            if len(self.critic_exteroception_obs_idx) > 0:
                observation[self.critic_exteroception_obs_idx] = np.clip((observation[self.critic_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0)

        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = np.clip(observation, -10.0, 10.0)

        return observation


    def _reset_ball_plate_state(self, qpos, qvel):
        if not self.use_ball_plate:
            return

        ball_plate_params = self._sample_ball_plate_domain_randomization()
        ball_radius = ball_plate_params["ball_radius"]
        ball_mass = ball_plate_params["ball_mass"]
        plate_size = ball_plate_params["plate_size"]
        plate_mass = ball_plate_params["plate_mass"]
        self._apply_ball_plate_domain_randomization(self.internal_state["mj_model"], ball_plate_params)

        data = mujoco.MjData(self.internal_state["mj_model"])
        data.qpos = qpos.copy()
        data.qvel = qvel.copy()
        mujoco.mj_forward(self.internal_state["mj_model"], data)

        plate_pos, plate_quat = self._get_ball_plate_supported_pose(
            data,
            plate_size,
            ball_plate_params["capsule_hand_radius"],
            ball_plate_params["capsule_hand_half_length"],
        )
        qpos[self.ball_plate_qposadr:self.ball_plate_qposadr + 3] = plate_pos
        qpos[self.ball_plate_qposadr + 3:self.ball_plate_qposadr + 7] = plate_quat
        qvel[self.ball_plate_qveladr:self.ball_plate_qveladr + 6] = 0.0

        data.qpos = qpos.copy()
        data.qvel = qvel.copy()
        mujoco.mj_forward(self.internal_state["mj_model"], data)

        plate_xmat = data.site_xmat[self.ball_plate_site_id].reshape(3, 3)
        ball_start_offset = ball_plate_params["ball_start_offset"].copy()
        ball_start_offset[2] = ball_radius
        ball_pos = data.site_xpos[self.ball_plate_site_id] + plate_xmat @ ball_start_offset
        qpos[self.ball_plate_ball_qposadr:self.ball_plate_ball_qposadr + 3] = ball_pos
        qpos[self.ball_plate_ball_qposadr + 3:self.ball_plate_ball_qposadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        ball_qvel = np.zeros(6)
        ball_qvel[:2] = ball_plate_params["ball_initial_linear_velocity_xy"]
        ball_qvel[3:] = ball_plate_params["ball_initial_angular_velocity"]
        qvel[self.ball_plate_ball_qveladr:self.ball_plate_ball_qveladr + 6] = ball_qvel
        self.internal_state["ball_plate_ball_radius"] = ball_radius
        self.internal_state["ball_plate_ball_mass"] = ball_mass
        self.internal_state["ball_plate_plate_size"] = plate_size
        self.internal_state["ball_plate_plate_mass"] = plate_mass
        self.internal_state["ball_plate_plate_ball_friction_tangential"] = ball_plate_params["plate_ball_friction_tangential"]
        self.internal_state["ball_plate_plate_support_friction_tangential"] = ball_plate_params["plate_support_friction_tangential"]
        self.internal_state["ball_plate_plate_robot_friction_tangential"] = ball_plate_params["plate_robot_friction_tangential"]
        self.internal_state["ball_plate_contact_timeconst"] = ball_plate_params["contact_timeconst"]
        self.internal_state["ball_plate_capsule_hand_radius"] = ball_plate_params["capsule_hand_radius"]
        self.internal_state["ball_plate_capsule_hand_half_length"] = ball_plate_params["capsule_hand_half_length"]
        self.internal_state["ball_plate_capsule_hand_mass"] = ball_plate_params["capsule_hand_mass"]
        self.internal_state["ball_plate_ball_dropped"] = False
        self.internal_state["ball_plate_plate_dropped"] = False
        self.internal_state["ball_plate_dropped"] = False
        self.internal_state["ball_plate_drop_latched"] = False
        self.internal_state["ball_plate_newly_dropped"] = False
        self.internal_state["ball_plate_time_since_drop"] = 0.0
        self.internal_state["ball_plate_post_drop_truncated"] = False


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
            return max(left_support[2], right_support[2]) + fist_radius

        left_body_xmat = data.xmat[self.ball_plate_left_fist_body_id].reshape(3, 3)
        right_body_xmat = data.xmat[self.ball_plate_right_fist_body_id].reshape(3, 3)
        fist_half_length = float(self.ball_plate_config["fist_half_length"]) if capsule_half_length is None else capsule_half_length
        left_capsule_axis = left_body_xmat[:, 0]
        right_capsule_axis = right_body_xmat[:, 0]
        left_endpoint_heights = np.array([
            left_support[2] - left_capsule_axis[2] * fist_half_length,
            left_support[2] + left_capsule_axis[2] * fist_half_length,
        ])
        right_endpoint_heights = np.array([
            right_support[2] - right_capsule_axis[2] * fist_half_length,
            right_support[2] + right_capsule_axis[2] * fist_half_length,
        ])
        capsule_axis_height = max(np.max(left_endpoint_heights), np.max(right_endpoint_heights))
        return capsule_axis_height + fist_radius


    def _get_ball_plate_supported_pose(self, data, plate_size=None, capsule_radius=None, capsule_half_length=None):
        if plate_size is None:
            plate_size = self.ball_plate_plate_size
        left_support, right_support = self._get_ball_plate_support_points(data)
        support_center = 0.5 * (left_support + right_support)
        plate_pos = support_center.copy()
        support_surface_height = self._get_ball_plate_support_surface_height(data, capsule_radius, capsule_half_length)
        plate_pos[2] = (
            support_surface_height
            + plate_size[2]
            + self.ball_plate_plate_support_clearance
        )
        return plate_pos, np.array(self.ball_plate_config["plate_home_quat"], dtype=float)


    def _sample_curriculum_scalar(self, nominal, value_range, curriculum_coeff, randomize):
        if not randomize:
            return float(nominal)
        sampled = self.np_rng.uniform(value_range[0], value_range[1])
        return float(nominal + curriculum_coeff * (sampled - nominal))


    def _sample_ball_plate_domain_randomization(self):
        curriculum_coeff = self.internal_state["env_curriculum_coeff"]
        if not self.ball_plate_config.get("randomize_ball_plate_domain", True):
            curriculum_coeff = 0.0

        ball_radius = self._sample_curriculum_scalar(
            self.ball_plate_ball_radius,
            self.ball_plate_ball_radius_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_ball_size_mass", True),
        )
        ball_mass = self._sample_curriculum_scalar(
            self.ball_plate_ball_mass,
            self.ball_plate_ball_mass_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_ball_size_mass", True),
        )
        plate_mass = self._sample_curriculum_scalar(
            self.ball_plate_plate_mass,
            self.ball_plate_plate_mass_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_plate_mass", True),
        )
        plate_size_scale = self._sample_curriculum_scalar(
            1.0,
            self.ball_plate_plate_size_scale_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_plate_size", True),
        )
        plate_size = self.ball_plate_plate_size.copy()
        plate_size[:2] *= plate_size_scale

        if self.ball_plate_config.get("randomize_ball_initial_state", True):
            ball_start_offset_xy = curriculum_coeff * self.np_rng.uniform(
                -self.ball_plate_ball_start_offset_xy_range,
                self.ball_plate_ball_start_offset_xy_range,
            )
            ball_initial_linear_velocity_xy = curriculum_coeff * self.np_rng.uniform(
                -self.ball_plate_ball_initial_linear_velocity_xy_range,
                self.ball_plate_ball_initial_linear_velocity_xy_range,
            )
            ball_initial_angular_velocity = curriculum_coeff * self.np_rng.uniform(
                -self.ball_plate_ball_initial_angular_velocity_range,
                self.ball_plate_ball_initial_angular_velocity_range,
            )
        else:
            ball_start_offset_xy = np.zeros(2)
            ball_initial_linear_velocity_xy = np.zeros(2)
            ball_initial_angular_velocity = np.zeros(3)
        ball_start_offset = self.ball_plate_ball_start_offset.copy()
        ball_start_offset[:2] += ball_start_offset_xy

        plate_ball_friction_tangential = self._sample_curriculum_scalar(
            self.ball_plate_nominal_plate_ball_friction,
            self.ball_plate_plate_ball_friction_tangential_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_plate_ball_friction", True),
        )
        plate_support_friction_tangential = self._sample_curriculum_scalar(
            self.ball_plate_nominal_support_friction,
            self.ball_plate_plate_support_friction_tangential_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_plate_support_friction", True),
        )
        plate_robot_friction_tangential = self._sample_curriculum_scalar(
            self.ball_plate_nominal_robot_friction,
            self.ball_plate_plate_robot_friction_tangential_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_plate_robot_friction", True),
        )
        contact_timeconst = self._sample_curriculum_scalar(
            self.ball_plate_nominal_contact_timeconst,
            self.ball_plate_contact_timeconst_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_plate_contact_stiffness", True),
        )
        capsule_hand_radius = self._sample_curriculum_scalar(
            self.ball_plate_capsule_hand_radius,
            self.ball_plate_capsule_hand_radius_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_capsule_hand_size", True),
        )
        capsule_hand_half_length = self._sample_curriculum_scalar(
            self.ball_plate_capsule_hand_half_length,
            self.ball_plate_capsule_hand_half_length_range,
            curriculum_coeff,
            self.ball_plate_config.get("randomize_capsule_hand_size", True),
        )
        capsule_hand_mass = self._sample_curriculum_scalar(
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


    def _set_pair_tangential_friction(self, model, pair_ids, value):
        if len(pair_ids) == 0:
            return
        model.pair_friction[pair_ids, 0] = value
        model.pair_friction[pair_ids, 1] = value


    def _get_capsule_hand_inertia(self, radius, half_length, mass):
        nominal_size = self.ball_plate_capsule_hand_radius + self.ball_plate_capsule_hand_half_length
        size_scale = (radius + half_length) / max(nominal_size, 1e-6)
        mass_scale = mass / max(self.ball_plate_capsule_hand_mass, 1e-6)
        return np.maximum(self.ball_plate_capsule_hand_nominal_body_inertia * mass_scale * size_scale ** 2, 1e-7)


    def _apply_ball_plate_domain_randomization(self, model, params):
        ball_radius = params["ball_radius"]
        ball_mass = params["ball_mass"]
        plate_size = params["plate_size"]
        plate_mass = params["plate_mass"]
        capsule_hand_radius = params["capsule_hand_radius"]
        capsule_hand_half_length = params["capsule_hand_half_length"]
        capsule_hand_mass = params["capsule_hand_mass"]

        model.geom_size[self.ball_plate_ball_geom_id, 0] = ball_radius
        model.geom_size[self.ball_plate_geom_id] = plate_size
        model.geom_rbound[self.ball_plate_ball_geom_id] = ball_radius
        model.geom_rbound[self.ball_plate_geom_id] = np.linalg.norm(plate_size)
        model.site_pos[self.ball_plate_site_id, 2] = plate_size[2]

        model.body_mass[self.ball_plate_ball_body_id] = ball_mass
        model.body_mass[self.ball_plate_body_id] = plate_mass
        model.body_inertia[self.ball_plate_ball_body_id] = np.full(3, 0.4 * ball_mass * ball_radius ** 2)
        model.body_inertia[self.ball_plate_body_id] = (plate_mass / 3.0) * np.array([
            plate_size[1] ** 2 + plate_size[2] ** 2,
            plate_size[0] ** 2 + plate_size[2] ** 2,
            plate_size[0] ** 2 + plate_size[1] ** 2,
        ])
        if self.use_capsule_hand:
            nr_capsule_hands = len(self.ball_plate_capsule_hand_geom_ids)
            model.geom_size[self.ball_plate_capsule_hand_geom_ids] = np.tile(
                np.array([capsule_hand_radius, capsule_hand_half_length, 0.0]),
                (nr_capsule_hands, 1),
            )
            model.geom_rbound[self.ball_plate_capsule_hand_geom_ids] = capsule_hand_radius + capsule_hand_half_length
            model.body_mass[self.ball_plate_capsule_hand_body_ids] = np.full(nr_capsule_hands, capsule_hand_mass)
            model.body_inertia[self.ball_plate_capsule_hand_body_ids] = np.tile(
                self._get_capsule_hand_inertia(capsule_hand_radius, capsule_hand_half_length, capsule_hand_mass),
                (nr_capsule_hands, 1),
            )

        self._set_pair_tangential_friction(model, self.ball_plate_ball_pair_ids, params["plate_ball_friction_tangential"])
        self._set_pair_tangential_friction(model, self.ball_plate_support_pair_ids, params["plate_support_friction_tangential"])
        self._set_pair_tangential_friction(model, self.ball_plate_robot_pair_ids, params["plate_robot_friction_tangential"])
        if len(self.ball_plate_all_pair_ids) > 0:
            model.pair_solref[self.ball_plate_all_pair_ids, 0] = params["contact_timeconst"]


    def get_ball_plate_metrics(self):
        if not self.use_ball_plate:
            return {
                "plate_support_relative_position": np.zeros(3),
                "plate_velocity": np.zeros(3),
                "ball_dropped": False,
                "plate_dropped": False,
                "relative_position": np.zeros(3),
                "relative_velocity": np.zeros(3),
                "radial_distance": 0.0,
                "on_plate": False,
                "dropped": False,
            }

        data = self.internal_state["data"]
        plate_pos = data.site_xpos[self.ball_plate_site_id]
        plate_xmat = data.site_xmat[self.ball_plate_site_id].reshape(3, 3)
        ball_pos = data.xpos[self.ball_plate_ball_body_id]
        left_support, right_support = self._get_ball_plate_support_points(data)
        support_center = 0.5 * (left_support + right_support)
        support_height = max(left_support[2], right_support[2])
        relative_position = plate_xmat.T @ (ball_pos - plate_pos)
        relative_velocity = plate_xmat.T @ data.qvel[self.ball_plate_ball_qveladr:self.ball_plate_ball_qveladr + 3]
        plate_support_relative_position = plate_pos - support_center
        plate_velocity = data.qvel[self.ball_plate_qveladr:self.ball_plate_qveladr + 3]
        plate_up = plate_xmat[:, 2]

        plate_size = self.internal_state["ball_plate_plate_size"]
        outside_x = np.abs(relative_position[0]) > (plate_size[0] + self.ball_plate_drop_margin)
        outside_y = np.abs(relative_position[1]) > (plate_size[1] + self.ball_plate_drop_margin)
        below_plate = relative_position[2] < -self.ball_plate_drop_height
        ball_radius = self.internal_state.get("ball_plate_ball_radius", self.ball_plate_ball_radius)
        near_plate_height = relative_position[2] <= (ball_radius * 3.0)
        plate_below_support = data.xpos[self.ball_plate_body_id, 2] < (support_height - self.ball_plate_drop_height)
        plate_tilted = plate_up[2] < self.ball_plate_tilt_drop_threshold
        ball_dropped = outside_x | outside_y | below_plate
        plate_dropped = plate_below_support | plate_tilted
        dropped = ball_dropped | plate_dropped
        on_plate = (~outside_x) & (~outside_y) & (~below_plate) & near_plate_height
        self.internal_state["ball_plate_ball_dropped"] = ball_dropped
        self.internal_state["ball_plate_plate_dropped"] = plate_dropped
        self.internal_state["ball_plate_dropped"] = dropped

        return {
            "plate_support_relative_position": plate_support_relative_position,
            "plate_velocity": plate_velocity,
            "ball_dropped": ball_dropped,
            "plate_dropped": plate_dropped,
            "relative_position": relative_position,
            "relative_velocity": relative_velocity,
            "radial_distance": np.linalg.norm(relative_position[:2]),
            "on_plate": on_plate,
            "dropped": dropped,
        }


    def get_ball_plate_observation(self):
        if not self.use_ball_plate:
            return np.zeros(0)

        metrics = self.get_ball_plate_metrics()
        return np.concatenate([
            metrics["plate_support_relative_position"],
            metrics["plate_velocity"],
            metrics["relative_position"],
            metrics["relative_velocity"],
        ])


    def get_ball_plate_observation_names(self):
        return [
            "plate_support_relative_position_x",
            "plate_support_relative_position_y",
            "plate_support_relative_position_z",
            "plate_velocity_x",
            "plate_velocity_y",
            "plate_velocity_z",
            "ball_relative_position_x",
            "ball_relative_position_y",
            "ball_relative_position_z",
            "ball_relative_velocity_x",
            "ball_relative_velocity_y",
            "ball_relative_velocity_z",
        ]


    def denormalize_ball_plate_observation(self, observation):
        observation = np.asarray(observation, dtype=float).copy()
        if not self.use_ball_plate or observation.size == 0:
            return observation

        if (
            getattr(self.observation_noise_function, "handles_normalization", False)
            and not getattr(self.observation_noise_function, "handles_ball_plate_normalization", False)
        ):
            return observation

        plate_size = self.internal_state.get("ball_plate_plate_size", self.ball_plate_plate_size)
        position_scale = np.maximum(np.max(plate_size[:2]), 1e-6)
        observation[:3] *= position_scale
        observation[3:6] *= 5.0
        observation[6:9] *= position_scale
        observation[9:] *= 5.0
        return observation


    def _update_prediction_visualization_overlay(self):
        if not self.should_render or not self.use_ball_plate:
            return

        visualization = self.internal_state.get("next_step_prediction_visualization")
        if not visualization:
            self.viewer.extra_overlay = None
            self.viewer.set_extra_geoms([])
            return

        predicted = np.asarray(visualization.get("ball_plate_prediction", []), dtype=float).reshape(-1)
        if predicted.size != self.nr_ball_plate_observations:
            self.viewer.extra_overlay = None
            self.viewer.set_extra_geoms([])
            return

        actual = np.asarray(self.internal_state["last_state"][self.ball_plate_obs_idx], dtype=float).reshape(-1)
        predicted = self.denormalize_ball_plate_observation(predicted)
        actual = self.denormalize_ball_plate_observation(actual)
        error = predicted - actual
        self.viewer.set_extra_geoms(self._get_ball_plate_prediction_geoms(predicted))

        def fmt(vec):
            return "[" + ", ".join(f"{value:+.3f}" for value in vec) + "]"

        title = "Next-step ball plate prediction\n"
        values = "\n"
        title += "ball rel pos pred/real/error\n"
        values += f"{fmt(predicted[6:9])} / {fmt(actual[6:9])} / {fmt(error[6:9])}\n"
        title += "ball rel vel pred/real/error\n"
        values += f"{fmt(predicted[9:12])} / {fmt(actual[9:12])} / {fmt(error[9:12])}\n"
        title += "plate support rel pred/real/error\n"
        values += f"{fmt(predicted[:3])} / {fmt(actual[:3])} / {fmt(error[:3])}\n"
        title += "plate vel pred/real/error\n"
        values += f"{fmt(predicted[3:6])} / {fmt(actual[3:6])} / {fmt(error[3:6])}"
        self.viewer.extra_overlay = [title, values]


    def _get_ball_plate_prediction_geoms(self, predicted_observation):
        data = self.internal_state["data"]
        plate_xmat = data.site_xmat[self.ball_plate_site_id].reshape(3, 3)
        left_support, right_support = self._get_ball_plate_support_points(data)
        support_center = 0.5 * (left_support + right_support)

        plate_size = np.asarray(self.internal_state["ball_plate_plate_size"], dtype=float)
        ball_radius = float(self.internal_state.get("ball_plate_ball_radius", self.ball_plate_ball_radius))
        predicted_plate_top_pos = support_center + predicted_observation[:3]
        predicted_plate_geom_pos = predicted_plate_top_pos - plate_xmat[:, 2] * plate_size[2]
        predicted_ball_pos = predicted_plate_top_pos + plate_xmat @ predicted_observation[6:9]
        predicted_rgba = [1.0, 0.55, 0.05, 0.55]

        return [
            {
                "type": "box",
                "pos": predicted_plate_geom_pos,
                "size": plate_size,
                "mat": plate_xmat.reshape(-1),
                "rgba": predicted_rgba,
            },
            {
                "type": "sphere",
                "pos": predicted_ball_pos,
                "size": [ball_radius, 0.0, 0.0],
                "rgba": predicted_rgba,
            },
        ]


    def ball_plate_has_dropped(self):
        return bool(self.get_ball_plate_metrics()["dropped"])


    def ball_plate_ball_has_dropped(self):
        return bool(self.get_ball_plate_metrics()["ball_dropped"])


    def ball_plate_plate_has_dropped(self):
        return bool(self.get_ball_plate_metrics()["plate_dropped"])


    def handle_domain_randomization(self, is_episode_start=False):
        should_randomize_domain_episode_start = self.domain_randomization_sampling_function.setup()
        should_randomize_domain_perturbation_episode_start = self.domain_randomization_perturbation_sampling_function.setup()
        should_randomize_domain_step = self.domain_randomization_sampling_function.step()
        should_randomize_domain_perturbation_step = self.domain_randomization_perturbation_sampling_function.step()
        should_randomize_domain = np.where(is_episode_start, should_randomize_domain_episode_start | self.internal_state["in_eval_mode"], should_randomize_domain_step)
        should_randomize_domain_perturbation = np.where(is_episode_start, should_randomize_domain_perturbation_episode_start, should_randomize_domain_perturbation_step)

        if should_randomize_domain:
            self.domain_randomization_unseen_robot_function.sample()
            self.domain_randomization_seen_robot_function.sample()
            self.domain_randomization_mujoco_model_function.sample()
            self.domain_randomization_action_delay_function.sample()
            self.joint_dropout_function.sample()
            self.reward_function.handle_model_change()
            if self.use_ball_plate:
                self._apply_ball_plate_domain_randomization(self.internal_state["mj_model"], {
                    "ball_radius": self.internal_state["ball_plate_ball_radius"],
                    "ball_mass": self.internal_state["ball_plate_ball_mass"],
                    "plate_size": self.internal_state["ball_plate_plate_size"],
                    "plate_mass": self.internal_state["ball_plate_plate_mass"],
                    "plate_ball_friction_tangential": self.internal_state["ball_plate_plate_ball_friction_tangential"],
                    "plate_support_friction_tangential": self.internal_state["ball_plate_plate_support_friction_tangential"],
                    "plate_robot_friction_tangential": self.internal_state["ball_plate_plate_robot_friction_tangential"],
                    "contact_timeconst": self.internal_state["ball_plate_contact_timeconst"],
                    "capsule_hand_radius": self.internal_state["ball_plate_capsule_hand_radius"],
                    "capsule_hand_half_length": self.internal_state["ball_plate_capsule_hand_half_length"],
                    "capsule_hand_mass": self.internal_state["ball_plate_capsule_hand_mass"],
                })
        
        if should_randomize_domain_perturbation:
            self.domain_randomization_perturbation_function.sample()
    

    def get_observation_space(self):
        current_observation_idx = 0

        self.joint_positions_obs_idx = np.array([current_observation_idx + i for i in range(self.nr_actuator_joints)], dtype=int)
        current_observation_idx += self.nr_actuator_joints
        self.joint_velocities_obs_idx = np.array([current_observation_idx + i for i in range(self.nr_actuator_joints)], dtype=int)
        current_observation_idx += self.nr_actuator_joints
        self.joint_previous_actions_obs_idx = np.array([current_observation_idx + i for i in range(self.nr_actuator_joints)], dtype=int)
        current_observation_idx += self.nr_actuator_joints
        self.feet_ground_contact_obs_idx = np.array([current_observation_idx + i for i in range(self.nr_feet)], dtype=int)
        current_observation_idx += self.nr_feet
        self.feet_time_on_ground_obs_idx = np.array([current_observation_idx + i for i in range(self.nr_feet)], dtype=int)
        current_observation_idx += self.nr_feet
        self.feet_time_in_air_obs_idx = np.array([current_observation_idx + i for i in range(self.nr_feet)], dtype=int)
        current_observation_idx += self.nr_feet
        self.imu_linear_vel_obs_idx = np.array([current_observation_idx + i for i in range(self.imu_linear_velocity_sensor_dim)], dtype=int)
        current_observation_idx += self.imu_linear_velocity_sensor_dim
        self.imu_angular_vel_obs_idx = np.array([current_observation_idx + i for i in range(self.imu_angular_velocity_sensor_dim)], dtype=int)
        current_observation_idx += self.imu_angular_velocity_sensor_dim
        self.goal_velocities_obs_idx = np.array([current_observation_idx + i for i in range(self.command_observation_size)], dtype=int)
        current_observation_idx += self.command_observation_size
        self.gravity_vector_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3
        self.ball_plate_obs_idx = np.array([current_observation_idx + i for i in range(self.nr_ball_plate_observations)], dtype=int)
        current_observation_idx += self.nr_ball_plate_observations
        self.policy_exteroception_obs_idx = np.array([current_observation_idx + i for i in range(self.policy_exteroceptive_observation_function.nr_exteroceptive_observations)], dtype=int)
        current_observation_idx += self.policy_exteroceptive_observation_function.nr_exteroceptive_observations
        self.critic_exteroception_obs_idx = np.array([current_observation_idx + i for i in range(self.critic_exteroceptive_observation_function.nr_exteroceptive_observations)], dtype=int)
        current_observation_idx += self.critic_exteroceptive_observation_function.nr_exteroceptive_observations

        self.qpos_observation_idx = np.array([current_observation_idx + i for i in range(self.nr_actuator_joints + 7)])
        current_observation_idx += self.nr_actuator_joints + 7

        self.qvel_observation_idx = np.array([current_observation_idx + i for i in range(self.nr_actuator_joints + 6)])
        current_observation_idx += self.nr_actuator_joints + 6

        self.contact_obs_idx = np.array([current_observation_idx + i for i in range(self.nr_feet)])
        current_observation_idx += self.nr_feet

        self.body_height_obs_idx = np.array([current_observation_idx], dtype=int)
        current_observation_idx += 1

        self.body_tilt_obs_idx = np.array([current_observation_idx], dtype=int)
        current_observation_idx += 1

        if self.use_ball_plate:
            self.ball_not_falling_obs_idx = np.array([current_observation_idx], dtype=int)
            current_observation_idx += 1

            self.plate_not_falling_obs_idx = np.array([current_observation_idx], dtype=int)
            current_observation_idx += 1

            self.ball_plate_dropped_obs_idx = np.array([current_observation_idx], dtype=int)
            current_observation_idx += 1
        else:
            self.ball_not_falling_obs_idx = np.array([], dtype=int)
            self.plate_not_falling_obs_idx = np.array([], dtype=int)
            self.ball_plate_dropped_obs_idx = np.array([], dtype=int)

        ball_plate_policy_obs_idx = self.ball_plate_obs_idx if self.include_ball_plate_observations else np.array([], dtype=int)
        ball_plate_policy_state_obs_idx = np.concatenate([
            ball_plate_policy_obs_idx,
            self.ball_plate_dropped_obs_idx,
        ], dtype=int) if self.use_ball_plate else np.array([], dtype=int)

        self.policy_observation_indices = np.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.goal_velocities_obs_idx,
            self.gravity_vector_obs_idx,
            ball_plate_policy_state_obs_idx,
            self.policy_exteroception_obs_idx,
        ], dtype=int)

        self.critic_observation_indices = np.concatenate([
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
            self.critic_exteroception_obs_idx,
        ], dtype=int)

        self.dynamics_observation_indices = np.concatenate([
            self.qpos_observation_idx,
            self.qvel_observation_idx,
        ])

        if self.env_config.ncbf_use_policy_observations:
            self.ncbf_observation_indices = self.policy_observation_indices
        else:
            actuator_qpos_idx = self.qpos_observation_idx[7:]
            actuator_qvel_idx = self.qvel_observation_idx[6:]

            self.ncbf_observation_indices = np.concatenate([actuator_qpos_idx, actuator_qvel_idx], dtype=int)
        # note all obs here is not normalized or clipped to pass into the forward step function for dynamics models

        self.ncbf_obs_in_dynamics_state_idx = np.concatenate([
            self.actuator_joint_mask_qpos,
            self.actuator_joint_mask_qvel + self.initial_mj_model.nq
        ], dtype=int
        )

        self.next_state_indices = np.concatenate([
            self.qvel_observation_idx[:3],
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.ball_plate_obs_idx,
        ], dtype=int
        )

        ncbf_target_indices = [
            self.body_height_obs_idx,
            self.body_tilt_obs_idx,
        ]
        if self.use_ball_plate:
            ncbf_target_indices.extend([
                self.ball_not_falling_obs_idx,
                self.plate_not_falling_obs_idx,
            ])
        self.ncbf_target_indices = np.concatenate(ncbf_target_indices, dtype=int)

        observation_space_low = -np.ones(current_observation_idx) * np.inf
        observation_space_high = np.ones(current_observation_idx) * np.inf

        return gym.spaces.Box(low=observation_space_low, high=observation_space_high, shape=(current_observation_idx,), dtype=np.float32)


    def close(self):
        if self.should_render:
            self.viewer.close()
            pygame.quit()
