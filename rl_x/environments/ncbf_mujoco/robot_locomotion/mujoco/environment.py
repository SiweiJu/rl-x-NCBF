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
        self.nr_envs = nr_envs
        self.nr_history_steps = env_config["nr_history_steps"]

        self.np_rng = np.random.default_rng(seed)

        xml_path = (self.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
        xml_handle = mjcf.from_path(xml_path)

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
            trunk = xml_handle.find("body", "trunk")
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
            trunk = xml_handle.find("body", "trunk")
            trunk.add("geom", name="safety_light", pos="0 0 1", type="sphere", dclass="visual", size="0.05", rgba="1 0 0 1", group="0")
            # light_vec = xml_handle.find("body", "safety_light_body")
            # light_vec.add("site", name="safety_light", type="sphere", size="0.2", pos="0 0 0.6", rgba="0 1 0 1")
            #
            # trunk = xml_handle.worldbody
            # trunk.add("site", name="safety_light", type="sphere", size="0.5", pos="0 0 1.5", rgba="1 0 0 1")

        self.initial_mj_model = mujoco.MjModel.from_xml_string(xml=xml_handle.to_xml_string(), assets=xml_handle.get_assets())
        self.initial_mj_model.opt.timestep = env_config["timestep"]
        self.data = mujoco.MjData(self.initial_mj_model)
        self.c_model = deepcopy(self.initial_mj_model)
        self.c_data = mujoco.MjData(self.c_model)
        self.c_data.qpos = self.initial_mj_model.keyframe("home").qpos
        mujoco.mj_forward(self.c_model, self.c_data)
        
        self.imu_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.trunk_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
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
        all_feet_are_box = np.all(self.initial_mj_model.geom_type[self.foot_geom_indices] == 6)
        if not all_feet_are_sphere | all_feet_are_box:
            raise ValueError("Foot geoms are not all of type sphere or box.")
        self.foot_type = "sphere" if all_feet_are_sphere else "box"
        self.foot_type_int = 0 if self.foot_type == "sphere" else 1

        feet_global_linear_velocity_sensor_ids = [self.initial_mj_model.sensor(f"{foot_name}_global_linear_velocity").id for foot_name in self.feet_names]
        self.feet_global_linear_velocity_sensor_adrs_start = np.array([self.initial_mj_model.sensor_adr[sensor_id] for sensor_id in feet_global_linear_velocity_sensor_ids])

        body_to_parentid = np.array([self.initial_mj_model.body(body_id).parentid[0] for body_id in range(self.initial_mj_model.nbody)])
        body_to_children_count = np.array([np.sum(body_to_parentid == body_id) for body_id in range(self.initial_mj_model.nbody)])
        self.body_ids_of_actuator_joints = np.array([self.initial_mj_model.joint(joint_name).bodyid[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_nr_direct_child_actuator_joints = body_to_children_count[self.body_ids_of_actuator_joints]

        self.floor_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        self.reward_collision_sphere_geom_ids = np.array([geom.id for geom in [self.initial_mj_model.geom(geom_id) for geom_id in range(self.initial_mj_model.ngeom)] if geom.group[0] == 5])
        
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
        self.command_sampling_function = get_sampling_function(env_config["command"]["sampling_type"], self)
        self.initial_state_function = get_initial_state_function(env_config["domain_randomization"]["initial_state"]["type"], self)
        self.reward_function = get_reward_function(env_config["reward"]["type"], self)
        self.termination_function = get_termination_function(env_config["termination"]["type"], self)
        self.policy_exteroceptive_observation_function = get_exteroceptive_observation_function(env_config["policy_exteroceptive_observation_type"], self)
        self.critic_exteroceptive_observation_function = get_exteroceptive_observation_function(env_config["critic_exteroceptive_observation_type"], self)
        self.terrain_function = get_terrain_function(env_config["terrain"]["type"], self)
        self.domain_randomization_sampling_function = get_sampling_function(env_config["domain_randomization"]["sampling_type"], self)
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
        self.internal_state = {
            "mj_model": deepcopy(self.initial_mj_model),
            "data": mujoco.MjData(self.initial_mj_model),
            "in_eval_mode": eval_mode,
            "env_curriculum_coeff": np.where(eval_mode, 1.0, 0.0),
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
            "nr_collisions_in_nominal": 0,
            "info": {
                "rollout/episode_return": 0.0,
                "rollout/episode_length": 0,
                "env_curriculum/coefficient": np.where(eval_mode, 1.0, 0.0),
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

        # for debugging, get the system dynamics function
        # model = deepcopy(self.initial_mj_model)
        # nr_substeps = self.nr_substeps
        # template_data = mujoco.MjData(self.internal_state["mj_model"])
        #
        # def system_dynamics(model_data, u):
        #     # harded coded indexing for now
        #     # qpos : pos(3), quat(4), joint_pos(n_joints)
        #     # qvel : vel(3), ang_vel(3), joint_vel(n_joints
        #     # pos(3) is not necessary
        #     # qpos = x[:nq]
        #     # qvel = x[nq:nq + nv]
        #
        #     # data = model_data.replace(qpos=qpos, qvel=qvel, ctrl=u)
        #
        #     model_data.ctrl = u
        #     mujoco.mj_forward(model, model_data)
        #     mujoco.mj_step(model, model_data, nr_substeps)
        #     return np.concatenate([model_data.qpos[self.actuator_joint_mask_qpos], model_data.qvel[self.actuator_joint_mask_qvel]], axis=0)

        # def system_dynamics_with_model(model, model_data, u):
        #     # harded coded indexing for now
        #     # qpos : pos(3), quat(4), joint_pos(n_joints)
        #     # qvel : vel(3), ang_vel(3), joint_vel(n_joints
        #     # pos(3) is not necessary
        #     # qpos = x[:nq]
        #     # qvel = x[nq:nq + nv]
        #
        #     # data = model_data.replace(qpos=qpos, qvel=qvel, ctrl=u)
        #
        #     model_data.ctrl = u
        #     mujoco.mj_forward(model, model_data)
        #     mujoco.mj_step(model, model_data, nr_substeps)
        #     return np.concatenate([model_data.qpos[self.actuator_joint_mask_qpos], model_data.qvel[self.actuator_joint_mask_qvel]], axis=0)
        #
        # def system_dynamics_with_only_qpos_and_qvel(qpos, qvel, u):
        #     template_data.qpos = qpos
        #     template_data.qvel = qvel
        #     template_data.ctrl = u
        #     mujoco.mj_forward(model, template_data)
        #     mujoco.mj_step(model, template_data, nr_substeps)
        #     return np.concatenate([template_data.qpos[self.actuator_joint_mask_qpos], template_data.qvel[self.actuator_joint_mask_qvel]], axis=0)
        # self.system_dynamics = system_dynamics
        # self.system_dynamics_with_model = system_dynamics_with_model
        # self.system_dynamics_with_only_qpos_and_qvel = system_dynamics_with_only_qpos_and_qvel

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
        arrow_base_pos = data.body("trunk").xpos + np.array([0.0, 0.0, 0.5], dtype=np.float64)
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
        self.internal_state["env_curriculum_coeff"] = np.where(self.internal_state["in_eval_mode"], 1.0, self.internal_state["env_curriculum_coeff"])
        
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

        should_sample_commands = self.command_sampling_function.setup()
        if should_sample_commands:
            self.command_function.get_next_command()

        next_observation = self.get_observation(np.zeros(self.nr_actuator_joints))
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

        should_sample_commands = self.command_sampling_function.step()
        if should_sample_commands or self.command_function_type == "random_trajectory":
            self.command_function.get_next_command()

        next_observation = self.get_observation(chosen_action)
        terminated = self.termination_function.should_terminate() | np.any(np.abs(self.internal_state["data"].qvel[:3]) == 100.0)
        truncated = self.internal_state["info_episode_store"]["episode_step"] >= (self.horizon - 1)
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
        robot_height_threshold = self.env_config["termination"]["height_percentage_threshold"] * self.internal_state["robot_nominal_imu_height_over_ground"]
        robot_height_safe = np.float32(robot_height >= robot_height_threshold)

        body_roll = self.internal_state["imu_orientation_euler"][0]
        body_pitch = self.internal_state["imu_orientation_euler"][1]
        body_tilt = np.sqrt(body_roll ** 2 + body_pitch ** 2)
        body_tilt_safe = np.float32(body_tilt <= self.internal_state["body_tilt_threshold"])

        observation = np.concatenate([
            self.internal_state["data"].qpos[self.actuator_joint_mask_qpos],
            self.internal_state["data"].qvel[self.actuator_joint_mask_qvel],
            action.squeeze(),
            feet_ground_contact,
            self.internal_state["feet_time_on_ground"],
            self.internal_state["feet_time_in_air"],
            self.internal_state["data"].sensordata[self.imu_linear_velocity_sensor_adr:self.imu_linear_velocity_sensor_adr + self.imu_linear_velocity_sensor_dim],
            self.internal_state["data"].sensordata[self.imu_angular_velocity_sensor_adr:self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim],
            self.internal_state["goal_velocities"],
            self.internal_state["imu_orientation_rotation_inverse"].apply(np.array([0.0, 0.0, -1.0])),
            np.array([self.policy_exteroceptive_observation_function.get_exteroceptive_observation()]).reshape(-1),
            np.array([self.critic_exteroceptive_observation_function.get_exteroceptive_observation()]).reshape(-1),
            qpos,    # qpos all
            qvel,    # qvel all
            feet_ground_contact,
            np.array([robot_height_safe]),
            np.array([body_tilt_safe]),
        ])

        # Add noise
        self.observation_noise_function.modify_observation(observation)

        # Normalize and clip
        observation[self.joint_positions_obs_idx] = (observation[self.joint_positions_obs_idx] - self.internal_state["actuator_joint_nominal_positions"]) / 3.14
        observation[self.joint_velocities_obs_idx] /= 100.0
        observation[self.joint_previous_actions_obs_idx] /= 10.0
        observation[self.feet_ground_contact_obs_idx] = (observation[self.feet_ground_contact_obs_idx] / 0.5) - 1.0
        observation[self.feet_time_on_ground_obs_idx] = np.clip((observation[self.feet_time_on_ground_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0)
        observation[self.feet_time_in_air_obs_idx] = np.clip((observation[self.feet_time_in_air_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0)
        observation[self.imu_linear_vel_obs_idx] = np.clip(observation[self.imu_linear_vel_obs_idx] / 10.0, -1.0, 1.0)
        observation[self.imu_angular_vel_obs_idx] = np.clip(observation[self.imu_angular_vel_obs_idx] / 50.0, -1.0, 1.0)
        if len(self.policy_exteroception_obs_idx) > 0:
            observation[self.policy_exteroception_obs_idx] = np.clip((observation[self.policy_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0)
        if len(self.critic_exteroception_obs_idx) > 0:
            observation[self.critic_exteroception_obs_idx] = np.clip((observation[self.critic_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0)

        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = np.clip(observation, -10.0, 10.0)

        return observation


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
        self.goal_velocities_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3
        self.gravity_vector_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3
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

        self.policy_observation_indices = np.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.goal_velocities_obs_idx,
            self.gravity_vector_obs_idx,
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
            self.critic_exteroception_obs_idx,
        ], dtype=int)

        self.dynamics_observation_indices = np.concatenate([
            self.qpos_observation_idx,
            self.qvel_observation_idx,
        ])

        if self.env_config.ncbf_use_policy_observations:
            self.ncbf_observation_indices = self.policy_observation_indices
        else:
            actuator_qpos_idx = self.qpos_observation_idx[self.actuator_joint_mask_qpos]
            actuator_qvel_idx = self.qvel_observation_idx[self.actuator_joint_mask_qvel]

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
        ], dtype=int
        )

        self.ncbf_target_indices = np.concatenate([
            self.body_height_obs_idx,
            self.body_tilt_obs_idx,
        ], dtype=int
        )

        observation_space_low = -np.ones(current_observation_idx) * np.inf
        observation_space_high = np.ones(current_observation_idx) * np.inf

        return gym.spaces.Box(low=observation_space_low, high=observation_space_high, shape=(current_observation_idx,), dtype=np.float32)


    def close(self):
        if self.should_render:
            self.viewer.close()
            pygame.quit()
