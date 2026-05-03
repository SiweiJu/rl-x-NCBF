import numpy as np


class BoosterDRSeenRobotFunction:
    def __init__(self, env):
        self.env = env
        config = env.env_config["domain_randomization"]["seen_robot"]

        self.randomize_joint_damping = config.get("randomize_joint_damping", True)
        self.joint_damping_range = np.array(config.get("joint_damping_range", [0.005, 0.015]))
        self.randomize_joint_friction_loss = config.get("randomize_joint_friction_loss", True)
        self.joint_friction_loss_range = np.array(config.get("joint_friction_loss_range", [0.0, 0.5]))
        self.randomize_joint_armature = config.get("randomize_joint_armature", True)
        self.joint_armature_range = np.array(config.get("joint_armature_range", [0.007, 0.013]))
        self.randomize_com_displacement = config.get("randomize_com_displacement", True)
        self.com_displacement_range = np.array(config.get("com_displacement_range", [-0.05, 0.05]))
        self.randomize_link_mass = config.get("randomize_link_mass", True)
        link_mass_range = config.get("link_mass_multiplier_range", {
            "root_body": [0.8, 1.2],
            "other_bodies": [0.9, 1.1],
        })
        self.root_body_mass_range = np.array(link_mass_range.get("root_body", [0.8, 1.2]))
        self.other_body_mass_range = np.array(link_mass_range.get("other_bodies", [0.9, 1.1]))
        self.add_p_gains_noise = config.get("add_p_gains_noise", True)
        self.add_d_gains_noise = config.get("add_d_gains_noise", True)
        self.p_gains_noise_scale = config.get("p_gains_noise_scale", 0.15)
        self.d_gains_noise_scale = config.get("d_gains_noise_scale", 0.15)

        self.default_body_mass = env.initial_mj_model.body_mass.copy()
        self.default_body_inertia = env.initial_mj_model.body_inertia.copy()
        self.default_body_ipos = env.initial_mj_model.body_ipos.copy()
        self.default_dof_damping = env.initial_mj_model.dof_damping.copy()
        self.default_dof_frictionloss = env.initial_mj_model.dof_frictionloss.copy()
        self.default_dof_armature = env.initial_mj_model.dof_armature.copy()
        self.default_p_gain = env.initial_mj_model.actuator_gainprm[:, 0].copy()
        self.default_d_gain = -env.initial_mj_model.actuator_biasprm[:, 2].copy()
        self.default_scaling_factor = env.robot_config["scaling_factor"]
        self.default_actuator_joint_nominal_positions = env.initial_qpos[env.actuator_joint_mask_qpos].copy()
        self.default_actuator_joint_max_velocities = env.actuator_joint_max_velocities.copy()

        self.root_body_id = env.trunk_body_id
        other_body_ids = np.arange(env.initial_mj_model.nbody)
        self.other_body_ids = other_body_ids[(other_body_ids != 0) & (other_body_ids != self.root_body_id)]
        self.nr_bodies_without_world = env.initial_mj_model.nbody - 1
        self.nr_non_free_dofs = env.initial_mj_model.nv - 6
        self.nr_non_free_joints = env.initial_mj_model.njnt - 1


    @staticmethod
    def _lerp(value_range, interpolation):
        return value_range[0] + (value_range[1] - value_range[0]) * interpolation


    def init(self):
        self.env.internal_state["seen_body_masses"] = self.default_body_mass[1:]
        self.env.internal_state["seen_body_inertias"] = self.default_body_inertia[1:]
        self.env.internal_state["seen_body_coms"] = self.default_body_ipos[1:]
        self.env.internal_state["seen_body_positions"] = self.env.initial_mj_model.body_pos[1:]
        self.env.internal_state["seen_torque_limits"] = self.env.initial_mj_model.actuator_forcerange[:, 1]
        self.env.internal_state["seen_joint_ranges"] = self.env.initial_mj_model.jnt_range[1:]
        self.env.internal_state["seen_joint_dampings"] = self.default_dof_damping[6:]
        self.env.internal_state["seen_joint_armatures"] = self.default_dof_armature[6:]
        self.env.internal_state["seen_joint_stiffnesses"] = self.env.initial_mj_model.jnt_stiffness[1:]
        self.env.internal_state["seen_joint_frictionlosses"] = self.default_dof_frictionloss[6:]
        self.env.internal_state["seen_p_gain"] = self.default_p_gain
        self.env.internal_state["seen_d_gain"] = self.default_d_gain
        self.env.internal_state["scaling_factor"] = self.default_scaling_factor
        self.env.internal_state["partial_actuator_gainprm_without_dropout"] = self.default_p_gain
        self.env.internal_state["partial_actuator_biasprm_without_dropout"] = self.env.initial_mj_model.actuator_biasprm[:, 1:3].copy()
        self.env.internal_state["robot_nominal_qpos_height_over_ground"] = self.env.initial_qpos[2]
        self.env.internal_state["robot_nominal_imu_height_over_ground"] = self.env.initial_imu_height

        self.env.internal_state["mass_inertia_noise_factors"] = np.ones(self.nr_bodies_without_world)
        self.env.internal_state["com_noise_factors"] = np.ones((self.nr_bodies_without_world, 3))
        self.env.internal_state["body_position_noise_factors"] = np.ones((self.nr_bodies_without_world, 3))
        self.env.internal_state["joint_damping_noise_factors"] = np.ones(self.nr_non_free_dofs)
        self.env.internal_state["joint_armature_noise_factors"] = np.ones(self.nr_non_free_dofs)
        self.env.internal_state["joint_stiffness_noise_factors"] = np.ones(self.nr_non_free_joints)
        self.env.internal_state["joint_friction_loss_noise_factors"] = np.ones(self.nr_non_free_dofs)
        self.env.internal_state["p_gain_noise_factors"] = np.ones(self.env.nr_actuator_joints)
        self.env.internal_state["d_gain_noise_factors"] = np.ones(self.env.nr_actuator_joints)
        self.env.internal_state["position_offsets"] = np.zeros(self.env.nr_actuator_joints)


    def sample(self):
        model = self.env.internal_state["mj_model"]

        body_mass = self.default_body_mass.copy()
        if self.randomize_link_mass:
            root_multiplier = self._lerp(self.root_body_mass_range, self.env.np_rng.uniform())
            other_multipliers = self._lerp(
                self.other_body_mass_range,
                self.env.np_rng.uniform(size=(self.other_body_ids.shape[0],)),
            )
            body_mass[self.root_body_id] = self.default_body_mass[self.root_body_id] * root_multiplier
            body_mass[self.other_body_ids] = self.default_body_mass[self.other_body_ids] * other_multipliers

        body_ipos = self.default_body_ipos.copy()
        if self.randomize_com_displacement:
            body_ipos[self.root_body_id] = (
                self.default_body_ipos[self.root_body_id] +
                self._lerp(self.com_displacement_range, self.env.np_rng.uniform(size=(3,)))
            )

        joint_damping = self.default_dof_damping[6:].copy()
        if self.randomize_joint_damping:
            joint_damping = self._lerp(self.joint_damping_range, self.env.np_rng.uniform(size=joint_damping.shape))
        dof_damping = self.default_dof_damping.copy()
        dof_damping[6:] = joint_damping

        joint_friction_loss = self.default_dof_frictionloss[6:].copy()
        if self.randomize_joint_friction_loss:
            joint_friction_loss = self._lerp(
                self.joint_friction_loss_range,
                self.env.np_rng.uniform(size=joint_friction_loss.shape),
            )
        dof_frictionloss = self.default_dof_frictionloss.copy()
        dof_frictionloss[6:] = joint_friction_loss

        joint_armature = self.default_dof_armature[6:].copy()
        if self.randomize_joint_armature:
            joint_armature = self._lerp(self.joint_armature_range, self.env.np_rng.uniform(size=joint_armature.shape))
        dof_armature = self.default_dof_armature.copy()
        dof_armature[6:] = joint_armature

        p_gain = self.default_p_gain.copy()
        if self.add_p_gains_noise:
            p_gain += (
                self.env.np_rng.uniform(low=-1.0, high=1.0, size=p_gain.shape) *
                self.p_gains_noise_scale * self.default_p_gain
            )
        d_gain = self.default_d_gain.copy()
        if self.add_d_gains_noise:
            d_gain += (
                self.env.np_rng.uniform(low=-1.0, high=1.0, size=d_gain.shape) *
                self.d_gains_noise_scale * self.default_d_gain
            )

        model.body_mass[:] = body_mass
        model.body_ipos[:] = body_ipos
        model.dof_damping[:] = dof_damping
        model.dof_frictionloss[:] = dof_frictionloss
        model.dof_armature[:] = dof_armature
        model.actuator_gainprm[:, 0] = p_gain
        model.actuator_biasprm[:, 1] = -p_gain
        model.actuator_biasprm[:, 2] = -d_gain

        self.env.internal_state["seen_body_masses"] = body_mass[1:]
        self.env.internal_state["seen_body_coms"] = body_ipos[1:]
        self.env.internal_state["seen_joint_dampings"] = joint_damping
        self.env.internal_state["seen_joint_armatures"] = joint_armature
        self.env.internal_state["seen_joint_frictionlosses"] = joint_friction_loss
        self.env.internal_state["seen_p_gain"] = p_gain
        self.env.internal_state["seen_d_gain"] = d_gain
        self.env.internal_state["actuator_joint_nominal_positions"] = self.default_actuator_joint_nominal_positions.copy()
        self.env.internal_state["actuator_joint_max_velocities"] = self.default_actuator_joint_max_velocities.copy()
        self.env.internal_state["scaling_factor"] = self.default_scaling_factor
        self.env.internal_state["partial_actuator_gainprm_without_dropout"] = p_gain
        self.env.internal_state["partial_actuator_biasprm_without_dropout"] = np.stack([-p_gain, -d_gain], axis=1)
