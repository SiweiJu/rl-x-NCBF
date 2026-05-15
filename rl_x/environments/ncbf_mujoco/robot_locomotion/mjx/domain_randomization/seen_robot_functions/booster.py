import numpy as np
import jax
import jax.numpy as jnp


class BoosterDRSeenRobotFunction:
    def __init__(self, env):
        self.env = env
        config = env.env_config["domain_randomization"]["seen_robot"]

        self.randomize_joint_damping = config.get("randomize_joint_damping", True)
        self.joint_damping_range = jnp.array(config.get("joint_damping_range", [0.005, 0.015]))
        self.randomize_joint_friction_loss = config.get("randomize_joint_friction_loss", True)
        self.joint_friction_loss_range = jnp.array(config.get("joint_friction_loss_range", [0.0, 0.5]))
        self.randomize_joint_armature = config.get("randomize_joint_armature", True)
        self.joint_armature_range = jnp.array(config.get("joint_armature_range", [0.007, 0.013]))
        self.randomize_com_displacement = config.get("randomize_com_displacement", True)
        self.com_displacement_range = self._axis_range(config.get("com_displacement_range", [-0.05, 0.05]))
        self.randomize_link_mass = config.get("randomize_link_mass", True)
        link_mass_range = config.get("link_mass_multiplier_range", {
            "root_body": [0.8, 1.2],
            "other_bodies": [0.9, 1.1],
        })
        self.root_body_mass_range = jnp.array(link_mass_range.get("root_body", [0.8, 1.2]))
        self.other_body_mass_range = jnp.array(link_mass_range.get("other_bodies", [0.9, 1.1]))
        self.add_p_gains_noise = config.get("add_p_gains_noise", True)
        self.add_d_gains_noise = config.get("add_d_gains_noise", True)
        self.p_gains_noise_scale = config.get("p_gains_noise_scale", 0.15)
        self.d_gains_noise_scale = config.get("d_gains_noise_scale", 0.15)
        self.add_actuator_joint_nominal_position = config.get("add_actuator_joint_nominal_position", 0.0)
        self.randomize_actuator_joint_nominal_position = config.get(
            "randomize_actuator_joint_nominal_position",
            self.add_actuator_joint_nominal_position > 0.0,
        )

        self.default_body_mass = env.initial_mjx_model.body_mass
        self.default_body_inertia = env.initial_mjx_model.body_inertia
        self.default_body_ipos = env.initial_mjx_model.body_ipos
        self.default_dof_damping = env.initial_mjx_model.dof_damping
        self.default_dof_frictionloss = env.initial_mjx_model.dof_frictionloss
        self.default_dof_armature = env.initial_mjx_model.dof_armature
        self.default_p_gain = env.actuator_joint_stiffness if env.use_torque_pd_control else env.initial_mjx_model.actuator_gainprm[:, 0]
        self.default_d_gain = env.actuator_joint_damping if env.use_torque_pd_control else -env.initial_mjx_model.actuator_biasprm[:, 2]
        self.default_effort_limits = env.actuator_joint_effort_limits
        self.default_velocity_limits = env.actuator_joint_velocity_limits
        self.default_knee_point_velocities = env.actuator_joint_knee_point_velocities
        self.default_scaling_factor = env.scaling_factor
        self.default_actuator_joint_nominal_positions = env.initial_qpos[env.actuator_joint_mask_qpos]
        self.default_actuator_joint_max_velocities = env.actuator_joint_max_velocities

        self.root_body_id = env.trunk_body_id
        other_body_ids = np.arange(env.initial_mj_model.nbody)
        other_body_ids = other_body_ids[(other_body_ids != 0) & (other_body_ids != self.root_body_id)]
        self.other_body_ids = jnp.array(other_body_ids)
        self.nr_bodies_without_world = env.initial_mj_model.nbody - 1
        self.nr_non_free_dofs = env.initial_mj_model.nv - 6
        self.nr_non_free_joints = env.initial_mj_model.njnt - 1


    @staticmethod
    def _lerp(value_range, interpolation):
        return value_range[0] + (value_range[1] - value_range[0]) * interpolation


    @staticmethod
    def _axis_range(value_range):
        if hasattr(value_range, "get"):
            return jnp.array([value_range.get(axis, [0.0, 0.0]) for axis in ("x", "y", "z")])
        value_range = jnp.array(value_range)
        if value_range.ndim == 1:
            return jnp.tile(value_range, (3, 1))
        return value_range


    def init(self, internal_state):
        internal_state["seen_body_masses"] = self.default_body_mass[1:]
        internal_state["seen_body_inertias"] = self.default_body_inertia[1:]
        internal_state["seen_body_coms"] = self.default_body_ipos[1:]
        internal_state["seen_body_positions"] = self.env.initial_mjx_model.body_pos[1:]
        internal_state["seen_torque_limits"] = self.default_effort_limits
        internal_state["seen_joint_ranges"] = self.env.initial_mjx_model.jnt_range[1:]
        internal_state["seen_joint_dampings"] = self.default_dof_damping[6:]
        internal_state["seen_joint_armatures"] = self.default_dof_armature[6:]
        internal_state["seen_joint_stiffnesses"] = self.env.initial_mjx_model.jnt_stiffness[1:]
        internal_state["seen_joint_frictionlosses"] = self.default_dof_frictionloss[6:]
        internal_state["seen_p_gain"] = self.default_p_gain
        internal_state["seen_d_gain"] = self.default_d_gain
        internal_state["scaling_factor"] = self.default_scaling_factor
        internal_state["actuator_p_gains"] = self.default_p_gain
        internal_state["actuator_d_gains"] = self.default_d_gain
        internal_state["actuator_effort_limits"] = self.default_effort_limits
        internal_state["actuator_velocity_limits"] = self.default_velocity_limits
        internal_state["actuator_knee_point_velocities"] = self.default_knee_point_velocities
        internal_state["partial_actuator_gainprm_without_dropout"] = self.default_p_gain
        internal_state["partial_actuator_biasprm_without_dropout"] = self.env.initial_mjx_model.actuator_biasprm[:, 1:3]
        internal_state["robot_nominal_qpos_height_over_ground"] = self.env.initial_qpos[2]
        internal_state["robot_nominal_imu_height_over_ground"] = self.env.initial_imu_height
        internal_state["nr_collisions_in_nominal"] = 0

        internal_state["mass_inertia_noise_factors"] = jnp.ones(self.nr_bodies_without_world)
        internal_state["com_noise_factors"] = jnp.ones((self.nr_bodies_without_world, 3))
        internal_state["body_position_noise_factors"] = jnp.ones((self.nr_bodies_without_world, 3))
        internal_state["joint_damping_noise_factors"] = jnp.ones(self.nr_non_free_dofs)
        internal_state["joint_armature_noise_factors"] = jnp.ones(self.nr_non_free_dofs)
        internal_state["joint_stiffness_noise_factors"] = jnp.ones(self.nr_non_free_joints)
        internal_state["joint_friction_loss_noise_factors"] = jnp.ones(self.nr_non_free_dofs)
        internal_state["p_gain_noise_factors"] = jnp.ones(self.env.nr_actuator_joints)
        internal_state["d_gain_noise_factors"] = jnp.ones(self.env.nr_actuator_joints)
        internal_state["position_offsets"] = jnp.zeros(self.env.nr_actuator_joints)


    def sample(self, internal_state, mjx_model, data, should_randomize, key):
        keys = jax.random.split(key, 9)

        root_multiplier = self._lerp(self.root_body_mass_range, jax.random.uniform(keys[0]))
        other_multipliers = self._lerp(
            self.other_body_mass_range,
            jax.random.uniform(keys[1], shape=(self.other_body_ids.shape[0],)),
        )
        body_mass = self.default_body_mass
        if self.randomize_link_mass:
            body_mass = body_mass.at[self.root_body_id].set(self.default_body_mass[self.root_body_id] * root_multiplier)
            body_mass = body_mass.at[self.other_body_ids].set(self.default_body_mass[self.other_body_ids] * other_multipliers)

        com_displacement = self._lerp(self.com_displacement_range.T, jax.random.uniform(keys[2], shape=(3,)))
        body_ipos = self.default_body_ipos
        if self.randomize_com_displacement:
            body_ipos = body_ipos.at[self.root_body_id].set(self.default_body_ipos[self.root_body_id] + com_displacement)

        joint_damping = self.default_dof_damping[6:]
        if self.randomize_joint_damping:
            joint_damping = self._lerp(self.joint_damping_range, jax.random.uniform(keys[3], shape=joint_damping.shape))
        dof_damping = self.default_dof_damping.at[6:].set(joint_damping)

        joint_friction_loss = self.default_dof_frictionloss[6:]
        if self.randomize_joint_friction_loss:
            joint_friction_loss = self._lerp(self.joint_friction_loss_range, jax.random.uniform(keys[4], shape=joint_friction_loss.shape))
        dof_frictionloss = self.default_dof_frictionloss.at[6:].set(joint_friction_loss)

        joint_armature = self.default_dof_armature[6:]
        if self.randomize_joint_armature:
            joint_armature = self._lerp(self.joint_armature_range, jax.random.uniform(keys[5], shape=joint_armature.shape))
        dof_armature = self.default_dof_armature.at[6:].set(joint_armature)

        p_gain = self.default_p_gain
        if self.add_p_gains_noise:
            p_gain = self.default_p_gain + (
                jax.random.uniform(keys[6], shape=self.default_p_gain.shape, minval=-1.0, maxval=1.0) *
                self.p_gains_noise_scale * self.default_p_gain
            )
        d_gain = self.default_d_gain
        if self.add_d_gains_noise:
            d_gain = self.default_d_gain + (
                jax.random.uniform(keys[7], shape=self.default_d_gain.shape, minval=-1.0, maxval=1.0) *
                self.d_gains_noise_scale * self.default_d_gain
            )

        actuator_joint_nominal_positions = self.default_actuator_joint_nominal_positions
        if self.randomize_actuator_joint_nominal_position:
            actuator_joint_nominal_positions = actuator_joint_nominal_positions + jax.random.uniform(
                keys[8],
                shape=actuator_joint_nominal_positions.shape,
                minval=-self.add_actuator_joint_nominal_position,
                maxval=self.add_actuator_joint_nominal_position,
            )
            joint_ranges = mjx_model.jnt_range[self.env.actuator_joint_mask_joints]
            actuator_joint_nominal_positions = jnp.clip(
                actuator_joint_nominal_positions,
                joint_ranges[:, 0],
                joint_ranges[:, 1],
            )

        if self.env.use_torque_pd_control:
            actuator_gainprm = mjx_model.actuator_gainprm
            actuator_biasprm = mjx_model.actuator_biasprm
            actuator_forcerange = mjx_model.actuator_forcerange.at[:, 0].set(-self.default_effort_limits)
            actuator_forcerange = actuator_forcerange.at[:, 1].set(self.default_effort_limits)
        else:
            actuator_gainprm = mjx_model.actuator_gainprm.at[:, 0].set(p_gain)
            actuator_biasprm = mjx_model.actuator_biasprm.at[:, 1].set(-p_gain)
            actuator_biasprm = actuator_biasprm.at[:, 2].set(-d_gain)
            actuator_forcerange = mjx_model.actuator_forcerange

        new_mjx_model = mjx_model.tree_replace({
            "body_mass": body_mass,
            "body_ipos": body_ipos,
            "dof_damping": dof_damping,
            "dof_frictionloss": dof_frictionloss,
            "dof_armature": dof_armature,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
            "actuator_forcerange": actuator_forcerange,
        })
        mjx_model = jax.lax.cond(should_randomize, lambda _: new_mjx_model, lambda _: mjx_model, None)

        internal_state["seen_body_masses"] = jnp.where(should_randomize, body_mass[1:], internal_state["seen_body_masses"])
        internal_state["seen_body_coms"] = jnp.where(should_randomize, body_ipos[1:], internal_state["seen_body_coms"])
        internal_state["seen_joint_dampings"] = jnp.where(should_randomize, joint_damping, internal_state["seen_joint_dampings"])
        internal_state["seen_joint_armatures"] = jnp.where(should_randomize, joint_armature, internal_state["seen_joint_armatures"])
        internal_state["seen_joint_frictionlosses"] = jnp.where(should_randomize, joint_friction_loss, internal_state["seen_joint_frictionlosses"])
        internal_state["seen_p_gain"] = jnp.where(should_randomize, p_gain, internal_state["seen_p_gain"])
        internal_state["seen_d_gain"] = jnp.where(should_randomize, d_gain, internal_state["seen_d_gain"])
        internal_state["actuator_joint_nominal_positions"] = jnp.where(
            should_randomize,
            actuator_joint_nominal_positions,
            internal_state["actuator_joint_nominal_positions"],
        )
        internal_state["actuator_joint_max_velocities"] = jnp.where(
            should_randomize,
            self.default_actuator_joint_max_velocities,
            internal_state["actuator_joint_max_velocities"],
        )
        internal_state["scaling_factor"] = jnp.where(should_randomize, self.default_scaling_factor, internal_state["scaling_factor"])
        internal_state["actuator_p_gains"] = jnp.where(should_randomize, p_gain, internal_state["actuator_p_gains"])
        internal_state["actuator_d_gains"] = jnp.where(should_randomize, d_gain, internal_state["actuator_d_gains"])
        internal_state["actuator_effort_limits"] = jnp.where(
            should_randomize,
            self.default_effort_limits,
            internal_state["actuator_effort_limits"],
        )
        internal_state["actuator_velocity_limits"] = jnp.where(
            should_randomize,
            self.default_velocity_limits,
            internal_state["actuator_velocity_limits"],
        )
        internal_state["actuator_knee_point_velocities"] = jnp.where(
            should_randomize,
            self.default_knee_point_velocities,
            internal_state["actuator_knee_point_velocities"],
        )
        internal_state["partial_actuator_gainprm_without_dropout"] = jnp.where(
            should_randomize,
            p_gain,
            internal_state["partial_actuator_gainprm_without_dropout"],
        )
        internal_state["partial_actuator_biasprm_without_dropout"] = jnp.where(
            should_randomize,
            jnp.stack([-p_gain, -d_gain], axis=1),
            internal_state["partial_actuator_biasprm_without_dropout"],
        )

        return mjx_model, data
