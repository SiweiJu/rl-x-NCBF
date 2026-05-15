import jax.numpy as jnp


class NoneDRUnseenRobotFunction:
    def __init__(self, env):
        self.env = env


    def init(self, internal_state):
        nr_bodies = self.env.initial_mjx_model.body_mass[1:].shape[0]
        nr_joint_dampings = self.env.initial_mjx_model.dof_damping[6:].shape[0]
        nr_joint_armatures = self.env.initial_mjx_model.dof_armature[6:].shape[0]
        nr_joint_stiffnesses = self.env.initial_mjx_model.jnt_stiffness[1:].shape[0]
        nr_joint_frictionlosses = self.env.initial_mjx_model.dof_frictionloss[6:].shape[0]

        internal_state["mass_inertia_noise_factors"] = jnp.ones(nr_bodies)
        internal_state["com_noise_factors"] = jnp.ones((nr_bodies, 3))
        internal_state["body_position_noise_factors"] = jnp.ones((nr_bodies, 3))
        internal_state["joint_damping_noise_factors"] = jnp.ones(nr_joint_dampings)
        internal_state["joint_armature_noise_factors"] = jnp.ones(nr_joint_armatures)
        internal_state["joint_stiffness_noise_factors"] = jnp.ones(nr_joint_stiffnesses)
        internal_state["joint_friction_loss_noise_factors"] = jnp.ones(nr_joint_frictionlosses)
        internal_state["p_gain_noise_factors"] = jnp.ones(self.env.nr_actuator_joints)
        internal_state["d_gain_noise_factors"] = jnp.ones(self.env.nr_actuator_joints)
        internal_state["position_offsets"] = jnp.zeros(self.env.nr_actuator_joints)


    def sample(self, internal_state, should_randomize, key):
        return
