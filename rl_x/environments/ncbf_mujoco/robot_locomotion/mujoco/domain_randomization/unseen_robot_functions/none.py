import numpy as np


class NoneDRUnseenRobotFunction:
    def __init__(self, env):
        self.env = env


    def init(self):
        nr_bodies = self.env.initial_mj_model.body_mass[1:].shape[0]
        nr_joint_dampings = self.env.initial_mj_model.dof_damping[6:].shape[0]
        nr_joint_armatures = self.env.initial_mj_model.dof_armature[6:].shape[0]
        nr_joint_stiffnesses = self.env.initial_mj_model.jnt_stiffness[1:].shape[0]
        nr_joint_frictionlosses = self.env.initial_mj_model.dof_frictionloss[6:].shape[0]

        self.env.internal_state["mass_inertia_noise_factors"] = np.ones(nr_bodies)
        self.env.internal_state["com_noise_factors"] = np.ones((nr_bodies, 3))
        self.env.internal_state["body_position_noise_factors"] = np.ones((nr_bodies, 3))
        self.env.internal_state["joint_damping_noise_factors"] = np.ones(nr_joint_dampings)
        self.env.internal_state["joint_armature_noise_factors"] = np.ones(nr_joint_armatures)
        self.env.internal_state["joint_stiffness_noise_factors"] = np.ones(nr_joint_stiffnesses)
        self.env.internal_state["joint_friction_loss_noise_factors"] = np.ones(nr_joint_frictionlosses)
        self.env.internal_state["p_gain_noise_factors"] = np.ones(self.env.nr_actuator_joints)
        self.env.internal_state["d_gain_noise_factors"] = np.ones(self.env.nr_actuator_joints)
        self.env.internal_state["position_offsets"] = np.zeros(self.env.nr_actuator_joints)


    def sample(self):
        return
