import numpy as np


class PDControl:
    def __init__(self, env, control_frequency_hz=50):
        self.env = env
        self.control_frequency_hz = control_frequency_hz


    def process_action(self, action):
        scaled_action = action * self.env.internal_state["scaling_factor"]
        target_joint_positions = self.env.internal_state["actuator_joint_nominal_positions"] + scaled_action
        noisy_target_joint_positions = target_joint_positions + self.env.internal_state["position_offsets"]

        if not self.env.use_torque_pd_control:
            return noisy_target_joint_positions

        qpos = self.env.internal_state["data"].qpos[self.env.actuator_joint_mask_qpos]
        qvel = self.env.internal_state["data"].qvel[self.env.actuator_joint_mask_qvel]
        p_gains = self.env.internal_state["actuator_p_gains"]
        d_gains = self.env.internal_state["actuator_d_gains"]
        efforts = p_gains * (noisy_target_joint_positions - qpos) - d_gains * qvel
        max_efforts = self._speed_limited_efforts(qvel)
        
        return np.clip(efforts, -max_efforts, max_efforts)


    def _speed_limited_efforts(self, qvel):
        effort_limits = self.env.internal_state["actuator_effort_limits"]
        velocity_limits = self.env.internal_state["actuator_velocity_limits"]
        knee_point_velocities = np.minimum(
            self.env.internal_state["actuator_knee_point_velocities"],
            velocity_limits,
        )
        abs_velocities = np.abs(qvel)
        has_speed_limit = np.isfinite(velocity_limits) & (velocity_limits > 0.0)
        has_reduction_region = knee_point_velocities < velocity_limits
        slope_denominator = np.where(
            has_reduction_region,
            knee_point_velocities - velocity_limits,
            -1.0,
        )
        slopes = np.where(
            has_reduction_region,
            effort_limits / slope_denominator,
            0.0,
        )
        max_efforts = effort_limits + slopes * (abs_velocities - knee_point_velocities)
        max_efforts = np.clip(max_efforts, 0.0, effort_limits)
        max_efforts = np.where(has_speed_limit, max_efforts, effort_limits)
        max_efforts = np.where(velocity_limits <= 0.0, np.zeros_like(max_efforts), max_efforts)
        return max_efforts
