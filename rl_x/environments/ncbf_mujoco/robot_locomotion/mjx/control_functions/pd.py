import jax.numpy as jnp


class PDControl:
    def __init__(self, env, control_frequency_hz=50):
        self.env = env
        self.control_frequency_hz = control_frequency_hz


    def process_action(self, action, internal_state, data=None):
        scaled_action = action * internal_state["scaling_factor"]
        target_joint_positions = internal_state["actuator_joint_nominal_positions"] + scaled_action
        noisy_target_joint_positions = target_joint_positions + internal_state["position_offsets"]

        if not self.env.use_torque_pd_control:
            return noisy_target_joint_positions

        qpos = data.qpos[self.env.actuator_joint_mask_qpos]
        qvel = data.qvel[self.env.actuator_joint_mask_qvel]
        p_gains = internal_state["actuator_p_gains"]
        d_gains = internal_state["actuator_d_gains"]
        efforts = p_gains * (noisy_target_joint_positions - qpos) - d_gains * qvel
        max_efforts = self._speed_limited_efforts(qvel, internal_state)
        
        return jnp.clip(efforts, -max_efforts, max_efforts)


    def _speed_limited_efforts(self, qvel, internal_state):
        effort_limits = internal_state["actuator_effort_limits"]
        velocity_limits = internal_state["actuator_velocity_limits"]
        knee_point_velocities = jnp.minimum(
            internal_state["actuator_knee_point_velocities"],
            velocity_limits,
        )
        abs_velocities = jnp.abs(qvel)
        has_speed_limit = jnp.isfinite(velocity_limits) & (velocity_limits > 0.0)
        has_reduction_region = knee_point_velocities < velocity_limits
        slope_denominator = jnp.where(
            has_reduction_region,
            knee_point_velocities - velocity_limits,
            -1.0,
        )
        slopes = jnp.where(
            has_reduction_region,
            effort_limits / slope_denominator,
            0.0,
        )
        max_efforts = effort_limits + slopes * (abs_velocities - knee_point_velocities)
        max_efforts = jnp.clip(max_efforts, 0.0, effort_limits)
        max_efforts = jnp.where(has_speed_limit, max_efforts, effort_limits)
        max_efforts = jnp.where(velocity_limits <= 0.0, jnp.zeros_like(max_efforts), max_efforts)
        return max_efforts
