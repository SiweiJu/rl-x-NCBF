from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CommandState:
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0
    height: float = 0.67
    gait_frequency: float = 0.0
    gait_phase: float = 0.0


class T1ObservationBuilder:
    """Build the Booster policy observation used by this repository."""

    def __init__(self, cfg: dict):
        obs_cfg = cfg.get("observation", {})
        policy_cfg = cfg.get("policy", {})
        ball_cfg = obs_cfg.get("ball_plate", {})

        self.num_dof = int(cfg.get("robot", {}).get("num_dof", 23))
        self.clip_observation = float(obs_cfg.get("clip_observation", 10.0))
        self.joint_position_scale = float(obs_cfg.get("joint_position_scale", 1.0))
        self.joint_velocity_scale = float(obs_cfg.get("joint_velocity_scale", 0.1))
        self.previous_action_scale = float(obs_cfg.get("previous_action_scale", 1.0))
        self.angular_velocity_scale = float(obs_cfg.get("angular_velocity_scale", 1.0))
        self.gravity_scale = float(obs_cfg.get("gravity_scale", 1.0))

        self.include_ball_plate = bool(ball_cfg.get("enabled", False))
        self.ball_plate_observation_size = int(ball_cfg.get("observation_size", 12))
        self.include_ball_plate_drop_signal = bool(ball_cfg.get("include_drop_signal", False))

        self.expected_observation_size = int(policy_cfg.get("observation_size", 81))

    def build(
        self,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        previous_action: np.ndarray,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        command: CommandState,
        ball_plate_observation: np.ndarray | None = None,
        ball_plate_dropped: float = 0.0,
    ) -> np.ndarray:
        dof_pos = self._as_vector(dof_pos, self.num_dof, "dof_pos")
        dof_vel = self._as_vector(dof_vel, self.num_dof, "dof_vel")
        previous_action = self._as_vector(previous_action, self.num_dof, "previous_action")
        base_ang_vel = self._as_vector(base_ang_vel, 3, "base_ang_vel")
        projected_gravity = self._as_vector(projected_gravity, 3, "projected_gravity")

        command_obs = np.array([
            command.vx,
            command.vy,
            command.yaw,
            command.height,
            np.cos(2.0 * np.pi * command.gait_phase) * (command.gait_frequency > 1.0e-8),
            np.sin(2.0 * np.pi * command.gait_phase) * (command.gait_frequency > 1.0e-8),
        ], dtype=np.float32)

        pieces = [
            dof_pos * self.joint_position_scale,
            dof_vel * self.joint_velocity_scale,
            previous_action * self.previous_action_scale,
            base_ang_vel * self.angular_velocity_scale,
            command_obs,
            projected_gravity * self.gravity_scale,
        ]

        if self.include_ball_plate:
            if ball_plate_observation is None:
                ball_plate_observation = np.zeros(self.ball_plate_observation_size, dtype=np.float32)
            pieces.append(self._as_vector(
                ball_plate_observation,
                self.ball_plate_observation_size,
                "ball_plate_observation",
            ))
            if self.include_ball_plate_drop_signal:
                pieces.append(np.array([ball_plate_dropped], dtype=np.float32))

        observation = np.concatenate(pieces).astype(np.float32)
        if observation.shape[0] != self.expected_observation_size:
            raise ValueError(
                f"Built observation has size {observation.shape[0]}, "
                f"but policy.observation_size is {self.expected_observation_size}."
            )
        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(observation, -self.clip_observation, self.clip_observation)

    @staticmethod
    def _as_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape[0] != size:
            raise ValueError(f"{name} must have size {size}, got {array.shape[0]}.")
        return array

