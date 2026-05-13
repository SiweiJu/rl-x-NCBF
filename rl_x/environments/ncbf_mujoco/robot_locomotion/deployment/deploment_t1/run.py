from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from observation import CommandState, T1ObservationBuilder
from policy import load_policy


class T1DeploymentController:
    def __init__(self, node, cfg: dict, config_dir: Path):
        from robot_bridge_py.robot_client import RobotClient

        self.node = node
        robot_cfg = cfg["robot"]
        self.robot = RobotClient(
            node=self.node,
            robot_type=robot_cfg.get("robot_type", "T1"),
            num_dof=int(robot_cfg.get("num_dof", 23)),
            control_frequency=float(robot_cfg.get("control_frequency", 50.0)),
            interpolation_order=float(robot_cfg.get("interpolation_order", 0.8)),
        )

        self.cfg = cfg
        self.policy = load_policy(cfg["policy"], config_dir=config_dir)
        self.observation_builder = T1ObservationBuilder(cfg)

        common_cfg = cfg["common"]
        self.default_qpos = self._array(common_cfg["default_qpos"], self.robot.num_dof, "default_qpos")
        self.kp = self._array(common_cfg["stiffness"], self.robot.num_dof, "stiffness")
        self.kd = self._array(common_cfg["damping"], self.robot.num_dof, "damping")
        self.joint_limit_low = self._array(common_cfg["joint_limit_low"], self.robot.num_dof, "joint_limit_low")
        self.joint_limit_high = self._array(common_cfg["joint_limit_high"], self.robot.num_dof, "joint_limit_high")

        policy_cfg = cfg["policy"]
        command_cfg = cfg["command"]
        safety_cfg = cfg["safety"]

        self.action_scale = float(policy_cfg.get("action_scale", 1.0))
        self.max_x_velocity = float(command_cfg.get("max_x_velocity", 0.8))
        self.max_y_velocity = float(command_cfg.get("max_y_velocity", 0.5))
        self.max_yaw_velocity = float(command_cfg.get("max_yaw_velocity", 1.0))
        self.velocity_step = float(command_cfg.get("velocity_step", 0.1))
        self.yaw_step = float(command_cfg.get("yaw_step", 0.1))
        self.zero_clip_threshold = float(command_cfg.get("zero_clip_threshold", 0.02))
        self.standing_height = float(command_cfg.get("standing_height", 0.67))
        self.moving_gait_frequency = float(command_cfg.get("gait_frequency", 1.25))

        self.max_abs_joint_velocity = float(safety_cfg.get("max_abs_joint_velocity", 25.0))
        self.max_target_delta_per_step = float(safety_cfg.get("max_target_delta_per_step", 0.35))
        self.clip_target_to_joint_limits = bool(safety_cfg.get("clip_target_to_joint_limits", True))

        self.agent_started = False
        self.command = CommandState(height=self.standing_height)
        self.previous_action = np.zeros(self.robot.num_dof, dtype=np.float32)
        self.last_target = self.default_qpos.copy()
        self.timer = self.node.create_timer(1.0 / float(robot_cfg.get("control_frequency", 50.0)), self.step)

        self.node.get_logger().info(
            "T1 deployment ready. Use LT+START for bridge control, LT+A for policy, "
            "BACK to stop, d-pad for vx/vy, right stick x for yaw."
        )

    def step(self) -> None:
        self.check_state()
        if self.robot.control_started and self.agent_started:
            self.policy_step()

    def policy_step(self) -> None:
        if np.max(np.abs(self.robot.q_vel)) > self.max_abs_joint_velocity:
            self.node.get_logger().error("Joint velocity safety limit exceeded; stopping policy.")
            self.agent_started = False
            return

        self.command.gait_phase = np.fmod(
            self.command.gait_phase + (1.0 / float(self.cfg["robot"].get("control_frequency", 50.0))) * self.command.gait_frequency,
            1.0,
        )

        projected_gravity = self._quat_to_projected_gravity(
            np.asarray(self.robot.quat, dtype=np.float32),
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        observation = self.observation_builder.build(
            dof_pos=np.asarray(self.robot.q_pos, dtype=np.float32),
            dof_vel=np.asarray(self.robot.q_vel, dtype=np.float32),
            previous_action=self.previous_action,
            base_ang_vel=np.asarray(self.robot.angular_velocity, dtype=np.float32),
            projected_gravity=projected_gravity,
            command=self.command,
        )
        policy_output = self.policy.act(observation)
        self.previous_action[:] = policy_output.action

        target = self.default_qpos + self.action_scale * policy_output.action
        if self.clip_target_to_joint_limits:
            target = np.clip(target, self.joint_limit_low, self.joint_limit_high)
        if self.max_target_delta_per_step > 0.0:
            delta = np.clip(target - self.last_target, -self.max_target_delta_per_step, self.max_target_delta_per_step)
            target = self.last_target + delta
        self.last_target[:] = target

        self.robot.send_cmd(q_target_pos=target, target_kp=self.kp, target_kd=self.kd)

    def check_state(self) -> None:
        self.robot.update_robot_state()
        joy_key = self.robot.joy_key
        if joy_key is None:
            if not self.robot.control_started:
                self.agent_started = False
            return

        if getattr(joy_key, "lt", False) and getattr(joy_key, "a", False) and self.robot.key_count == 2:
            if self.robot.control_started:
                self._start_policy()
            else:
                self.node.get_logger().warn("Start bridge control first with LT+START.")

        if getattr(joy_key, "back", False) and self.robot.key_count == 1:
            self._stop_policy()

        if self.agent_started:
            if getattr(joy_key, "hat_u", False) and self.robot.key_count == 1:
                self.command.vx += self.velocity_step
            elif getattr(joy_key, "hat_d", False) and self.robot.key_count == 1:
                self.command.vx -= self.velocity_step
            elif getattr(joy_key, "hat_l", False) and self.robot.key_count == 1:
                self.command.vy += self.velocity_step
            elif getattr(joy_key, "hat_r", False) and self.robot.key_count == 1:
                self.command.vy -= self.velocity_step
            elif getattr(joy_key, "rx", 0.0) * -1 >= 1.0:
                self.command.yaw += self.yaw_step
            elif getattr(joy_key, "rx", 0.0) * -1 <= -1.0:
                self.command.yaw -= self.yaw_step
            if (getattr(joy_key, "ls", False) or getattr(joy_key, "rs", False)) and self.robot.key_count == 1:
                self.command.vx = 0.0
                self.command.vy = 0.0
                self.command.yaw = 0.0

            self.command.vx = float(np.clip(self.command.vx, -self.max_x_velocity, self.max_x_velocity))
            self.command.vy = float(np.clip(self.command.vy, -self.max_y_velocity, self.max_y_velocity))
            self.command.yaw = float(np.clip(self.command.yaw, -self.max_yaw_velocity, self.max_yaw_velocity))
            self._update_gait_frequency()
            self.node.get_logger().info(
                f"cmd vx={self.command.vx:.2f}, vy={self.command.vy:.2f}, yaw={self.command.yaw:.2f}, "
                f"gait={self.command.gait_frequency:.2f}"
            )

        if not self.robot.control_started:
            self.agent_started = False

        self.robot.joy_key = None
        self.robot.key_count = 0

    def _start_policy(self) -> None:
        self.agent_started = True
        self.command = CommandState(height=self.standing_height)
        self.previous_action[:] = 0.0
        self.last_target[:] = np.asarray(self.robot.q_pos, dtype=np.float32)
        projected_gravity = self._quat_to_projected_gravity(
            np.asarray(self.robot.quat, dtype=np.float32),
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        initial_observation = self.observation_builder.build(
            dof_pos=np.asarray(self.robot.q_pos, dtype=np.float32),
            dof_vel=np.asarray(self.robot.q_vel, dtype=np.float32),
            previous_action=self.previous_action,
            base_ang_vel=np.asarray(self.robot.angular_velocity, dtype=np.float32),
            projected_gravity=projected_gravity,
            command=self.command,
        )
        self.policy.reset(initial_observation)
        self.node.get_logger().info("Policy started.")

    def _stop_policy(self) -> None:
        self.agent_started = False
        self.command = CommandState(height=self.standing_height)
        self.previous_action[:] = 0.0
        self.node.get_logger().info("Policy stopped.")

    def _update_gait_frequency(self) -> None:
        commands = np.array([self.command.vx, self.command.vy, self.command.yaw], dtype=np.float32)
        commands[np.abs(commands) < self.zero_clip_threshold] = 0.0
        self.command.vx, self.command.vy, self.command.yaw = [float(x) for x in commands]
        self.command.gait_frequency = self.moving_gait_frequency if np.linalg.norm(commands) > 0.0 else 0.0

    @staticmethod
    def _quat_to_projected_gravity(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
        quat_wxyz = quat_wxyz / max(np.linalg.norm(quat_wxyz), 1.0e-8)
        w, x, y, z = quat_wxyz
        rot_mat = np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ], dtype=np.float32)
        return rot_mat.T @ np.asarray(vector, dtype=np.float32)

    @staticmethod
    def _array(values, size: int, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.shape[0] != size:
            raise ValueError(f"{name} must have size {size}, got {array.shape[0]}.")
        return array


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CURRENT_DIR / "config.yaml")
    args = parser.parse_args()

    cfg_path = args.config.expanduser().resolve()
    cfg = load_config(cfg_path)

    import rclpy

    rclpy.init()
    node = rclpy.create_node("rlx_t1_deployment")
    controller = T1DeploymentController(node, cfg, cfg_path.parent)
    try:
        rclpy.spin(node)
    finally:
        controller._stop_policy()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
