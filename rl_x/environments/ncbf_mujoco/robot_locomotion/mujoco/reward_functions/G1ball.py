import numpy as np

from rl_x.environments.ncbf_mujoco.robot_locomotion.mujoco.reward_functions.defaultG1 import DefaultG1Reward


class G1BallReward(DefaultG1Reward):
    def __init__(self, env):
        super().__init__(env)

        reward_config = env.env_config["reward"]
        self.ball_plate_centering_coeff = reward_config.get("ball_plate_centering_coeff", 0.0) * env.dt
        self.ball_plate_center_bonus_coeff = reward_config.get("ball_plate_center_bonus_coeff", 0.0) * env.dt
        self.ball_plate_center_bonus_temperature = reward_config.get("ball_plate_center_bonus_temperature", 0.01)
        self.ball_plate_velocity_coeff = reward_config.get("ball_plate_velocity_coeff", 0.0) * env.dt
        self.ball_plate_on_plate_coeff = reward_config.get("ball_plate_on_plate_coeff", 0.0) * env.dt
        self.ball_plate_alive_coeff = reward_config.get("ball_plate_alive_coeff", 0.0) * env.dt
        self.ball_plate_drop_penalty_coeff = reward_config.get("ball_plate_drop_penalty_coeff", 0.0) * env.dt


    def extra_reward_terms(self, action):
        if self.env.use_ball_plate:
            metrics = self.env.get_ball_plate_metrics()
            if self.env.stand_after_ball_plate_drop:
                was_dropped = bool(self.env.internal_state["ball_plate_drop_latched"])
                dropped_now = bool(metrics["dropped"])
                newly_dropped = dropped_now and not was_dropped
                task_active = 0.0 if (was_dropped or dropped_now) else 1.0
                drop_penalty_scale = float(newly_dropped)
            else:
                newly_dropped = bool(metrics["dropped"])
                task_active = 1.0
                drop_penalty_scale = float(metrics["dropped"])

            centering_norm = np.sum(np.square(metrics["relative_position"][:2]))
            velocity_norm = np.sum(np.square(metrics["relative_velocity"][:2]))
            centering_reward = task_active * self.ball_plate_centering_coeff * -centering_norm
            velocity_reward = task_active * self.ball_plate_velocity_coeff * -velocity_norm
            on_plate_reward = task_active * self.ball_plate_on_plate_coeff * float(metrics["on_plate"])
            center_bonus_reward = self.ball_plate_center_bonus_coeff * np.exp(
                -centering_norm / max(self.ball_plate_center_bonus_temperature, 1e-6)
            ) * float(metrics["on_plate"]) * task_active
            alive_reward = self.ball_plate_alive_coeff * task_active
            drop_penalty_reward = self.ball_plate_drop_penalty_coeff * -drop_penalty_scale
        else:
            centering_reward = 0.0
            velocity_reward = 0.0
            on_plate_reward = 0.0
            center_bonus_reward = 0.0
            alive_reward = 0.0
            drop_penalty_reward = 0.0
            metrics = {"radial_distance": 0.0, "on_plate": False, "ball_dropped": False, "plate_dropped": False, "dropped": False}
            newly_dropped = False

        self.env.internal_state["ball_plate_ball_dropped"] = metrics["ball_dropped"]
        self.env.internal_state["ball_plate_plate_dropped"] = metrics["plate_dropped"]
        self.env.internal_state["ball_plate_dropped"] = metrics["dropped"]
        self.env.internal_state["ball_plate_newly_dropped"] = newly_dropped
        self.env.internal_state["info"][f"reward/ball_plate_centering"] = centering_reward
        self.env.internal_state["info"][f"reward/ball_plate_center_bonus"] = center_bonus_reward
        self.env.internal_state["info"][f"reward/ball_plate_velocity"] = velocity_reward
        self.env.internal_state["info"][f"reward/ball_plate_on_plate"] = on_plate_reward
        self.env.internal_state["info"][f"reward/ball_plate_alive"] = alive_reward
        self.env.internal_state["info"][f"reward/ball_plate_drop_penalty"] = drop_penalty_reward
        self.env.internal_state["info"][f"env_info/ball_plate_radial_distance"] = metrics["radial_distance"]
        self.env.internal_state["info"][f"env_info/ball_plate_on_plate"] = float(metrics["on_plate"])
        self.env.internal_state["info"][f"env_info/ball_plate_ball_dropped"] = float(metrics["ball_dropped"])
        self.env.internal_state["info"][f"env_info/ball_plate_plate_dropped"] = float(metrics["plate_dropped"])
        self.env.internal_state["info"][f"env_info/ball_plate_dropped"] = float(metrics["dropped"])

        penalty = centering_reward + velocity_reward + drop_penalty_reward
        return alive_reward, on_plate_reward + center_bonus_reward, penalty
