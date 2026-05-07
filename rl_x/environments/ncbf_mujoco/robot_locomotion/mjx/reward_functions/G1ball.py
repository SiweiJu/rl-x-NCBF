import jax.numpy as jnp

from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.reward_functions.defaultG1 import DefaultG1Reward


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


    def extra_reward_terms(self, data, mjx_model, internal_state, action, info):
        if self.env.use_ball_plate:
            metrics = self.env.get_ball_plate_metrics(data, internal_state)
            if self.env.stand_after_ball_plate_drop:
                was_dropped = jnp.asarray(internal_state["ball_plate_drop_latched"])
                dropped_now = metrics["dropped"]
                newly_dropped = dropped_now & (~was_dropped)
                task_active = (~(was_dropped | dropped_now)).astype(jnp.float32)
                drop_penalty_scale = newly_dropped.astype(jnp.float32)
            else:
                newly_dropped = metrics["dropped"]
                task_active = jnp.asarray(1.0, dtype=jnp.float32)
                drop_penalty_scale = metrics["dropped"].astype(jnp.float32)

            centering_norm = jnp.sum(jnp.square(metrics["relative_position"][:2]))
            velocity_norm = jnp.sum(jnp.square(metrics["relative_velocity"][:2]))
            centering_reward = task_active * self.ball_plate_centering_coeff * -centering_norm
            velocity_reward = task_active * self.ball_plate_velocity_coeff * -velocity_norm
            on_plate_reward = task_active * self.ball_plate_on_plate_coeff * metrics["on_plate"].astype(jnp.float32)
            center_bonus_reward = self.ball_plate_center_bonus_coeff * jnp.exp(
                -centering_norm / jnp.maximum(self.ball_plate_center_bonus_temperature, 1e-6)
            ) * metrics["on_plate"].astype(jnp.float32) * task_active
            alive_reward = self.ball_plate_alive_coeff * task_active
            drop_penalty_reward = self.ball_plate_drop_penalty_coeff * -drop_penalty_scale
            on_plate_info = metrics["on_plate"].astype(jnp.float32)
            ball_dropped_info = metrics["ball_dropped"].astype(jnp.float32)
            plate_dropped_info = metrics["plate_dropped"].astype(jnp.float32)
            dropped_info = metrics["dropped"].astype(jnp.float32)
        else:
            centering_reward = 0.0
            velocity_reward = 0.0
            on_plate_reward = 0.0
            center_bonus_reward = 0.0
            alive_reward = 0.0
            drop_penalty_reward = 0.0
            metrics = {"radial_distance": 0.0, "on_plate": False, "ball_dropped": False, "plate_dropped": False, "dropped": False}
            newly_dropped = False
            on_plate_info = 0.0
            ball_dropped_info = 0.0
            plate_dropped_info = 0.0
            dropped_info = 0.0

        internal_state["ball_plate_ball_dropped"] = metrics["ball_dropped"]
        internal_state["ball_plate_plate_dropped"] = metrics["plate_dropped"]
        internal_state["ball_plate_dropped"] = metrics["dropped"]
        internal_state["ball_plate_newly_dropped"] = newly_dropped
        info[f"reward/ball_plate_centering"] = centering_reward
        info[f"reward/ball_plate_center_bonus"] = center_bonus_reward
        info[f"reward/ball_plate_velocity"] = velocity_reward
        info[f"reward/ball_plate_on_plate"] = on_plate_reward
        info[f"reward/ball_plate_alive"] = alive_reward
        info[f"reward/ball_plate_drop_penalty"] = drop_penalty_reward
        info[f"env_info/ball_plate_radial_distance"] = metrics["radial_distance"]
        info[f"env_info/ball_plate_on_plate"] = on_plate_info
        info[f"env_info/ball_plate_ball_dropped"] = ball_dropped_info
        info[f"env_info/ball_plate_plate_dropped"] = plate_dropped_info
        info[f"env_info/ball_plate_dropped"] = dropped_info

        penalty = centering_reward + velocity_reward + drop_penalty_reward
        return alive_reward, on_plate_reward + center_bonus_reward, penalty
