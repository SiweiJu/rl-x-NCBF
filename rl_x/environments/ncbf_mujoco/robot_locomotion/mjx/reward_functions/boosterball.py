import jax.numpy as jnp

from rl_x.environments.ncbf_mujoco.robot_locomotion.mjx.reward_functions.booster import BoosterReward


class BoosterBallReward(BoosterReward):
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


    def ball_plate_reward_and_info(self, data, internal_state, info):
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
            on_plate = metrics["on_plate"].astype(jnp.float32) * task_active
            dropped = metrics["dropped"].astype(jnp.float32)

            centering_reward = task_active * self.ball_plate_centering_coeff * -centering_norm
            center_bonus_reward = self.ball_plate_center_bonus_coeff * jnp.exp(
                -centering_norm / jnp.maximum(self.ball_plate_center_bonus_temperature, 1e-6)
            ) * on_plate
            velocity_reward = task_active * self.ball_plate_velocity_coeff * -velocity_norm
            on_plate_reward = self.ball_plate_on_plate_coeff * on_plate
            alive_reward = self.ball_plate_alive_coeff * task_active
            drop_penalty_reward = self.ball_plate_drop_penalty_coeff * -drop_penalty_scale
            ball_dropped_info = metrics["ball_dropped"].astype(jnp.float32)
            plate_dropped_info = metrics["plate_dropped"].astype(jnp.float32)
        else:
            metrics = {"radial_distance": 0.0, "ball_dropped": False, "plate_dropped": False, "dropped": False}
            centering_reward = 0.0
            center_bonus_reward = 0.0
            velocity_reward = 0.0
            on_plate_reward = 0.0
            alive_reward = 0.0
            drop_penalty_reward = 0.0
            on_plate = 0.0
            ball_dropped_info = 0.0
            plate_dropped_info = 0.0
            dropped = 0.0
            newly_dropped = False

        internal_state["ball_plate_ball_dropped"] = metrics["ball_dropped"]
        internal_state["ball_plate_plate_dropped"] = metrics["plate_dropped"]
        internal_state["ball_plate_dropped"] = metrics["dropped"]
        internal_state["ball_plate_newly_dropped"] = newly_dropped
        ball_plate_reward = (
            alive_reward + on_plate_reward + center_bonus_reward +
            centering_reward + velocity_reward + drop_penalty_reward
        )

        info["reward/ball_plate_centering"] = centering_reward
        info["reward/ball_plate_center_bonus"] = center_bonus_reward
        info["reward/ball_plate_velocity"] = velocity_reward
        info["reward/ball_plate_on_plate"] = on_plate_reward
        info["reward/ball_plate_alive"] = alive_reward
        info["reward/ball_plate_drop_penalty"] = drop_penalty_reward
        info["reward/ball_plate_total"] = ball_plate_reward
        info["env_info/ball_plate_radial_distance"] = metrics["radial_distance"]
        info["env_info/ball_plate_on_plate"] = on_plate
        info["env_info/ball_plate_ball_dropped"] = ball_dropped_info
        info["env_info/ball_plate_plate_dropped"] = plate_dropped_info
        info["env_info/ball_plate_dropped"] = dropped
        return ball_plate_reward


    def reward_and_info(self, data, mjx_model, internal_state, action, info):
        reward = super().reward_and_info(data, mjx_model, internal_state, action, info)
        ball_plate_reward = self.ball_plate_reward_and_info(data, internal_state, info)
        reward = jnp.nan_to_num(reward + ball_plate_reward, nan=0.0, posinf=0.0, neginf=0.0)
        info["reward/total"] = reward
        return reward
