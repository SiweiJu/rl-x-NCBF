import jax
import jax.numpy as jnp


class BoosterDRPerturbation:
    def __init__(self, env):
        self.env = env
        config = env.env_config["domain_randomization"]["perturbation"]
        self.kick_robots = config.get("kick_robots", True)
        self.kick_min_vel = config.get("kick_min_vel", 0.0)
        self.kick_max_vel = config.get("kick_max_vel", 0.4)
        velocity_range = config.get("velocity_range")
        self.velocity_range = None if velocity_range is None else jnp.array([
            velocity_range.get(axis, [0.0, 0.0])
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        ])


    def sample(self, internal_state, mjx_model, data, should_randomize, key):
        if not self.kick_robots:
            return data

        if self.velocity_range is not None:
            kick = jax.random.uniform(
                key,
                shape=(6,),
                minval=self.velocity_range[:, 0],
                maxval=self.velocity_range[:, 1],
            )
            qvel = data.qvel.at[:6].set(jnp.where(should_randomize, kick, data.qvel[:6]))
            return data.replace(qvel=qvel)

        kick = jax.random.uniform(key, shape=(3,), minval=-1.0, maxval=1.0) * self.kick_max_vel
        kick = jnp.sign(kick) * jnp.clip(jnp.abs(kick), a_min=self.kick_min_vel, a_max=self.kick_max_vel)
        qvel = data.qvel.at[:3].add(jnp.where(should_randomize, kick, jnp.zeros(3)))
        return data.replace(qvel=qvel)
