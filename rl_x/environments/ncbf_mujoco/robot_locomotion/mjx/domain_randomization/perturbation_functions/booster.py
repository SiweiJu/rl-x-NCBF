import jax
import jax.numpy as jnp


class BoosterDRPerturbation:
    def __init__(self, env):
        self.env = env
        config = env.env_config["domain_randomization"]["perturbation"]
        self.kick_robots = config.get("kick_robots", True)
        self.kick_min_vel = config.get("kick_min_vel", 0.0)
        self.kick_max_vel = config.get("kick_max_vel", 0.4)


    def sample(self, internal_state, mjx_model, data, should_randomize, key):
        if not self.kick_robots:
            return data

        kick = jax.random.uniform(key, shape=(3,), minval=-1.0, maxval=1.0) * self.kick_max_vel
        kick = jnp.sign(kick) * jnp.clip(jnp.abs(kick), a_min=self.kick_min_vel, a_max=self.kick_max_vel)
        qvel = data.qvel.at[:3].add(jnp.where(should_randomize, kick, jnp.zeros(3)))
        return data.replace(qvel=qvel)
