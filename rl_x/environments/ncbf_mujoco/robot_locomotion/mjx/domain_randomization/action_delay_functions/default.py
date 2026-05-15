import jax
import jax.numpy as jnp


class DefaultActionDelay:
    def __init__(self, env):
        self.env = env

        config = env.env_config["domain_randomization"]["action_delay"]
        self.max_nr_delay_steps = config["max_nr_delay_steps"]
        self.min_nr_delay_steps = config.get("min_nr_delay_steps", 0)
        self.mixed_chance = config["mixed_chance"]
        self.use_curriculum = config.get("use_curriculum", True)

        self.current_mixed = False
        self.current_nr_delay_steps = 0


    def init(self, internal_state):
        internal_state["action_current_mixed"] = False
        internal_state["action_current_nr_delay_steps"] = 0


    def setup(self, internal_state):
        internal_state["action_history"] = jnp.zeros((self.max_nr_delay_steps + 1, self.env.nr_actuator_joints))


    def sample(self, internal_state, should_randomize, key):
        mixed_key, delay_key = jax.random.split(key)
        internal_state["action_current_mixed"] = jnp.where(should_randomize, jax.random.uniform(mixed_key) < self.mixed_chance, internal_state["action_current_mixed"])
        internal_state["action_current_nr_delay_steps"] = jnp.where(
            should_randomize,
            jax.random.randint(delay_key, (), self.min_nr_delay_steps, self.max_nr_delay_steps + 1),
            internal_state["action_current_nr_delay_steps"],
        )


    def delay_action(self, action, internal_state, key):
        delay_curriculum_scale = internal_state["env_curriculum_coeff"] if self.use_curriculum else 1.0
        current_nr_delay_steps = jnp.ceil(jnp.where(
            internal_state["action_current_mixed"],
            jax.random.randint(key, (1,), 0, self.max_nr_delay_steps+1)[0],
            internal_state["action_current_nr_delay_steps"]
        ) * delay_curriculum_scale).astype(jnp.int32)

        internal_state["action_history"] = jnp.roll(internal_state["action_history"], -1, axis=0)
        internal_state["action_history"] = internal_state["action_history"].at[-1].set(action)

        chosen_action = internal_state["action_history"][-1-current_nr_delay_steps]

        return chosen_action
