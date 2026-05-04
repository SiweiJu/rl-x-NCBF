from typing import Sequence
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))
    use_history_latent = getattr(config.algorithm, "use_history_latent_for_value", False)

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Critic(critic_observation_indices, config.algorithm.hidden_layers, use_history_latent)


class Critic(nn.Module):
    critic_observation_indices: Sequence[int]
    hidden_layers: Sequence[int]
    use_history_latent: bool = False

    @nn.compact
    def __call__(self, x, history_latent=None):
        x = x[..., self.critic_observation_indices]
        if self.use_history_latent:
            if history_latent is None:
                raise ValueError("Critic was configured with use_history_latent_for_value=True, but no history_latent was passed.")
            x = jnp.concatenate([x, history_latent], axis=-1)
        critic = x
        for hidden_units in self.hidden_layers:
            critic = nn.Dense(hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
            critic = nn.elu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(critic)
        return critic
