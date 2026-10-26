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
    use_decoder_output = getattr(config.algorithm, "use_decoder_output_for_policy", False)

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Critic(critic_observation_indices, config.algorithm.hidden_layers, use_history_latent, use_decoder_output,
                      bool(getattr(config.algorithm, "value_layer_norm", False)))


class Critic(nn.Module):
    critic_observation_indices: Sequence[int]
    hidden_layers: Sequence[int]
    use_history_latent: bool = False
    use_decoder_output: bool = False
    layer_norm: bool = False  # T06c: paper code (ncbf-q) had LayerNorm after the first hidden layer

    @nn.compact
    def __call__(self, x, history_latent=None, decoder_output=None):
        x = x[..., self.critic_observation_indices]
        if self.use_history_latent:
            if history_latent is None:
                raise ValueError("Critic was configured with use_history_latent_for_value=True, but no history_latent was passed.")
            x = jnp.concatenate([x, history_latent], axis=-1)
        if self.use_decoder_output:
            if decoder_output is None:
                raise ValueError("Critic was configured with use_decoder_output_for_policy=True, but no decoder_output was passed.")
            x = jnp.concatenate([x, decoder_output], axis=-1)
        critic = x
        for layer_index, hidden_units in enumerate(self.hidden_layers):
            critic = nn.Dense(hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
            if self.layer_norm and layer_index == 0:
                critic = nn.LayerNorm()(critic)
            critic = nn.elu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(critic)
        return critic
