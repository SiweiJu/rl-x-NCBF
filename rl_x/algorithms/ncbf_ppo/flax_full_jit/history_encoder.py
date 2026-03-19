from typing import Sequence

import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax.numpy as jnp

def get_history_encoder(config, env):
    """
    Factory function to get the appropriate history encoder based on config.

    Args:
        config: Configuration object with encoder type specification
        env: Environment object

    Returns:
        An encoder instance (FFNN_encoder or GRU_encoder)
    """
    encoder_type = getattr(config.algorithm.next_step_predictor, 'history_encoder_type', 'FFNN')
    hidden_size = getattr(config.algorithm.next_step_predictor, 'history_encoder_hidden_size', 128)
    observation_indices = getattr(env, "policy_observation_indices")
    history_length = getattr(env, "nr_history_steps", 10)

    if encoder_type == 'FFNN':
        return FFNN_encoder(hidden_size=hidden_size, observation_indices=observation_indices, history_length=history_length)
    elif encoder_type == 'GRU':
        return GRU_encoder(hidden_size=hidden_size)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


class FFNN_encoder(nn.Module):
    """
    Feed-Forward Neural Network encoder for processing history stacks.

    Flattens the input history and processes it through multiple dense layers.
    Input shape: (batch_size, history_length, observation_dim)
    Output shape: (batch_size, hidden_size)
    """
    hidden_size: int
    observation_indices: Sequence[int]
    history_length: int

    @nn.compact
    def __call__(self, x):
        # x: (batch_size, history_length, observation_dim)
        # Apply observation indices to each timestep in the history
        x = x[..., self.observation_indices]  # (batch_size, history_length, filtered_observation_dim)
        # Flatten the history dimension

        D = len(self.observation_indices)

        x = x.reshape(*x.shape[:-2], self.history_length * D)  # (batch_size, history_length * filtered_observation_dim)

        # Process through dense layers
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.elu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.elu(x)
        z = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)

        return z


class GRU_encoder(nn.Module):
    """
    Gated Recurrent Unit (GRU) encoder for processing history stacks.

    Processes sequence of observations through a GRU cell to extract temporal features.
    Returns the final hidden state as the encoded representation.
    Input shape: (batch_size, history_length, observation_dim)
    Output shape: (batch_size, hidden_size)
    """
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        # x: (batch_size, history_length, observation_dim)
        gru = nn.RNN(
            nn.GRUCell(features=self.hidden_size),
            return_carry=True,     # return final carry (hidden state)
            time_major=False,      # input is (batch, time, features)
        )
        carry, _ = gru(x)
        return carry
