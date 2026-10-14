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
    observation_indices = getattr(env, "history_observation_indices", None)
    if observation_indices is None:
        observation_indices = env.policy_observation_indices
    history_length = getattr(env, "nr_history_steps", 10)

    if encoder_type == 'FFNN':
        return FFNN_encoder(hidden_size=hidden_size, observation_indices=observation_indices, history_length=history_length)
    elif encoder_type == 'GRU':
        return GRU_encoder(hidden_size=hidden_size, observation_indices=observation_indices)
    elif encoder_type == 'scan_conv':
        observation_indices = np.asarray(observation_indices)
        scan_indices = np.asarray(getattr(env, "obstacle_scan_obs_idx", np.zeros(0, dtype=int)))
        scan_indices = scan_indices[np.isin(scan_indices, observation_indices)]
        if scan_indices.size == 0:
            raise ValueError("history_encoder_type='scan_conv' needs the obstacle range scan in the safety critic's observations")
        other_indices = observation_indices[~np.isin(observation_indices, scan_indices)]
        nr_sectors = getattr(config.algorithm.next_step_predictor, 'scan_encoder_sectors', 8)
        if scan_indices.size % nr_sectors != 0:
            raise ValueError(f"the number of scan rays ({scan_indices.size}) must be divisible by scan_encoder_sectors ({nr_sectors})")
        return ScanConvEncoder(hidden_size=hidden_size, observation_indices=tuple(other_indices.tolist()),
                               scan_indices=tuple(scan_indices.tolist()), history_length=history_length, nr_sectors=nr_sectors)
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
    observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        # x: (batch_size, history_length, observation_dim)
        x = x[..., self.observation_indices]
        gru = nn.RNN(
            nn.GRUCell(features=self.hidden_size),
            return_carry=True,     # return final carry (hidden state)
            time_major=False,      # input is (batch, time, features)
        )
        carry, _ = gru(x)
        return carry


class ScanConvEncoder(nn.Module):
    """
    History encoder for observations that contain a planar range scan.

    The scan part of the history is processed by a circular convolution over the rays with the
    frames as channels, so motion between frames is seen locally and the same filters apply in
    every direction. The result is max-pooled into nr_sectors angular sectors, which keeps the
    direction of an obstacle relative to the heading. The other observations go through an MLP over
    the flattened history, as in FFNN_encoder.
    Input shape: (..., history_length, observation_dim)
    Output shape: (..., hidden_size)
    """
    hidden_size: int
    observation_indices: Sequence[int]
    scan_indices: Sequence[int]
    history_length: int
    nr_sectors: int = 8
    channels: int = 32

    @nn.compact
    def __call__(self, x):
        batch_shape = x.shape[:-2]
        x = x.reshape((-1,) + x.shape[-2:])
        scan = jnp.swapaxes(x[..., jnp.asarray(self.scan_indices)], 1, 2)  # (batch, rays, history)
        scan = nn.elu(nn.Conv(self.channels, (5,), padding="CIRCULAR")(scan))
        scan = nn.elu(nn.Conv(self.channels, (5,), padding="CIRCULAR")(scan))
        rays_per_sector = len(self.scan_indices) // self.nr_sectors
        scan = scan.reshape(scan.shape[0], self.nr_sectors, rays_per_sector, self.channels).max(axis=2)
        scan = scan.reshape(scan.shape[0], -1)

        features = [scan]
        if len(self.observation_indices) > 0:
            other = x[..., jnp.asarray(self.observation_indices)].reshape(x.shape[0], -1)
            features.append(nn.elu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(other)))
        h = nn.elu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(jnp.concatenate(features, axis=-1)))
        z = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h)
        return z.reshape(batch_shape + (self.hidden_size,))
