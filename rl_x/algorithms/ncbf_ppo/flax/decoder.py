import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax.numpy as jnp

def get_decoder(config, env):
    """
    Factory function to get a decoder that reconstructs from latent representation.

    Args:
        config: Configuration object with decoder specifications
        env: Environment to get observation dimension

    Returns:
        A decoder instance (Feedforward_decoder)
    """
    history_encoder_hidden_size = getattr(config.algorithm.next_step_predictor, 'history_encoder_hidden_size', 128)
    prediction_indices = getattr(env, 'next_state_indices', None)
    decoder_output_dim = prediction_indices.shape[0]
    return Feedforward_decoder(hidden_size=history_encoder_hidden_size, output_dim=decoder_output_dim)


class Feedforward_decoder(nn.Module):
    """
    Feedforward decoder for reconstructing observations from latent representations.

    Takes a latent vector and reconstructs it back to the original observation space
    through multiple dense layers with activation functions.

    Input shape: (batch_size, hidden_size)
    Output shape: (batch_size, output_dim)
    """
    hidden_size: int
    output_dim: int

    @nn.compact
    def __call__(self, z, x, a):
        # z: (batch_size, hidden_size)
        # Expand latent through dense layers
        input = jnp.concatenate([z, a, x], axis=-1)  # Concatenate action if needed
        x = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(input)
        x = nn.elu(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.elu(x)
        # Final output layer to reach desired output dimension
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)

        return x

