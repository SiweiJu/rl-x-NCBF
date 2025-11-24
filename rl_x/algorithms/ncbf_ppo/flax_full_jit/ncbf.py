from copy import deepcopy
from typing import Sequence, Callable, Optional
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


def get_ncbf(config, env):
    # TODO: make different types of ncbf
    ncbf_type = config.algorithm.ncbf.type
    use_safety_layer = config.algorithm.ncbf.use_safety_layer

    ncbf_observation_indices = getattr(env, "ncbf_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    NCBF = NCBF_FFNN(config.algorithm.nr_hidden_units, ncbf_observation_indices)
    if use_safety_layer:
        dynamics_step_function = get_dynamics_step_function(config)
        safety_layer_function = make_get_safe_action(NCBF.apply, dynamics_step_function, env.dynamics_observation_indices)
    else:
        # dummy safety layer that does nothing
        safety_layer_function = lambda x: x
    return (NCBF, safety_layer_function)


class NCBF_FFNN(nn.Module):
    nr_hidden_units: int
    observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x[..., self.observation_indices]
        # Two hidden layers
        x = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        x = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        # Scalar CBF output h(x)
        h = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        return jnp.squeeze(h, -1)  # shape ()


def make_get_safe_action(apply, dynamics_step_function, observation_indices):
    raise NotImplementedError