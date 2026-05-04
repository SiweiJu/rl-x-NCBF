from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))
    use_history_latent = getattr(config.algorithm, "use_history_latent_for_policy", False)

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (Policy(env.single_action_space.shape, config.algorithm.std_dev, policy_observation_indices, config.algorithm.hidden_layers, use_history_latent),
                get_processed_action_function(
                    config.algorithm.action_clipping_and_rescaling,
                    getattr(config.algorithm, "action_clip", 0.0),
                    jnp.array(env.single_action_space.low), jnp.array(env.single_action_space.high)
                ))


class Policy(nn.Module):
    as_shape: Sequence[int]
    std_dev: float
    policy_observation_indices: Sequence[int]
    hidden_layers: Sequence[int]
    use_history_latent: bool = False

    @nn.compact
    def __call__(self, x, history_latent=None):
        x = x[..., self.policy_observation_indices]
        if self.use_history_latent:
            if history_latent is None:
                raise ValueError("Policy was configured with use_history_latent_for_policy=True, but no history_latent was passed.")
            x = jnp.concatenate([x, history_latent], axis=-1)
        policy_mean = x
        for hidden_units in self.hidden_layers:
            policy_mean = nn.Dense(hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(policy_mean)
            policy_mean = nn.elu(policy_mean)
        policy_mean = nn.Dense(np.prod(self.as_shape).item(), kernel_init=orthogonal(0.01), bias_init=constant(0.0))(policy_mean)
        policy_logstd = self.param("policy_logstd", constant(jnp.log(self.std_dev)), (1, np.prod(self.as_shape).item()))
        return policy_mean, policy_logstd


def get_processed_action_function(action_clipping_and_rescaling, action_clip, env_as_low, env_as_high):
    if action_clipping_and_rescaling:
        def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
            clipped_action = jnp.clip(action, -1, 1)
            return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))
        return jax.jit(get_clipped_and_scaled_action)
    elif action_clip > 0.0:
        return jax.jit(lambda x: jnp.clip(x, -action_clip, action_clip))
    else:
        return jax.jit(lambda x: x)
