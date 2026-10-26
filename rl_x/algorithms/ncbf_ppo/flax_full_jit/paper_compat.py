"""T06c: switches for the PPO-side differences between the paper code (ncbf-q @ cacffe7) and pncbf-v2.

Each switch defaults to the pncbf-v2 behavior; the paper value is given per switch.
  algorithm.policy_layer_norm   False | paper True: LayerNorm after the first hidden layer of the policy (policy.py)
  algorithm.value_layer_norm    False | paper True: same for the value network (critic.py)
  algorithm.gae_stop_at_truncation  True | paper False: the GAE recursion also stops at truncations (1 - dones);
                                the paper stopped only at terminations (1 - terminations), so the advantage of the
                                last steps before a time-out leaked in the first steps of the next episode
  algorithm.policy_logstd_clip  True | paper False: clip the policy log-std to [-5, 2]
  algorithm.ncbf.anticipation_style  "mean_action" | paper "sampled_action": the anticipation (behavior cloning) loss
                                shields the sampled rollout action instead of the policy mean and averages over the
                                action dimensions instead of summing. With the shield off (PPO), the paper loss is
                                0.5 * mean((mu - a_sampled)^2), a weak pull of the mean toward the rollout actions;
                                v2's is identically 0.
"""
import jax
import jax.numpy as jnp

ANTICIPATION_STYLES = ("mean_action", "sampled_action")
LOGSTD_MIN, LOGSTD_MAX = -5.0, 2.0

# Values that make a shield-off pncbf-v2 PPO run as close to the paper code as v2 can get (bisect variant "b").
PAPER_PPO_SWITCHES = {
    "algorithm.policy_layer_norm": True,
    "algorithm.value_layer_norm": True,
    "algorithm.gae_stop_at_truncation": False,
    "algorithm.policy_logstd_clip": False,
    "algorithm.ncbf.anticipation_style": "sampled_action",
}


def maybe_clip_logstd(logstd, enabled):
    return jnp.clip(logstd, LOGSTD_MIN, LOGSTD_MAX) if enabled else logstd


def gae_advantages(delta, terminations, dones, gamma, gae_lambda, stop_at_truncation=True):
    """GAE over a [T, ...] rollout from the TD errors delta (which already bootstrap with 1 - terminations).

    stop_at_truncation=True (v2): the recursion is cut at every episode end (1 - dones).
    stop_at_truncation=False (paper code): it is cut only at terminations (1 - terminations).
    Returns the advantages [T, ...]."""
    cut = dones if stop_at_truncation else terminations
    not_cut = 1.0 - cut.astype(delta.dtype)

    def step(next_advantage, inputs):
        delta_t, not_cut_t = inputs
        advantage = delta_t + gamma * gae_lambda * not_cut_t * next_advantage
        return advantage, advantage

    _, advantages = jax.lax.scan(step, delta[-1], (delta[:-1], not_cut[:-1]), reverse=True)
    return jnp.concatenate([advantages, delta[-1:]], axis=0)


def anticipation_squared_error(action_mean, safe_action, style):
    """Per-sample anticipation error: sum over action dims (v2) or mean over them (paper)."""
    squared = jnp.square(action_mean - safe_action)
    return jnp.mean(squared, axis=-1) if style == "sampled_action" else jnp.sum(squared, axis=-1)
