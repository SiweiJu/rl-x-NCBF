class Batch:
    def __init__(self, states, next_states, actions, rewards, values, dones, terminations, log_probs, advantages, returns,
                 masks, y_targets, constraint_violated, delta_u, history_stacks, last_states, last_actions, env_actions=None):
        self.states = states
        self.next_states = next_states
        self.actions = actions
        self.env_actions = actions if env_actions is None else env_actions
        self.rewards = rewards
        self.values = values
        self.terminations = terminations
        self.log_probs = log_probs
        self.advantages = advantages
        self.returns = returns
        self.dones = dones
        self.masks = masks
        self.y_targets = y_targets
        self.constraint_violated = constraint_violated
        self.delta_u = delta_u
        self.history_stacks = history_stacks
        self.last_states = last_states
        self.last_actions = last_actions
