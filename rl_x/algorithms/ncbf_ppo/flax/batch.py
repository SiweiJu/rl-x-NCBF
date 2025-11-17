class Batch:
    def __init__(self, states, next_states, actions, rewards, values, dones, terminations, log_probs, advantages, returns,
                 masks, y_targets):
        self.states = states
        self.next_states = next_states
        self.actions = actions
        self.rewards = rewards
        self.values = values
        self.terminations = terminations
        self.log_probs = log_probs
        self.advantages = advantages
        self.returns = returns
        self.dones = dones
        self.masks = masks
        self.y_targets = y_targets
