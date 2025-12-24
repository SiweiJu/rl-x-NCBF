import numpy as np

class DualBuffer():
    """dual buffer to store positive and negative samples separately"""
    def __init__(self, pos_capacity, nr_env, neg_capacity, os_shape, as_shape, neg_sampling_ratio, rng):
        self.sampling_ratio = neg_sampling_ratio
        self.nr_env = nr_env
        self.pos_buffer = ReplayBuffer(pos_capacity, nr_env, os_shape, as_shape, rng)
        self.neg_buffer = ReplayBuffer(neg_capacity, nr_env, os_shape, as_shape, rng)

    def add(self, states, next_states, actions, rewards, terminations, masks, y_targets):
        assert states.ndim == 2, "States should only have batch dimension when adding a batch of transitions."

        pos_mask = y_targets == 1
        if np.any(pos_mask):
            idx = np.nonzero(pos_mask)[0]
            self.pos_buffer.add_batch(states[idx], next_states[idx], actions[idx],
                                      rewards[idx], terminations[idx], masks[idx], y_targets[idx])
        neg_mask = ~pos_mask
        if np.any(neg_mask):
            idx = np.nonzero(neg_mask)[0]
            self.neg_buffer.add_batch(states[idx], next_states[idx], actions[idx],
                                      rewards[idx], terminations[idx], masks[idx], y_targets[idx])

    def sample(self, nr_samples):
        nr_neg_samples = int(nr_samples * self.sampling_ratio)
        nr_pos_samples = nr_samples - nr_neg_samples

        pos_samples = self.pos_buffer.sample(nr_pos_samples)
        neg_samples = self.neg_buffer.sample(nr_neg_samples)

        states = np.concatenate([pos_samples[0], neg_samples[0]], axis=0)
        next_states = np.concatenate([pos_samples[1], neg_samples[1]], axis=0)
        actions = np.concatenate([pos_samples[2], neg_samples[2]], axis=0)
        rewards =  np.concatenate([pos_samples[3], neg_samples[3]], axis=0)
        terminations = np.concatenate([pos_samples[4], neg_samples[4]], axis=0)
        y_targets = np.concatenate([pos_samples[5], neg_samples[5]], axis=0)
        masks = np.concatenate([pos_samples[6], neg_samples[6]], axis=0)

        # shuffle the samples
        perm = np.random.permutation(nr_samples)
        states = states[perm]
        next_states = next_states[perm]
        actions = actions[perm]
        rewards = rewards[perm]
        terminations = terminations[perm]
        y_targets = y_targets[perm]
        masks = masks[perm]
        return states, next_states, actions, rewards, terminations, y_targets, masks


class ReplayBuffer():
    def __init__(self, capacity, nr_envs, os_shape, as_shape, rng):
        self.os_shape = os_shape
        self.as_shape = as_shape
        self.capacity = capacity // nr_envs
        self.nr_envs = nr_envs
        self.rng = rng
        self.states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.actions = np.zeros((self.capacity, nr_envs) + as_shape, dtype=np.float32)
        self.rewards = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.terminations = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.y_targets = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.masks = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, states, next_states, actions, rewards, terminations, masks, y_targets):
        self.states[self.pos] = states
        self.next_states[self.pos] = next_states
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.terminations[self.pos] = terminations
        self.y_targets[self.pos] = y_targets
        self.masks[self.pos] = masks
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, states, next_states, actions, rewards, terminations, masks, y_targets):
        assert states.ndim == 2, "States should only have batch dimension when adding a batch of transitions."

        batch_size = states.shape[0]

        # Case 1: Enough space from current position to end of buffer
        if self.pos + batch_size <= self.capacity:
            sl = slice(self.pos, self.pos + batch_size)
            self.states[sl] = states
            self.next_states[sl] = next_states
            self.actions[sl] = actions
            self.rewards[sl] = rewards
            self.terminations[sl] = terminations
            self.masks[sl] = masks
            self.y_targets[sl] = y_targets
        else:
            # Case 2: Must wrap around the ring buffer
            end_space = self.capacity - self.pos
            start_space = batch_size - end_space

            # Fill until end of buffer
            sl1 = slice(self.pos, self.capacity)
            self.states[sl1] = states[:end_space]
            self.next_states[sl1] = next_states[:end_space]
            self.actions[sl1] = actions[:end_space]
            self.rewards[sl1] = rewards[:end_space]
            self.terminations[sl1] = terminations[:end_space]
            self.masks[sl1] = masks[:end_space]
            self.y_targets[sl1] = y_targets[:end_space]

            # Wrap to the beginning
            sl2 = slice(0, start_space)
            self.states[sl2] = states[end_space:]
            self.next_states[sl2] = next_states[end_space:]
            self.actions[sl2] = actions[end_space:]
            self.rewards[sl2] = rewards[end_space:]
            self.terminations[sl2] = terminations[end_space:]
            self.masks[sl2] = masks[end_space:]
            self.y_targets[sl2] = y_targets[end_space:]

        # Update pointer and size
        self.pos = (self.pos + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, nr_samples):
        idx1 = self.rng.integers(self.size, size=nr_samples)
        idx2 = self.rng.integers(self.nr_envs, size=nr_samples)
        states = self.states[idx1, idx2].reshape((nr_samples,) + self.os_shape)
        next_states = self.next_states[idx1, idx2].reshape((nr_samples,) + self.os_shape)
        actions = self.actions[idx1, idx2].reshape((nr_samples,) + self.as_shape)
        rewards = self.rewards[idx1, idx2].reshape((nr_samples,))
        terminations = self.terminations[idx1, idx2].reshape((nr_samples,))
        y_targets = self.y_targets[idx1, idx2].reshape((nr_samples,))
        masks = self.masks[idx1, idx2].reshape((nr_samples,))
        return states, next_states, actions, rewards, terminations, y_targets, masks
