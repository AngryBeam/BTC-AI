import numpy as np
from logger import get_logger
import ray

logger, _ = get_logger()
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

    def _get_priority(self, error):
        return min((abs(error) + self.epsilon) ** self.alpha, self.max_priority)

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            # Use logarithmic scaling to handle large values
            log_a, log_b = np.log(max(1, a)), np.log(max(1, b))
            s = np.exp(np.random.uniform(log_a, log_b))

            idx, p, data = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return batch, idxs, is_weights

    def update(self, idx, error):
        priority = self._get_priority(error)
        self.tree.update(idx, priority)
        self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries

    def increase_beta(self):
        self.beta = min(1.0, self.beta + self.beta_increment)

@ray.remote
class RayCompatibleSumTree(SumTree):
    def __init__(self, capacity):
        super().__init__(capacity)

    def add(self, p, data):
        super().add(p, data)

    def update(self, idx, p):
        super().update(idx, p)

    def get(self, s):
        return super().get(s)

    def total(self):
        return super().total()

@ray.remote
class RayCompatiblePrioritizedReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.tree = RayCompatibleSumTree.remote(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

    def _get_priority(self, error):
        return min((abs(error) + self.epsilon) ** self.alpha, self.max_priority)

    def add(self, error, sample):
        p = self._get_priority(error)
        ray.get(self.tree.add.remote(p, sample))

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = ray.get(self.tree.total.remote()) / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            log_a, log_b = np.log(max(1, a)), np.log(max(1, b))
            s = np.exp(np.random.uniform(log_a, log_b))

            idx, p, data = ray.get(self.tree.get.remote(s))
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / ray.get(self.tree.total.remote())
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return batch, idxs, is_weights

    def update(self, idx, error):
        priority = self._get_priority(error)
        ray.get(self.tree.update.remote(idx, priority))
        self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return ray.get(self.tree.n_entries)

    def increase_beta(self):
        self.beta = min(1.0, self.beta + self.beta_increment)