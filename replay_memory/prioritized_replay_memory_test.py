import numpy as np
from prioritized_replay_memory import PrioritizedReplayMemory

# Size test.
memory = PrioritizedReplayMemory(8)

for i in range(8):
  assert memory.size() == i
  memory.add(i)
assert memory.size() == 8
memory.add(8)
assert memory.size() == 8

# Add test.
memory = PrioritizedReplayMemory(8)

for i in range(8):
  assert memory.add(i) == i + 7
assert memory.add(8) == 7

# Prio test.
memory = PrioritizedReplayMemory(8)

assert memory.get_max_prio() == 1

for i in range(8):
  memory.add(i)
for i in range(8):
  assert memory.get_prio(7 + i) == 1

assert memory.get_max_prio() == 1

memory.update(7, 1.2)
memory.update(8, 0.4)

assert memory.get_max_prio() == (1.2 + memory.epsilon) ** memory.alpha
assert memory.get_min_prio() == (0.4 + memory.epsilon) ** memory.alpha

# Random sample test.
memory = PrioritizedReplayMemory(8, 0, 1, .5)

for i in range(8):
  memory.add(i)

for i in range(8):
  memory.update(7 + i, i + 1)

np.random.seed(0)
batch = memory.get_random_sample(4)

# Indices.
assert batch[0][0] == 12
assert batch[1][0] == 13
assert batch[2][0] == 13
assert batch[3][0] == 12

# Priorities.
assert batch[0][1] == 6.0
assert batch[1][1] == 7.0
assert batch[2][1] == 7.0
assert batch[3][1] == 6.0

# IS weights.
max_weight = (1.0 / 28.0 * 8) ** -.5 # (lowest prio) / (sum of prios) * (total samples) ^ -beta

assert abs(batch[0][2] - (6.0 / 28.0 * 8) ** -.5 / max_weight) < .0001
assert abs(batch[1][2] - (7.0 / 28.0 * 8) ** -.5 / max_weight) < .0001
assert abs(batch[2][2] - (7.0 / 28.0 * 8) ** -.5 / max_weight) < .0001
assert abs(batch[3][2] - (6.0 / 28.0 * 8) ** -.5 / max_weight) < .0001

