import numpy as np
from replay_memory.prioritized_replay_memory import PrioritizedReplayMemory

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

max_prio = 1.000001 ** .5
assert memory.get_max_prio() == max_prio

for i in range(8):
  memory.add(i)
for i in range(8):
  assert memory.get_prio(7 + i) == max_prio

assert memory.get_max_prio() == max_prio

memory.update(7, 1.2)
memory.update(8, 0.4)

max_prio = (1.2 + .000001) ** .5
min_prio = (0.4 + .000001) ** .5

assert memory.get_max_prio() == max_prio
assert memory.get_min_prio() == min_prio

memory.add(4)
assert memory.get_prio(7) == max_prio

# Max prio also moves down when the old max falls off the tail.
memory = PrioritizedReplayMemory(8)

for i in range(8):
  memory.add(i)

memory.update(7,  .9)
memory.update(8,  .8)
memory.update(9,  .7)
memory.update(10, .6)
memory.update(11, .5)
memory.update(12, .4)
memory.update(13, .3)
assert memory.get_max_prio() == (1 + .000001) ** .5
memory.update(14, .2)
assert memory.get_max_prio() == (.9 + .000001) ** .5

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
max_weight = (1.0 / 36.0 * 8) ** -.5 # (lowest prio) / (sum of prios) * (total samples) ^ -beta

assert batch[0][2] == (6.0 / 36.0 * 8) ** -.5 / max_weight
assert batch[1][2] == (7.0 / 36.0 * 8) ** -.5 / max_weight
assert batch[2][2] == (7.0 / 36.0 * 8) ** -.5 / max_weight
assert batch[3][2] == (6.0 / 36.0 * 8) ** -.5 / max_weight

