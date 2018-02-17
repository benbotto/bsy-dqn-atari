import numpy as np
from replay_memory import ReplayMemory

# Get tests.
mem = ReplayMemory(5)

for i in range(5):
  mem.add(i + 1)

assert mem.get(0) == 1
assert mem.get(1) == 2
assert mem.get(2) == 3
assert mem.get(3) == 4
assert mem.get(4) == 5

mem.add(6)
assert mem.get(0) == 6

# Random sample tests.
mem = ReplayMemory(5)

for i in range(5):
  mem.add(i)

sample = mem.get_random_sample(5)
count  = np.zeros(5)

# All sampled, so the sample is unique.
for i in range(5):
  count[sample[i]] += 1
for i in range(5):
  assert count[i] == 1

