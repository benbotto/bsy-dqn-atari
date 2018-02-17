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

# Remove tests.
# Removes the one and only item.
mem = ReplayMemory(7)

mem.add(1)
mem.delete(0)
assert mem.size() == 0

# Remove the last item.  Not full.
mem = ReplayMemory(7)

for i in range(5):
  mem.add(i + 1)

# [1 2 3 4 5 0 0] write at 5
mem.delete(4)

assert mem.size() == 4
for i in range(4):
  assert mem.get(i) == i + 1
mem.add(42)
assert mem.get(4) == 42

# Remove the last item.  Full.
mem = ReplayMemory(7)

for i in range(7):
  mem.add(i + 1)
# [1 2 3 4 5 6 7] write at 0.

mem.delete(6)
assert mem.size() == 6
for i in range(4):
  assert mem.get(i) == i + 1

mem.add(42)
assert mem.get(6) == 42

# Remove from the end after wrapping.
mem = ReplayMemory(7)

for i in range(8):
  mem.add(i + 1)
# [8 2 3 4 5 6 7] write at 1.  Index 0 (8) is the end.
mem.delete(0)

assert mem.size() == 6

for i in range(1, 7):
  assert mem.get(i) == i + 1

mem.add(42)
assert mem.get(0) == 42

# Remove from the middle.  Not full.
mem = ReplayMemory(7)

for i in range(5):
  mem.add(i + 1)
# [1 2 3 4 5 0 0] write at 5

mem.delete(2)

assert mem.get(0) == 1
assert mem.get(1) == 2
assert mem.get(2) == 5
assert mem.get(3) == 4
assert mem.size() == 4

mem.add(42)
assert mem.get(4) == 42

# Remove from the middle.  Full.
mem = ReplayMemory(7)

for i in range(7):
  mem.add(i + 1)
# [1 2 3 4 5 6 7] write at 0

mem.delete(2)

assert mem.get(0) == 1
assert mem.get(1) == 2
assert mem.get(2) == 7
assert mem.get(3) == 4
assert mem.get(4) == 5
assert mem.get(5) == 6
assert mem.size() == 6

mem.add(42)
assert mem.get(6) == 42

# Remove from the middle after wrapping.
mem = ReplayMemory(7)

for i in range(10):
  mem.add(i + 1)
# [8 9 10 4 5 6 7] write at 3

mem.delete(1)

assert mem.get(0) == 8
assert mem.get(1) == 10
assert mem.get(3) == 4
assert mem.get(4) == 5
assert mem.get(5) == 6
assert mem.get(6) == 7
assert mem.size() == 6

mem.add(42)
assert mem.get(2) == 42

# Remove from the middle after wrapping.  After write pointer.
mem = ReplayMemory(7)

for i in range(10):
  mem.add(i + 1)
# [8 9 10 4 5 6 7] write at 3

mem.delete(5)

assert mem.get(0) == 8
assert mem.get(1) == 9
assert mem.get(3) == 4
assert mem.get(4) == 5
assert mem.get(5) == 10
assert mem.get(6) == 7
assert mem.size() == 6

mem.add(42)
assert mem.get(2) == 42

# Remove from the beginning.  Not full.
mem = ReplayMemory(7)

for i in range(5):
  mem.add(i + 1)
# [1 2 3 4 5 0 0] write at 5

mem.delete(0)

assert mem.size() == 4
assert mem.get(0) == 5
assert mem.get(1) == 2
assert mem.get(2) == 3
assert mem.get(3) == 4

# Remove from the beginning.  Full.
mem = ReplayMemory(7)

for i in range(7):
  mem.add(i + 1)
# [1 2 3 4 5 6 7] write at 0

mem.delete(0)

assert mem.size() == 6
assert mem.get(0) == 7
assert mem.get(1) == 2
assert mem.get(2) == 3
assert mem.get(3) == 4
assert mem.get(4) == 5
assert mem.get(5) == 6

# Remove from the beginning after wrapping.
mem = ReplayMemory(7)

for i in range(8):
  mem.add(i + 1)
# [8 2 3 4 5 6 7] write at 0

mem.delete(6)

assert mem.size() == 6
assert mem.get(1) == 2
assert mem.get(2) == 3
assert mem.get(3) == 4
assert mem.get(4) == 5
assert mem.get(5) == 6
assert mem.get(6) == 8

