import numpy as np
from sum_segment_tree import SumSegmentTree

# Tests use the following tree.
# Indices top to bottom left to right, 0 - 14.  Sums are in the
# leaf nodes and are the values 1 - 8.
#            ____36____
#           /          \
#         _10_        _26_
#        /    \      /    \
#       3      7    11    15
#      / \    / \   / \   / \   
#     1   2  3   4 5   6 7   8

# Add tests.
tree = SumSegmentTree(8)
for i in range(8):
  assert tree.add(i, 1) == i + 7
assert tree.add(8, 1) == 7

# Get tests.
tree = SumSegmentTree(8)

for i in range(8):
  tree.add(i+1, i+1)

for i in range(7, 15):
  assert tree.get_prio(i) == i - 7 + 1
tree.add(8, 8)
assert tree.get_prio(7) == 8

# Get parent sum tests.
tree = SumSegmentTree(8)

for i in range(8):
  tree.add(i, i+1)

assert tree.get_parent_prio(1)  == 36
assert tree.get_parent_prio(2)  == 36
assert tree.get_parent_prio(3)  == 10
assert tree.get_parent_prio(4)  == 10
assert tree.get_parent_prio(5)  == 26
assert tree.get_parent_prio(6)  == 26
assert tree.get_parent_prio(7)  == 3
assert tree.get_parent_prio(8)  == 3
assert tree.get_parent_prio(9)  == 7
assert tree.get_parent_prio(10) == 7
assert tree.get_parent_prio(11) == 11
assert tree.get_parent_prio(12) == 11
assert tree.get_parent_prio(13) == 15
assert tree.get_parent_prio(14) == 15

# Overwrites the first leaf (index 7).
tree.add(1, 10)
assert tree.get_prio(7) == 10
assert tree.get_parent_prio(7) == 12
assert tree.get_parent_prio(8) == 12
assert tree.get_parent_prio(3) == 19
assert tree.get_parent_prio(1) == 45

# Child indices.
tree = SumSegmentTree(8)

assert tree.get_left_child_index(0)  == 1
assert tree.get_right_child_index(0) == 2
assert tree.get_left_child_index(1)  == 3
assert tree.get_right_child_index(1) == 4
assert tree.get_left_child_index(6)  == 13
assert tree.get_right_child_index(6) == 14

# Update.
tree = SumSegmentTree(8)

for i in range(8):
  tree.add(i, i+1)

tree.update(7, 10)
assert tree.get_prio(7) == 10
assert tree.get_prio(3) == 12
assert tree.get_prio(1) == 19
assert tree.get_prio(0) == 45

# Get.
tree = SumSegmentTree(8)

for i in range(8):
  tree.add(i, i+1)

counts = np.zeros(8)

for i in range(10000):
  _, prio, _ = tree.find(np.random.uniform(0, tree.get_base_prio()))
  counts[int(prio - 1)] += 1

# Each leaf should be retrieved proportionately to the total.
# E.g. 1 should be chosen 1 in 36 times.
for i in range(8):
  # There could be some failures here but it would be rare.
  assert abs(counts[i] / 10000 - (i+1) / 36.0) < .015

# Random sample tests (seeded for reproducibility).
np.random.seed(0)
tree = SumSegmentTree(8)

for i in range(8):
  tree.add(i, i+1)

sample = tree.get_random_sample(3)
assert sample[0][0] == 12
assert sample[1][0] == 13
assert sample[2][0] == 13

sample = tree.get_random_sample(3)
assert sample[0][0] == 12
assert sample[1][0] == 12
assert sample[2][0] == 13

sample = tree.get_random_sample(3)
assert sample[0][0] == 12
assert sample[1][0] == 14
assert sample[2][0] == 14

# Purely random samples.
np.random.seed(0)
tree = SumSegmentTree(8)

for i in range(8):
  tree.add(i+1, (i+1)**i)

sample = tree.get_random_sample(6, False)
# If prioritized 8 (index 14) would come back frequently since its priority is
# so high.
assert sample[0][0] == 13
assert sample[1][0] == 9
assert sample[2][0] == 8
assert sample[3][0] == 14
assert sample[4][0] == 10
assert sample[5][0] == 7

