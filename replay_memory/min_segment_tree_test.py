import numpy as np
from replay_memory.min_segment_tree import MinSegmentTree

# Tests use the following tree.
# Indices top to bottom left to right, 0 - 14.  Priorities are in the leaf
# nodes and are the values 1 - 8.
#            _____1____
#           /          \
#         _1_         _5_
#        /   \       /   \
#       1     3     5     7
#      / \   / \   / \   / \   
#     1   2 3   4 5   6 7   8

# Add tests.
tree = MinSegmentTree(8)
for i in range(8):
  assert tree.add(i, i+1) == i + 7
assert tree.add(9, 9) == 7 # Circular, at index 7.

# Get priority tests.
tree = MinSegmentTree(8)
for i in range(8):
  tree.add(i+1, i+1)
tree.add(9, 9)

assert tree.get_prio(0) == 2
assert tree.get_prio(1) == 2
assert tree.get_prio(2) == 5
assert tree.get_prio(3) == 2
assert tree.get_prio(4) == 3
assert tree.get_prio(5) == 5
assert tree.get_prio(6) == 7
assert tree.get_prio(7) == 9
assert tree.get_prio(8) == 2
assert tree.get_prio(9) == 3
assert tree.get_prio(10) == 4
assert tree.get_prio(11) == 5
assert tree.get_prio(12) == 6
assert tree.get_prio(13) == 7
assert tree.get_prio(14) == 8

tree.add(10, 1)

assert tree.get_prio(0) == 1
assert tree.get_prio(1) == 1
assert tree.get_prio(2) == 5
assert tree.get_prio(3) == 1
assert tree.get_prio(4) == 3
assert tree.get_prio(5) == 5
assert tree.get_prio(6) == 7
assert tree.get_prio(7) == 9
assert tree.get_prio(8) == 1
assert tree.get_prio(9) == 3
assert tree.get_prio(10) == 4
assert tree.get_prio(11) == 5
assert tree.get_prio(12) == 6
assert tree.get_prio(13) == 7
assert tree.get_prio(14) == 8

# Propagation tests.
tree = MinSegmentTree(8)
for i in range(8):
  tree.add(i+1, (i+1)*2)

assert tree.get_prio(0) == 2
assert tree.get_prio(1) == 2
assert tree.get_prio(2) == 10
assert tree.get_prio(3) == 2
assert tree.get_prio(4) == 6
assert tree.get_prio(5) == 10
assert tree.get_prio(6) == 14
assert tree.get_prio(7) == 2
assert tree.get_prio(8) == 4
assert tree.get_prio(9) == 6
assert tree.get_prio(10) == 8
assert tree.get_prio(11) == 10
assert tree.get_prio(12) == 12
assert tree.get_prio(13) == 14
assert tree.get_prio(14) == 16

tree.add(1, 2)
tree.add(2, 1)
tree.update(12, 4)

assert tree.get_prio(0) == 1
assert tree.get_prio(1) == 1
assert tree.get_prio(2) == 4
assert tree.get_prio(3) == 1
assert tree.get_prio(4) == 6
assert tree.get_prio(5) == 4
assert tree.get_prio(6) == 14
assert tree.get_prio(7) == 2
assert tree.get_prio(8) == 1
assert tree.get_prio(9) == 6
assert tree.get_prio(10) == 8
assert tree.get_prio(11) == 10
assert tree.get_prio(12) == 4
assert tree.get_prio(13) == 14
assert tree.get_prio(14) == 16

# Find tests.
tree = MinSegmentTree(8)

for i in range(8):
  tree.add(i+1, (i+1) * 2)

assert tree.find(1)[0] == 7
assert tree.find(2)[0] == 7
assert tree.find(3)[0] == 7
assert tree.find(4)[0] == 8
assert tree.find(5)[0] == 8
assert tree.find(6)[0] == 9
assert tree.find(7)[0] == 9
assert tree.find(8)[0] == 10
assert tree.find(9)[0] == 10
assert tree.find(10)[0] == 11
assert tree.find(11)[0] == 11
assert tree.find(12)[0] == 12
assert tree.find(14)[0] == 13
assert tree.find(15)[0] == 13
assert tree.find(16)[0] == 14
assert tree.find(17)[0] == 14
assert tree.find(15.999)[0] == 13
assert tree.find(2.999)[0] == 7

