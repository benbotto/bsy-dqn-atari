import numpy as np
import math

'''
 ' Data structure for retrieving values proportionately based on their values.
 ' Inspired by https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
 ' (Maybe a better name would be SegmentTree, but I followed the naming
 ' from the referenced article.)
 ' This data structure differes, however, in that new items are added to a
 ' holding area ("pergatory") and treated as if they have infinite priority.
 ' These items are sampled first.
'''
class SumTree:
  '''
   ' Init.
  '''
  def __init__(self, capacity, pergatory_capacity):
    # A binary tree is used, where all the items are stored as leaves.  The
    # capacity is required to be a power of 2.
    assert math.log(capacity, 2).is_integer()

    # Maximum number of leaf nodes.
    self._capacity = capacity

    # Total size of the tree (e.g. if there are 8 leaf nodes, the tree has
    # a height of 4 and a size of 2^4-1 = 15).
    self._tree_size = capacity * 2 - 1
    self._memory    = np.zeros(self._capacity, dtype=object)
    self._sum_tree  = np.zeros(self._tree_size)

    # Circular write index.
    self._write = 0

    # Number of leaves currently in the tree.
    self._size = 0

    # New, unprioritized items (sampled first).
    self._pergatory_capacity = pergatory_capacity;
    self._pergatory          = np.zeros(self._pergatory_capacity, dtype=int)
    self._pergatory_size     = 0

  '''
   ' Get the number of items in memory.
  '''
  def size(self):
    return self._size

  '''
   ' Select an item proportionally based on sum.
  '''
  def get(self, sum):
    ind = self._get(0, sum)

    return (ind, self._sum_tree[ind], self._memory[ind - self._capacity + 1])

  '''
   ' Get a random sample of memory.
  '''
  def get_random_sample(self, sample_size):
    sample = np.zeros((sample_size, 3), dtype=object)

    # Unprioritized items first.
    u_sample_size = min(sample_size, self._pergatory_size)

    if u_sample_size != 0:
      # Sampled from pergatory without replacement.  pergatory contains
      # indices into the tree, and perg_inds is a random sample of indices
      # into pergatory.
      perg_inds = np.random.choice(self._pergatory_size, u_sample_size, False)

      # These are sum tree indices.
      u_sample_inds = [self._pergatory[i] for i in perg_inds]

      for i in range(u_sample_size):
        ind       = u_sample_inds[i]
        item      = self._memory[ind - self._capacity + 1]
        sample[i] = (ind, 0, item)

    # Prioritized samples next.
    p_sample_size = sample_size - u_sample_size

    for i in range(p_sample_size):
      sample[u_sample_size + i] = self.get(np.random.uniform(0, self.get_total_sum()))

    return sample

  '''
   ' Recursive retrieve based on sum.
  '''
  def _get(self, ind, sum):
    if ind >= self.get_leaf_start_ind():
      return ind

    lInd = self.get_left_child_index(ind)
    rInd = self.get_right_child_index(ind)

    if sum <= self._sum_tree[lInd]:
      return self._get(lInd, sum)
    return self._get(rInd, sum - self._sum_tree[lInd])

  '''
   ' Get the sum at an index in the tree.
  '''
  def get_sum(self, ind):
    start = self.get_leaf_start_ind()

    assert ind >= 0 and ind < start + self.size()

    return self._sum_tree[ind]

  '''
   ' Get the tree total.
  '''
  def get_total_sum(self):
    return self.get_sum(0)

  '''
   ' Add an item to memory and return its tree index.
   ' If prio is 0, it's considered to be an unprioritized
   ' item and is sampled first (with infinite priority).
  '''
  def add(self, item, prio = 0):
    assert self._pergatory_size < self._pergatory_capacity

    # Circular array -- capacity is never exceeded.
    if self._write == self._capacity:
      self._write = 0

    if self._size < self._capacity:
      self._size += 1

    # Store the item.
    self._memory[self._write] = item

    # Store the priority in the tree, and propagate the sum up to the parents.
    tree_ind = self.get_leaf_start_ind() + self._write
    self.update(tree_ind, prio)

    # Keep track of items with infinite priority.
    if prio == 0:
      self._pergatory[self._pergatory_size] = tree_ind
      self._pergatory_size += 1

    self._write += 1

    return tree_ind

  '''
   ' Update the priority of the item associated with ind.
  '''
  def update(self, ind, prio):
    start = self.get_leaf_start_ind()

    assert ind >= start and ind < start + self.size()

    # The change in priority needs to be propagated up the tree.
    old_sum = self.get_sum(ind)
    delta   = prio - old_sum

    self._sum_tree[ind] = prio
    self._update_sums(ind, delta)

    # If the item was previously unprioritized, prioritize it.
    if old_sum == 0:
      for i in range(self._pergatory_size):
        if self._pergatory[i] == ind:
          # Rather than deleting the item, the last item in pergatory simply
          # replaces the newly-prioritized item.  This was done for efficiency.
          self._pergatory_size -= 1
          self._pergatory[i] = self._pergatory[self._pergatory_size]
          self._pergatory[self._pergatory_size] = 0
          break

  '''
   ' Update the sums starting at ind and moving up the tree.
  '''
  def _update_sums(self, ind, delta):
    pInd = self.get_parent_index(ind)

    self._sum_tree[pInd] += delta

    if pInd != 0:
      self._update_sums(pInd, delta)

  '''
   ' Get the leaf starting point.
  '''
  def get_leaf_start_ind(self):
    # The sum data is stored in the sum_tree, and the priorities are in the
    # leaves (e.g. at the end).  For example, if the capacity is 8, then the
    # 8 leaves are at indices 7 through 14, inclusive.
    return self._capacity - 1

  '''
   ' Given an index, get the parent index.
  '''
  def get_parent_index(self, ind):
    return (ind - 1) // 2

  '''
   ' Given an index, get the parent item sum.
  '''
  def get_parent_sum(self, ind):
    return self.get_sum(self.get_parent_index(ind))

  '''
   ' Given an index, get the left child's index.
  '''
  def get_left_child_index(self, ind):
    return ind * 2 + 1

  '''
   ' Given an index, get the right child's index.
  '''
  def get_right_child_index(self, ind):
    return ind * 2 + 2

