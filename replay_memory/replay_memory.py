import numpy as np

class ReplayMemory:
  '''
   ' Init.
  '''
  def __init__(self, capacity):
    self._capacity = capacity
    self._size     = 0
    self._write    = 0
    self._memory   = np.zeros(capacity, dtype=object)

  '''
   ' Get the number of items in memory.
  '''
  def size(self):
    return self._size

  '''
   ' Add an item to memory.
  '''
  def add(self, item):
    if self._size < self._capacity:
      self._size += 1

    self._memory[self._write] = item
    self._write += 1

    if self._write == self._capacity:
      self._write = 0

  '''
   ' Get the item at index.
  '''
  def get(self, ind):
    return self._memory[ind]

  '''
   ' Get a random sample of memory.
  '''
  def get_random_sample(self, sample_size):
    # Sample from memory without replacement (unique).
    return np.random.choice(self._memory, sample_size, False)

  '''
   ' Remove an item from memory.
  '''
  def delete(self, ind):
    item         = self.get(ind)
    self._size  -= 1

    # Move the write pointer back one element (it's circular).
    self._write -= 1

    if self._write == -1:
      self._write = self._capacity - 1

    # Replace the item with the last thing in the list.
    self._memory[ind] = self._memory[self._write]

    # This isn't strictly needed, it's just housekeeping.
    self._memory[self._write] = 0

    return item

