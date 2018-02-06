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
    if self._write == self._capacity:
      self._write = 0

    if self._size < self._capacity:
      self._size += 1

    self._memory[self._write] = item
    self._write += 1

  '''
   ' Get the item at index.
  '''
  def get(self, ind):
    return self._memory[ind]

  '''
   ' Get a random sample of memory.
  '''
  def get_random_sample(self, sample_size):
    sample = np.zeros(sample_size, dtype=object)

    for i in range(sample_size):
      sample[i] = self._memory[np.random.randint(0, self._size)]

    return sample
    
