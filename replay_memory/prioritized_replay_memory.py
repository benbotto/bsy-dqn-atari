from replay_memory.sum_tree import SumTree

'''
 ' Prioritized experience replay using a SumTree.
'''
class PrioritizedReplayMemory(SumTree):
  '''
   ' Init.
  '''
  def __init__(self, capacity, epsilon = .01, alpha = .6):
    super().__init__(capacity)

    self.epsilon = epsilon
    self.alpha   = alpha

  '''
   ' Update the error (and hence, priority) of the item associated with ind.
  '''
  def update_error(self, ind, error):
    # Convert the error to a priority.  Epsilon is used so that
    # no item has 0 priority.  Alpha is used to determine how much
    # priority is weighted.  For example, an alpha of 0 would mean
    # items are sampled uniformly.
    prio = (error + self.epsilon) ** self.alpha

    return super().update(ind, prio)

