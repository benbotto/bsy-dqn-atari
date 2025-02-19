from abc import ABC, abstractmethod
import numpy as np
from collections import deque

# Keep a running average reward over this may episodes.
AVG_REWARD_EPISODES = 100

'''
 ' Abstract base class for Deep-Q Learning Agents.
'''
class Agent(ABC):
  '''
   ' Init.
  '''
  def __init__(self, env, model):
    self._env   = env
    self._model = model

    # For keeping the average reward over time.
    self._reward_que = deque([], maxlen=AVG_REWARD_EPISODES)

    # For keeping track of the maximum reward over time.
    self._max_reward = -1000000
    self._max_avg    = -1000000

  '''
   ' Get epsilon, which may be based on timestep.
  '''
  @abstractmethod
  def get_epsilon(self, total_t):
    pass

  '''
   ' Process a reward.
   ' By default rewards are clipped to -1, 0, and 1 per the Nature paper.
  '''
  def process_reward(self, reward):
    return np.sign(reward)

  '''
   ' Track episode rewards.
  '''
  def track_reward(self, reward):
    self._reward_que.append(reward)

    if reward > self._max_reward:
      self._max_reward = reward

    avg = self.get_average_reward()

    if avg > self._max_avg:
      self._max_avg = avg

  '''
   ' Get the maximum reward.
  '''
  def get_max_reward(self):
    return self._max_reward

  '''
   ' Get the maximum average.
  '''
  def get_max_average(self):
    return self._max_avg

  '''
   ' Get the average reward.
  '''
  def get_average_reward(self):
    return sum(self._reward_que) / len(self._reward_que)

  '''
   ' Reset the reward tracking.
  '''
  def reset_rewards(self):
    self._reward_que.clear()
    self._max_reward = -1000000

  '''
   ' Run the agent.
  '''
  @abstractmethod
  def run(self, num_episodes):
    pass

