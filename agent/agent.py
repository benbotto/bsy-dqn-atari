from abc import ABC, abstractmethod
import numpy as np
from collections import deque

# Replays are stored in the format (s, a, r, s', done).
REP_LASTOBS = 0
REP_ACTION  = 1
REP_REWARD  = 2
REP_NEWOBS  = 3
REP_DONE    = 4

# Keep a running average reward over this may episodes.
AVG_REWARD_EPISODES = 500

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
    self._max_reward = 0

  '''
   ' Epsilon-greedy policy is used.
  '''
  def get_epsilon(self, total_t):
    return .005

  '''
   ' Process an observation.
  '''
  def process_obs(self, obs):
    return obs

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

  '''
   ' Get the maximum reward.
  '''
  def get_max_reward(self):
    return self._max_reward

  '''
   ' Get the average reward.
  '''
  def get_average_reward(self):
    return sum(self._reward_que) / len(self._reward_que)

  '''
   ' Run the agent.
  '''
  @abstractmethod
  def run(self):
    pass

