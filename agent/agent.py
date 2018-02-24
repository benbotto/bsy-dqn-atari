from abc import ABC, abstractmethod
import numpy as np
from collections import deque

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
    self._max_reward = -1000000

    # Number of random no-ops when a game starts.
    self.start_no_ops = 10

    # Helper actions for starting the game (see reset() below).
    act_meanings = self._env.unwrapped.get_action_meanings()

    self._no_op_act = None
    self._fire_act  = None

    for i in range(len(act_meanings)):
      if act_meanings[i] == 'NOOP':
        self._no_op_act = i
      if act_meanings[i] == 'FIRE':
        self._fire_act = i

    # Some games, like Breakout, have a life counter.  Lives are kept
    # track of.
    self.lives = 0

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
   ' Reset the environment.  If there is a 'FIRE' action it is sent to start
   ' the game.
   ' Initially a series of no-ops are sent to make the environment stochastic.
  '''
  def reset(self):
    done       = True
    obs        = self._env.reset()
    self.lives = 0

    while done:
      # Fire to start the game.
      if self._fire_act is not None:
        obs, _, done, _ = self._env.step(self._fire_act)

      # Send a random number of no-ops.
      if self._no_op_act is not None and not done:
        num_no_ops = np.random.randint(0, self.start_no_ops)

        for i in range(num_no_ops):
          obs, _, done, _ = self._env.step(self._no_op_act)

          if done:
            break

    return obs

  '''
   ' Step the environment.
  '''
  def step(self, action):
    new_obs, reward, done, info = self._env.step(action)

    # For games that have a life counter, losing a life is considered terminal.
    # This is how the Nature paper on DQN trained.
    if 'ale.lives' in info:
      if info['ale.lives'] < self.lives and self._fire_act is not None:
        new_obs, reward, done, info = self._env.step(self._fire_act)

      self.lives = info['ale.lives']

    return new_obs, reward, done, info

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
  def run(self, num_episodes):
    pass

