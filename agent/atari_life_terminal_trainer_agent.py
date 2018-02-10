import numpy as np
from agent.atari_trainer_agent import AtariTrainerAgent

class AtariLifeTerminalTrainerAgent(AtariTrainerAgent):
  '''
   ' Init.
  '''
  def __init__(self, env, model, target_model, memory):
    AtariTrainerAgent.__init__(self, env, model, target_model, memory)

    self._lives = 0

  '''
   ' Reset the environment.
  '''
  def reset(self):
    self._lives = 0
    return self._env.reset()

  '''
   ' Step the environment.
  '''
  def step(self, action):
    new_obs, reward, done, info = super(AtariTrainerAgent, self).step(action)

    # For games that have a life counter, losing a life is considered terminal.
    # This is how the Nature paper on DQN trained.
    if 'ale.lives' in info:
      if info['ale.lives'] > self._lives:
        self._lives = info['ale.lives']
      elif info['ale.lives'] < self._lives:
        done = True

    return new_obs, reward, done, info

