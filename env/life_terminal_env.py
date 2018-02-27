from gym import Wrapper
from util.life_tracker import LifeTracker

'''
 ' Wrapper environment with episodic life (per the Nature paper).
'''
class LifeTerminalEnv(Wrapper):
  '''
   ' Init.
  '''
  def __init__(self, env):
    super().__init__(env)

    self.life_tracker = LifeTracker()

  '''
   ' Reset the environment and life counter.
  '''
  def reset(self):
    self.life_tracker.reset()
    return self.env.reset()

  '''
   ' Step the environment.  Flag done if a life is lost.
  '''
  def step(self, action):
    lives = self.life_tracker.lives
    obs, reward, done, info = self.env.step(action)
    new_lives = self.life_tracker.track_lives(info)

    if new_lives < lives:
      done = True

    return obs, reward, done, info

