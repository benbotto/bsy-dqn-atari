from gym import Wrapper
from util.life_tracker import LifeTracker

'''
 ' Wrapper environment that fires to start a game.
'''
class FireToStartEnv(Wrapper):
  '''
   ' Init.
  '''
  def __init__(self, env):
    super().__init__(env)

    # Lives are tracked for games with lives.
    self.life_tracker = LifeTracker()

    # Fire action for starting the game (see reset() below).
    act_meanings   = self.env.unwrapped.get_action_meanings()
    self._fire_act =  None

    for i in range(len(act_meanings)):
      if act_meanings[i] == 'FIRE':
        self._fire_act = i

  '''
   ' Reset the environment.
  '''
  def reset(self):
    self.life_tracker.reset()

    obs = self.env.reset()

    # Fire to start the game.
    if self._fire_act is not None:
      obs, _, _, info = self.env.step(self._fire_act)
      self.life_tracker.track_lives(info)

    return obs

  '''
   ' Step the environment.  Fire if a life is lost.
  '''
  def step(self, action):
    lives = self.life_tracker.lives
    obs, reward, done, info = self.env.step(action)
    new_lives = self.life_tracker.track_lives(info)

    if not done and new_lives < lives:
      obs, reward2, done, info = self.env.step(self._fire_act)
      self.life_tracker.track_lives(info)
      reward += reward2

    return obs, reward, done, info

