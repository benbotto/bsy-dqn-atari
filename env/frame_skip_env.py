import numpy as np
from gym import Wrapper
from util.frame_buffer import FrameBuffer

'''
 ' Wrapper environment that does frame skipping.
'''
class FrameSkipEnv(Wrapper):
  '''
   ' Init.
  '''
  def __init__(self, env, frame_skip=4):
    super().__init__(env)

    self.frame_skip = frame_skip

  '''
   ' Step the environment.  The action is repeated frame_skip times, and the
   ' total reward is returned.  The maximum pixel value of the last two
   ' frames makes up the returned observation.
  '''
  def step(self, action):
    frames       = 0
    total_reward = 0
    done         = False
    max_buff     = FrameBuffer(2)

    while not done and frames < self.frame_skip:
      obs, reward, done, info = self.env.step(action)

      frames       += 1
      total_reward += reward

      max_buff.add_frame(obs)

    frame_stack = max_buff.get_frame_stack()

    # Per the Nature paper, take the maximum pixel values from the last two
    # frames.  This is needed because some games render sprites ever other
    # frame (like the blinking ghosts in pacman).  (The FrameStack class
    # guarantees that there will be two frames.)
    obs = np.maximum(frame_stack[0], frame_stack[1])

    return obs, total_reward, done, info

