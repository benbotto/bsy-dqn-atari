import numpy as np
from gym import Wrapper, spaces
from util.image_utils import preprocess_frame

'''
 ' Wrapper environment that does frame skipping.
'''
class PreprocessedFrameEnv(Wrapper):
  '''
   ' Init.
  '''
  def __init__(self, env, frame_width=84, frame_height=84):
    super().__init__(env)

    # Redefine the observation space using the frame width and height.
    self.observation_space = spaces.Box(
      low=0,
      high=255,
      shape=(frame_width, frame_height),
      dtype=np.uint8)

  '''
   ' Helper function to preprocess a frame.
  '''
  def preprocess_frame(self, obs):
    return preprocess_frame(
      obs,
      self.observation_space.shape[0],
      self.observation_space.shape[1])

  '''
   ' Reset the environment and preprocess the frame..
  '''
  def reset(self):
    return self.preprocess_frame(self.env.reset())

  '''
   ' Step the environment and process the frame by converting it to grayscale
   ' and scaling it down to frame_width x frame_height.
  '''
  def step(self, action):
    obs, reward, done, info = self.env.step(action)

    obs = self.preprocess_frame(obs)

    return obs, reward, done, info

