from gym import Wrapper, spaces
from util.frame_buffer import FrameBuffer

'''
 ' Wrapper environment that does frame stacking.
'''
class FrameStackEnv(Wrapper):
  '''
   ' Init.
  '''
  def __init__(self, env, stacked_frames=4):
    super().__init__(env)

    self.stacked_frames = stacked_frames

    # Stores and stacks the frames for easy access.  The DQN paper use the last
    # four frames as an observation.
    self._frames = FrameBuffer(self.stacked_frames)

    # Redefine the observation space to have stacked_frames frames.
    env_obs_space = self.env.observation_space

    self.observation_space = spaces.Box(0, 255, 
      (self.stacked_frames, env_obs_space.shape[0], env_obs_space.shape[1]))

  '''
   ' Step the environment and stack the observation.  This way the agent can
   ' determine volocity.
  '''
  def step(self, action):
    obs, reward, done, info = self.env.step(action)

    # Adding the frame returns the most recent for frames.
    obs = self._frames.add_frame(obs)

    return obs, reward, done, info

