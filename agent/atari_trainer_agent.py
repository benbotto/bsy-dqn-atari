import numpy as np
from agent.trainer_agent import TrainerAgent
from util.image_utils import preprocess_frame, display_frame
from util.frame_buffer import FrameBuffer
from agent.atari_tester_agent import AtariTesterAgent

class AtariTrainerAgent(TrainerAgent):
  '''
   ' Init.
  '''
  def __init__(self, env, model, target_model, memory):
    super().__init__(env, model, target_model, memory)

    # Store and stacks the frames for easy access.  The DQN paper use the last
    # four frames as an observation.
    self._frames = FrameBuffer(self._model.stacked_frames)

    # Number of frames to skip.  The Nature paper skips 4 frames.  Rewards
    # are summed over the skipped frames.
    self.frame_skip = 4

  '''
   ' Process an observation.  They are converted to grayscale and scaled to
   ' 84x84 (per the Nature paper), then stacked.
  '''
  def process_obs(self, obs):
    preprocessed = preprocess_frame(obs, self._model.frame_width, self._model.frame_height)

    return self._frames.add_frame(preprocessed)

  '''
   ' Step the environment.
  '''
  def step(self, action):
    frames       = 0
    total_reward = 0
    done         = False
    max_buff     = FrameBuffer(2)

    while not done and frames < self.frame_skip:
      obs, reward, done, info = super().step(action)

      frames       += 1
      total_reward += reward

      max_buff.add_frame(obs)

    frame_stack = max_buff.get_frame_stack()

    # Per the Nature paper, take the maximum pixel values from the last two
    # frames.  This is needed because some games render sprites ever other
    # frame (like the blinking ghosts in pacman).
    if len(frame_stack) == 2:
      obs = np.maximum(frame_stack[0], frame_stack[1])

    return obs, total_reward, done, info

  '''
   ' Test the model.
  '''
  def test(self):
    test_agent = AtariTesterAgent(self._env, self._target_model)
    test_agent.run(self.test_episodes)

