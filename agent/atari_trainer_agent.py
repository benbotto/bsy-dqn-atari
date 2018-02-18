import numpy as np
from agent.trainer_agent import TrainerAgent
from util.image_utils import preprocess_frame
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

  '''
   ' Process an observation.  They are converted to grayscale, scaled down by
   ' a factor of two, then stacked.
  '''
  def process_obs(self, obs):
    return self._frames.add_frame(
      preprocess_frame(obs, self._model.frame_width, self._model.frame_height))

  '''
   ' Test the model.
  '''
  def test(self):
    test_agent = AtariTesterAgent(self._env, self._target_model)
    test_agent.run(self.test_episodes)

