import numpy as np
from image_utils import preprocess_frame
from FrameBuffer import FrameBuffer
from TrainerAgent import TrainerAgent

class AtariTrainerAgent(TrainerAgent):
  '''
   ' Init.
  '''
  def __init__(self, env, model, target_model, memory):
    TrainerAgent.__init__(self, env, model, target_model, memory)

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

