import numpy as np
from TrainerAgent import TrainerAgent

class AtariRamTrainerAgent(TrainerAgent):
  '''
   ' Init.
  '''
  def __init__(self, env, model, target_model, memory):
    TrainerAgent.__init__(self, env, model, target_model, memory)

