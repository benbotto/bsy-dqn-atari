import numpy as np
from agent.trainer_agent import TrainerAgent

class AtariRamTrainerAgent(TrainerAgent):
  '''
   ' Init.
  '''
  def __init__(self, env, model, target_model, memory):
    TrainerAgent.__init__(self, env, model, target_model, memory)

