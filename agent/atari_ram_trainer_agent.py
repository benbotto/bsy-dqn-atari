import numpy as np
from agent.trainer_agent import TrainerAgent
from agent.atari_ram_tester_agent import AtariRamTesterAgent

class AtariRamTrainerAgent(TrainerAgent):
  '''
   ' Init.
  '''
  def __init__(self, env, model, target_model, memory):
    super().__init__(env, model, target_model, memory)

  '''
   ' Test the model.
  '''
  def test(self):
    test_agent = AtariRamTesterAgent(self._env, self._target_model)
    test_agent.run(self.test_episodes)

