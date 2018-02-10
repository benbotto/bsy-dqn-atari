import numpy as np
from agent.tester_agent import TesterAgent

class AtariRamTesterAgent(TesterAgent):
  '''
   ' Init.
  '''
  def __init__(self, env, model):
    TesterAgent.__init__(self, env, model)

