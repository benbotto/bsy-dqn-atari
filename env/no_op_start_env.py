import numpy as np
from gym import Wrapper

'''
 ' Wrapper environment that performs random no-ops on start.
'''
class NoOpStartEnv(Wrapper):
  '''
   ' Init.
  '''
  def __init__(self, env, start_no_ops=10):
    super().__init__(env)

    self.start_no_ops = start_no_ops

    # No-op action.
    act_meanings    = self.env.unwrapped.get_action_meanings()
    self._no_op_act = None

    for i in range(len(act_meanings)):
      if act_meanings[i] == 'NOOP':
        self._no_op_act = i

  '''
   ' Reset the environment, then send some no-ops to make the environment
   ' stochastic.
  '''
  def reset(self):
    done = True

    while done:
      obs = self.env.reset()

      if self._no_op_act is None:
        return obs

      # Send a random number of no-ops.
      num_no_ops = np.random.randint(0, self.start_no_ops)

      for i in range(num_no_ops):
        obs, _, done, _ = self.env.step(self._no_op_act)

        if done:
          break

    return obs

