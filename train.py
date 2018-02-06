import sys
import gym
from NetworkModel import NetworkModel
from ReplayMemory import ReplayMemory
from Agent import Agent

REP_SIZE = 600000

def main(argv):
  if len(argv) != 3:
    print('Usage: {} <environment-name> <weights-file>'.format(argv[0]))
    return

  # Create the environment.
  env = gym.make(argv[1])

  # Where to save the network weights.
  model_file_name = argv[2]

  # Build the network model for the environment.  The inputs are images, which
  # are preprocessed (grayscaled and scaled down by a factor of two), then
  # stacked so that the agent can determine things like velocity.
  model        = NetworkModel(env)
  target_model = NetworkModel(env)

  model.copy_weights_to(target_model)

  # The buffer for replay memory.
  memory = ReplayMemory(REP_SIZE)

  # Create the agent and start training.
  agent = Agent(env, memory, model, target_model, model_file_name)
  agent.train()

if __name__ == "__main__":
  main(sys.argv)

