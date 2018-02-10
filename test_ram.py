import sys
import gym
from network_model.atari_ram_network_model import AtariRamNetworkModel
from agent.atari_ram_tester_agent import AtariRamTesterAgent

def main(argv):
  if len(argv) != 3:
    print('Usage: {} <environment-name> <weights-file>'.format(argv[0]))
    return

  # Create the environment.
  env = gym.make(argv[1])

  # Model file to load weights from.
  model_file_name = argv[2]

  # Load the model.
  model = AtariRamNetworkModel(model_file_name, env).create_network()

  # Create the agent and test the model.
  agent = AtariRamTesterAgent(env, model)
  agent.run()

if __name__ == "__main__":
  main(sys.argv)

