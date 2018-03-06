import sys
import gym
from network_model.atari_ram_network_model import AtariRamNetworkModel
from env.fire_to_start_env import FireToStartEnv
from env.no_op_start_env import NoOpStartEnv
from agent.tester_agent import TesterAgent

def main(argv):
  if len(argv) != 3:
    print('Usage: {} <environment-name> <weights-file>'.format(argv[0]))
    return

  # Create the environment, wrapped per the Nature paper.
  env = gym.make(argv[1])
  env = FireToStartEnv(env)
  env = NoOpStartEnv(env)

  # Model file to load weights from.
  model_file_name = argv[2]

  # Load the model.
  model = AtariRamNetworkModel(model_file_name, env, 'model').create_network()

  # Create the agent and test the model.
  agent = TesterAgent(env, model)
  agent.run()

if __name__ == "__main__":
  main(sys.argv)

