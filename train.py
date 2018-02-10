import sys
import gym
from network_model.atari_network_model import AtariNetworkModel
from replay_memory.replay_memory import ReplayMemory
from agent.atari_trainer_agent import AtariTrainerAgent

REP_SIZE = 1000000

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
  model        = AtariNetworkModel(model_file_name, env).create_network()
  target_model = AtariNetworkModel(model_file_name, env).create_network()

  model.copy_weights_to(target_model)

  # The buffer for replay memory.
  memory = ReplayMemory(REP_SIZE)

  # Create the agent and start training.
  agent = AtariTrainerAgent(env, model, target_model, memory)
  agent.run()

if __name__ == "__main__":
  main(sys.argv)

