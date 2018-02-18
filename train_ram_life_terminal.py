import sys
import gym
from network_model.atari_ram_network_model import AtariRamNetworkModel
from replay_memory.prioritized_replay_memory import PrioritizedReplayMemory
from agent.atari_ram_life_terminal_trainer_agent import AtariRamLifeTerminalTrainerAgent

# Number of transitions to store in memory (must be a power of 2).
REP_SIZE  = 1048576

# Maximum number of unprioritized items in memory.
PERG_SIZE = 50000

def main(argv):
  if len(argv) != 3:
    print('Usage: {} <environment-name> <weights-file>'.format(argv[0]))
    return

  # Create the environment.
  env = gym.make(argv[1])

  # Where to save the network weights.
  model_file_name = argv[2]

  # Build the network model for the environment.
  model        = AtariRamNetworkModel(model_file_name, env).create_network()
  target_model = AtariRamNetworkModel(model_file_name, env).create_network()

  model.copy_weights_to(target_model)

  # The buffer for replay memory.
  memory = PrioritizedReplayMemory(REP_SIZE, PERG_SIZE)

  # Create the agent and start training.
  agent = AtariRamLifeTerminalTrainerAgent(env, model, target_model, memory)
  agent.run()

if __name__ == "__main__":
  main(sys.argv)

