import sys
import gym
from network_model.atari_ram_network_model import AtariRamNetworkModel
from replay_memory.prioritized_replay_memory import PrioritizedReplayMemory
from env.life_terminal_env import LifeTerminalEnv
from env.fire_to_start_env import FireToStartEnv
from env.no_op_start_env import NoOpStartEnv
from agent.trainer_agent import TrainerAgent
from agent.tester_agent import TesterAgent

# Number of transitions to store in memory (must be a power of 2).
REP_SIZE  = 1048576

def main(argv):
  if len(argv) != 3:
    print('Usage: {} <environment-name> <weights-file>'.format(argv[0]))
    return

  # Create the environments, one for training and one for testing,
  # wrapped per the Nature paper.
  train_env = gym.make(argv[1])
  train_env = LifeTerminalEnv(train_env)
  train_env = FireToStartEnv(train_env)
  train_env = NoOpStartEnv(train_env)

  test_env = gym.make(argv[1])
  test_env = FireToStartEnv(test_env)
  test_env = NoOpStartEnv(test_env)

  # Where to save the network weights.
  model_file_name = argv[2]

  # Build the network model for the environment.
  model        = AtariRamNetworkModel(model_file_name, train_env, 'model').create_network()
  target_model = AtariRamNetworkModel(model_file_name, train_env, 'target').create_network()

  model.copy_weights_to(target_model)

  # The buffer for replay memory.
  memory = PrioritizedReplayMemory(REP_SIZE, PERG_SIZE)

  # Create the agent and start training.
  test_agent  = TesterAgent(test_env, target_model)
  train_agent = TrainerAgent(train_env, model, target_model, memory, test_agent)
  train_agent.run()

if __name__ == "__main__":
  main(sys.argv)

