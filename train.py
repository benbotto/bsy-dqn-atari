import sys
import gym
from network_model.atari_network_model import AtariNetworkModel
from replay_memory.prioritized_replay_memory import PrioritizedReplayMemory
from env.life_terminal_env import LifeTerminalEnv
from env.fire_to_start_env import FireToStartEnv
from env.no_op_start_env import NoOpStartEnv
from env.frame_skip_env import FrameSkipEnv
from env.preprocessed_frame_env import PreprocessedFrameEnv
from env.frame_stack_env import FrameStackEnv
from agent.trainer_agent import TrainerAgent
from agent.tester_agent import TesterAgent

# Number of transitions to store in memory (must be a power of 2).
REP_SIZE  = 1048576

# Maximum number of unprioritized items in memory.
PERG_SIZE = 50000

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
  train_env = FrameSkipEnv(train_env)
  train_env = PreprocessedFrameEnv(train_env)
  train_env = FrameStackEnv(train_env)

  test_env = gym.make(argv[1])
  test_env = FireToStartEnv(test_env)
  test_env = NoOpStartEnv(test_env)
  test_env = FrameSkipEnv(test_env)
  test_env = PreprocessedFrameEnv(test_env)
  test_env = FrameStackEnv(test_env)

  # Where to save the network weights.
  model_file_name = argv[2]

  # Build the network model for the environment.  The inputs are images, which
  # are preprocessed (grayscaled and scaled down by a factor of two), then
  # stacked so that the agent can determine things like velocity.
  model        = AtariNetworkModel(model_file_name, train_env).create_network()
  target_model = AtariNetworkModel(model_file_name, train_env).create_network()

  model.copy_weights_to(target_model)

  # The buffer for replay memory.
  memory = PrioritizedReplayMemory(REP_SIZE, PERG_SIZE)

  # Create the agent and start training.
  test_agent  = TesterAgent(test_env, target_model)
  train_agent = TrainerAgent(train_env, model, target_model, memory, test_agent)
  train_agent.run()

if __name__ == "__main__":
  main(sys.argv)

