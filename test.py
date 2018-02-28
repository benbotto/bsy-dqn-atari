import sys
import gym
from network_model.atari_network_model import AtariNetworkModel
from env.fire_to_start_env import FireToStartEnv
from env.no_op_start_env import NoOpStartEnv
from env.frame_skip_env import FrameSkipEnv
from env.preprocessed_frame_env import PreprocessedFrameEnv
from env.frame_stack_env import FrameStackEnv
from agent.trainer_agent import TrainerAgent
from agent.tester_agent import TesterAgent

def main(argv):
  if len(argv) != 3:
    print('Usage: {} <environment-name> <weights-file>'.format(argv[0]))
    return

  # Create the environment, wrapped per the Nature paper.
  env = gym.make(argv[1])
  env = FireToStartEnv(env)
  env = NoOpStartEnv(env)
  env = FrameSkipEnv(env)
  env = PreprocessedFrameEnv(env)
  env = FrameStackEnv(env)

  # Model file to load weights from.
  model_file_name = argv[2]

  # Load the model.
  model = AtariNetworkModel(model_file_name, env).create_network()

  # Create the agent and test the model.
  agent = TesterAgent(env, model)
  agent.run()

if __name__ == "__main__":
  main(sys.argv)

