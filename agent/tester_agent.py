import numpy as np
from agent.agent import Agent

class TesterAgent(Agent):
  '''
   ' Init.
  '''
  def __init__(self, env, model):
    Agent.__init__(self, env, model)

    ##
    # Tunable parameters.
    ##

    # Epsilon value.  Some random actions are needed because the model
    # may not learn to start games, and hence get stuck.  The Nature
    # paper uses .05, but that seems high to me.
    self.epsilon = .05

  '''
   ' Get the fixed epsilon.
  '''
  def get_epsilon(self, total_t):
    return self.epsilon

  '''
   ' Run the agent.
  '''
  def run(self, num_episodes=10):
    # Number of full games played.
    episode = 0

    # Total timesteps.
    total_t = 0

    while episode < num_episodes:
      episode       += 1
      done           = False
      t              = 0 # t is the timestep.
      episode_reward = 0

      # Reset the environment to get the initial observation.
      last_obs = self.process_obs(self.reset())

      while not done:
        t       += 1
        total_t += 1

        self._env.render()

        # Choose an action randomly sometimes for exploration, but other times
        # predict the action using the network model.
        epsilon = self.get_epsilon(total_t)

        if np.random.rand() < epsilon:
          action = self._env.action_space.sample()
        else:
          # Run the inputs through the network to predict an action and get the Q
          # table (the estimated rewards for the current state).
          Q = self._model.predict(np.array([last_obs]))

          # Action is the index of the element with the highest predicted reward.
          action = np.argmax(Q)

        # Apply the action.
        new_obs, reward, done, _ = self.step(action)
        new_obs = self.process_obs(new_obs)
        episode_reward += reward

        last_obs = new_obs

      # Episode complete.  Track reward info.
      self.track_reward(episode_reward)

      print('Episode: {} Timesteps: {} Total timesteps: {} Reward: {} Best reward: {} Average: {}'
        .format(episode, t, total_t, episode_reward, self.get_max_reward(), self.get_average_reward()))

