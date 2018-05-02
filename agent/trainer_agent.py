import numpy as np
from agent.agent import Agent
from agent.tester_agent import TesterAgent
from util.math_utils import get_annealing_rate, get_annealed_value

# Replay memory returns (ind, prio, is_weights, transition)
REP_IND     = 0
REP_PRIO    = 1
REP_WEIGHTS = 2
REP_TRANS   = 3

# Replays are stored in the format (s, a, r, s', d).
REP_LASTOBS = 0
REP_ACTION  = 1
REP_REWARD  = 2
REP_NEWOBS  = 3
REP_DONE    = 4

class TrainerAgent(Agent):
  '''
   ' Init.
  '''
  def __init__(self, env, model, target_model, memory, tester_agent):
    super().__init__(env, model)

    self._target_model = target_model
    self._memory       = memory
    self.tester_agent  = tester_agent

    # Keeps track of the best rewards during testing.
    self._best_max_test_reward = -1e6
    self._best_avg_test_reward = -1e6

    ##
    # Tunable parameters.
    ##

    # When to start training.
    self.train_start = 50000

    # Training interval, in frames.  E.g. when to train on a batch of transitions.
    self.train_interval = 4

    # The size of the replay batch to train on.
    self.replay_batch_size = 32

    # Discount factor.
    self.gamma = .99

    # How often to update the target network (in timesteps).
    self.target_upd_interval = 10000

    # How often to save network weights.
    self.save_weights_interval = 100000

    # Per the Nature paper, an epsilon-greedy method is used to explore, and
    # epsilon decays from 1 to epsilon_min over epsilon_decay_over frames.  The
    # OpenAI baselines, however, decrease epsilon further over the next
    # epsilon_decay_over2 frames.
    self._epsilon_decay_over  = 1e6
    self.epsilon_min          = .1
    self.epsilon_decay_rate   = get_annealing_rate(1, self.epsilon_min, self._epsilon_decay_over)

    self._epsilon_decay_over2 = 24e6
    self.epsilon_min2         = .01
    self.epsilon_decay_rate2  = get_annealing_rate(self.epsilon_min, self.epsilon_min2, self._epsilon_decay_over2)

    # How often to test the target model (in episodes).
    self.test_interval = 100

    # When to start testing.
    self.test_start = 1e6

    # How many episodes to test for.
    self.test_episodes = 20

  '''
   ' Decaying epsilon based on total timesteps.
  '''
  def get_epsilon(self, total_t):
    if total_t < self._epsilon_decay_over:
      return max(get_annealed_value(self.epsilon_decay_rate, 1, total_t), self.epsilon_min)
    else:
      return max(
        get_annealed_value(self.epsilon_decay_rate2, self.epsilon_min, total_t - self._epsilon_decay_over),
        self.epsilon_min2)

  '''
   ' PER's beta is increased from per_beta_min to 1 over the duration of the training.
  '''
  def get_per_beta(self, total_t):
    return min(get_annealed_value(self.per_beta_rate, self.per_beta_min, total_t), 1)

  '''
   ' Run the agent.
  '''
  def run(self, num_frames=200e6):
    # Number of full games played.
    episode = 0

    # Total timesteps.
    total_t = 0

    while total_t < num_frames:
      episode       += 1
      done           = False
      t              = 0 # t is the timestep.
      episode_reward = 0

      # Reset the environment to get the initial observation.
      last_obs = self._env.reset()

      while not done:
        t       += 1
        total_t += 1

        self._env.render()

        # Choose an action randomly sometimes for exploration, but other times
        # predict the action using the network model.
        epsilon = self.get_epsilon(total_t)

        if self._memory.size() < self.train_start or np.random.rand() < epsilon:
          action = self._env.action_space.sample()
        else:
          # Run the inputs through the network to predict an action and get the Q
          # table (the estimated rewards for the current state).
          Q = self._model.predict(np.array([last_obs]))
          #print('Q: {}'.format(Q))

          # Action is the index of the element with the highest predicted reward.
          action = np.argmax(Q)

        # Apply the action.
        new_obs, reward, done, _ = self._env.step(action)
        episode_reward += reward
        reward = self.process_reward(reward)

        # Store the replay memory in (s, a, r, s', d) form.
        self._memory.add((last_obs, action, reward, new_obs, done))

        if self._memory.size() >= self.train_start and total_t % self.train_interval == 0:
          # Update the beta parameter in the replay memory (for IS weights).
          self._memory.beta = self.get_per_beta(total_t)

          # Random sample from replay memory to train on.
          batch       = self._memory.get_random_sample(self.replay_batch_size)
          indices     = np.take(batch, REP_IND,     1)
          is_weights  = np.take(batch, REP_WEIGHTS, 1)
          transitions = np.take(batch, REP_TRANS,   1)

          # Array of old and new observations.
          last_observations = np.array([rep[REP_LASTOBS] for rep in transitions])
          new_observations  = np.array([rep[REP_NEWOBS] for rep in transitions])

          # Predictions from the old states, which will be updated to act as the
          # training target. Using Double DQN.
          target  = self._model.predict(last_observations)
          new_q   = self._target_model.predict(new_observations)
          act_sel = self._model.predict(new_observations)

          for i in range(self.replay_batch_size):
            act = np.argmax(act_sel[i])

            # This was the predicted reward for the taken action.  It's used
            # for updating the priority below (priority is based on error).
            predicted = target[i][transitions[i][REP_ACTION]]

            if transitions[i][REP_DONE]:
              # For terminal states, the target is simply the reward received for
              # the action taken.
              target[i][transitions[i][REP_ACTION]] = transitions[i][REP_REWARD]
            else:
              # Non-terminal targets get the reward and the estimated future
              # reward, discounted.  A discount factor (gamma) of one weighs
              # heavily toward future rewards, whereas a discount factor of
              # zero only considers immediate rewards.
              target[i][transitions[i][REP_ACTION]] = transitions[i][REP_REWARD] + self.gamma * new_q[i][act]

            # The error is the update reward minus the prediction.  If there is
            # a large error, then the transition was unexpected and thus a lot
            # can be learned from it.
            error = np.abs(target[i][transitions[i][REP_ACTION]] - predicted)
            #print('Error on index {}: {}'.format(indices[i], error))
            self._memory.update(indices[i], error)

          loss = self._model.train_on_batch(last_observations, target)
          #print('Loss: {}'.format(loss))

          # Periodically update the target network.
          if total_t % self.target_upd_interval == 0:
            self._model.copy_weights_to(self._target_model)

          # Periodically save the network weights.
          if total_t % self.save_weights_interval == 0:
            self._model.save()

        last_obs = new_obs

      # Episode complete.  Track reward info.
      self.track_reward(episode_reward)

      print('Episode: {} Timesteps: {} Total timesteps: {} Reward: {} Best reward: {} Average: {}'
        .format(episode, t, total_t, episode_reward, self.get_max_reward(), self.get_average_reward()))

      if total_t >= self.test_start and episode % self.test_interval == 0:
        print('Testing model.')
        self.test()
        print('Max Test Reward: {} Max Test Average: {}'
          .format(self._best_max_test_reward, self._best_avg_test_reward))

  '''
   ' Test the model.
  '''
  def test(self):
    self.tester_agent.reset_rewards()
    self.tester_agent.run(self.test_episodes)

    if self.tester_agent.get_max_reward() > self._best_max_test_reward:
      self._best_max_test_reward = self.tester_agent.get_max_reward()

      # Save the weights.
      model_file_name = self._target_model.model_file_name.replace('.h5', '.max.h5')
      self._target_model.save(model_file_name)

    if self.tester_agent.get_average_reward() > self._best_avg_test_reward:
      self._best_avg_test_reward = self.tester_agent.get_average_reward()
      model_file_name = self._target_model.model_file_name.replace('.h5', '.avg.h5')
      self._target_model.save(model_file_name)

