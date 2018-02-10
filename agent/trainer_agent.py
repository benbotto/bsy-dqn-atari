import numpy as np
from agent.agent import Agent

# Replays are stored in the format (s, a, r, s', done).
REP_LASTOBS = 0
REP_ACTION  = 1
REP_REWARD  = 2
REP_NEWOBS  = 3
REP_DONE    = 4

class TrainerAgent(Agent):
  '''
   ' Init.
  '''
  def __init__(self, env, model, target_model, memory):
    Agent.__init__(self, env, model)

    self._target_model = target_model
    self._memory       = memory

    ##
    # Tunable parameters.
    ##

    # When to start training.
    self.train_start = 50000

    # The size of the replay batch to train on.
    self.replay_batch_size = 32

    # Discount factor.
    self.gamma = .99

    # How often to update the target network (in timesteps).
    self.target_upd_interval = 10000

    # How often to save network weights.
    self.save_weights_interval = 500000

    # Per the Nature paper, an epsilon-greedy method is used to explore, and
    # epsilon decays from 1 to .1 over 1 million frames.
    EPSILON_DECAY_OVER       = 1000000
    self.epsilon_min         = .1
    self.epsilon_decay_rate  = (self.epsilon_min - 1) / EPSILON_DECAY_OVER

  '''
   ' Decaying epsilon based on total timesteps.
  '''
  def get_epsilon(self, total_t):
    return max(self.epsilon_decay_rate * total_t + 1, self.epsilon_min)

  '''
   ' Run the agent.
  '''
  def run(self):
    # Number of full games played.
    episode = 0

    # Total timesteps.
    total_t = 0

    while True:
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

        if self._memory.size() < self.train_start or np.random.rand() < epsilon:
          action = self._env.action_space.sample()
        else:
          # Run the inputs through the network to predict an action and get the Q
          # table (the estimated rewards for the current state).
          Q = self._model.predict(np.array([last_obs]))
          print('Q: {}'.format(Q))

          # Action is the index of the element with the highest predicted reward.
          action = np.argmax(Q)

        # Apply the action.
        new_obs, reward, done, _ = self.step(action)
        new_obs = self.process_obs(new_obs)
        episode_reward += reward
        reward = self.process_reward(reward)

        # Store the replay memory in (s, a, r, s', t) form.
        self._memory.add((last_obs, action, reward, new_obs, done))

        if self._memory.size() >= self.train_start:
          # Random sample from replay memory to train on.
          batch = self._memory.get_random_sample(self.replay_batch_size)

          # Array of old and new observations.
          last_observations = np.array([rep[REP_LASTOBS] for rep in batch])
          new_observations  = np.array([rep[REP_NEWOBS] for rep in batch])

          # Predictions from the old states, which will be updated to act as the
          # training target. Using Double DQN.
          target  = self._model.predict(last_observations)
          new_q   = self._target_model.predict(new_observations)
          act_sel = self._model.predict(new_observations)

          for i in range(len(batch)):
            act = np.argmax(act_sel[i])

            if batch[i][REP_DONE]:
              # For terminal states, the target is simply the reward received for
              # the action taken.
              target[i][batch[i][REP_ACTION]] = batch[i][REP_REWARD]
            else:
              # Non-terminal targets get the reward and the estimated future
              # reward, discounted.  A discount factor (gamma) of one weighs
              # heavily toward future rewards, whereas a discount factor of
              # zero only considers immediate rewards.
              target[i][batch[i][REP_ACTION]] = batch[i][REP_REWARD] + self.gamma * new_q[i][act]

          mse = self._model.train_on_batch(last_observations, target)
          #print(mse)

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
