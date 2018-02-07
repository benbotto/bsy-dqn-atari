import numpy as np
from image_utils import preprocess_frame
from FrameBuffer import FrameBuffer
from collections import deque

# Per the Nature paper, an epsilon-greedy method is used to explore, and
# epsilon decays from 1 to .1 over 1 million frames.
EPSILON_MIN         = .1
EPSILON_DECAY_OVER  = 1000000
EPSILON_DECAY_RATE  = (EPSILON_MIN - 1) / EPSILON_DECAY_OVER

# When to start training.
TRAIN_START = 50000

# The size of the replay batch to train on.
REP_BATCH_SIZE = 32

# Replays are stored in the format (s, a, r, s', done).
REP_LASTOBS = 0
REP_ACTION  = 1
REP_REWARD  = 2
REP_NEWOBS  = 3
REP_DONE    = 4

# Discount factor.
GAMMA = .99

# How often to update the target network.
TARGET_UPD_INTERVAL = 10000

# How often to save network weights.
SAVE_WEIGHTS_INTERVAL = 500000

# Keep a running average reward over this may episodes.
AVG_REWARD_EPISODES = 500

class Agent:
  '''
   ' Init.
  '''
  def __init__(self, env, memory, model, target_model, model_file_name):
    self._env            = env
    self._memory         = memory
    self._model          = model
    self._target_model   = target_model
    self.model_file_name = model_file_name
    self._reward_que     = deque([], maxlen=AVG_REWARD_EPISODES)

  '''
   ' Decaying epsilon based on total timesteps.
  '''
  def get_epsilon(self, total_t):
    return max(EPSILON_DECAY_RATE * total_t + 1, EPSILON_MIN)

  '''
   ' Train the model.
  '''
  def train(self):
    # Store and stacks the frames for easy access.  The DQN paper use the last
    # four frames as an observation.
    frames = FrameBuffer(self._model.stacked_frames)

    # Number of full games played.
    episode = 0

    # Maximum reward for all games.
    max_reward = 0

    # Total timesteps.
    total_t = 0

    while True:
      episode       += 1
      done           = False
      t              = 0 # t is the timestep.
      episode_reward = 0

      # Frames get preprocessed (converted to grayscale and scaled down by a
      # factor of two) then stacked.
      last_obs = frames.add_frame(preprocess_frame(self._env.reset(),
        self._model.frame_width, self._model.frame_height))

      while not done:
        t       += 1
        total_t += 1

        self._env.render()

        # Choose an action randomly sometimes for exploration, but other times
        # predict the action using the network model.
        epsilon = self.get_epsilon(total_t)

        if self._memory.size() < TRAIN_START or np.random.rand() < epsilon:
          action = self._env.action_space.sample()
        else:
          # Run the inputs through the network to predict an action and get the Q
          # table (the estimated rewards for the current state).
          Q = self._model.predict(np.array([last_obs]))
          print('Q: {}'.format(Q))

          # Action is the index of the element with the highest predicted reward.
          action = np.argmax(Q)

        # Apply the action.
        new_obs, reward, done, _ = self._env.step(action)
        new_obs = frames.add_frame(preprocess_frame(new_obs,
          self._model.frame_width, self._model.frame_height))
        episode_reward += reward

        # Rewards are clipped to -1, 0, and 1 per the Nature paper.
        reward = np.sign(reward)

        # Store the replay memory in (s, a, r, s', t) form.
        self._memory.add((last_obs, action, reward, new_obs, done))

        if self._memory.size() >= TRAIN_START:
          # Random sample from replay memory to train on.
          batch = self._memory.get_random_sample(REP_BATCH_SIZE)

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
              target[i][batch[i][REP_ACTION]] = batch[i][REP_REWARD] + GAMMA * new_q[i][act]

          mse = self._model.train_on_batch(last_observations, target)
          #print(mse)

          if total_t % TARGET_UPD_INTERVAL == 0:
            self._model.copy_weights_to(self._target_model)

          if total_t % SAVE_WEIGHTS_INTERVAL == 0:
            self._model.save(self.model_file_name)

        last_obs = new_obs

      if episode_reward > max_reward:
        max_reward = episode_reward

      self._reward_que.append(episode_reward)
      avg_reward = sum(self._reward_que) / len(self._reward_que)

      print('Episode: {} Timesteps: {} Total timesteps: {} Reward: {} Best reward: {} Average: {}'
        .format(episode, t, total_t, episode_reward, max_reward, avg_reward))

