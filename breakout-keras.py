import gym
import math
import numpy as np
import tensorflow as tf
import time
from skimage.transform import resize
from skimage.color import rgb2gray

env = gym.make('Breakout-v0')

ACT_SIZE            = env.action_space.n
STACKED_FRAMES      = 4
OBS_SHAPE           = env.observation_space.shape
INPUT_SHAPE         = (STACKED_FRAMES, int(OBS_SHAPE[0] / 2), int(OBS_SHAPE[1] / 2))
LEARN_RATE          = 0.00025
REP_SIZE            = 600000
REP_BATCH_SIZE      = 32
GAMMA               = .99
TEST_INTERVAL       = 100
MAX_EPISODES        = 1000000
EPSILON_MIN         = .1
EPSILON_DECAY_OVER  = 1000000
EPSILON_DECAY_RATE  = (EPSILON_MIN - 1) / EPSILON_DECAY_OVER
TEST_INTERVAL       = 100
TARGET_UPD_INTERVAL = 10000
MODEL_FILE_NAME     = "weights_ddqn_breakout_2018_02_01_20_10.h5"

# Decaying epsilon based on total timesteps.
def getEpsilon(totalT):
  return max(EPSILON_DECAY_RATE * totalT + 1, EPSILON_MIN)

# Convert an image to grayscale and downsize it to half the size.
def preprocess(img):
  # https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
  #return np.mean(img[::2, ::2], axis=2).astype(np.uint8)
  return np.uint8(resize(rgb2gray(img), (INPUT_SHAPE[1], INPUT_SHAPE[2])) * 255)

# Get a stack of observations from the replay array.
# key is one of lastObs or newObs.
def getObsStack(replay, ind, key):
  if len(replay) < STACKED_FRAMES:
    return [replay[0][key], replay[0][key], replay[0][key], replay[0][key]]
  return [rep[key] for rep in replay[ind - STACKED_FRAMES + 1: ind + 1]]

# Define the network model.
def buildModel():
  # Define the input model (takes in the state, which is pixels).
  model = tf.keras.models.Sequential()

  model.add(tf.keras.layers.Lambda(lambda x: x / 255.0, input_shape=INPUT_SHAPE))
  model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=8, strides=4, activation="relu", data_format="channels_first"))
  model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation="relu"))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(256, activation="relu"))
  model.add(tf.keras.layers.Dense(ACT_SIZE, activation="linear"))

  opt = tf.keras.optimizers.RMSprop(lr=LEARN_RATE)

  model.compile(loss="mean_squared_error", optimizer=opt)
  return model

# Copy the weights form model to targetModel.
def updateTargetModel(model, targetModel):
  targetModel.set_weights(model.get_weights())
  model.save(MODEL_FILE_NAME)

def main():
  model = buildModel()
  model.summary()

  targetModel = buildModel()
  updateTargetModel(model, targetModel)

  # Array for holding past results.  This is "replayed" in the network to help
  # train.
  replay    = []
  episode   = 0
  maxReward = 0
  totalT    = 0

  while episode < MAX_EPISODES:
    episode      += 1
    done          = False
    t             = 0 # t is the timestep.
    episodeReward = 0
    lastObs       = preprocess(env.reset())

    while not done:
      t      += 1
      totalT += 1
      env.render()

      if len(replay) == 0:
        lastObsStack = [lastObs, lastObs, lastObs, lastObs]
      else:
        lastObsStack = getObsStack(replay, len(replay) - 1, 'lastObs')

      epsilon = getEpsilon(totalT)

      if np.random.rand() < epsilon and episode % TEST_INTERVAL != 0:
        action = env.action_space.sample()
      elif episode % TEST_INTERVAL != 0:
        # Run the inputs through the target network to predict an action and
        # get the Q table (the estimated rewards for the current state).
        Q = targetModel.predict(np.array([lastObsStack]))
        print('Q: {}'.format(Q))

        # Action is the index of the element with the hightest predicted reward.
        action = np.argmax(Q)
      else:
        # Run the inputs through the network to predict an action and get the Q
        # table (the estimated rewards for the current state).
        Q = model.predict(np.array([lastObsStack]))
        print('Q: {}'.format(Q))

        # Action is the index of the element with the hightest predicted reward.
        action = np.argmax(Q)

      # Apply the action.
      newObs, reward, done, _ = env.step(action)
      newObs = preprocess(newObs)
      episodeReward += reward
      reward = np.sign(reward)
      #print('t, newobs, reward, done')
      #print(t, newObs, reward, done)

      if episode % TEST_INTERVAL != 0:
        # Save the result for replay.
        replay.append({
          'lastObs' : lastObs,
          'newObs'  : newObs,
          'reward'  : reward,
          'done'    : done,
          'action'  : action
        })

        if len(replay) > REP_SIZE:
          replay.pop(0)

        if len(replay) >= REP_BATCH_SIZE * STACKED_FRAMES:
          lastObses = []
          newObses  = []
          rewards   = np.empty(REP_BATCH_SIZE)
          dones     = np.empty(REP_BATCH_SIZE, dtype=bool)
          actions   = np.empty(REP_BATCH_SIZE, dtype=int)

          # Create a randomly sampled replay batch from memory.
          batchStarts = np.random.randint(STACKED_FRAMES - 1, len(replay), REP_BATCH_SIZE)

          for i in range(REP_BATCH_SIZE):
            ind = batchStarts[i]

            lastObses.append(getObsStack(replay, ind, 'lastObs'))
            newObses.append(getObsStack(replay, ind, 'newObs'))
            rewards[i]   = replay[ind]['reward']
            dones[i]     = replay[ind]['done']
            actions[i]   = replay[ind]['action']

          # Using Double DQN.
          target = model.predict(lastObses)
          newQ   = targetModel.predict(newObses)
          actSel = model.predict(newObses)

          for i in range(REP_BATCH_SIZE):
            if dones[i]:
              target[i][actions[i]] = rewards[i]
            else:
              target[i][actions[i]] = rewards[i] + GAMMA * newQ[i][np.argmax(actSel[i])]

          mse = model.train_on_batch(lastObses, target)
          print('MSE: {}'.format(mse))

          if totalT % TARGET_UPD_INTERVAL == 0:
            print("Updating target model.")
            updateTargetModel(model, targetModel)

      lastObs = newObs

      #time.sleep(.1)

    if episodeReward > maxReward:
      maxReward = episodeReward

    print('Episode {} went for {} timesteps, {} total.  Episode reward: {}.  Best reward: {}.  Epsilon: {}.'
      .format(episode, t, totalT, episodeReward, maxReward, epsilon))

if __name__ == "__main__":
  main()

