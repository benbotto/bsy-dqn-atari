import numpy as np
import os.path
import tensorflow as tf

class NetworkModel:
  '''
   ' Init.
  '''
  def __init__(self, model_file_name, env, learn_rate=0.00025):
    self.obs_shape       = env.observation_space.shape
    self.stacked_frames  = 4
    self.frame_width     = int(self.obs_shape[0] / 2)
    self.frame_height    = int(self.obs_shape[1] / 2)
    self.input_shape     = (self.stacked_frames, self.frame_width, self.frame_height)
    self.act_size        = env.action_space.n
    self.learn_rate      = learn_rate
    self.model_file_name = model_file_name

    # Load the network from a save file.
    if os.path.isfile(self.model_file_name):
      print('Loading model from file {}.'.format(self.model_file_name))
      self.load()
    else:
      print('Building model.')
      self.build()

  '''
   ' Build the model.  Override this in specific classes.
  '''
  def build(self):
    # Build the model, as defined by the Nature paper on DQN by
    # David Silver et. al.
    self.network = tf.keras.models.Sequential()

    self.network.add(tf.keras.layers.Lambda(lambda x: x / 255.0, input_shape=self.input_shape))
    self.network.add(tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation="relu", data_format="channels_first"))
    self.network.add(tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu"))
    self.network.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu"))
    self.network.add(tf.keras.layers.Flatten())
    self.network.add(tf.keras.layers.Dense(512, activation="relu"))
    self.network.add(tf.keras.layers.Dense(self.act_size, activation="linear"))

    opt = tf.keras.optimizers.RMSprop(lr=self.learn_rate)

    self.network.compile(loss="mean_squared_error", optimizer=opt)

  '''
   ' Copy the weights from one NetworkModel to another.
  '''
  def copy_weights_to(self, target):
    target.network.set_weights(self.network.get_weights())

  '''
   ' Save the weights to a file.
  '''
  def save(self):
    self.network.save(self.model_file_name)

  '''
   ' Load the weights from a file.
  '''
  def load(self):
    self.network = tf.keras.models.load_model(self.model_file_name)

  '''
   ' Make a prediction based on an observation.
  '''
  def predict(self, obs):
    return self.network.predict(obs)

  '''
   ' Train the model on a batch of inputs (observations) and expectations (reward
   ' predictions).
  '''
  def train_on_batch(self, observations, expectations):
    return self.network.train_on_batch(observations, expectations)

