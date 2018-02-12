import numpy as np
import tensorflow as tf
from network_model.network_model import NetworkModel
from network_model.loss import huber_loss

class AtariRamNetworkModel(NetworkModel):
  '''
   ' Init.
  '''
  def __init__(self, model_file_name, env, learn_rate=0.00025):
    NetworkModel.__init__(self, model_file_name, env, learn_rate)

  '''
   ' Build the model.
  '''
  def build(self):
    print('Building model.')
    self.network = tf.keras.models.Sequential()

    self.network.add(tf.keras.layers.Lambda(lambda x: x / 255.0, input_shape=self.get_input_shape()))
    self.network.add(tf.keras.layers.Dense(128, activation="relu"))
    self.network.add(tf.keras.layers.Dense(128, activation="relu"))
    self.network.add(tf.keras.layers.Dense(128, activation="relu"))
    self.network.add(tf.keras.layers.Dense(128, activation="relu"))
    self.network.add(tf.keras.layers.Dense(self.act_size, activation="linear"))

    opt = tf.keras.optimizers.RMSprop(lr=self.learn_rate)

    self.network.compile(loss="logcosh", optimizer=opt)

