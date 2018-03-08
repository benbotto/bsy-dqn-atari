import numpy as np
import tensorflow as tf
from network_model.network_model import NetworkModel
from network_model.loss import huber_loss, huber_loss_mean

class AtariRamNetworkModel(NetworkModel):
  '''
   ' Init.
  '''
  def __init__(self, model_file_name, env, name, learn_rate=1e-4, decay=9e-7):
    super().__init__(model_file_name, env, name, learn_rate, decay)

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

    opt = tf.keras.optimizers.Adam(lr=self.learn_rate, decay=self.decay)

    self.network.compile(loss=huber_loss_mean, optimizer=opt)

