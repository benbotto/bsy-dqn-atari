import numpy as np
import tensorflow as tf
from network_model.network_model import NetworkModel
from network_model.loss import huber_loss, huber_loss_mean

class AtariRamNetworkModel(NetworkModel):
  '''
   ' Init.
  '''
  def __init__(self, model_file_name, env, name, learn_rate=5e-5, decay=0.0):
    super().__init__(model_file_name, env, name, learn_rate, decay)

  '''
   ' Build the model.
  '''
  def build(self):
    print('Building model.')

    inObs  = tf.keras.layers.Input(shape=self.get_input_shape())
    norm   = tf.keras.layers.Lambda(lambda x: x / 255.0)(inObs)
    dense1 = tf.keras.layers.Dense(128, activation="relu")(norm)
    dense2 = tf.keras.layers.Dense(128, activation="relu")(dense1)
    dense3 = tf.keras.layers.Dense(128, activation="relu")(dense2)
    dense4 = tf.keras.layers.Dense(128, activation="relu")(dense3)
    out    = tf.keras.layers.Dense(self.act_size, activation="linear")(dense4)

    opt    = tf.keras.optimizers.Adam(lr=self.learn_rate, decay=self.decay)

    self.network = tf.keras.models.Model(inputs=inObs, outputs=out)
    self.network.compile(optimizer=opt, loss=huber_loss_mean)

