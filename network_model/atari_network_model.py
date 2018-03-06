import numpy as np
import tensorflow as tf
from network_model.network_model import NetworkModel
from network_model.loss import huber_loss, huber_loss_mean

class AtariNetworkModel(NetworkModel):
  '''
   ' Init.
  '''
  def __init__(self, model_file_name, env, name, learn_rate=5e-5):
    super().__init__(model_file_name, env, name, learn_rate)

  '''
   ' Build the model.
  '''
  def build(self):
    print('Building model.')

    # Build the model, as defined by the Nature paper on DQN by
    # David Silver et. al.
    self.network = tf.keras.models.Sequential()

    self.network.add(tf.keras.layers.Lambda(lambda x: x / 255.0, input_shape=self.get_input_shape()))
    self.network.add(tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation="relu", data_format="channels_first"))
    self.network.add(tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu"))
    self.network.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu"))
    self.network.add(tf.keras.layers.Flatten())
    self.network.add(tf.keras.layers.Dense(512, activation="relu"))
    self.network.add(tf.keras.layers.Dense(self.act_size, activation="linear"))

    opt = tf.keras.optimizers.Adam(lr=self.learn_rate)

    self.network.compile(loss=huber_loss_mean, optimizer=opt)

