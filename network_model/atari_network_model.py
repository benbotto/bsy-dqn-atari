import numpy as np
import tensorflow as tf
from network_model.network_model import NetworkModel
from network_model.loss import huber_loss, huber_loss_mean

class AtariNetworkModel(NetworkModel):
  '''
   ' Init.
  '''
  def __init__(self, model_file_name, env, learn_rate=1e-4):
    super().__init__(model_file_name, env, learn_rate)

    self.stacked_frames  = 4
    self.frame_width     = int(self.obs_shape[0] / 2)
    self.frame_height    = int(self.obs_shape[1] / 2)

  '''
   ' Get the input shape.
  '''
  def get_input_shape(self):
    return (self.stacked_frames, self.frame_width, self.frame_height)

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

    opt = tf.keras.optimizers.Adam(lr=self.learn_rate, epsilon=1e-4)

    self.network.compile(loss=huber_loss_mean, optimizer=opt)

