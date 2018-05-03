import numpy as np
import tensorflow as tf
from network_model.network_model import NetworkModel
from network_model.loss import huber_loss_mean_weighted, huber_loss_mean

class AtariNetworkModel(NetworkModel):
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

    # The observation input.
    in_obs = tf.keras.layers.Input(shape=self.get_input_shape())

    # Reference atari_ram_network_model for notes about IS weights and
    # in_actual, and the custom loss that's used.
    in_is_weights = tf.keras.layers.Input(shape=(1,))
    in_actual     = tf.keras.layers.Input(shape=(self.act_size,))

    # Normalize the observation to the range of [0, 1].
    norm = tf.keras.layers.Lambda(lambda x: x / 255.0)(in_obs)

    # Convolutional layers per the Nature paper on DQN.
    conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4,
      activation="relu", data_format="channels_first")(norm)
    conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2,
      activation="relu")(conv1)
    conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
      activation="relu")(conv2)

    # Flatten, and move to the fully-connected part of the network.
    flatten = tf.keras.layers.Flatten()(conv3)
    dense1  = tf.keras.layers.Dense(512, activation="relu")(flatten)

    # Output prediction.
    out_pred = tf.keras.layers.Dense(self.act_size, activation="linear")(dense1)

    # Using Adam optimizer, RMSProp's successor.
    opt = tf.keras.optimizers.Adam(lr=self.learn_rate, decay=self.decay)

    # This network is used for training.
    self.train_network = tf.keras.models.Model(
      inputs=[in_obs, in_actual, in_is_weights],
      outputs=out_pred)

    # The custom loss, which is Huber Loss weighted by IS weights.
    self.train_network.add_loss(
      huber_loss_mean_weighted(out_pred, in_actual, in_is_weights))

    self.train_network.compile(optimizer=opt, loss=None)

    # This network is use for predicting.
    self.pred_network = tf.keras.models.Model(inputs=in_obs, outputs=out_pred)

    #self.pred_network.summary()

