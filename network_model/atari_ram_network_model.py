import numpy as np
import tensorflow as tf
from network_model.network_model import NetworkModel
from network_model.loss import huber_loss_mean_weighted

class AtariRamNetworkModel(NetworkModel):
  '''
   ' Init.
  '''
  def __init__(self, model_file_name, env, name, learn_rate=1e-4, decay=0.0):
    super().__init__(model_file_name, env, name, learn_rate, decay)

  '''
   ' Build the model.
  '''
  def build(self):
    print('Building model.')

    # The observation input.
    in_obs = tf.keras.layers.Input(shape=self.get_input_shape())

    # When training, prioritized experience replay specifies that IS weights
    # need to be multiplied by the gradients.  As such, a custom loss function
    # is used, and that loss function depends on the IS weights and the
    # labels (i.e. y_true: what to compare the predictions to).
    in_is_weights = tf.keras.layers.Input(shape=(1,))
    in_actual     = tf.keras.layers.Input(shape=(self.act_size,))

    # Normalize the observation to the range of [0, 1].
    norm = tf.keras.layers.Lambda(lambda x: x / 255.0)(in_obs)

    # Hidden layers.
    dense1 = tf.keras.layers.Dense(128, activation="relu")(norm)
    dense2 = tf.keras.layers.Dense(128, activation="relu")(dense1)
    dense3 = tf.keras.layers.Dense(128, activation="relu")(dense2)
    dense4 = tf.keras.layers.Dense(128, activation="relu")(dense3)

    # Output prediction (i.e. y_pred).
    out_pred = tf.keras.layers.Dense(self.act_size, activation="linear")(dense4)

    # The DQN paper uses RMSProp, but its successor is used here.
    opt = tf.keras.optimizers.Adam(lr=self.learn_rate, decay=self.decay)

    # This network is used for training, and has three inputs (descibed above).
    self.train_network = tf.keras.models.Model(
      inputs=[in_obs, in_actual, in_is_weights],
      outputs=out_pred)

    # The custom loss, which is Huber Loss weighted by IS weights.
    self.train_network.add_loss(
      huber_loss_mean_weighted(out_pred, in_actual, in_is_weights))

    self.train_network.compile(optimizer=opt, loss=None)

    # This network is use for predicting, and just takes an observation.
    # Because it's not trained, it doesn't need to be compiled.
    self.pred_network = tf.keras.models.Model(inputs=in_obs, outputs=out_pred)

