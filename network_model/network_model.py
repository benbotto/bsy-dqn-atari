import numpy as np
import os.path
import tensorflow as tf
from abc import ABC, abstractmethod

class NetworkModel(ABC):
  '''
   ' Init.
  '''
  def __init__(self, model_file_name, env, name, learn_rate=5e-5, decay=0.0):
    self.obs_shape       = env.observation_space.shape
    self.act_size        = env.action_space.n
    self.model_file_name = model_file_name
    self.name            = name
    self.learn_rate      = learn_rate
    self.decay           = decay

  '''
   ' Create the underlying networks.  There are two: one for training and one
   ' for predicting.
  '''
  def create_network(self):
    with tf.name_scope(self.name):
      self.build()

      # Load the network from a save file.
      if os.path.isfile(self.model_file_name):
        self.load()

    return self

  '''
   ' Get the input shape.
  '''
  def get_input_shape(self):
    return self.obs_shape

  '''
   ' Build the model.  Override this in specific classes.
  '''
  @abstractmethod
  def build(self):
    pass

  '''
   ' Copy the weights from one NetworkModel to another.
  '''
  def copy_weights_to(self, target):
    print('Copying weights to target.')

    # Note that pred_network and train_network share layers and hence weights,
    # so it doesn't matter if pred_network or train_network is used here.
    target.pred_network.set_weights(self.pred_network.get_weights())

  '''
   ' Save the weights to a file.  The default file name, which is passed in to
   ' the ctor, can be overridden.
  '''
  def save(self, model_file_name=None):
    if model_file_name is None:
      print('Saving model weights to {}.'.format(self.model_file_name))
      self.pred_network.save_weights(self.model_file_name)
    else:
      print('Saving model weights to {}.'.format(model_file_name))
      self.pred_network.save_weights(model_file_name)

  '''
   ' Load the weights from a file.
  '''
  def load(self):
    print('Loading model weights from {}.'.format(self.model_file_name))
    self.pred_network.load_weights(self.model_file_name)

  '''
   ' Make a prediction based on an observation.
  '''
  def predict(self, obs):
    return self.pred_network.predict(obs)

  '''
   ' Train the model on a batch of inputs (observations) and expectations (reward
   ' predictions).  Gradients are weighted by importance sampling weights
   ' (is_weights).
  '''
  def train_on_batch(self, observations, expectations, is_weights):
    return self.train_network.fit(
      x=[observations, expectations, is_weights],
      batch_size=len(observations),
      verbose=0)

