import numpy as np
import os.path
import tensorflow as tf
from abc import ABC, abstractmethod
from network_model.loss import huber_loss, huber_loss_mean
from copy import deepcopy

class NetworkModel(ABC):
  '''
   ' Init.
  '''
  def __init__(self, model_file_name, env, name, learn_rate=1e-4, decay=9e-7):
    self.obs_shape       = env.observation_space.shape
    self.act_size        = env.action_space.n
    self.model_file_name = model_file_name
    self.name            = name
    self.learn_rate      = learn_rate
    self.decay           = decay

  '''
   ' Create the underlying network.
  '''
  def create_network(self):
    with tf.variable_scope(self.name):
      with tf.name_scope(self.name):
        # Load the network from a save file.
        if os.path.isfile(self.model_file_name):
          self.load()
        else:
          self.build()

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
    weights = deepcopy(self.network.get_weights())
    target.network.set_weights(weights)

  '''
   ' Save the weights to a file.  The default file name, which is passed in to
   ' the ctor, can be overridden.
  '''
  def save(self, model_file_name=None):
    if model_file_name is None:
      print('Saving model weights to {}.'.format(self.model_file_name))
    else:
      print('Saving model weights to {}.'.format(model_file_name))
      self.network.save(model_file_name)

  '''
   ' Load the weights from a file.
  '''
  def load(self):
    print('Loading model weights from {}.'.format(self.model_file_name))
    self.network = tf.keras.models.load_model(self.model_file_name,
      custom_objects={'huber_loss': huber_loss, 'huber_loss_mean': huber_loss_mean})

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

