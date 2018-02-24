import numpy as np
import os.path
import tensorflow as tf
from abc import ABC, abstractmethod
from network_model.loss import huber_loss, huber_loss_mean

class NetworkModel(ABC):
  '''
   ' Init.
  '''
  def __init__(self, model_file_name, env, learn_rate=5e-5):
    self.obs_shape       = env.observation_space.shape
    self.act_size        = env.action_space.n
    self.learn_rate      = learn_rate
    self.model_file_name = model_file_name

  '''
   ' Create the underlying network.
  '''
  def create_network(self):
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
    target.network.set_weights(self.network.get_weights())

  '''
   ' Save the weights to a file.
  '''
  def save(self):
    print('Saving model weights to {}.'.format(self.model_file_name))
    self.network.save(self.model_file_name)

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

