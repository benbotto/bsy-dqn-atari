import numpy as np
import tensorflow as tf

'''
 ' Huber loss.
 ' http://srome.github.io/A-Tour-Of-Gotchas-When-Implementing-Deep-Q-Networks-With-Keras-And-OpenAi-Gym/
 ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
 ' https://en.wikipedia.org/wiki/Huber_loss
 ' The first link has a nice math trick to avoid a tf.where.
'''
def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < clip_delta

  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

  return tf.where(cond, squared_loss, linear_loss)

'''
 ' Same as above but returns the mean loss.
'''
def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
  return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))

'''
 ' Importance Sampling weighted huber loss.
'''
def huber_loss_mean_weighted(y_true, y_pred, is_weights, clip_delta=1.0):
  error = huber_loss(y_true, y_pred, clip_delta)

  return tf.keras.backend.mean(error * is_weights)

