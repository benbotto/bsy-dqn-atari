import numpy as np
import tensorflow as tf

'''
 ' Huber loss, importance sampling weighted.
 ' http://srome.github.io/A-Tour-Of-Gotchas-When-Implementing-Deep-Q-Networks-With-Keras-And-OpenAi-Gym/
 ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
 ' https://en.wikipedia.org/wiki/Huber_loss
 ' The first link has a nice math trick to avoid a tf.keras.backend.switch.
'''
def huber_loss_mean_weighted(y_true, y_pred, is_weights):
  error = tf.keras.backend.abs((y_true - y_pred) * is_weights)
  cond  = error <= 1.0

  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = error - 0.5

  return tf.keras.backend.mean(
    tf.keras.backend.switch(cond, squared_loss, linear_loss))

