import numpy as np

'''
 ' Huber loss.
 ' http://srome.github.io/A-Tour-Of-Gotchas-When-Implementing-Deep-Q-Networks-With-Keras-And-OpenAi-Gym/
 ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
 ' https://en.wikipedia.org/wiki/Huber_loss
 ' The first link has a nice math trick to avoid a tf.where.
'''
def huber_loss(y_true, y_pred, clip_delta=1.0):
  error          = np.abs(y_true - y_pred)
  quadratic_part = np.minimum(error, clip_delta)

  return 0.5 * np.square(quadratic_part) + clip_delta * (error - quadratic_part)

