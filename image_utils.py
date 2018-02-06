import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray

'''
 ' Preprocess an observation frame.
'''
def preprocess_frame(obs, width, height):
  return np.uint8(resize(rgb2gray(obs), (width, height)) * 255)

