import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from matplotlib import pyplot as plt

'''
 ' Preprocess an observation frame.
'''
def preprocess_frame(obs, width, height):
  return np.uint8(resize(rgb2gray(obs), (width, height)) * 255)

'''
 ' Preprocess an observation frame with cropping.
'''
def preprocess_frame_crop(obs, width, height):
  gray = rgb2gray(obs)
  down = resize(gray, (110, 84)) * 255
  crop = down[26:110, 0:84]
  return np.uint8(crop)

'''
 ' Helper function to show a processed frame.abs.
'''
def display_frame(frame):
  plt.imshow(frame, cmap=plt.cm.gray)
  plt.show()

