'''
 ' Helper for tracking lives.
'''
class LifeTracker:
  '''
   ' Init with no lives.
  '''
  def __init__(self):
    self.lives = 0

  '''
   ' Store the life count, if available.  Info comes from an ale step.
  '''
  def track_lives(self, info):
    if 'ale.lives' in info:
      self.lives = info['ale.lives']

    return self.lives

  '''
   ' Reset the life count.
  '''
  def reset(self):
    self.lives = 0
  
