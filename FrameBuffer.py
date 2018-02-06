from collections import deque

class FrameBuffer:
  '''
   ' Ini.t
  '''
  def __init__(self, stack_size):
    self.stack_size = stack_size
    self._frames    = deque([], maxlen=self.stack_size)

  '''
   ' Add a frame and return the stack.
  '''
  def add_frame(self, frame):
    if len(self._frames) == 0:
      for _ in range(self.stack_size):
        self._frames.appendleft(frame)
    else:
      self._frames.appendleft(frame)

    return self.get_frame_stack()

  '''
   ' Get a stack of frames.
  '''
  def get_frame_stack(self):
    return [frame for frame in self._frames]

