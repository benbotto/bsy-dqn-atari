from ReplayMemory import ReplayMemory
from FrameBuffer import FrameBuffer

def main():
  frames = FrameBuffer(4)

  frames.add_frame(1)

  stack = frames.get_frame_stack()
  print(stack)

  frames.add_frame(2)
  stack = frames.get_frame_stack()
  print(stack)

  frames.add_frame(3)
  frames.add_frame(4)
  frames.add_frame(5)
  stack = frames.get_frame_stack()
  print(stack)

  print(frames.add_frame(6))

if __name__ == "__main__":
  main()

