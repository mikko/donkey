from collections import deque

class History:

  def __init__(self, buffer_size):
    self.on = True
    self.history_buffer = deque(maxlen=buffer_size)

  def run(self, value):
    self.history_buffer.append(value)
    return list(self.history_buffer)
