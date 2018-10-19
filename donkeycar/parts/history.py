from collections import deque

class History:

  def __init__(self, buffer_size):
    self.on = True
    self.history_buffer = deque(maxlen=buffer_size)
    for i in range(buffer_size):
      self.history_buffer.append(0)

  def run(self, value):
    if (value != None):
      self.history_buffer.append(value)
    return list(self.history_buffer)
