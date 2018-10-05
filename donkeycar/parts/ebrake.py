from collections import deque
from statistics import median

class EBrake:

  def __init__(self):
    self.buffer = deque(maxlen=5)
    self.abs = True # Used for alternating reverse and zero throttle
    print('Emergency brake started')

  def run(self, time_to_impact, raw_throttle):
    self.buffer.append(time_to_impact)
    # Use median from buffered values to avoid noise
    filtered_time = median(list(self.buffer))

    if (filtered_time > 0 and filtered_time < 1):
      throttle = 0
      if (self.abs):
        throttle = -1
      self.abs = not self.abs
      print('\n\nEmergency brake used!!', throttle, '\n\n')
      return throttle, True
    return raw_throttle, False

  def shutdown(self):
    print('Emergency brake stopped')
