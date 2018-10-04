from collections import deque
from statistics import median

class EBrake:

  def __init__(self):
    self.buffer = deque(maxlen=3)
    print('Emergency brake started')

  def run(self, time_to_impact, raw_throttle):
    self.buffer.append(time_to_impact)
    # Use median from buffered values to avoid noise
    filtered_time = median(list(self.buffer))

    if (filtered_time > 0 and filtered_time < 1):
      print('\n\nEmergency brake used!!\n\n')
      return -0.2, True
    return raw_throttle, False

  def shutdown(self):
    print('Emergency brake stopped')
