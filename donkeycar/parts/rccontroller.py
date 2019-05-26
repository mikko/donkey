from serial import Serial
from numpy import interp

#TODO: move to configuration
MIN_STEERING = 2310
MAX_STEERING = 3720
MIN_THROTTLE = 2210
MAX_THROTTLE = 3900
MIN_MODE = 2210
MAX_MODE = 3820

MODES = ["local_angle", "user", "local"]

class RCController:

  def __init__(self, device='/dev/ttyUSB0'):
    self.device = device
    self.on = True
    self.serial = Serial(device, 115200, timeout=1)
    
    self.state = {
      "steering": 0,
      "throttle": 0,
      "mode": MODES[1],
      "recording": False
    }

  def run_threaded(self):
    return self.state["angle"], self.state["throttle"], self.state["mode"], self.state["recording"]

  def run(self):
    array = self.serial.readline().decode("utf-8").split(",")
    if len(array) >= 2:
      values = list(map(int, array))
      steering = interp(values[0], [MIN_STEERING, MAX_STEERING], [-1, 1])
      throttle = interp(values[1], [MIN_THROTTLE, MAX_THROTTLE], [-1, 1])
      mode = MODES[round(interp(values[2], [MIN_MODE, MAX_MODE], [0, 2]))]
      return {
        "steering": steering,
        "throttle": throttle,
        "mode": mode
      }

  def shutdown(self):
    self.on = False

  def update(self):
    while self.on:
      array = self.serial.readline().decode("utf-8").split(",")
      if len(array) >= 2:
        values = list(map(int, array))
        self.state["steering"] = interp(values[0], [MIN_STEERING, MAX_STEERING], [-1, 1])
        self.state["throttle"] = interp(values[1], [MIN_THROTTLE, MAX_THROTTLE], [-1, 1])
        self.state["mode"] = MODES[round(interp(values[2], [MIN_MODE, MAX_MODE], [0, 2]))]

if __name__ == "__main__":
  import time
  iter = 0
  p = RCController()
  while True:
    data = p.run()
    print("{}".format(data))
    iter += 1