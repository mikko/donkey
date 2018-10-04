import serial

class Sonar:

  def __init__(self, device='/dev/ttyACM0'):
    self.device = device
    self.on = True
    self.state = {
      "left": 0,
      "center": 0,
      "right": 0,
      "timeToImpact": -1,
    }

  def run_threaded(self):
    return self.state["left"], self.state["center"], self.state["right"], self.state["timeToImpact"]

  def run(self):
    with serial.Serial(self.device, 115200, timeout=1) as ser:
      return ser.readline().decode("utf-8").split(" ")

  def shutdown(self):
    # indicate that the thread should be stopped
    self.on = False
    print('stopping sonar')

  def update(self):
    with serial.Serial(self.device, 115200, timeout=1) as ser:
      while self.on:
        line = ser.readline()   # read a '\n' terminated line
        print(line)
        values = line.split(" ")
        self.state = {
          "left": values[0],
          "center": values[1],
          "right": values[2],
          "timeToImpact": values[3]
        }

if __name__ == "__main__":
  import time
  iter = 0
  p = Sonar()
  while iter < 100:
    data = p.run()
    print(data)
    time.sleep(0.1)
    iter += 1
