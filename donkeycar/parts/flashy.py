import os
# http://pyserial.sourceforge.net/
import serial

class Flashy:
  def flash_on(self):
    print('Should turn on flashy lights')
    self.ser.write(b'1')

  def flash_off(self):
    print('Should turn off flashy lights')
    self.ser.write(b'-1')

  def __init__(self, device='/dev/ttyACM0'):
    self.on = True
    self.mode = ''
    self.error = False
    try:
       self.ser = serial.Serial(device, 9600)
    except:
       self.error = True
    print("Starting flashy")

  def run(self, mode):
    if self.on and self.error is False:
        prev_mode = self.mode
        self.mode = mode

        # Start AI music if mode changed and is now "local"
        if (prev_mode != mode): # local mode means AI driven
            if ((mode == "local" or mode == "local_angle")):
                self.flash_on()
            else:
                self.flash_off()

  def shutdown(self):
    # indicate that the thread should be stopped
    self.on = False
    print('stopping flashy')

if __name__ == "__main__":
    print("Instantiating directly not implemented")

