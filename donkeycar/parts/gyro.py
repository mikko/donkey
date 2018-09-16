#!/usr/bin/python
import smbus
 
class Gyro:

  def __init__(self, address=0x68, bus=1):
    self.address = address
    self.power_mgmt_1 = 0x6b
    self.power_mgmt_2 = 0x6c
    self.acceleration_scale_modifier_2g = 16384.0
    self.bus = smbus.SMBus(bus)
    self.bus.write_byte_data(address, self.power_mgmt_1, 0)

  def run(self):
    gyro_x = self.read_word_2c(0x43)
    gyro_y = self.read_word_2c(0x45)
    gyro_z = self.read_word_2c(0x47)
    acceleration_x = self.read_word_2c(0x3b) / self.acceleration_scale_modifier_2g
    acceleration_y = self.read_word_2c(0x3d) / self.acceleration_scale_modifier_2g
    acceleration_z = self.read_word_2c(0x3f) / self.acceleration_scale_modifier_2g
    return {
      "gyro": {
        "x": gyro_x,
        "y": gyro_y,
        "z": gyro_z
      },
      "acceleration": {
        "x": acceleration_x,
        "y": acceleration_y,
        "z": acceleration_z
      }
    }
 
  def read_byte(self, reg):
    return self.bus.read_byte_data(self.address, reg)
 
  def read_word(self, reg):
    h = self.bus.read_byte_data(self.address, reg)
    l = self.bus.read_byte_data(self.address, reg+1)
    value = (h << 8) + l
    return value
 
  def read_word_2c(self, reg):
    val = self.read_word(reg)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val
