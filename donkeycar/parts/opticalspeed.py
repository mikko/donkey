import numpy as np
import cv2
import math

MAX_ANG = math.pi * 0.8
MIN_ANG = math.pi * 0.2
MIN_MAG = 0.05

# TODO Self_threaded?

class OpticalSpeed:

  def __init__(self):
    self.prev_img = None
    self.speed = 0.0

  def run(self, img):
    img = np.array(img) # check if needed
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if self.prev_img is None:
      self.prev_img = img
    else:
      flow = cv2.calcOpticalFlowFarneback(self.prev_img, img, None,  0.5, 3, 15, 3, 5, 1.2, 0)
      mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

      ang = ang.flatten()
      mag = mag.flatten()

      # Filter only those points going from front to back
      indices = np.argwhere((ang > MIN_ANG) & (ang < MAX_ANG))
      target_mag = np.take(mag, indices)
      target_ind = np.argwhere(target_mag > MIN_MAG)

      # Filter out those magnitudes that are not relevant
      speed = np.average(np.take(target_mag, target_ind))
      self.speed = speed

    return self.speed
