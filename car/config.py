"""
CAR CONFIG

This file is read by your car application's manage.py script to change the car
performance.

EXAMPPEL
-----------
import dk
cfg = dk.load_config(config_path='~/mycar/config.py')
print(cfg.CAMERA_RESOLUTION)

"""


import os

#PATHS
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, './data')
MODELS_PATH = os.path.join(CAR_PATH, './models')

#DEFAULT MODEL
DEFAULT_MODEL = "CustomSequential-205_teipattu_2_11"

#CAMERA
CAMERA_RESOLUTION = (180, 240) #(height, width)

#STEERING
STEERING_CHANNEL = 1
STEERING_LEFT_PWM = 400
STEERING_RIGHT_PWM = 290

#THROTTLE
THROTTLE_CHANNEL = 0
THROTTLE_FORWARD_PWM = 500
THROTTLE_STOPPED_PWM = 360
THROTTLE_REVERSE_PWM = 300

#TRAINING
BATCH_SIZE = 128
TRAIN_TEST_SPLIT = 0.8


#JOYSTICK
USE_JOYSTICK_AS_DEFAULT = False
JOYSTICK_MAX_THROTTLE = 0.5
JOYSTICK_STEERING_SCALE = 1.0
AUTO_RECORD_ON_THROTTLE = False
CRUISING_MODE_THROTTLE = 0.25


TUB_PATH = os.path.join(CAR_PATH, 'tub') # if using a single tub

#ROPE.DONKEYCAR.COM
ROPE_TOKEN="GET A TOKEN AT ROPE.DONKEYCAR.COM"
