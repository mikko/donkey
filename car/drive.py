#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it.

Usage:
    drive.py [--model=<model>] [--js]

Options:
    -h --help        Show this screen.
    --js             Use physical joystick.
    --noebrake       Disable emergency brake

"""
import os
import logging
from docopt import docopt

import donkeycar as dk

#import parts
from donkeycar.parts.camera import PiCamera
from donkeycar.parts.transform import Lambda
from donkeycar.parts.keras import CustomSequential
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
from donkeycar.parts.datastore import TubGroup, TubWriter
from donkeycar.parts.controller import LocalWebController, JoystickController
from donkeycar.parts.clock import Timestamp
from donkeycar.parts.imu import Mpu6050
from donkeycar.parts.sonar import Sonar
from donkeycar.parts.ebrake import EBrake
from donkeycar.parts.subwoofer import Subwoofer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

def _drive(cfg, model_path=None, use_joystick=False, no_ebrake=False):
    """
    Construct a working robotic vehicle from many parts.
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a timely manner.
    Parts may have named outputs and inputs. The framework handles passing named outputs
    to parts requesting the same named input.
    """

    V = dk.vehicle.Vehicle()

    clock = Timestamp()
    V.add(clock, outputs='timestamp')

    cam = PiCamera(resolution=cfg.CAMERA_RESOLUTION)
    V.add(cam, outputs=['cam/image_array'], threaded=True)

    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
        ctr = JoystickController(max_throttle=cfg.JOYSTICK_MAX_THROTTLE,
                                 steering_scale=cfg.JOYSTICK_STEERING_SCALE,
                                 auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE)
    else:
        # This web controller will create a web server that is capable
        # of managing steering, throttle, and modes, and more.
        ctr = LocalWebController(use_chaos=False)

    V.add(ctr,
          inputs=['cam/image_array'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)

    # See if we should even run the pilot module.
    # This is only needed because the part run_condition only accepts boolean
    def pilot_condition(mode):
        if mode == 'user':
            return False
        else:
            return True

    pilot_condition_part = Lambda(pilot_condition)
    V.add(pilot_condition_part, inputs=['user/mode'],
                                outputs=['run_pilot'])

    # Run the pilot if the mode is not user.
    kl = CustomSequential()
    if model_path:
        kl.load(model_path)

    V.add(kl, inputs=['cam/image_array'],
              outputs=['pilot/angle', 'pilot/throttle'],
              run_condition='run_pilot')

    # Choose what inputs should change the car.
    def drive_mode(mode,
                   user_angle, user_throttle,
                   pilot_angle, pilot_throttle):
        if mode == 'user':
            return user_angle, user_throttle

        elif mode == 'local_angle':
            return pilot_angle, user_throttle

        else:
            return pilot_angle, pilot_throttle

    drive_mode_part = Lambda(drive_mode)
    V.add(drive_mode_part,
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'],
          outputs=['angle', 'raw_throttle'])

    sonar = Sonar() # What if device changes?
    V.add(sonar, outputs=['sonar/left', 'sonar/center', 'sonar/right', 'sonar/time_to_impact'], threaded=True)

    steering_controller = PCA9685(cfg.STEERING_CHANNEL)
    steering = PWMSteering(controller=steering_controller,
                           left_pulse=cfg.STEERING_LEFT_PWM,
                           right_pulse=cfg.STEERING_RIGHT_PWM)

    throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL)
    throttle = PWMThrottle(controller=throttle_controller,
                           max_pulse=cfg.THROTTLE_FORWARD_PWM,
                           zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                           min_pulse=cfg.THROTTLE_REVERSE_PWM)

    if (no_ebrake):
        V.add(throttle, inputs=['raw_throttle'])
    else:
        emergency_brake = EBrake()
        V.add(emergency_brake, inputs=['sonar/time_to_impact', 'raw_throttle'], outputs=['throttle', 'emergency_brake'])
        V.add(throttle, inputs=['throttle'])

    V.add(steering, inputs=['angle'])


    mpu6050 = Mpu6050()
    V.add(mpu6050, outputs=['acceleration/x', 'acceleration/y', 'acceleration/z', 'gyro/x', 'gyro/y', 'gyro/z', 'temperature'], threaded=True)

    subwoofer = Subwoofer()
    V.add(subwoofer, inputs=['user/mode', 'recording', 'emergency_brake'])

    # add tub to save data
    inputs = ['cam/image_array', 'user/angle', 'user/throttle', 'user/mode', 'timestamp', 'acceleration/x', 'acceleration/y', 'acceleration/z', 'gyro/x', 'gyro/y', 'gyro/z', 'temperature', 'sonar/left', 'sonar/center', 'sonar/right', 'sonar/time_to_impact']
    types = ['image_array', 'float', 'float',  'str', 'str', 'str', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']

    #multiple tubs
    #th = TubHandler(path=cfg.DATA_PATH)
    #tub = th.new_tub_writer(inputs=inputs, types=types)

    # single tub
    tub = TubWriter(path=cfg.TUB_PATH, inputs=inputs, types=types)
    V.add(tub, inputs=inputs, run_condition='recording')

    # run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS)

def start_drive(model_path=None, use_joystick=True, no_ebrake=False):
    if (not "donkey_config" in os.environ):
        logging.info('Environment variable donkey_config missing')
        return
    config_path = os.environ['donkey_config']
    logging.info('Config path: {}'.format(config_path))
    cfg = dk.load_config(config_path=config_path)
    _drive(cfg, model_path, use_joystick, no_ebrake)

if __name__ == '__main__':
    args = docopt(__doc__)

    start_drive(model_path = args['--model'],
                use_joystick=args['--js'],
                no_ebrake=args['--noebrake'])
