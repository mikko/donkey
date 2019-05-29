import os
import logging
import platform
from threading import Lock

import donkeycar as dk

# NOTE: Not a bullet proof solution
isRaspberryPi = platform.machine()[:3] == 'arm'

#import parts
from donkeycar.util.loader import create_instance
from donkeycar.parts.transform import Lambda
from donkeycar.parts.datastore import DynamicTubWriter
from donkeycar.parts.controller import LocalWebController, JoystickController
from donkeycar.parts.clock import Timestamp
from donkeycar.parts.sonar import Sonar
from donkeycar.parts.ebrake import EBrake
#from donkeycar.parts.subwoofer import Subwoofer
from donkeycar.parts.history import History
from donkeycar.parts.T265 import T265

if isRaspberryPi:
    from donkeycar.parts.camera import PiCamera as Camera
    from donkeycar.parts.actuator import PCA9685 as Servoshield
    from donkeycar.parts.actuator import PWMSteering as Servo
    from donkeycar.parts.actuator import PWMThrottle as ESC
#    from donkeycar.parts.imu import Mpu6050 as IMU
else:
    from donkeycar.parts.camera import MockCamera as Camera
    from donkeycar.parts.actuator import MockPCA9685 as Servoshield
    from donkeycar.parts.actuator import MockServo as Servo
    from donkeycar.parts.actuator import MockEsc as ESC
    from donkeycar.parts.imu import MockImu as IMU

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

DEFAULT_PILOT_MODULE = "donkeycar.parts.keras"
DEFAULT_PILOT_CLASS = "CustomWithHistory"

class Garage:
    __instance = None

    @staticmethod
    def get_instance():
        if Garage.__instance == None:
            Garage()
        return Garage.__instance 

    def __init__(self):
        if Garage.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Garage.__instance = self
        self.vehicle = None
        self.configuration = self._load_configuration()
        self.lock = Lock()

    def _load_configuration(self):
        if (not "donkey_config" in os.environ):
            logging.info('Environment variable donkey_config missing')
            return
        config_path = os.environ['donkey_config']
        logging.info('Config path: {}'.format(config_path))
        return dk.load_config(config_path=config_path)

    def get_vehicle(self):
      return self.vehicle

    def create_vehicle(self, model_path=None, use_joystick=True, no_ebrake=False, module_name=None, class_name=None):
        """
        Construct a working robotic vehicle from many parts.
        Each part runs as a job in the Vehicle loop, calling either
        it's run or run_threaded method depending on the constructor flag `threaded`.
        All parts are updated one after another at the 20 fps framerate assuming each
        part finishes processing in a timely manner.
        Parts may have named outputs and inputs. The framework handles passing named outputs
        to parts requesting the same named input.
        """
        self.lock.acquire()
        if self.vehicle is not None:
          print("\n\nStopping existing vehicle\n\n")
          self.vehicle.stop()

        if not module_name:
            module_name = DEFAULT_PILOT_MODULE
        if not class_name:
            class_name = DEFAULT_PILOT_CLASS

        self.vehicle = dk.vehicle.Vehicle()

        clock = Timestamp()
        self.vehicle.add(clock, outputs=['timestamp'])

        cam = Camera(resolution=self.configuration.CAMERA_RESOLUTION)
        self.vehicle.add(cam, outputs=['cam/image_array'], threaded=True)

        t265 = T265()
        self.vehicle.add(t265,
            outputs=['t265/frame_l', 't265/self.frame_r', 't265/translation', 't265/acceleration', 't265/velocity', 't265/rotation'], 
            threaded=True)

        if use_joystick or self.configuration.USE_JOYSTICK_AS_DEFAULT:
            ctr = JoystickController(max_throttle=self.configuration.JOYSTICK_MAX_THROTTLE,
                                    steering_scale=self.configuration.JOYSTICK_STEERING_SCALE,
                                    auto_record_on_throttle=self.configuration.AUTO_RECORD_ON_THROTTLE)
        else:
            # This web controller will create a web server that is capable
            # of managing steering, throttle, and modes, and more.
            ctr = LocalWebController(use_chaos=False)

        self.vehicle.add(ctr,
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
        self.vehicle.add(pilot_condition_part, inputs=['user/mode'],
                                    outputs=['run_pilot'])

        if model_path is None and self.configuration.DEFAULT_MODEL:
            model_name = self.configuration.DEFAULT_MODEL
            class_name = model_name.split('-', 1)[0]
            model_path = "{}/{}".format(self.configuration.MODELS_PATH, model_name)

        # Run the pilot if the mode is not user.
        kl = create_instance(module_name, class_name)
        if model_path:
            kl.load(model_path)

 #       imu = IMU()
 #       self.vehicle.add(imu, outputs=['acceleration/x', 'acceleration/y', 'acceleration/z', 'gyro/x', 'gyro/y', 'gyro/z', 'temperature'], threaded=True)

        sonar = Sonar() # What if device changes?
        self.vehicle.add(sonar, outputs=['sonar/left', 'sonar/center', 'sonar/right', 'sonar/time_to_impact'], threaded=True)

        history_values = ['user/angle',
                          'user/throttle',
                          'acceleration/x',
                          'acceleration/y',
                          'acceleration/z',
                          'sonar/left',
                          'sonar/right',
                          'sonar/center',
                          'pilot/angle',
                          'pilot/throttle']

        for hist in history_values:
            # TODO: history length to constants
            hist_buffer = History(50)
            self.vehicle.add(hist_buffer, inputs=[hist], outputs=['history/%s' % hist])

        self.vehicle.add(kl,
                         inputs=kl.drive_inputs(),
                         outputs=['pilot/angle', 'pilot/throttle'],
                         run_condition='run_pilot')

        # Choose what inputs should change the car.
        def drive_mode(mode,
                      user_angle, user_throttle,
                      pilot_angle, pilot_throttle):
            if mode == 'user':
                return user_angle, user_throttle
            elif mode == 'local_angle':
                return pilot_angle, self.configuration.CRUISING_MODE_THROTTLE
            else:
                return pilot_angle, pilot_throttle

        drive_mode_part = Lambda(drive_mode)
        self.vehicle.add(drive_mode_part,
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'],
          outputs=['angle', 'raw_throttle'])

        steering_controller = Servoshield(self.configuration.STEERING_CHANNEL)
        steering = Servo(controller=steering_controller,
                        left_pulse=self.configuration.STEERING_LEFT_PWM,
                        right_pulse=self.configuration.STEERING_RIGHT_PWM)

        throttle_controller = Servoshield(self.configuration.THROTTLE_CHANNEL)
        throttle = ESC(controller=throttle_controller,
                    max_pulse=self.configuration.THROTTLE_FORWARD_PWM,
                    zero_pulse=self.configuration.THROTTLE_STOPPED_PWM,
                    min_pulse=self.configuration.THROTTLE_REVERSE_PWM)

        if (no_ebrake):
            self.vehicle.add(throttle, inputs=['raw_throttle'])
        else:
            emergency_brake = EBrake()
            self.vehicle.add(emergency_brake, inputs=['sonar/time_to_impact', 'raw_throttle'], outputs=['throttle', 'emergency_brake'])
            self.vehicle.add(throttle, inputs=['throttle'])

        self.vehicle.add(steering, inputs=['angle'])

 #       subwoofer = Subwoofer()
 #       self.vehicle.add(subwoofer, inputs=['user/mode', 'recording', 'emergency_brake'])

        # add tub to save data
        inputs = ['cam/image_array', 'user/angle', 'user/throttle', 'user/mode', 'timestamp', 'acceleration/x', 'acceleration/y', 'acceleration/z', 'gyro/x', 'gyro/y', 'gyro/z', 'temperature', 'sonar/left', 'sonar/center', 'sonar/right', 'sonar/time_to_impact', 't265/velocity']
        types = ['image_array', 'float', 'float',  'str', 'str', 'str', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', '3d']

        #multiple tubs
        #th = TubHandler(path=self.configuration.DATA_PATH)
        #tub = th.new_tub_writer(inputs=inputs, types=types)

        tub_inputs = ['recording'] + inputs
        tub = DynamicTubWriter(path=self.configuration.TUB_PATH, inputs=inputs, types=types)
        self.vehicle.add(tub, inputs=tub_inputs)

        self.lock.release()

        return self.vehicle
