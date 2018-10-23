#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it.

Usage:
    drive.py [--module=<module_name>] [--class=<class_name>] [--model=<model>] [--js] [--noebrake]

Options:
    -h --help        Show this screen.
    --js             Use physical joystick.
    --noebrake       Disable emergency brake

"""
from docopt import docopt
from garage import Garage

if __name__ == '__main__':
    args = docopt(__doc__)
    model_path = args['--model']
    use_joystick = args['--js']
    no_ebrake = args['--noebrake']
    module_name=args['--module']
    class_name=args['--class']
    vehicle = Garage.get_instance().create_vehicle(model_path, use_joystick, no_ebrake, module_name, class_name)
    vehicle.start()
    