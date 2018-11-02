"""
Script to augment teaching data

Usage:
    augment.py --path=<records_dir> --out=<target_dir>

Options:
    -h --help        Show this screen.
    --path TUBPATHS   Path of the record directory
    --out MODELPATH  Path of the model file
"""

from docopt import docopt
from PIL import Image

import numpy as np

import cv2

import glob
import json
import re
import copy
import shutil
import os
from donkeycar.parts.opticalspeed import OpticalSpeed
from collections import deque

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def print_progress(count, total, name='', bar_length=20):
    if count % 10 == 0 or count == total:
        percent = 100 * (count / total)
        filled_length = int(bar_length * count / total)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        print('\r  %s\t |%s| %.1f%% %s' % (name, bar, percent, 'done'), end='\r')
    if count == total:
        print()

def initialize_records(records, path, out, target_dir):
    sum = 0

    if path is not out:
        target_path = '%s/%s' % (out, target_dir)
        ensure_directory(target_path)
        shutil.copy('%s/meta.json' % path, target_path)
    else:
        target_path = path

    for _, record in records:
        sum = sum + 1
        if path is not out:
            with open(record, 'r') as record_file:
                data = json.load(record_file)
                img_path = data['cam/image_array']
            shutil.copy(record, target_path)
            shutil.copy('%s/%s' % (path, img_path), target_path)

    return (sum, target_path)

# TODO: better place for global stuff
round_number = 0

def augmentation_round(in_path, out, total, name, augment_function, meta_function=None):
    global round_number
    round_number += 1
    target = '%s/%s_%s' % (out, round_number, name)
    records = glob.glob('%s/record*.json' % in_path)
    records = ((int(re.search('.+_(\d+).json', path).group(1)), path) for path in records)

    ensure_directory(target)
    if (meta_function is not None):
        with open('%s/meta.json' % in_path, 'r') as meta_file:
            raw_data = json.load(meta_file)
            new_data = meta_function(raw_data)
            with open('%s/meta.json' % target, 'w') as outfile:
                json.dump(new_data, outfile)
    else:
        shutil.copy('%s/meta.json' % in_path, target)

    count = 0

    for _, record in sorted(records):
        with open(record, 'r') as record_file:
            data = json.load(record_file)
            img_path = data['cam/image_array']
        img = Image.open('%s/%s' % (in_path, img_path))
        img = np.array(img)
        write(target, _, img, data, name, augment_function)
        count = count + 1
        print_progress(count, total, name)

    return (count, target)


def write(out, id, img, data, name, augment_function):

    new_img, new_data = augment_function(img, data)

    # Augment function can return None if this item should be skipped in the return set
    if (new_img is None or new_data is None):
        return

    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    record_path = '%s/record_%d.json' % (out, id)
    image_name = '%d_%s.jpg' % (id, name)
    image_path = '%s/%s' % (out, image_name)

    new_data['cam/image_array'] = image_name

    cv2.imwrite(image_path, new_img)

    with open(record_path, 'w') as outfile:
        json.dump(new_data, outfile)

optical_speed = None


def augment_speed(img, data):
    global optical_speed

    if optical_speed is None:
        optical_speed = OpticalSpeed()

    speed = optical_speed.run(img)

    data['optic/speed'] = float(speed)

    return (img, data)


def gen_meta_speed(old_meta):
    meta_with_speed = copy.deepcopy(old_meta)
    meta_with_speed['inputs'].append('optic/speed')
    meta_with_speed['types'].append('float')

    return meta_with_speed


def augment(target, out = None):

    print('Start adding speed to records')

    records = glob.glob('%s/record*.json' % target)
    records = ((int(re.search('.+_(\d+).json', path).group(1)), path) for path in records)

    # Directories starting with underscore are skipped in training. Originals have no history augmented so have to be skipped
    size, init_path = initialize_records(records, target, out, "_original")

    count = size

    if not out:
        out = target
    print('  Augmenting %d records from "%s". Target folder: "%s"' % (count, target, out))
    if target is not out:
        print('  Original files copies to "%s"', init_path)
    print('  -------------------------------------------------')
    size, history_path = augmentation_round(init_path, out, count, 'speed', augment_speed, gen_meta_speed)
    count = count + size
    print('  -------------------------------------------------')
    print('Augmentation done. Total records %s.' % count)


def is_empty(dir):
    return not os.listdir(dir)


if __name__ == '__main__':
    args = docopt(__doc__)

    target_path = args['--path']
    out_path = args['--out']

    if out_path:
        ensure_directory(out_path)

    if out_path and target_path is not out_path and not is_empty(out_path):
        print(' Target folder "%s" must be empty' % out_path)
    else:
        augment(target_path, out_path)
