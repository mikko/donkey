"""
Script to augment teaching data

Usage:
    augment.py [--path=<records_dir>] [--out=<target_dir>]

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

def print_progress(count, total, name='', bar_length=20):
    if count % 10 == 0 or count == total:
        percent = 100 * (count / total)
        filled_length = int(bar_length * count / total)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        print('\r  %s\t |%s| %.1f%% %s' % (name, bar, percent, 'done'), end='\r')
    if count == total:
        print()

def initialize_records(records, path, out):
    min_count = None
    max_count = 0
    sum = 0
    for _, record in records:
        max_count = max(max_count, _)
        if not min_count:
            min_count = _
        else:
            min_count = min(min_count, _)
        sum = sum + 1
        if path is not out:
            with open(record, 'r') as record_file:
                data = json.load(record_file)
                img_path = data['cam/image_array']
            shutil.copy(record, out)
            shutil.copy('%s/%s' % (path, img_path), out)

    return (min_count, max_count, sum)


def augmentation_round(out, start_number, total, name, augment_function):
    records = glob.glob('%s/record*.json' % out)
    records = ((int(re.search('.+_(\d+).json', path).group(1)), path) for path in records)

    current_id = start_number
    count = 0

    for _, record in sorted(records):
        with open(record, 'r') as record_file:
            data = json.load(record_file)
            imgPath = data['cam/image_array']

        img = Image.open('%s/%s' % (out, imgPath))
        img = np.array(img)
        write(out, current_id, img, data, name, augment_function)
        current_id = current_id + 1
        count = count + 1
        print_progress(count, total, name)

    return current_id


def write(out, id, img, data, name, augment_function):

    new_img, new_data = augment_function(img, data)

    record_path = '%s/record_%d.json' % (out, id)
    image_name = '%d_%s.jpg' % (id, name)
    image_path = '%s/%s' % (out, image_name)

    new_data['cam/image_array'] = image_name

    cv2.imwrite(image_path, new_img)

    with open(record_path, 'w') as outfile:
        json.dump(new_data, outfile)


def augment_flip(img, data):
    data = copy.deepcopy(data)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img, 1)

    data['user/angle'] = 0 - data['user/angle']
    data['gyro']['acceleration']['y'] = 0 - data['gyro']['acceleration']['y']
    data['gyro']['gyro']['y'] = 0 - data['gyro']['gyro']['y']

    return (img, data)


def augment_brightness(img, data):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    img[:,:,2] = img[:,:,2]*random_bright
    img[:,:,2][img[:,:,2]>255]  = 255
    img = np.array(img, dtype = np.uint8)
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)

    return (img, data)


def augment_shadow(img, data):
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][0]
    Y_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][1]

    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    # random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
    img = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return (img, data)


def augment(target, out = None):

    print('Start augmentation')

    records = glob.glob('%s/record*.json' % target)
    records = ((int(re.search('.+_(\d+).json', path).group(1)), path) for path in records)

    first_record, max_record, count = initialize_records(records, target, out)
    next_record = max_record + 1

    if not out:
        out = target
    print('  Augmenting %d records from "%s". Target folder: "%s"' % (count, target, out))

    print('  ---------------------')
    next_record = augmentation_round(out, next_record, count, 'flipped', augment_flip)
    count = next_record - first_record
    next_record = augmentation_round(out, next_record, count, 'bright', augment_brightness)
    count = next_record - first_record
    next_record = augmentation_round(out, next_record, count, 'shadow', augment_shadow)
    count = next_record - first_record
    print('  ---------------------')
    print('Augmentation done. %s new records created. Total records %s.' % (next_record - max_record - 1, count))


def is_empty(dir):
    return not os.listdir(dir)


if __name__ == '__main__':
    args = docopt(__doc__)

    target_path = args['--path']
    out_path = args['--out']

    if out_path and target_path is not out_path and not is_empty(out_path):
        print(' Target folder "%s" must be empty' % out_path)
    else:
        augment(target_path, out_path)
