"""
Scripts to drive a donkey 2 car and train a model for it.

Usage:
    testbench.py [--path=<records_dir>] [--model=<model>]
    
Options:
    -h --help        Show this screen.
    --path TUBPATHS   Path of the record directory
    --model MODELPATH  Path of the model file
"""

from docopt import docopt
from donkeycar.parts.keras import CustomSequential
from PIL import Image

import numpy as np
import cv2
import glob
import json
import re

maxZ = 0.0
minZ = 0.0
maxX = 0.0
minX = 0.0
maxY = 0.0
minY = 0.0

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 160*4, 120*4)

font = cv2.FONT_HERSHEY_SIMPLEX


def drawOverlay(img, angle, throttle, acc_x = 0, acc_y = 0, acc_z = 0):

    height, width, channels = img.shape

    angle_color = (255, 0, 0) if angle > 0 else (255, 255, 0)
    angle_end = int(40 + (angle * 30))
    cv2.line(img, (10, 10), (70, 10), (255, 255, 255), 2)
    cv2.line(img, (40, 10), (angle_end, 10), angle_color, 2)
    cv2.putText(img, str(round(angle, 2)), (30, 20), font, 0.3, angle_color, 2)

    throttle_end = int(60 - throttle * 40)
    cv2.line(img, (10, 20), (10, 60), (255, 255, 255), 2)
    cv2.line(img, (10, throttle_end), (10, 60), (255, 0, 255), 2)

    cv2.putText(img, str(round(throttle, 2)), (20, 40), font, 0.3, (255, 0, 255), 2)

    origo_x = 40
    origo_y = height - 40
    radius = 30

    cv2.circle(img, (origo_x, origo_y), radius, (255,255,255), 1)

    pos_x = origo_x - int(round(acc_y * radius))
    pos_y = origo_y - int(round(acc_z * radius))
    size = max(1, 3 - int(round(acc_x * 2)))

    cv2.circle(img, (pos_x, pos_y), size, (255,0,0), -1)

    return img


def test(path, model_path = None):

    global minX, maxX, minY, maxY, maxZ, minZ

    kl = CustomSequential()
    if model_path:
        kl.load(model_path)

    acc_x = 0
    acc_y = 0
    acc_z = 0

    records = glob.glob('%s/record*.json' % path)
    records = ((int(re.search('.+_(\d+).json', path).group(1)), path) for path in records)

    for _, record in sorted(records):
        with open(record, 'r') as record_file:
            data = json.load(record_file)
            imgPath = data['cam/image_array']
            if not model_path:
                angle = data['user/angle']
                throttle = data['user/throttle']
                acc = data['gyro']['acceleration']
                acc_x = acc['x']
                acc_y = acc['y']
                acc_z = acc['z']
        img = Image.open('%s/%s' % (path, imgPath))
        img = np.array(img)
        if model_path:
            angle, throttle = kl.run(img)

        minY = min(minY, acc_y)
        maxY = max(maxY, acc_y)
        minZ = min(minZ, acc_z)
        maxZ = max(maxZ, acc_z)
        minX = min(minX, acc_x)
        maxX = max(maxX, acc_x)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (0,0), fx=2, fy=2)
        img = drawOverlay(img, angle, throttle, acc_x, acc_y, acc_z)
        cv2.imshow('image', img)

        # Draw overlay

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    args = docopt(__doc__)

    path = args['--path']
    model_path = args['--model']
    test(path, model_path)
    print(f'z: {minZ}, {maxZ}')
    print(f'y: {minY}, {maxY}')
    print(f'x: {minX}, {maxX}')