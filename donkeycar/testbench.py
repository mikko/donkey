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

def drawAngleBar(img, data):
    angle = data['user/angle']

    angle_color = (255, 0, 0) if angle > 0 else (255, 255, 0)
    angle_end = int(40 + (angle * 30))
    cv2.line(img, (10, 10), (70, 10), (255, 255, 255), 2)
    cv2.line(img, (40, 10), (angle_end, 10), angle_color, 2)
    cv2.putText(img, str(round(angle, 2)), (30, 20), font, 0.3, angle_color, 2)


def drawThrottleBar(img, data):
    throttle = data['user/throttle']

    throttle_end = int(60 - throttle * 40)
    cv2.line(img, (10, 20), (10, 60), (255, 255, 255), 2)
    cv2.line(img, (10, throttle_end), (10, 60), (255, 0, 255), 2)

    cv2.putText(img, str(round(throttle, 2)), (20, 40), font, 0.3, (255, 0, 255), 2)


def drawAccRadar(img, height, data):
    global minY, maxY, minZ, maxZ, minX, maxX

    acc_x = data['acceleration/x']
    acc_y = data['acceleration/y']
    acc_z = data['acceleration/z']

    origo_x = 40
    origo_y = height - 40
    radius = 30

    cv2.circle(img, (origo_x, origo_y), radius, (255, 255, 255), 1)

    pos_x = origo_x - int(round((acc_y / 10) * radius))
    pos_y = origo_y - int(round((acc_z / 10) * radius))
    size = max(1, 3 - int(round((acc_x / 10) * 2)))

    cv2.circle(img, (pos_x, pos_y), size, (255, 0, 0), -1)

    minY = min(minY, acc_y)
    maxY = max(maxY, acc_y)
    minZ = min(minZ, acc_z)
    maxZ = max(maxZ, acc_z)
    minX = min(minX, acc_x)
    maxX = max(maxX, acc_x)


def drawSector(img, origo, startAngle, endAngle, colors):
    s_size = (10, 10)
    m_size = (25, 25)
    l_size = (40, 40)

    cv2.ellipse(img, origo, l_size, 180, startAngle, endAngle, colors[2], -1)
    cv2.ellipse(img, origo, m_size, 180, startAngle, endAngle, colors[1], -1)
    cv2.ellipse(img, origo, s_size, 180, startAngle, endAngle, colors[0], -1)


def getSectorColors(value):
    blank_color = (128, 128, 128)
    far_color = (96, 255, 96)
    med_color = (96, 255, 255)
    near_color = (64, 64, 255)

    if value == 0:
        return [far_color, far_color, far_color]
    elif value <= 5:
        return [near_color, blank_color, blank_color]
    elif value <= 100:
        return [med_color, med_color, blank_color]
    else:
        return [far_color, far_color, far_color]


def drawProximitySensor(img, width, height, data):

    left = data['sonar/left']
    center = data['sonar/center']
    right = data['sonar/right']
    impact_time = data['sonar/time_to_impact']

    origo_x = width - 40
    origo_y = height - 30

    drawSector(img, (origo_x-2, origo_y), 45, 75, getSectorColors(left))
    drawSector(img, (origo_x, origo_y), 75, 105, getSectorColors(center))
    drawSector(img, (origo_x+2, origo_y), 105, 135, getSectorColors(right))

    if impact_time >= 0 and impact_time < 1:
        cv2.putText(img, 'STOP', (origo_x - 20, origo_y + 15), font, 0.5, (0,0,255), 2)


def drawOverlay(img, data, only_outputs):

    height, width, channels = img.shape

    # Angle bar on top
    drawAngleBar(img, data)

    # Throttle bar at right
    drawThrottleBar(img, data)

    if not only_outputs:
        # Acceleration radar at bottom left
        drawAccRadar(img, height, data)

        # Proximity sensor at bottom right
        drawProximitySensor(img, width, height, data)

    return img


def test(path, model_path = None):

    kl = CustomSequential()
    if model_path:
        kl.load(model_path)

    records = glob.glob('%s/record*.json' % path)
    records = ((int(re.search('.+_(\d+).json', path).group(1)), path) for path in records)

    for _, record in sorted(records):
        with open(record, 'r') as record_file:
            data = json.load(record_file)
            img_path = data['cam/image_array']
        img = Image.open('%s/%s' % (path, img_path))
        img = np.array(img)
        only_outputs = False
        if model_path:
            data = kl.run(img)
            only_outputs = True

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (0,0), fx=2, fy=2)
        img = drawOverlay(img, data, only_outputs)
        cv2.imshow('image', img)

        # Draw overlay
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    args = docopt(__doc__)

    path = args['--path']
    model_path = args['--model']
    test(path, model_path)
    # y up-down
    # z left-right
    # x gravity
    print(f'z: {minZ}, {maxZ}')
    print(f'y: {minY}, {maxY}')
    print(f'x: {minX}, {maxX}')