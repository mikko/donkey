"""
Scripts to drive a donkey 2 car and train a model for it.

Usage:
    testbench.py [--path=<records_dir>]
    
Options:
    -h --help        Show this screen.
    --path TUBPATHS   Path of the record directory
"""

from docopt import docopt

from donkeycar.parts.keras import CustomSequential


import numpy as np
import cv2
import glob
import json
import re
from PIL import Image

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 160*4, 120*4)

font = cv2.FONT_HERSHEY_SIMPLEX

def drawOverlay(img, angle, throttle):
    angle_color = (255, 0, 0) if angle > 0 else (255, 255, 0)
    angle_end = int(40 + (angle * 30))
    cv2.line(img, (10, 10), (70, 10), (255, 255, 255), 2)
    cv2.line(img, (40, 10), (angle_end, 10), angle_color, 2)
    cv2.putText(img, str(round(angle, 2)), (30, 20), font, 0.3, angle_color, 2)

    throttle_end = int(60 - throttle * 40)
    cv2.line(img, (10, 20), (10, 60), (255, 255, 255), 2)
    cv2.line(img, (10, throttle_end), (10, 60), (255, 0, 255), 2)

    cv2.putText(img, str(round(throttle, 2)), (20, 40), font, 0.3, (255, 0, 255), 2)

    return img

def test(path):
    kl = CustomSequential()

    records = glob.glob('%s/record*.json' % path)
    records = ((int(re.search('.+_(\d+).json', path).group(1)), path) for path in records)

    for _, record in sorted(records):
        with open(record, 'r') as record_file:
            data = json.load(record_file)
            imgPath = data['cam/image_array']
            angle = data['user/angle']
            throttle = data['user/throttle']
        img = Image.open('%s/%s' % (path, imgPath))
        img = np.array(img)
        # angle, throttle = kl.run(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = drawOverlay(img, angle, throttle)
        cv2.imshow('image', img)

        # Draw overlay

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    args = docopt(__doc__)

    path = args['--path']
    test(path)
