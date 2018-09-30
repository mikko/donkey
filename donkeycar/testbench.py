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
from pyoptflow import HornSchunck

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 160*4, 120*4)

cv2.namedWindow('filtered', cv2.WINDOW_NORMAL)
cv2.resizeWindow('filtered', 160*4, 120*4)


font = cv2.FONT_HERSHEY_SIMPLEX


def opticalFlowDense(image_current, image_next):
    """
    input: image_current, image_next (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    * Set the saturation to the saturation value of image_next
    * Set the hue to the angles returned from computing the flow params
    * set the value to the magnitude returned from computing the flow params
    * Convert from HSV to RGB and return RGB image with same size as original image
    """

    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)

    hsv = np.zeros(image_current.shape)
    # set saturation
    hsv[:, :, 1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:, :, 1]

    # Flow Parameters
    flow_mat = cv2.CV_32FC2
    # flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 5
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0
    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,
                                        flow_mat,
                                        image_scale,
                                        nb_images,
                                        win_size,
                                        nb_iterations,
                                        deg_expansion,
                                        STD,
                                        0)

    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    #filtered = list(map(lambda m, a: m, mag, ang))
    #print(filtered)
    print(np.average(mag))

    # hue corresponds to direction
    hsv[:, :, 0] = ang * (180 / np.pi / 2)

    # hsv[:, :, 0] = ang * (90 / np.pi)

    # value corresponds to magnitude
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)


    # convert HSV to int32's
    hsv = np.asarray(hsv, dtype=np.float32)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # return rgb_flow
    return flow

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

def crop_lower(img,height):
    return img[img.shape[0] - height:, :]

def test(path):
    kl = CustomSequential()

    records = glob.glob('%s/record*.json' % path)
    records = ((int(re.search('.+_(\d+).json', path).group(1)), path) for path in records)
    prev_image = None
    prev_cropped = None
    for _, record in sorted(records):
        with open(record, 'r') as record_file:
            data = json.load(record_file)
            imgPath = data['cam/image_array']
            angle = data['user/angle']
            throttle = data['user/throttle']
        original_img = Image.open('%s/%s' % (path, imgPath))
        original_img = np.array(original_img)
        img = np.copy(original_img)
        img_crop = crop_lower(img, 60)
        # angle, throttle = kl.run(img)

        img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        img = drawOverlay(img, angle, throttle)
        cv2.imshow('image', img)
        if (prev_image is not None):
            opt_flow = opticalFlowDense(prev_image, original_img)
            filtered = np.copy(original_img)
            for y in range(0, prev_image.shape[0], 30):
                for x in range(0, prev_image.shape[1], 30):
                    flow = opt_flow[y][x]
                    print(flow)
                    cv2.circle(filtered, (x, y), 1, (0, 0, 255))
                    end_point = (int(x + flow[0] * 5), int(y + flow[1] * 5))
                    #print(end_point)
                    #cv2.line(filtered, (x, y), end_point, (255, 0, 0), 2)
                    # end_point = (x, int(y + flow[1] * 5))
                    cv2.line(filtered, (x, y), end_point, (0, 0, 255), 1)

            cv2.imshow('filtered', filtered)
        prev_image = original_img
        prev_cropped = img_crop

        # Draw overlay

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    args = docopt(__doc__)

    path = args['--path']
    test(path)
