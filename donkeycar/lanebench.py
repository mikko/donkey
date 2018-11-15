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
from parts.keras import CustomSequential
from PIL import Image

import numpy as np
import cv2
import glob
import json
import re
#import pyflow
import time
import math

# pyFlow Options:
alpha = 0.012
ratio = 0.75
minWidth = 10
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

MAX = math.pi * 0.8
MIN = math.pi * 0.2
MIN_MAG = 0.05

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 160*4, 120*4)

#src_tr = [169, 5]
#src_br = [239, 50]
#src_bl = [0, 50]
#src_tl = [69, 5]
src_tr = [160, 15]
src_br = [239, 50]
src_bl = [0, 50]
src_tl = [80, 15]

dst_tr = [229, 0]
dst_br = [229, 99]
dst_bl = [15, 99]
dst_tl = [15, 0]

def nothing(x):
    pass

cv2.createTrackbar("Speed", "image", 5, 10, nothing)

cv2.createTrackbar("S_Min", "image", 170, 255, nothing)
cv2.createTrackbar("S_Max", "image", 255, 255, nothing)

cv2.createTrackbar("H", "image", 0, 255, nothing)
cv2.createTrackbar("S", "image", 0, 255, nothing)
cv2.createTrackbar("L", "image", 0, 255, nothing)

font = cv2.FONT_HERSHEY_SIMPLEX

calib_mtx = None
calib_dist = None


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    isX = True if orient == 'x' else False
    sobel = cv2.Sobel(gray, cv2.CV_64F, isX, not isX)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    cv2.imshow("Sobel tresh", grad_binary)

    return grad_binary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    cv2.imshow("Mag tresh", mag_binary)

    return mag_binary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

    cv2.imshow("Dir tresh", dir_binary)

    return dir_binary

def apply_color_threshold(image, min = 170, max = 255):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_thresh_min = min
    s_thresh_max = max
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    cv2.imshow("Color tresh", s_binary)

    return s_binary

def combine_threshold(s_binary, combined):
    combined_binary = np.zeros_like(combined)
    combined_binary[(s_binary == 1) | (combined == 1)] = 1

    cv2.imshow("Combine tresh", combined_binary)

    return combined_binary


def apply_thresholds(image, ksize=3):
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    # print(gradx.shape)
    # print(grady.shape)
    # print(mag_binary.shape)
    # print(dir_binary.shape)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    cv2.imshow("Applied tresh", combined)

    return combined


def get_histogram(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    return histogram

def warp(img):
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
        [src_tr,
         src_br,
         src_bl,
         src_tl])

    dst = np.float32(
        [dst_tr,
         dst_br,
         dst_bl,
         dst_tl])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    binary_warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return binary_warped, Minv


def draw_guide_lines(img, color = (255, 0, 0)):
    cv2.line(img, (src_tr[0], src_tr[1]), (src_br[0], src_br[1]), color, 1)
    cv2.line(img, (src_br[0], src_br[1]), (src_bl[0], src_bl[1]), color, 1)
    cv2.line(img, (src_bl[0], src_bl[1]), (src_tl[0], src_tl[1]), color, 1)
    cv2.line(img, (src_tl[0], src_tl[1]), (src_tr[0], src_tr[1]), color, 1)
    return img


def check_side(points, side, debug=False, prefix="Log:"):
    intersection = np.intersect1d(np.unique(points), side)

    if debug:
        print(prefix, len(intersection), len(np.unique(points)))

    return len(intersection) > 20 or len(points) < 300

def slide_window(binary_warped, histogram):
    shape = binary_warped.shape
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = np.int(histogram.shape[0] / 2)
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = np.int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()

    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_x_current = left_x_base
    right_x_current = right_x_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    print("Start")
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = left_x_current - margin
        win_xleft_high = left_x_current + margin
        win_xright_low = right_x_current - margin
        win_xright_high = right_x_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            left_x_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            right_x_current = np.int(np.mean(nonzerox[good_right_inds]))
    print("End")

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_side_x = np.empty(dst_bl[1] - dst_tl[1] + 1, dtype=int)
    left_side_x.fill(round(dst_tl[0]))
    left_side_y = np.arange(dst_tl[1], dst_bl[1] + 1)
    right_side_x = np.empty(dst_br[1] - dst_tr[1] + 1, dtype=int)
    right_side_x.fill(round(dst_tr[0]))
    right_side_y = np.arange(dst_tr[1], dst_br[1] + 1)

    left_all_x = np.arange(0, midpoint)
    right_all_x = np.arange(midpoint + 1, shape[1])

    print(midpoint + 1, shape[1], "<>", left_all_x, right_all_x, len(left_all_x), len(right_all_x))

    fill_right = check_side(rightx, left_all_x, True, "right")
    fill_left = check_side(leftx, right_all_x, True, "left")

    if len(lefty) == 0 or len(leftx) == 0 or fill_left:
        leftx = left_side_x
        lefty = left_side_y
        print("FILL LEFT", fill_left)
    if len(righty) == 0 or len(rightx) == 0 or fill_right:
        rightx = right_side_x
        righty = right_side_y
        print("FILL RIGHT", fill_right)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    cv2.imshow("Out", out_img)
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)

    return ploty, left_fit, right_fit


def skip_sliding_window(binary_warped, left_fit, right_fit):
    shape = binary_warped.shape
    midpoint = shape[0] / 2
    print("Left_fit", left_fit)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    left_side_x = np.empty(dst_bl[1] - dst_tl[1] + 1, dtype=int)
    left_side_x.fill(dst_tl[0])
    left_side_y = np.arange(dst_tl[1], dst_bl[1] + 1)
    right_side_x = np.empty(dst_br[1] - dst_tr[1] + 1, dtype=int)
    right_side_x.fill(dst_tr[0])
    right_side_y = np.arange(dst_tr[1], dst_br[1] + 1)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_points = np.stack((leftx, lefty), axis=1)
    right_points = np.stack((rightx, righty), axis=1)

    left_all_x = np.arange(0, midpoint)
    right_all_x = np.arange(midpoint + 1, shape[0])

    fill_right = check_side(rightx, left_all_x)
    fill_left = check_side(leftx, right_all_x)

    if len(leftx) is 0 or len(lefty) is 0 or fill_left:
        leftx = left_side_x
        lefty = left_side_y
        print("FILL L")

    if len(rightx) is 0 or len(righty) is 0 or fill_right:
        rightx = right_side_x
        righty = right_side_y
        print("FILL R")

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ################################
    ## Visualization
    ################################

    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    cv2.imshow("fill", result)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)

    ret = {}
    ret['leftx'] = leftx
    ret['rightx'] = rightx
    ret['left_fitx'] = left_fitx
    ret['right_fitx'] = right_fitx
    ret['ploty'] = ploty

    return ret


def measure_curvature(ploty, lines_info):
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700

    leftx = lines_info['left_fitx']
    rightx = lines_info['right_fitx']

    leftx = leftx[::-1]
    rightx = rightx[::-1]

    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # print(left_curverad, 'm', right_curverad, 'm')

    return left_curverad, right_curverad


def draw_lane_lines(original_image, warped_image, m_inv, draw_info):
    leftx = draw_info['leftx']
    rightx = draw_info['rightx']
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    newwarp = cv2.warpPerspective(color_warp, m_inv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

    return result


def draw_lanes(image, s_min, s_max, h_setting, s_setting, l_setting):
    global calib_mtx, calib_dist

    if calib_mtx is None:
        data = np.load('calib.npz')
        calib_mtx = data['mtx']
        calib_dist = data['dist']

    # Undistort image
    image = cv2.undistort(image, calib_mtx, calib_dist, None, calib_mtx)

    img2, thing = warp(image)
    cv2.imshow("Warped", img2)

    # Gradient thresholding
    gradient_combined = apply_thresholds(image)

    # Color thresholding
    s_binary = apply_color_threshold(image, s_min, s_max)

    # Combine Gradient and Color thresholding
    combined_binary = combine_threshold(s_binary, gradient_combined)
    cv2.imshow("Combined Tresholds", combined_binary)

    # Transforming Perspective
    binary_warped, Minv = warp(combined_binary)
    cv2.imshow("Combined Warped", binary_warped)

    # Getting Histogram
    histogram = get_histogram(binary_warped)

    # Sliding Window to detect lane lines
    ploty, left_fit, right_fit = slide_window(binary_warped, histogram)

    # Skipping Sliding Window
    ret = skip_sliding_window(binary_warped, left_fit, right_fit)

    image = draw_lane_lines(image, binary_warped, Minv, ret)

    image = draw_guide_lines(image)

    return image


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

    if 0 <= impact_time < 1:
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

def draw_flow(img, flow, ang, mag, step=16):
    max = MAX
    min = MIN

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    val = ang[y, x]
    spd = mag[y, x]
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(img, lines, 0, (0, 0, 255))
    for i, line in enumerate(lines):
        line_angle = val[i]
        line_spd = spd[i]
        (x1, y1), (x2, y2) = line
        if min < line_angle < max and line_spd > MIN_MAG:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.circle(img, (x1, y1), 1, (0, 0, 255), -1)
        else:
            cv2.line(img, (x1, y1), (x2, y2), (96, 96, 96), 1)
            cv2.circle(img,(x1, y1), 1, (96, 96, 96), -1)



def calculate_angle(angle):
    max = MAX
    min = MIN

    angle = angle.flatten()

    indices = np.argwhere((angle >= min) & (angle <= max))

    return np.average(np.take(angle, indices))



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

        s_min = cv2.getTrackbarPos("S_Min", "image")
        s_max = cv2.getTrackbarPos("S_Max", "image")

        h_setting = cv2.getTrackbarPos("H", "image")
        s_setting = cv2.getTrackbarPos("S", "image")
        l_setting = cv2.getTrackbarPos("L", "image")

        s = time.time()
        img = draw_lanes(img, s_min, s_max, h_setting, s_setting, l_setting)
        e = time.time()
        print('Lane calculation took: %.2f sesconds' % (e - s))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (0,0), fx=2, fy=2)
        img = drawOverlay(img, data, only_outputs)
        cv2.imshow('image', img)

        speed_setting = cv2.getTrackbarPos("Speed", "image")
        delay = 1 + ((10 - speed_setting) * 20)

        # Draw overlay
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    args = docopt(__doc__)

    path = args['--path']
    model_path = args['--model']
    test(path, model_path)
