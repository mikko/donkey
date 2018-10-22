"""
Scripts to drive a donkey 2 car and train a model for it.

Usage:
    testbench.py (--path=<validations_dir>) (--model=<model>)

Options:
    -h --help        Show this screen.
    --path TUBPATHS   Path of the validation record directory. Can be partial and does not require meta.json
    --model MODELPATH  Path of the model file
"""

from docopt import docopt
#from donkeycar.parts.keras import CustomWithHistory
from parts.keras import CustomWithHistory
from PIL import Image

import numpy as np
import glob
import json
import re
import math
import os

# in radians
START_ANGLE = math.radians(-90)
START_X = 0
START_Y = 0
MAX_ROTATION_ANGLE = math.radians(4)
MIN_THROTTLE = 0
THROTTLE_MULTIPLIER = 8

MAX_ANGLE_DIFF = 45
MAX_DISTANCE_DIFF = 200

PADDING = 5
MIN_WIDTH = 100
MIN_HEIGHT = 100

LINE_TEMPLATE = '<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1" marker-end="url(#%s)" />'

SVG_TEMPLATE = '''
<svg viewBox="%s %s %s %s" height="150" width="150">
  <defs>
    <marker id="arrow" markerWidth="3" markerHeight="6" refX="3" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L3,3 z" fill="#f33" />
    </marker>
  </defs>
  <defs>
    <marker id="arrow2" markerWidth="3" markerHeight="6" refX="3" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L3,3 z" fill="#333" />
    </marker>
  </defs>
  %s
  %s
</svg>
'''

RESULT_TEMPLATE = '''
    <div>
        <h3>%s</h3>
        %s
    </div>
'''

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: Roboto, Verdana, Sans-Serif;
            padding: 10px;
        }
    </style>
</head>
<body>
    <h1>Results</h1>
    %s
</body>
</html>                            
'''


def write_report(results):

    results_array = list(map(result_to_html, results))
    results_html = '\n<br>'.join(results_array)

    textFile = open("route.html", "w")
    textFile.write(HTML_TEMPLATE % results_html)
    textFile.close()


def result_to_html(result):

    min_x = 0.0
    max_x = 0.0
    min_y = 0.0
    max_y = 0.0

    last_point = None
    orig_lines = []
    model_lines = []

    for point in result["orig"]["points"]:
        if not last_point:
            min_x = point[0]
            max_x = point[0]
            min_y = point[1]
            max_y = point[1]
            last_point = point
        else:
            min_x = min(min_x, point[0])
            min_y = min(min_y, point[1])
            max_x = max(max_x, point[0])
            max_y = max(max_y, point[1])
            orig_lines.append(LINE_TEMPLATE % (last_point[0], last_point[1], point[0], point[1], "#f33", "arrow"))
            last_point = point

    last_point = None

    for point in result["model"]["points"]:
        if not last_point:
            last_point = point
            # starting point is same as with original. No need to check min & max
        else:
            min_x = min(min_x, point[0])
            min_y = min(min_y, point[1])
            max_x = max(max_x, point[0])
            max_y = max(max_y, point[1])
            orig_lines.append(LINE_TEMPLATE % (last_point[0], last_point[1], point[0], point[1], "#333", "arrow2"))
            last_point = point

    orig_html = '\n'.join(orig_lines)
    model_html = '\n'.join(model_lines)

    top = min_y - PADDING
    left = min_x - PADDING
    width = max(MIN_WIDTH, abs(max_x + PADDING - left))
    height = max(MIN_HEIGHT, abs(max_y + PADDING - top))

    print(min_x, min_y, max_x, max_y, result["name"], top, left, width, height)

    svg = SVG_TEMPLATE % (left, top, width, height, orig_html, model_html)

    html = RESULT_TEMPLATE % (result["name"], svg)

    return html


def drawVector(svgLines, currentAngle, currentPoint, speed, data, color, target):
    global MAX_ROTATION_ANGLE, MIN_THROTTLE, THROTTLE_MULTIPLIER

    angle = data['user/angle']
    throttle = data['user/throttle']

    newAngle = currentAngle + angle * MAX_ROTATION_ANGLE

    speedChange = throttle * MAX_SPEED_CHANGE - FRICTION

    newSpeed = min(MAX_SPEED, max(0, speed + speedChange))

    length = max(0, throttle * MAX_SPEED)

    dx = math.cos(newAngle) * length
    dy = math.sin(newAngle) * length

    newPoint = (currentPoint[0] + dx, currentPoint[1] + dy)

    svgLine = LINE_TEMPLATE % (currentPoint[0] * ZOOM, currentPoint[1] * ZOOM, newPoint[0] * ZOOM, newPoint[1] * ZOOM, color, target)

    svgLines.append(svgLine)

    return svgLines, newAngle, newPoint, newSpeed


def get_vectors(vectors, current_angle, current_point, data):
    global MAX_ROTATION_ANGLE, MIN_THROTTLE, THROTTLE_MULTIPLIER

    angle = data['user/angle']
    throttle = data['user/throttle']

    new_angle = current_angle + angle * MAX_ROTATION_ANGLE

    length = max(0, throttle * THROTTLE_MULTIPLIER)

    dx = math.cos(new_angle) * length
    dy = math.sin(new_angle) * length

    new_point = (current_point[0] + dx, current_point[1] + dy)
    vectors.append(new_point)

    return vectors, new_angle, new_point


def test_run(path, model_path = None):
    global START_ANGLE, START_X, START_Y, HTML_TEMPLATE

    kl = CustomWithHistory()
    if model_path:
        return ("a", 1, 2, 3, 4, 5, 6)

    records = glob.glob('%s/record*.json' % path)
    records = ((int(re.search('.+_(\d+).json', path).group(1)), path) for path in records)

    svgLines = []
    angle = START_ANGLE
    point = (START_X, START_Y)
    speed = 0.0

    svgLines2 = []
    angle2 = START_ANGLE
    point2 = (START_X, START_Y)
    speed2 = 0.0

    for _, record in sorted(records):
        with open(record, 'r') as record_file:
            data = json.load(record_file)
            img_path = data['cam/image_array']

        if model_path:
            img = Image.open('%s/%s' % (path, img_path))
            img = np.array(img)

            modelData = kl.run(img,             data['cam/image_array'],
            data['history/pilot/angle'],
            data['history/pilot/throttle'],
            data['history/acceleration/x'],
            data['history/acceleration/y'],
            data['history/acceleration/z'],
            data['history/sonar/left'],
            data['history/sonar/right'],
            data['history/sonar/center'])
            data2 = {"user/angle": modelData[0], "user/throttle": modelData[1]}
            (svgLines2, angle2, point2, speed2) = drawVector(svgLines2, angle2, point2, speed2, data2, "#00f", "arrow2")

        (svgLines, angle, point, speed) = drawVector(svgLines, angle, point, speed, data, "#f00", "arrow")
        # (svgLines2, angle2, point2, speed2) = drawVector(svgLines2, angle, point, speed, data2, "#00f", "arrow2")

    lines = "\n".join(svgLines)
    lines2 = "\n".join(svgLines2)

    textFile = open("route.html", "w")
    textFile.write(HTML_TEMPLATE % (lines, lines2))
    textFile.close()


def distance(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) * math.pow(a[1] - b[1], 2))


def determine_passing(angle_diff, distance_diff):
    return angle_diff < MAX_ANGLE_DIFF and distance_diff < MAX_DISTANCE_DIFF


def run_validation(name, path, model):
    records = glob.glob('%s/record*.json' % path)
    records = ((int(re.search('.+_(\d+).json', path).group(1)), path) for path in records)

    orig_angle = START_ANGLE
    orig_point = (START_X, START_Y)
    orig_array = [orig_point]

    model_angle = START_ANGLE
    model_point = (START_X, START_Y)
    model_array = [model_point]

    images = []

    for _, record in sorted(records):
        with open(record, 'r') as record_file:
            data = json.load(record_file)
            img_name = data['cam/image_array']

        img_path = '%s/%s' % (path, img_name)

        img = Image.open(img_path)
        img = np.array(img)

        data2 = model.run(img,
            data['history/user/angle'],
            data['history/user/throttle'],
            data['history/acceleration/x'],
            data['history/acceleration/y'],
            data['history/acceleration/z'],
            data['history/sonar/left'],
            data['history/sonar/right'],
            data['history/sonar/center'])

        model_data = {"user/angle": data2[0], "user/throttle": data2[1]}

        images.append(img_path)

        (orig_array, orig_angle, orig_point) = get_vectors(orig_array, orig_angle, orig_point, data)
        (model_array, model_angle, model_point) = get_vectors(model_array, model_angle, model_point, model_data)

    result = {}
    result["name"] = name
    result["images"] = images
    result["orig"] = {
        'points': orig_array,
        'start_angle': START_ANGLE,
        'end_angle': orig_angle,
    }
    result["model"] = {
        'points': model_array,
        'start_angle': START_ANGLE,
        'end_angle': model_angle,
    }
    result["angle_diff"] = abs(orig_angle - model_angle)
    result["distance_diff"] = distance(orig_point, model_point)
    result["passed"] = determine_passing(result["angle_diff"], result["distance_diff"])

    return result


def test(path, model_path):
    if not os.path.isdir(path):
        print('Given path %s does not exists or is not a directory' % path)
        return
    if not os.path.isfile(model_path):
        print("Given model %s does not exists or is not a file" % model_path)
        return

    model = CustomWithHistory()
    model.load(model_path)

    # Get all subdirectories
    entries = [(o, os.path.join(path, o)) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]

    # Run validation for each subdirectories
    results = list(map(lambda entry: run_validation(entry[0], entry[1], model), entries))

    # Write the results of each validation
    write_report(results)

if __name__ == '__main__':
    args = docopt(__doc__)

    path = args['--path']
    model_path = args['--model']

    test(path, model_path)
