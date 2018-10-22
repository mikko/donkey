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
import glob
import json
import re
import math

# in radians
START_ANGLE = math.radians(-90)
START_X = 3000
START_Y = 3000
MAX_ROTATION_ANGLE = math.radians(4)
MIN_THROTTLE = 0

MAX_SPEED_CHANGE = 6
MAX_SPEED = 100
FRICTION = 1

ZOOM = 1

#LINE_TEMPLATE = '<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="2" marker-end="url(#arrow)" />'
LINE_TEMPLATE = '<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="2" marker-end="url(#%s)" />'
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<body>

<svg height="4000" width="4000">
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

</body>
</html>                            
'''


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


def test(path, model_path = None):
    global START_ANGLE, START_X, START_Y, HTML_TEMPLATE

    kl = CustomSequential()
    if model_path:
        kl.load(model_path)

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
            modelData = kl.run(img)
            data2 = {"user/angle": modelData[0], "user/throttle": modelData[1]}
            (svgLines2, angle2, point2, speed2) = drawVector(svgLines2, angle2, point2, speed2, data2, "#00f", "arrow2")

        (svgLines, angle, point, speed) = drawVector(svgLines, angle, point, speed, data, "#f00", "arrow")
        # (svgLines2, angle2, point2, speed2) = drawVector(svgLines2, angle, point, speed, data2, "#00f", "arrow2")

    lines = "\n".join(svgLines)
    lines2 = "\n".join(svgLines2)

    textFile = open("route.html", "w")
    textFile.write(HTML_TEMPLATE % (lines, lines2))
    textFile.close()

if __name__ == '__main__':
    args = docopt(__doc__)

    path = args['--path']
    model_path = args['--model']
    test(path, model_path)
