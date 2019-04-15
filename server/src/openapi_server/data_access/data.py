from os import listdir
from os.path import isdir, isfile, dirname, realpath, join
from datetime import datetime
import os.path
import time
import json
import glob
import re
from openapi_server.models.car import Car  # noqa: E501
from openapi_server.models.tub import Tub  # noqa: E501
from openapi_server.models.tub_data_point import TubDataPoint  # noqa: E501

data_path = join(dirname(realpath(__file__)), "../../../data")
print(data_path)
# data_path = 'data'

"""Return all available cars
"""


def load_cars():
    car_dirs = get_sub_dirs(data_path)
    return [Car(dir, dir) for dir in car_dirs]


"""Return a car with given id, or None
"""


def load_car_by_id(car_id):
    car_data_dir = join(data_path, car_id)
    if not isdir(car_data_dir):
        return None

    return Car(car_id, car_id)


def load_tubs(car_id):
    car_data_dir = join(data_path, car_id)
    if not isdir(car_data_dir):
        return None

    tub_dirs = get_sub_dirs(car_data_dir)
    tubs = []
    for tub_data_dir in tub_dirs:
        records = glob.glob('%s/record*.json' %
                            join(car_data_dir, tub_data_dir))
        records = ((int(re.search('.+_(\d+).json', path).group(1)), path)
                   for path in records)
        records = sorted(records)
        if len(records) == 0:
            continue

        first = records[0][1]
        timestamp = datetime.fromtimestamp(os.path.getmtime(first))
        tubs.append(
            Tub(tub_data_dir, tub_data_dir, timestamp, len(records))
        )

    return tubs


def load_tub_data(car_id, tub_id):
    tub_data_dir = join(data_path, car_id, tub_id)
    if not isdir(tub_data_dir):
        return None

    records = glob.glob('%s/record*.json' % tub_data_dir)
    records = ((int(re.search('.+_(\d+).json', path).group(1)), path)
               for path in records)
    records = sorted(records)

    allData = []
    for _, record in sorted(records):
        with open(record, 'r') as record_file:
            data = json.load(record_file)
            allData.append(TubDataPoint(
                data['timestamp'],
                data['user/angle'],
                data['user/throttle'],
                "/car/" + car_id + "/tub/" + tub_id +
                "/image/" + data['cam/image_array']
            ))

    return allData


def load_tub_data_by_id(car_id, tub_id, data_id):
    tub_data_dir = join(data_path, car_id, tub_id)
    if not isdir(tub_data_dir):
        return None

    records = glob.glob('%s/record*.json' % tub_data_dir)
    records = ((int(re.search('.+_(\d+).json', path).group(1)), path)
               for path in records)
    records = sorted(records)
    if data_id < 0 or data_id >= len(records):
        return None

    record = records[data_id][1]
    with open(record, 'r') as record_file:
        data = json.load(record_file)
        return TubDataPoint(
            data['timestamp'],
            data['user/angle'],
            data['user/throttle'],
            "/car/" + car_id + "/tub/" + tub_id +
            "/image/" + data['cam/image_array']
        )


def get_image_path(car_id, tub_id, image_id):
    image_path = join(data_path, car_id, tub_id, image_id)
    if not isfile(image_path):
        return None

    return image_path


def get_sub_dirs(dir):
    return [f for f in listdir(dir) if isdir(join(dir, f)) and not f.startswith(".")]
