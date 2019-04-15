# Mikko's original POC
import cv2
import flask
from time import sleep
from multiprocessing import Process
import glob
import re
import json
import numpy as np

from .. / import train

app = flask.Flask(__name__)
app.config["DEBUG"] = True

tub_path = '../../ai-markku/ds-meetup/testi/tub_19-02-22-14-19-58/'


def gen():
    global tub_path
    while True:
        records = glob.glob('%s/record*.json' % tub_path)
        records = ((int(re.search('.+_(\d+).json', path).group(1)), path)
                   for path in records)
        for _, record in sorted(records):
            with open(record, 'r') as record_file:
                data = json.load(record_file)
                img_path = data['cam/image_array']
            img = cv2.imread('%s/%s' % (tub_path, img_path))
            ret, jpeg = cv2.imencode('.jpg', img)
            sleep(0.02)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


def train_func():
    train(tub_path, 'api_model')


print_process = None


@app.route('/', methods=['GET'])
def home():
    return "<h1>HEPSKUKKUU</h1><p>Best API ever</p>"


@app.route('/start', methods=['GET'])
def start():
    global print_process
    # instantiating without any argument
    print_process = Process(target=train_func)
    print('Training start')
    print_process.start()
    return "<h1>Started</h1>"


@app.route('/stop', methods=['GET'])
def stop():
    global print_process
    print_process.terminate()
    print('Training terminated')
    return "<h1>Stopped<h1>"


@app.route('/tub_stream')
def video_feed():
    return flask.Response(gen(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')


app.run()
