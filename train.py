#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it.

Usage:
    train.py [--tub=<tub1,tub2,..tubn>]  (--model=<model>) [--base_model=<base_model>] [--no_cache]

Options:
    -h --help        Show this screen.
    --tub TUBPATHS   List of paths to tubs. Comma separated. Use quotes to use wildcards. ie "~/tubs/*"
"""
import glob
import json
import os

import numpy as np
from PIL import Image
from docopt import docopt

import donkeycar as dk
# import parts
from donkeycar.parts.keras import CustomSequential

# These used to live in config but not anymore
BATCH_SIZE = 128
TRAIN_TEST_SPLIT = 0.8


def load_image(path):
    img = Image.open(path)
    return np.array(img)

def get_generator(input_keys, output_keys, record_paths, meta, tub_path):
    while True:
        for record_path in record_paths:
            with open(record_path, 'r') as record_file:
                record = json.load(record_file)
                inputs = [record[key] for key in input_keys]
                outputs = [record[key] for key in output_keys]
                input_types = [meta[key] for key in input_keys]
                # output_types = [meta[key] for key in output_keys]
                for i in range(len(inputs)):
                    type = input_types[i]
                    if (type == 'image_array'):
                        inputs[i] = load_image("%s/%s" % (tub_path, inputs[i]))
                    elif (type == 'custom/prev_image'):
                        # Currently previous images are in array, but there is only one
                        imagePath = inputs[i][0]
                        inputs[i] = load_image("%s/%s" % (tub_path, imagePath))
                yield inputs, outputs

def get_batch_generator(input_keys, output_keys, records, meta, tub_path):
    # Yield here a tuple (inputs, outputs)
    # both having arrays with batch_size length like:
    # 0: [input_1[batch_size],input_2[batch_size]]
    # 1: [output_1[batch_size],output_2[batch_size]]
    record_gen = get_generator(input_keys, output_keys, records, meta, tub_path)
    while True:
        raw_batch = [next(record_gen) for _ in range(BATCH_SIZE)]
        inputs = [[] for _ in range(len(input_keys))]
        outputs = [[] for _ in range(len(output_keys))]
        for rec in raw_batch:
            for i in range(len(input_keys)):
                inputs[i].append(rec[0][i])
            for i in range(len(output_keys)):
                outputs[i].append(rec[1][i])
        numpyInputs = [np.asarray(ar) for ar in inputs]
        numpyOutputs = [np.asarray(ar) for ar in outputs]
        yield numpyInputs, numpyOutputs

def get_meta(path):
    with open('%s/meta.json' % path, 'r') as f:
        meta = json.load(f)
        meta_dict = {}
        for i, key in enumerate(meta['inputs']):
            meta_dict[key] = meta['types'][i]
        return meta_dict
            # TODO: filter out values not listed in inputs or outputs


def get_train_val_gen(inputs, outputs, tub_names):
    print('Loading data', tub_names)
    print('Inputs', inputs)
    print('Outputs', outputs)
    tubs = glob.glob(str(tub_names))
    print(tubs)
    for tub in tubs:
        meta = get_meta(tub)
        print(meta)
        # TODO: Check if meta.json specs match with given inputs and outputs
        record_files = glob.glob('%s/record*.json' % tub)
        np.random.shuffle(record_files)
        split = int(round(len(record_files) * TRAIN_TEST_SPLIT))
        train_files, validation_files = record_files[:split], record_files[split:]
    return get_batch_generator(inputs, outputs, train_files, meta, tub), get_batch_generator(inputs, outputs, validation_files, meta, tub), len(record_files)


def train(tub_names, new_model_path, base_model_path=None ):
    inputs = ['cam/image_array']
    outputs = ['user/angle', 'user/throttle']

    new_model_path = os.path.expanduser(new_model_path)

    kl = CustomSequential()
    # Load base model if given
    if base_model_path is not None:
        base_model_path = os.path.expanduser(base_model_path)
        kl.load(base_model_path)

    # Support for multiple paths
    print('tub_names', tub_names)
    if not tub_names:
        print('No tub path given')
        return
        # tub_names = os.path.join(cfg.DATA_PATH, '*')

    train_gen, val_gen, total_train = get_train_val_gen(inputs, outputs, tub_names)

    steps_per_epoch = total_train // BATCH_SIZE

    kl.train(train_gen,
             val_gen,
             saved_model_path=new_model_path,
             steps=steps_per_epoch,
             train_split=TRAIN_TEST_SPLIT)


if __name__ == '__main__':
    args = docopt(__doc__)
    tub = args['--tub']
    new_model_path = args['--model']
    base_model_path = args['--base_model']
    cache = not args['--no_cache']
    train(tub, new_model_path, base_model_path)





