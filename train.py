#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it.

Usage:
    train.py [--tub=<tub1,tub2,..tubn>] (--model=<model>) [--base_model=<base_model>] [--module=<module_name>] [--class=<class_name>] [--no_cache]

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
from datetime import datetime

import donkeycar as dk
# import parts
from donkeycar.util.loader import create_instance
from donkeycar.util.data import linear_bin

# These used to live in config but not anymore
BATCH_SIZE = 256
TRAIN_TEST_SPLIT = 0.9

DEFAULT_MODULE = 'donkeycar.parts.keras'
DEFAULT_CLASS = 'CustomSequential'

def load_image(path):
    img = Image.open(path)
    return np.array(img)

def get_generator(input_keys, output_keys, record_paths, meta):
    while True:
        for (record_path, tub_path) in record_paths:
            with open(record_path, 'r') as record_file:
                record = json.load(record_file)
                inputs = [record[key] for key in input_keys]
                outputs = [record[key] for key in output_keys]
                input_types = [meta[key] for key in input_keys]
                output_types = [meta[key] for key in output_keys]
                for i in range(len(inputs)):
                    type = input_types[i]
                    if (type == 'image_array'):
                        inputs[i] = load_image("%s/%s" % (tub_path, inputs[i])) / 255
                for i in range(len(outputs)):
                    key = output_keys[i]
                    if (key == 'user/angle' or key == 'user/throttle'):
                        outputs[i] = linear_bin(outputs[i])
                yield inputs, outputs

def get_batch_generator(input_keys, output_keys, records, meta):
    # Yield here a tuple (inputs, outputs)
    # both having arrays with batch_size length like:
    # 0: [input_1[batch_size],input_2[batch_size]]
    # 1: [output_1[batch_size],output_2[batch_size]]
    record_gen = get_generator(input_keys, output_keys, records, meta)
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
    try:
        with open('%s/meta.json' % path, 'r') as f:
            meta = json.load(f)
            meta_dict = {}
            for i, key in enumerate(meta['inputs']):
                meta_dict[key] = meta['types'][i]
            return meta_dict
                # TODO: filter out values not listed in inputs or outputs
    except:
        return None


def get_train_val_gen(inputs, outputs, tub_names):
    print('Loading data', tub_names)
    print('Inputs', inputs)
    print('Outputs', outputs)
    tubs = glob.glob(str('%s/**' % tub_names), recursive=True)
    record_count = 0
    all_train = []
    all_validation = []
    for tub in tubs:
        # _original skipped by design
        if ('_original' not in tub):
            # TODO: should all tubs have the exact same meta file??? Probably yes.
            meta = get_meta(tub)
            if (meta != None):
                first_meta = meta
                # TODO: Check if meta.json specs match with given inputs and outputs
                record_files = glob.glob('%s/record*.json' % tub)
                files_and_paths = list(map(lambda rec: (rec, tub), record_files))
                np.random.shuffle(files_and_paths)
                split = int(round(len(files_and_paths) * TRAIN_TEST_SPLIT))
                train_files, validation_files = files_and_paths[:split], files_and_paths[split:]
                record_count += len(files_and_paths)
                all_train.extend(train_files)
                all_validation.extend(validation_files)
    return get_batch_generator(inputs, outputs, all_train, first_meta), get_batch_generator(inputs, outputs, all_validation, first_meta), record_count


def train(tub_names, new_model_path=None, base_model_path=None, module_name=None, class_name=None):

    if not module_name:
        module_name = DEFAULT_MODULE
    if not class_name:
        class_name = DEFAULT_CLASS

    kl = create_instance(module_name, class_name)

    inputs = kl.inputs()

    outputs = ['user/angle', 'user/throttle']

    new_model_path = os.path.expanduser(new_model_path)

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

    print("Amount of training data available", total_train)
    time = datetime.utcnow().strftime('%Y-%m-%d_%H:%M')

    kl.train(train_gen,
             val_gen,
             saved_model_path=f'{new_model_path}-{class_name}-{time}',
             steps=steps_per_epoch,
             train_split=TRAIN_TEST_SPLIT,
             use_early_stop=False)


if __name__ == '__main__':
    args = docopt(__doc__)
    tub = args['--tub']
    module_name = args['--module']
    class_name = args['--class']
    new_model_path = args['--model']
    base_model_path = args['--base_model']
    cache = not args['--no_cache']
    train(tub, new_model_path, base_model_path, module_name, class_name)





