#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it.

Usage:
    train.py [--tub=<tub1,tub2,..tubn>] (--model=<model>) [--base_model=<base_model>] [--module=<module_name>] [--class=<class_name>] [--no_augmentation] [--skip_flip] [--skip_brightness] [--skip_shadow]

Options:
    -h --help        Show this screen.
    --tub TUBPATHS   List of paths to tubs. Comma separated. Use quotes to use wildcards. ie "~/tubs/*"
"""
import glob
import json
import os

import numpy as np
import cv2

import re

from docopt import docopt
from datetime import datetime

import donkeycar as dk
# import parts
from donkeycar.util.loader import create_instance
from augment import aug_brightness, aug_shadow2, aug_flip

from PIL import Image

# These used to live in config but not anymore
BATCH_SIZE = 128
TRAIN_TEST_SPLIT = 0.9

DEFAULT_MODULE = 'donkeycar.parts.keras'
DEFAULT_CLASS = 'CNN_3D'

img_count = 0

image_resize = (120, 50) #defines the size image is resized to, set False to avoid reshaping

def write_img(img, type):
    global img_count
    name = 'output/file_' + str(img_count) + '_' + type + '.jpg'
    img_count = img_count + 1
    cv2.imwrite(name, img)

def load_image(path):
    if not image_resize:
        img = cv2.imread(path) #default without any reshaping
    else:
        img = Image.open(path)
        img = img.resize(image_resize, Image.BILINEAR)
    return np.array(img)

def get_generator(input_keys, output_keys, record_paths, meta, augmentations):
    prev_image = None
    while True:
        ls = []
        for (record_path, tub_path) in record_paths:
            with open(record_path, 'r') as record_file:
                record = json.load(record_file)
                inputs = [record[key] for key in input_keys]
                outputs = [record[key] for key in output_keys]
                input_types = [meta[key] for key in input_keys]
                # output_types = [meta[key] for key in output_keys]
                for i in range(len(inputs)):
                    type = input_types[i]
                    if (type == 'image_array'):
                        curr_image = load_image("%s/%s" % (tub_path, inputs[i]))
                        if (prev_image is None):
                            prev_image = curr_image
                        inputs[i] = np.stack([curr_image, prev_image], axis=0)
                        prev_image = curr_image
                yield inputs, outputs
                ls = ls + [(inputs, outputs)]
        for aug in augmentations:
            new_list = []
            for item in ls:
                # print(item[0][0][0].shape)
                new_tuple = aug([item[0][0][0]], item[1])

                # TUGGUMMI
                another_tuple = aug([item[0][0][1]], item[1])

                new_tuple[0][0] = np.stack([new_tuple[0][0], another_tuple[0][0]], axis=0)
                # print(new_tuple[0])
                new_list.append(new_tuple)
                yield new_tuple
            ls = ls + new_list

def get_batch_generator(input_keys, output_keys, records, meta, augmentation):
    # Yield here a tuple (inputs, outputs)
    # both having arrays with batch_size length like:
    # 0: [input_1[batch_size],input_2[batch_size]]
    # 1: [output_1[batch_size],output_2[batch_size]]
    record_gen = get_generator(input_keys, output_keys, records, meta, augmentation)
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


def get_train_val_gen(inputs, outputs, tub_names, augmentations):
    print('Loading data', tub_names)
    print('Inputs', inputs)
    print('Outputs', outputs)
    tubs = glob.glob(str('%s/**' % tub_names), recursive=True)
    record_count = 0
    all_train = []
    all_validation = []
    first_meta = None
    for tub in tubs:
        # _original skipped by design
        if ('_original' not in tub):
            # TODO: should all tubs have the exact same meta file??? Probably yes.
            meta = get_meta(tub)
            if (meta != None):
                first_meta = meta
                # TODO: Check if meta.json specs match with given inputs and outputs
                record_files = glob.glob('%s/record*.json' % tub)
                # Sort the frames for 3D CNN
                record_files = ((int(re.search('.+_(\d+).json', path).group(1)), path) for path in record_files)
                files_and_paths = list(map(lambda rec: (rec[1], tub), sorted(record_files)))

                # np.random.shuffle(files_and_paths)
                split = int(round(len(files_and_paths) * TRAIN_TEST_SPLIT))
                train_files, validation_files = files_and_paths[:split], files_and_paths[split:]
                record_count += len(files_and_paths) * (2 ** len(augmentations))
                if len(augmentations) > 0:
                    print("Record count (w/o augmentation): ", len(files_and_paths))
                print("Total count: ", record_count)

                all_train.extend(train_files)
                all_validation.extend(validation_files)
    return get_batch_generator(inputs, outputs, all_train, first_meta, augmentations), get_batch_generator(inputs, outputs, all_validation, first_meta, augmentations), record_count

def train(tub_names, new_model_path=None, base_model_path=None, module_name=None, class_name=None, augment=True, skip_flip=False, skip_brightness=False, skip_shadow=False):

    if not module_name:
        module_name = DEFAULT_MODULE
    if not class_name:
        class_name = DEFAULT_CLASS

    kl = create_instance(module_name, class_name)

    inputs = kl.inputs()

    outputs = ['user/angle', 'user/throttle']

    new_model_path = os.path.expanduser(new_model_path)

    augmentations = []

    # print('All augmentation temporarily disabled for 3DCNN')
    if (augment):
        if not skip_flip:
            augmentations.append(aug_flip)
        if not skip_brightness:
            augmentations.append(aug_brightness)
        if not skip_shadow:
            augmentations.append(aug_shadow2)

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

    train_gen, val_gen, total_train = get_train_val_gen(inputs, outputs, tub_names, augmentations)

    steps_per_epoch = total_train // BATCH_SIZE

    print("Amount of training data available", total_train)
    time = datetime.utcnow().strftime('%Y-%m-%d %H:%M')

    # running = True
    # count = 0
    # while running and count < 40:
    #    batch = next(train_gen)
    #    print('Start: ', len(batch[0][0]))
    #    for val in batch[1][0]:
    #        print('x-value', val)
    #    for img in batch[0][0]:
    #        write_img(img, 'output')
    #    count = count + 1

    while (True):
        batch = next(train_gen)
        inputs_batch = batch[0][0]
        outputs_batch = batch[1][0]
        for record in inputs_batch:
            prev_image = cv2.cvtColor(record[0], cv2.COLOR_BGR2RGB)
            cv2.imshow('prev', prev_image)

            curr_image = cv2.cvtColor(record[1], cv2.COLOR_BGR2RGB)
            cv2.imshow('curr', curr_image)

            if cv2.waitKey(200) & 0xFF == ord('q'):
                break

    #kl.train(train_gen,
    #         val_gen,
    #         saved_model_path=new_model_path,
    #         steps=steps_per_epoch,
    #         train_split=TRAIN_TEST_SPLIT,
    #         use_early_stop=False)

if __name__ == '__main__':
    args = docopt(__doc__)
    tub = args['--tub']
    module_name = args['--module']
    class_name = args['--class']
    new_model_path = args['--model']
    base_model_path = args['--base_model']
    augment = not args['--no_augmentation']
    skip_flip = args['--skip_flip']
    skip_brightness = args['--skip_brightness']
    skip_shadow = args['--skip_shadow']

    train(tub, new_model_path, base_model_path, module_name, class_name, augment, skip_flip, skip_brightness, skip_shadow)
