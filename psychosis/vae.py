# Inspiration https://medium.com/applied-data-science/how-to-build-your-own-world-model-using-python-and-keras-64fb388ba459

import numpy as np
from PIL import Image
import glob, random, os

from tensorflow.python.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard

import argparse

INPUT_DIM = (100,240,3)

CONV_FILTERS = [32,64,64,128]
CONV_KERNEL_SIZES = [4,4,4,4]
CONV_STRIDES = [2,2,2,2]
CONV_ACTIVATIONS = ['relu','relu','relu','relu']

DENSE_SIZE = 1024

CONV_T_FILTERS = [64,64,32,3]
CONV_T_KERNEL_SIZES = [5,5,6,6]
CONV_T_STRIDES = [2,2,2,2]
CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']

Z_DIM = 32

EPOCHS = 1
BATCH_SIZE = 32

TRAIN_SPLIT=0.9

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], Z_DIM), mean=0.,stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

class VAE():
    def __init__(self):
        self.models = self._build()
        self.model = self.models[0]
        self.encoder = self.models[1]
        self.decoder = self.models[2]

        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM


    def _build(self):
        vae_x = Input(shape=INPUT_DIM, name='img_in')
        vae_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0])(vae_x)
        vae_c2 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], activation=CONV_ACTIVATIONS[0])(vae_c1)
        vae_c3= Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], activation=CONV_ACTIVATIONS[0])(vae_c2)
        vae_c4= Conv2D(filters = CONV_FILTERS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], activation=CONV_ACTIVATIONS[0])(vae_c3)

        vae_z_in = Flatten()(vae_c4)

        vae_z_mean = Dense(Z_DIM)(vae_z_in)
        vae_z_log_var = Dense(Z_DIM)(vae_z_in)

        vae_z = Lambda(sampling)([vae_z_mean, vae_z_log_var])
        vae_z_input = Input(shape=(Z_DIM,))

        # we instantiate these layers separately so as to reuse them later
        vae_dense = Dense(1024)
        vae_dense_model = vae_dense(vae_z)

        vae_z_out = Reshape((1,1,DENSE_SIZE))
        vae_z_out_model = vae_z_out(vae_dense_model)

        vae_d1 = Conv2DTranspose(filters = CONV_T_FILTERS[0], kernel_size = CONV_T_KERNEL_SIZES[0] , strides = CONV_T_STRIDES[0], activation=CONV_T_ACTIVATIONS[0])
        vae_d1_model = vae_d1(vae_z_out_model)
        vae_d2 = Conv2DTranspose(filters = CONV_T_FILTERS[1], kernel_size = CONV_T_KERNEL_SIZES[1] , strides = CONV_T_STRIDES[1], activation=CONV_T_ACTIVATIONS[1])
        vae_d2_model = vae_d2(vae_d1_model)
        vae_d3 = Conv2DTranspose(filters = CONV_T_FILTERS[2], kernel_size = CONV_T_KERNEL_SIZES[2] , strides = CONV_T_STRIDES[2], activation=CONV_T_ACTIVATIONS[2])
        vae_d3_model = vae_d3(vae_d2_model)
        vae_d4 = Conv2DTranspose(filters = CONV_T_FILTERS[3], kernel_size = CONV_T_KERNEL_SIZES[3] , strides = CONV_T_STRIDES[3], activation=CONV_T_ACTIVATIONS[3])
        vae_d4_model = vae_d4(vae_d3_model)

        #### DECODER ONLY

        vae_dense_decoder = vae_dense(vae_z_input)
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)

        vae_d1_decoder = vae_d1(vae_z_out_decoder)
        vae_d2_decoder = vae_d2(vae_d1_decoder)
        vae_d3_decoder = vae_d3(vae_d2_decoder)
        vae_d4_decoder = vae_d4(vae_d3_decoder)

        # vae_reshaped_decoder = Reshape([-1, 100, 240, 3])(vae_d4_decoder)


        #### MODELS

        vae = Model(vae_x, vae_d4_model)
        vae_encoder = Model(vae_x, vae_z)
        vae_decoder = Model(vae_z_input, vae_d4_decoder)



        def vae_r_loss(y_true, y_pred):

            y_true_flat = K.flatten(y_true)
            y_pred_flat = K.flatten(y_pred)

            return 10 * K.mean(K.square(y_true_flat - y_pred_flat), axis = -1)

        def vae_kl_loss(y_true, y_pred):
            return - 0.5 * K.mean(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis = -1)

        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)

        vae.compile(optimizer='rmsprop', loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])

        return (vae,vae_encoder, vae_decoder)


    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, train_gen, val_gen, steps, train_split):

        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
        tbCallback = TensorBoard(log_dir=('./tensorboard_logs/vae'), histogram_freq=0, write_graph=True,
                    write_images=True)

        callbacks_list = [earlystop, tbCallback]

        print('STEPS', steps)

        self.model.fit_generator(
            train_gen,
            steps_per_epoch=steps,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks_list,
            validation_steps=steps * (1.0 - train_split) / train_split)

        self.model.save_weights('./vae/weights.h5')

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def generate_rnn_data(self, obs_data, action_data):

        rnn_input = []
        rnn_output = []

        for i, j in zip(obs_data, action_data):
            rnn_z_input = self.encoder.predict(np.array(i))
            conc = [np.concatenate([x,y]) for x, y in zip(rnn_z_input, j)]
            rnn_input.append(conc[:-1])
            rnn_output.append(np.array(rnn_z_input[1:]))

        rnn_input = np.array(rnn_input)
        rnn_output = np.array(rnn_output)

        return (rnn_input, rnn_output)

# Own data load stuff

def load_image(path):
    img = Image.open(path)
    return np.array(img)

def get_generator(record_paths):
    while True:
        for (record_path, tub_path) in record_paths:
            img = load_image(record_path)
            yield img

def get_batch_generator(records):
    record_gen = get_generator(records)
    while True:
        raw_batch = [next(record_gen) for _ in range(BATCH_SIZE)]
        yield np.array(raw_batch), np.array(raw_batch)

def get_train_val_gen():
    tubs = glob.glob('./data/**', recursive=True)
    record_count = 0
    all_train = []
    all_validation = []
    for tub in tubs:
        # TODO: Check if meta.json specs match with given inputs and outputs
        data_files = glob.glob('%s/*.jpg' % tub)
        files_and_paths = list(map(lambda rec: (rec, tub), data_files))
        split = int(round(len(files_and_paths) * TRAIN_SPLIT))
        train_files, validation_files = files_and_paths[:split], files_and_paths[split:]
        record_count += len(files_and_paths)
        all_train.extend(train_files)
        all_validation.extend(validation_files)
    return get_batch_generator(all_train), get_batch_generator(all_validation), record_count

# train_vae.py


def main(args):

    start_batch = args.start_batch
    max_batch = args.max_batch
    # new_model = args.new_model

    vae = VAE()

    print('Not using existing model ever')
    #if not new_model:
    #    try:
    #        vae.set_weights('./vae/weights.h5')
    #    except:
    #        print("Either set --new_model or ensure ./vae/weights.h5 exists")
    #        raise

    #for batch_num in range(start_batch, max_batch + 1):
    #    print('Building batch {}...'.format(batch_num))
    #    first_item = True

    #    for env_name in config.train_envs:
    #        try:
    #            new_data = np.load('./data/obs_data_' + env_name + '_' + str(batch_num) + '.npy')
    #            if first_item:
    #                data = new_data
    #                first_item = False
    #            else:
    #                data = np.concatenate([data, new_data])
    #            print('Found {}...current data size = {} episodes'.format(env_name, len(data)))
    #        except:
    #            pass

    #    if first_item == False: # i.e. data has been found for this batch number
    #        data = np.array([item for obs in data for item in obs])
    #        vae.train(data)
    #    else:
    #        print('no data found for batch number {}'.format(batch_num))

    train_gen, val_gen, total_train = get_train_val_gen()
    steps_per_epoch = total_train // BATCH_SIZE

    vae.train(train_gen,
              val_gen,
              steps=steps_per_epoch,
              train_split=TRAIN_SPLIT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train VAE'))
    parser.add_argument('--start_batch', type=int, default = 0, help='The start batch number')
    parser.add_argument('--max_batch', type=int, default = 0, help='The max batch number')
    parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
    args = parser.parse_args()

    main(args)
