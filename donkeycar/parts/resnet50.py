""""

resnet50.py

Implements resnet-50 like architecture with slight tunings for donkey car
Highly inspired by original Resnet paper: https://arxiv.org/abs/1512.03385

"""
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.python.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.models import Model, load_model

import datetime
import numpy as np

class KerasPilot:

    def load(self, model_path):
        self.model = load_model(model_path)

    def shutdown(self):
        pass

    def drive_inputs(self):
        inputs = self.inputs()
        inputs = [input.replace('/user/', '/pilot/') for input in inputs]
        return inputs

    def inputs(self):
        return ['cam/image_array']

    def train(self, train_gen, val_gen,
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=8, use_early_stop=True):
        """
        train_gen: generator that yields an array of images an array of

        """

        # checkpoint to save model after each epoch
        save_best = ModelCheckpoint(saved_model_path,
                                    monitor='val_loss',
                                    verbose=verbose,
                                    save_best_only=True,
                                    mode='min')

        # stop training if the validation error stops improving.
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=min_delta,
                                   patience=patience,
                                   verbose=verbose,
                                   mode='auto')

        date = datetime.datetime.now().strftime('%y-%m-%d-%H-%M')

        tbCallBack = TensorBoard(log_dir=('./tensorboard_logs/%s' % date), histogram_freq=0, write_graph=True,
                                                 write_images=True)

        callbacks_list = [save_best, tbCallBack]

        if use_early_stop:
            callbacks_list.append(early_stop)

        hist = self.model.fit_generator(
            train_gen,
            steps_per_epoch=steps,
            epochs=epochs,
            verbose=1,
            validation_data=val_gen,
            callbacks=callbacks_list,
            validation_steps=steps * (1.0 - train_split) / train_split)
        return hist

def id_block(X, f, filters):
    """
    Resnet Identity block
    """

    F1, F2, F3 = filters

    # Shortcut component
    X_sc = X 

    # First component
    X = Conv2D(F1, (1, 1), strides = (1, 1), padding = "valid")(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation("relu")(X)

    # Second component
    X = Conv2D(F2, (f, f), strides = (1, 1), padding = "same")(X)
    BatchNormalization(axis = 3)(X)
    X = Activation("relu")(X)

    # Third component
    X = Conv2D(F3, (1, 1), strides = (1, 1), padding = "valid")(X)
    X = BatchNormalization(axis = 3)(X)

    # SUM component
    X = Add()([X, X_sc])
    X = Activation("relu")(X)

    return X

def conv_block(X, f, filters, s):
    """
    Resnet convolutional block
    """
    F1, F2, F3 = filters

    # Shortcut component
    X_sc = X
    X_sc = Conv2D(F3, (1, 1), strides = (s, s), padding = "valid")(X_sc)
    X_sc = BatchNormalization(axis = 3)(X_sc)

    # First component
    X = Conv2D(F1, (1, 1), strides = (s, s))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation("relu")(X)

    # Second component
    X = Conv2D(F2, (f, f), strides = (1, 1), padding = "same")(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation("relu")(X)

    # Third component
    X = Conv2D(F3, (1, 1), strides = (1, 1), padding = "valid")(X)
    X = BatchNormalization(axis = 3)(X)

    # Third component
    X = Conv2D(F3, (1, 1), strides = (1, 1), padding = "valid")(X)
    X = BatchNormalization(axis = 3)(X)

    # SUM component
    X = Add()([X, X_sc])
    X = Activation("relu")(X)

    return X

def ResNet50(input_shape = (100, 240, 3)):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((3, 3), strides = (2, 2))(X)

    # Stage 2
    X = conv_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = id_block(X, 3, [64, 64, 256])
    X = id_block(X, 3, [64, 64, 256])

    # Stage 3
    X = conv_block(X, f = 3, filters = [128, 128, 512], s = 2)
    X = id_block(X, 3, [128, 128, 512])
    X = id_block(X, 3, [128, 128, 512])
    X = id_block(X, 3, [128, 128, 512])

    # Stage 4
    X = conv_block(X, f = 3, filters = [256, 256, 1024], s = 2)
    X = id_block(X, 3, [256, 256, 1024])
    X = id_block(X, 3, [256, 256, 1024])
    X = id_block(X, 3, [256, 256, 1024])
    X = id_block(X, 3, [256, 256, 1024])
    X = id_block(X, 3, [256, 256, 1024])

    # Stage 5
    X = conv_block(X, f = 3, filters = [512, 512, 2048], s = 2)
    X = id_block(X, 3, [512, 512, 2048])
    X = id_block(X, 3, [512, 512, 2048])

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size = (2, 2), name = "avg_pool")(X)

    # output layer 
    X = Flatten()(X)
    X = Dense(100, activation = "relu")(X)

    angle_out = Dense(units=1, activation='linear', name='angle_out')(X)
    throttle_out = Dense(units=1, activation='linear', name='throttle_out')(X)

    return Model(inputs=[X_input], outputs=[angle_out, throttle_out])


class Resnet50Model(KerasPilot):
    def __init__(self, model=None, *args, **kwargs):
        super(Resnet50Model, self).__init__(*args, **kwargs)
        
        if model: 
            self.model = model
        else: 
            self.model = self.create_model()

    def inputs(self):
        return ["cam/image_array"]

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)

        angle, throttle = self.model.predict([img_arr])

        return angle[0][0], throttle[0][0]

    def create_model(self):
        model = ResNet50()
        model.compile(optimizer='adam',
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_squared_error'})

        return model
