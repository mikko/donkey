""""

resnet50.py

Implements Deep residual architecture with slight tunings for donkey car
Highly inspired by original Resnet paper: https://arxiv.<org/abs/1512.03385

"""
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.python.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.models import Model, load_model

import datetime
import numpy as np

from donkeycar.util.data import linear_unbin

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

        #tbCallBack = TensorBoard(log_dir=('./tensorboard_logs/%s' % date), histogram_freq=0, write_graph=True,
        #                                        write_images=True)

        callbacks_list = [save_best]

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

def id_block(x, f, filters):
    """
    Resnet Identity block
    """

    F1, F2, F3 = filters

    # Shortcut component
    x_sc = x 

    # First component
    x = Conv2D(F1, (1, 1), 
            kernel_initializer='he_normal',
            strides = (1, 1), 
            padding = 'valid')(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation("relu")(x)

    # Second component
    x = Conv2D(F2, (f, f), 
            kernel_initializer='he_normal',
            strides = (1, 1), 
            padding = "same")(x)
    BatchNormalization(axis = 3)(x)
    x = Activation("relu")(x)

    # Third component
    x = Conv2D(F3, (1, 1), 
            kernel_initializer='he_normal',
            strides = (1, 1),
            padding = "valid")(x)
    x = BatchNormalization(axis = 3)(x)

    # SUM component
    x = Add()([x, x_sc])
    x = Activation("relu")(x)

    return x

def conv_block(x, f, filters, s=(2,2)):
    """
    Resnet convolutional block
    """
    F1, F2, F3 = filters

    # Shortcut component
    x_sc = x
    x_sc = Conv2D(F3, (1, 1), 
            kernel_initializer='he_normal',
            strides =(s,s), 
            padding="valid")(x_sc)
    x_sc = BatchNormalization(axis = 3)(x_sc)

    # First component
    x = Conv2D(F1, (1, 1), 
            kernel_initializer='he_normal',
            strides =(s,s))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)

    # Second component
    x = Conv2D(F2, (f, f), 
            kernel_initializer='he_normal',
            strides=(1,1), 
            padding="same")(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation("relu")(x)

    # Third component
    x = Conv2D(F3, (1, 1), 
            kernel_initializer='he_normal',
            strides=(1,1), 
            padding="valid")(x)
    x = BatchNormalization(axis = 3)(x)

    

    # SUM component
    x = Add()([x, x_sc])
    x = Activation("relu")(x)

    return x

def ResNet50(input_shape = (100, 240, 3)):
    x_input = Input(input_shape)

    # x = ZeroPadding2D((3, 3))(x_input)

    # # Stage 1
    # x = Conv2D(64, (7, 7), 
    #         kernel_initializer='he_normal',
    #         strides=(2,2),
    #         padding='valid')(x)
    # x = BatchNormalization(axis = 3)(x)
    # x = Activation("relu")(x)
    # x = ZeroPadding2D(padding=(1, 1))(x)
    # x = MaxPooling2D((3, 3), strides = (2, 2))(x)

    # # Stage 2
    # x = conv_block(x, f = 3, filters = [64, 64, 256], s = 1)
    # x = id_block(x, 3, [64, 64, 256])
    # x = id_block(x, 3, [64, 64, 256])

    # # Stage 3
    # x = conv_block(x, f = 3, filters = [128, 128, 512], s = 2)
    # x = id_block(x, 3, [128, 128, 512])
    # x = id_block(x, 3, [128, 128, 512])
    # x = id_block(x, 3, [128, 128, 512])

    # # Stage 4
    # x = conv_block(x, f = 3, filters = [256, 256, 1024], s = 2)
    # x = id_block(x, 3, [256, 256, 1024])
    # x = id_block(x, 3, [256, 256, 1024])
    # x = id_block(x, 3, [256, 256, 1024])
    # x = id_block(x, 3, [256, 256, 1024])
    # x = id_block(x, 3, [256, 256, 1024])

    # # Stage 5
    # x = conv_block(x, f = 3, filters = [512, 512, 2048], s = 2)
    # x = id_block(x, 3, [512, 512, 2048])
    # x = id_block(x, 3, [512, 512, 2048])

    # x = GlobalMaxPooling2D()(x)

    # output layer
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x) 
    output = Dense(30, activation='softmax')(x)
    

    return Model(inputs=[x_input], outputs=[output])

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
        print(np.argmax(angle[0]))
        return linear_unbin(angle[0]), linear_unbin(throttle[0])

    def create_model(self):
        model = ResNet50()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model
