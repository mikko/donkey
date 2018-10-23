""""

keras.py

functions to run and train autopilots using keras

"""

from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.python.keras.layers import Convolution2D, Concatenate
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import Input
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


class CustomWithHistory(KerasPilot):
    def __init__(self, model=None, *args, **kwargs):
        super(CustomWithHistory, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        else:
            self.model = custom_with_history(50)

    def inputs(self):
        return [
              'cam/image_array',
              'history/user/angle',
              'history/user/throttle',
              'history/acceleration/x',
              'history/acceleration/y',
              'history/acceleration/z',
              'history/sonar/left',
              'history/sonar/right',
              'history/sonar/center']

    def run(self,
            img_arr,
            angle_history,
            throttle_history,
            acceleration_x_history,
            acceleration_y_history,
            acceleration_z_history,
            sonar_left_history,
            sonar_right_history,
            sonar_center_history):

        angle_history = np.array(angle_history)
        throttle_history = np.array(throttle_history)
        acceleration_x_history = np.array(acceleration_x_history)
        acceleration_y_history = np.array(acceleration_y_history)
        acceleration_z_history = np.array(acceleration_z_history)
        sonar_left_history = np.array(sonar_left_history)
        sonar_right_history = np.array(sonar_right_history)
        sonar_center_history = np.array(sonar_center_history)

        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_history = angle_history.reshape((1,) + angle_history.shape)
        throttle_history = throttle_history.reshape((1,) + throttle_history.shape)
        acceleration_x_history = acceleration_x_history.reshape((1,) + acceleration_x_history.shape)
        acceleration_y_history = acceleration_y_history.reshape((1,) + acceleration_y_history.shape)
        acceleration_z_history = acceleration_z_history.reshape((1,) + acceleration_z_history.shape)
        sonar_left_history = sonar_left_history.reshape((1,) + sonar_left_history.shape)
        sonar_right_history = sonar_right_history.reshape((1,) + sonar_right_history.shape)
        sonar_center_history = sonar_center_history.reshape((1,) + sonar_center_history.shape)

        angle, throttle = self.model.predict([
            img_arr,
            angle_history,
            throttle_history,
            acceleration_x_history,
            acceleration_y_history,
            acceleration_z_history,
            sonar_left_history,
            sonar_right_history,
            sonar_center_history])
        return angle[0][0], throttle[0][0]

def custom_with_history(history_len):
    img_in = Input(shape=(100, 240, 3),
                   name='img_in')  # First layer, input layer, Shape comes from camera.py resolution, RGB

    angle_history = Input(shape=(history_len,),
                             name='angle_history_in')

    throttle_history = Input(shape=(history_len,),
                        name='throttle_history_in')

    acceleration_x_history = Input(shape=(history_len,), name='acceleration_x_history_in')
    acceleration__y_history = Input(shape=(history_len,), name='acceleration__y_history_in')
    acceleration__z_history = Input(shape=(history_len,), name='acceleration__z_history_in')
    sonar_left_history = Input(shape=(history_len,), name='sonar_left_history_in')
    sonar_right_history = Input(shape=(history_len,), name='sonar_right_history_in')
    sonar_center_history = Input(shape=(history_len,), name='sonar_center_history_in')

    # Current image convolution
    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(
        x)  # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(
        x)  # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(
        x)  # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(
        x)  # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(
        x)  # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)  # Flatten to 1D (Fully connected)
    x = Concatenate(name='with_history')([
        x,
        throttle_history,
        angle_history,
        acceleration_x_history,
        acceleration__y_history,
        acceleration__z_history,
        sonar_left_history,
        sonar_right_history,
        sonar_center_history
    ])
    x = Dense(50, activation='relu')(x)  # Classify the data into 100 features, make all negatives 0
    x = Dense(50, activation='relu')(x)  # Classify the data into 100 features, make all negatives 0
    x = Dense(50, activation='relu')(x)  # Classify the data into 100 features, make all negatives 0
    x = Dense(50, activation='relu')(x)  # Classify the data into 100 features, make all negatives 0

    angle_out = Dense(units=1, activation='linear', name='angle_out')(x)
    throttle_out = Dense(units=1, activation='linear', name='throttle_out')(x)

    model = Model(inputs=[
        img_in,
        angle_history,
        throttle_history,
        acceleration_x_history,
        acceleration__y_history,
        acceleration__z_history,
        sonar_left_history,
        sonar_right_history,
        sonar_center_history], outputs=[angle_out, throttle_out])

    model.compile(optimizer='adam',
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 0.6, 'throttle_out': 0.4})

    return model

class CustomSequential(KerasPilot):
    def __init__(self, model=None, *args, **kwargs):
        super(CustomSequential, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        else:
            self.model = custom_sequential()

        model_json = self.model.to_json(indent=2)
        with open("latest_model.json", "w") as json_file:
            json_file.write(model_json)
            print('Saved model to JSON')

    def inputs(self):
        return ['cam/image_array']

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)

        angle, throttle = self.model.predict([img_arr])
        return angle[0][0], throttle[0][0]

def custom_sequential():
    img_in = Input(shape=(100, 240, 3),
                   name='img_in')  # First layer, input layer, Shape comes from camera.py resolution, RGB

    # Current image convolution
    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu')(
        x)  # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')(
        x)  # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu')(
        x)  # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(2, 2), activation='relu')(
        x)  # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu')(
        x)  # 64 features, 3px3p kernal window, 1wx1h stride, relu

    x = Flatten(name='flattened')(x)  # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)  # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)  # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)  # Classify the data into 50 features, make all negatives 0
    x = Dropout(.1)(x)  # Randomly drop out 10% of the neurons (Prevent overfitting)

    angle_out = Dense(units=1, activation='linear', name='angle_out')(x)
    throttle_out = Dense(units=1, activation='linear', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 0.6, 'throttle_out': 0.4})

    return model
