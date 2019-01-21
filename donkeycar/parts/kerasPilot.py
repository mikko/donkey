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