import errno
import time
import logging
import os
import numpy as np
import pandas as pd
import sys
import math

filepath = os.path.abspath(__file__)
parentdir = os.path.dirname(os.path.dirname(filepath))
#adding the necessary directories to system paths.
sys.path.insert(0, parentdir)

from data_generator import PX4Generator
import tensorflow as tf

class CNN1DModel(object):
    def __init__(self, train_data, valid_data, train_suffix, valid_suffix,
                    batch_size = 48, epochs = 7, timesteps = 200,
                    kernel_size = 10, filters = 32, pool_size = 2,dropout = 0.5):

        self.batch_size = batch_size
        self.epochs = epochs
        self.timesteps = timesteps
        self.kernel_size = kernel_size
        self.filters = filters
        self.pool_size = 1
        self.dropout = dropout
        self.workers = 2
        self.callbacks = list()

        self.path_suffix = '-'.join([train_suffix,valid_suffix])

        self.props = {} # in case I wanna use it


        self.train_data = train_data
        self.valid_data = valid_data
        self.features = self.train_data.shape[1]
        self.info_x = '-'.join([str(self.kernel_size),str(self.filters),str(self.timesteps),str(dropout)])

    def add_callback(self,callback):
        """Adds a callback to the list of callbacks

        Argument:
            callback: callback to be added to the current callbacks list
        """

        self.callbacks.append(callback)

    def model(self):

        inputs = tf.keras.layers.Input(shape=(self.timesteps,self.features))
        x = tf.keras.layers.Conv1D(filters=self.filters,kernel_size=self.kernel_size,strides=1)(inputs)
        x = tf.keras.layers.ReLU()(x)        
        x = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Conv1D(filters=self.filters,kernel_size=self.kernel_size,strides=1)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        outputs = tf.keras.layers.Dense(self.features)(x)


        # outputs = tf.keras.layers.Dense(self.features,activation='softmax')(x)


        model = tf.keras.models.Model(
            inputs = inputs,
            outputs = outputs,
            name = "PX4_CNN1D"
        )

        # SGD is slow but give great results
        # optimizer = tf.keras.optimizers.SGD(lr=1e-7, momentum=0.9)

        # ADAM is fast but tends to over-fit
        # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,epsilon=1e-08,clipnorm=1.0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5,epsilon=1e-07,clipnorm=1.0)
        # loss  = tf.keras.losses.MeanAbsoluteError()
        # loss = tf.keras.losses.Huber(delta = 0.1)
        loss = tf.keras.losses.MeanSquaredError()
        start = time.time()
        model.compile(
            loss = loss,
            optimizer = optimizer,
            # optimizer= 'adam',
            # metrics = ['mse','mae']
            metrics = ['mae']
        )
        logging.getLogger().setLevel(logging.INFO)
        logging.info("Compilation Time : %s secs"%str(time.time() - start))

        model.summary()
        return model

    def fit(self):

        model = self.model()
        
        train = PX4Generator(
                    self.train_data,
                    batch_size=self.batch_size,timesteps=self.timesteps
                )
        validation = PX4Generator(
                        self.valid_data,
                        batch_size=self.batch_size,timesteps=self.timesteps
                    )

        start = time.time()

        hist = model.fit(
            x=train,
            validation_data=validation,
            epochs=self.epochs,
            use_multiprocessing=True,
            workers=self.workers,
            callbacks=self.callbacks,
            verbose=1,
            shuffle=False
        )

        # hist = model.fit_generator(
        #     generator=train,
        #     validation_data=validation,
        #     epochs=self.epochs,
        #     #tensorflow:multiprocessing can interact badly with TensorFlow, causing nondeterministic deadlocks.
        #     #For high performance data pipelines tf.data is recommended
        #     use_multiprocessing=True,
        #     workers=self.workers,
        #     callbacks=self.callbacks,
        #     verbose=1,
        #     shuffle=False
        # )

        model.save(
            os.path.join(
                parentdir,
                "logs/CNN1D/%s/%s/result.hdf5" % (self.path_suffix,self.info_x)
            )
        )

        logging.getLogger().setLevel(logging.INFO)
        logging.info("Training Time : %s secs"%str(time.time() - start))
        return hist


