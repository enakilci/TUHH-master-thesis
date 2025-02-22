import errno
import time
import logging
import os
import numpy as np 
import pandas as pd
import sys
import math
# import datetime as dt

filepath = os.path.abspath(__file__)
parentdir = os.path.dirname(os.path.dirname(filepath))
#adding the necessary directories to system paths.
sys.path.insert(0, parentdir)

from data_generator import PX4Generator
import tensorflow as tf
 
class LSTMModel(object):
    def __init__(self, train_data, valid_data, train_suffix, valid_suffix, 
                    batch_size = 48, epochs = 7,timesteps = 100, 
                    hidden = 20, dropout = 0.5):
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.timesteps = timesteps
        self.hidden = hidden
        self.dropout = dropout
        self.workers = 1
        self.callbacks = list()

        self.path_suffix = '-'.join([train_suffix,valid_suffix])

        # self.props = {} # in case I wanna use it

        self.train_data = train_data
        self.valid_data = valid_data
        self.features = self.train_data.shape[1]
        self.info_x = '-'.join([str(self.hidden),str(self.timesteps),str(self.dropout)])

    def add_callback(self,callback):
        """Adds a callback to the list of callbacks

        Argument:
            callback: callback to be added to the current callbacks list
        """
        
        self.callbacks.append(callback) 

    def model(self):

        inputs = tf.keras.layers.Input(shape=(self.timesteps,self.features))  
        x = tf.keras.layers.LSTM(self.hidden,return_sequences=True)(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.LSTM(self.hidden)(x)
        # x = tf.keras.layers.LeakyReLU(alpha=0.5)(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        outputs = tf.keras.layers.Dense(self.features)(x)


        model = tf.keras.models.Model(
            inputs = inputs,
            outputs = outputs,
            name = "PX4_LSTM"
        )
        
        # SGD is slow but give great results
        # optimizer = tf.keras.optimizers.SGD(lr=1e-7, momentum=0.9)

        # ADAM is fast but tends to over-fit
        # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,epsilon=1e-08,clipnorm=1.0)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5,epsilon=1e-08,clipnorm=1.0)
        # loss  = tf.keras.losses.MeanAbsoluteError()
        # loss = tf.keras.losses.Huber(delta=0.1)
        loss = tf.keras.losses.MeanSquaredError()
        start = time.time()
        model.compile(
            loss = loss,
            # optimizer = optimizer,
            optimizer= 'adam',
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

        hist = model.fit_generator(
            generator=train,
            validation_data=validation,
            epochs=self.epochs,
            use_multiprocessing=True,
            workers=self.workers,
            callbacks=self.callbacks,
            verbose=1,
            shuffle=False
        )

        model.save(
            os.path.join(
                parentdir,
                "logs/LSTM/%s/%s/result.hdf5" % (self.path_suffix,self.info_x)
            )
        )

        # save_fname =os.path.join(
        #     parentdir,
        #     "logs/LSTM/%s/%s/result.hdf5" % (self.path_suffix,self.info_x)
        # )
        
        # os.path.join(
        #     parentdir, 
        #     'logs/LSTM/%s/%s/%s-e%s.h5' % (self.path_suffix,self.info_x,dt.datetime.now().strftime('%d%m%Y-%H%M%S')))
        
        # model.save(save_fname)

        logging.getLogger().setLevel(logging.INFO),
        # print('[Model] Training Completed. Model saved as %s' % save_fname)
        logging.info("Training Time : %s secs"%str(time.time() - start))
        print('hist.history.keys(): ',hist.history.keys())
        return hist


