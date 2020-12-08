import errno
import time
import logging
import os
import numpy as np
import pandas as pd
import sys
import math

import inspect
from typing import List

# from tensorflow.keras import backend as K, Model, Input, optimizers
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Activation, SpatialDropout1D, Lambda
# from tensorflow.keras.layers import Layer, Conv1D, Dense, BatchNormalization, LayerNormalization

filepath = os.path.abspath(__file__)
parentdir = os.path.dirname(os.path.dirname(filepath))
#adding the necessary directories to system paths.
sys.path.insert(0, parentdir)

from data_generator import PX4Generator
import tensorflow as tf

class TCNParallelModel(object):
    def __init__(self, train_data, valid_data, train_suffix, valid_suffix,
                    batch_size = 48, epochs = 7, timesteps = 600,
                    kernel_size = 16, filters = 64, pool_size = 2, dropout = 0.5):

        self.batch_size = batch_size
        self.epochs = epochs
        self.timesteps = timesteps
        self.kernel_size = kernel_size
        self.filters = filters
        self.pool_size = pool_size
        self.dropout = dropout
        self.workers = 1
        self.callbacks = list()
        self.dilation = [1,2,4,8,16,32,64,128,256,512]

        self.path_suffix = '-'.join([train_suffix,valid_suffix])

        self.props = {} # in case I wanna use it


        self.train_data = train_data
        self.valid_data = valid_data
        self.features = self.train_data.shape[1]
        self.info_x = '-'.join([str(self.kernel_size),str(self.filters),str(self.timesteps)])

    def add_callback(self,callback):
        """Adds a callback to the list of callbacks

        Argument:
            callback: callback to be added to the current callbacks list
        """

        self.callbacks.append(callback)

    def model(self):
        
        inputs = tf.keras.layers.Input(shape=(self.timesteps,self.features))

        x10 = tf.keras.layers.Conv1D(
                    filters=32,kernel_size=self.kernel_size,
                    padding='causal', dilation_rate = 1,
                    activation='relu')(inputs)        
        x11 = tf.keras.layers.Conv1D(
                    filters=16,kernel_size=self.kernel_size,
                    padding='causal', dilation_rate = 1,
                    activation='relu')(inputs)
        x12 = tf.keras.layers.Conv1D(
                    filters=8,kernel_size=self.kernel_size,
                    padding='causal', dilation_rate = 1,
                    activation='relu')(inputs)
        x13 = tf.keras.layers.Conv1D(
                    filters=8,kernel_size=self.kernel_size,
                    padding='causal', dilation_rate = 1,
                    activation='relu')(inputs)
        identity1 = tf.keras.layers.Conv1D(
            filters=64, kernel_size=1,
            padding='same', name='identity1'
            )(inputs)
        concat1 = tf.keras.layers.Concatenate()([x10,x11,x12,x13])
        add1 =tf.keras.layers.Add()([concat1,identity1])
        dropout1 = tf.keras.layers.Dropout(self.dropout)(add1)
        x21 = tf.keras.layers.Conv1D(
                    filters=64,kernel_size=self.kernel_size,
                    padding='causal', dilation_rate = 2,
                    activation='relu')(dropout1)
        x22 = tf.keras.layers.Conv1D(
                    filters=32,kernel_size=self.kernel_size,
                    padding='causal', dilation_rate = 2,
                    activation='relu')(dropout1)
        x23 = tf.keras.layers.Conv1D(
                    filters=32,kernel_size=self.kernel_size,
                    padding='causal', dilation_rate = 2,
                    activation='relu')(dropout1)
        identity2 = tf.keras.layers.Conv1D(
            filters=128, kernel_size=1,
            padding='same', name='identity2'
            )(dropout1)
        concat2 = tf.keras.layers.Concatenate()([x21,x22,x23])
        add2 =tf.keras.layers.Add()([concat2,identity2])
        dropout2 = tf.keras.layers.Dropout(self.dropout)(add2)
        x31 = tf.keras.layers.Conv1D(
                    filters=128,kernel_size=self.kernel_size,
                    padding='causal', dilation_rate = 4,
                    activation='relu')(dropout2)
        x32 = tf.keras.layers.Conv1D(
                    filters=128,kernel_size=self.kernel_size,
                    padding='causal', dilation_rate = 4,
                    activation='relu')(dropout2)
        identity3 = tf.keras.layers.Conv1D(
            filters=256, kernel_size=1,
            padding='same', name='identity3'
            )(dropout2)
        concat3 = tf.keras.layers.Concatenate()([x31,x32])
        add3 =tf.keras.layers.Add()([concat3,identity3])
        dropout3 = tf.keras.layers.Dropout(self.dropout)(add3)

        last_cov = tf.keras.layers.Conv1D(
            filters=512, kernel_size=self.kernel_size,
            padding='causal', dilation_rate = 8,
            activation='relu')(dropout3)

        glob_avg = tf.keras.layers.GlobalAveragePooling1D()(last_cov)
        outputs = tf.keras.layers.Dense(self.features)(glob_avg)

        model = tf.keras.models.Model(
            inputs = inputs,
            outputs = outputs,
            name = "PX4_TCN_Parallel"
        )

        # SGD is slow but give great results
        # optimizer = tf.keras.optimizers.SGD(lr=1e-7, momentum=0.9)

        # ADAM is fast but tends to over-fit
        # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,epsilon=1e-08,clipnorm=1.0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6,epsilon=1e-07,clipnorm=1.0)
        # loss  = tf.keras.losses.MeanAbsoluteError()
        # loss = tf.keras.losses.Huber(delta=0.1)
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

        # hist = model.fit(
        #     x=train,
        #     validation_data=validation,
        #     epochs=self.epochs,
        #     use_multiprocessing=True,
        #     workers=self.workers,
        #     callbacks=self.callbacks,
        #     verbose=1,
        #     shuffle=False
        # )

        hist = model.fit_generator(
            generator=train,
            validation_data=validation,
            epochs=self.epochs,
            #tensorflow:multiprocessing can interact badly with TensorFlow, causing nondeterministic deadlocks.
            #For high performance data pipelines tf.data is recommended
            use_multiprocessing=True,
            workers=self.workers,
            callbacks=self.callbacks,
            verbose=1,
            shuffle=False
        )

        model.save(
            os.path.join(
                parentdir,
                "logs/TCN/%s/%s/result.hdf5" % (self.path_suffix,self.info_x)
            )
        )

        logging.getLogger().setLevel(logging.INFO)
        logging.info("Training Time : %s secs"%str(time.time() - start))
        return hist


class DilatedPooling1D(tf.keras.layers.Layer):
    def __init__(self, pooling_type, pool_size=2, padding='causal',
                 dilation_rate=1, name=None, **kwargs):
        super(DilatedPooling1D, self).__init__(name=name, **kwargs)
        self.pooling_type = pooling_type.upper()
        self.pool_size = pool_size
        self.padding = padding.upper()
        self.dilation_rate = dilation_rate
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)

    def call(self, inputs):
        # Input should have rank 3 and be in NWC format
        padding = self.padding
        if self.padding == 'CAUSAL':
            # Compute the causal padding
            left_pad = self.dilation_rate * (self.pool_size - 1)
            inputs = tf.pad(inputs, [[0, 0, ], [left_pad, 0], [0, 0]])
            padding = 'VALID'

        outputs = tf.nn.pool(inputs,
                             window_shape=[self.pool_size],
                             pooling_type=self.pooling_type,
                             padding=padding,
                             dilations=[self.dilation_rate],
                             strides=[1],
                             data_format='NWC')
        return outputs

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'pooling_type': self.pooling_type,
            'pool_size': self.pool_size,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'input_spec': self.input_spec,
        })
        return config


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size: int,
                 padding: str,
                 activation: str = 'relu',
                 dropout_rate: float = 0,
                 kernel_initializer: str = 'he_normal',
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 **kwargs):
        """Defines the residual block for the TCN
        Args:
            x: The previous layer in the model
            training: boolean indicating whether the layer should behave in training mode or in inference mode
            dilation_rate: The dilation power of 2 we are using for this residual block
            nb_filters: The number of convolutional filters to use in this block
            kernel_size: The size of the convolutional kernel
            padding: The padding used in the convolutional layers, 'same' or 'causal'.
            activation: The final activation used in o = Activation(x + F(x))
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
            kwargs: Any initializers for Layer class.
        """

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.kernel_initializer = kernel_initializer
        self.layers = []
        self.layers_outputs = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(ResidualBlock, self).__init__(**kwargs)

    def _add_and_activate_layer(self, layer):
        """Helper function for building layer
        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.
        """
        self.layers.append(layer)
        self.layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape)

    def build(self, input_shape):

        with tf.keras.name_scope(self.name):  # name scope used to make sure weights get unique names
            self.layers = []
            self.res_output_shape = input_shape

            for k in range(2):
                name = 'conv1D_{}'.format(k)
                with tf.keras.name_scope(name):  # name scope used to make sure weights get unique names
                    self._add_and_activate_layer(tf.keras.layers.Conv1D(filters=self.nb_filters,
                                                        kernel_size=self.kernel_size,
                                                        dilation_rate=self.dilation_rate,
                                                        padding=self.padding,
                                                        name=name,
                                                        kernel_initializer=self.kernel_initializer))

                with tf.keras.name_scope('norm_{}'.format(k)):
                    if self.use_batch_norm:
                        self._add_and_activate_layer(tf.keras.layers.BatchNormalization())
                    elif self.use_layer_norm:
                        self._add_and_activate_layer(tf.keras.layers.LayerNormalization())

                self._add_and_activate_layer(tf.keras.layers.Activation('relu'))
                self._add_and_activate_layer(tf.keras.layers.SpatialDropout1D(rate=self.dropout_rate))

            if self.nb_filters != input_shape[-1]:
                # 1x1 conv to match the shapes.
                name = 'matching_conv1D'
                with tf.keras.name_scope(name):
                    # make and build this layer separately because it directly uses input_shape
                    self.shape_match_conv = tf.keras.layers.Conv1D(filters=self.nb_filters,
                                                   kernel_size=1,
                                                   padding='same',
                                                   name=name,
                                                   kernel_initializer=self.kernel_initializer)

            else:
                name = 'matching_identity'
                self.shape_match_conv = tf.keras.layers.Lambda(lambda x: x, name=name)

            with tf.keras.name_scope(name):
                self.shape_match_conv.build(input_shape)
                self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            self.final_activation = tf.keras.layers.Activation(self.activation)
            self.final_activation.build(self.res_output_shape)  # probably isn't necessary

            # this is done to force Keras to add the layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)
            self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
            self.__setattr__(self.final_activation.name, self.final_activation)

            super(ResidualBlock, self).build(input_shape)  # done to make sure self.built is set True

    def call(self, inputs, training=None):
        """
        Returns: A tuple where the first element is the residual model tensor, and the second
                 is the skip connection tensor.
        """
        x = inputs
        self.layers_outputs = [x]
        for layer in self.layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            x = layer(x, training=training) if training_flag else layer(x)
            self.layers_outputs.append(x)
        x2 = self.shape_match_conv(inputs)
        self.layers_outputs.append(x2)
        res_x = layers.add([x2, x])
        self.layers_outputs.append(res_x)

        res_act_x = self.final_activation(res_x)
        self.layers_outputs.append(res_act_x)
        return [res_act_x, x]

    def compute_output_shape(self, input_shape):
        return [self.res_output_shape, self.res_output_shape]



def is_power_of_two(num: int):
    return num != 0 and ((num & (num - 1)) == 0)


def adjust_dilations(dilations: list):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations    

        