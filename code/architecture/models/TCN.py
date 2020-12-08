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

def ResidualLayer_Pooling(input_tensor,dilation = [1,2],
                    padding = 'causal', nb_filters=16,
                  return_sequences = False,
                    kernel_size = 3, name = 'residual_', name_suffix= '1'):
    name = name
    identity = input_tensor
    conv_x = input_tensor
    with tf.name_scope(name):
        for k in range(2):
            name1 = 'conv1D_{}_{}'.format(name_suffix,k)
            with tf.name_scope(name1):
                conv_x = tf.keras.layers.Conv1D(
                    filters=nb_filters,kernel_size=kernel_size,
                    padding='causal', dilation_rate = dilation[k],
                    activation='relu',name=name1
                )(conv_x)
        name_dilated = 'DilatedPooling1D_{}'.format(name_suffix)
        with tf.name_scope(name_dilated):
            dilatedpooling = DilatedPooling1D('max',dilation_rate=dilation[-1],name=name_dilated)(conv_x)
        name2 = 'identity_{}'.format(name_suffix)
        with tf.name_scope(name2):
            identity = tf.keras.layers.Conv1D(
            filters=nb_filters, kernel_size=1,
            padding='same', name=name2
            )(identity)
        residual_out = tf.keras.layers.Add()([dilatedpooling,identity])
        name3 = 'residual_out_{}'.format(name_suffix)
        with tf.name_scope(name3):            
            if not return_sequences:
                return tf.keras.layers.Lambda(lambda x: x[:,-1,:], name=name3)(residual_out)
            return residual_out

def ResidualLayer_Dropout(input_tensor,dilation = [1,2],
                    padding = 'causal', nb_filters=16,
                  return_sequences = False, dropout_rate = 0.5,
                    kernel_size = 3, name = 'residual_', name_suffix= '1'):
    name = name
    identity = input_tensor
    conv_x = input_tensor
    with tf.name_scope(name):
        for k in range(2):
            name1 = 'Conv1D_{}_{}'.format(name_suffix,k)            
            with tf.name_scope(name1):
                conv_x = tf.keras.layers.Conv1D(
                    filters=nb_filters,kernel_size=kernel_size,
                    padding='causal', dilation_rate = dilation[k],
                    name=name1)(conv_x)
                conv_x = tf.keras.layers.ReLU()(conv_x)
                conv_x = tf.keras.layers.LayerNormalization()(conv_x)
                conv_x = tf.keras.layers.Dropout(dropout_rate)(conv_x)

        name2 = 'Identity_{}'.format(name_suffix)
        with tf.name_scope(name2):
            identity = tf.keras.layers.Conv1D(
            filters=nb_filters, kernel_size=1,
            padding='same', name=name2
            )(identity)
        residual_out = tf.keras.layers.Add()([conv_x,identity])
        residual_out = tf.keras.layers.LayerNormalization()(residual_out)
        residual_out = tf.keras.layers.Dropout(dropout_rate)(residual_out)
        name3 = 'Residual_out_{}'.format(name_suffix)
        with tf.name_scope(name3):            
            if not return_sequences:
                return tf.keras.layers.Lambda(lambda x: x[:,-1,:], name=name3)(residual_out)
            return residual_out

def ResidualLayer(input_tensor,dilation = [1,2],
                    padding = 'causal', nb_filters=16,
                  return_sequences = False,
                    kernel_size = 3, name = 'residual_', name_suffix= '1'):
    name = name
    identity = input_tensor
    conv_x = input_tensor
    with tf.name_scope(name):
        for k in range(2):
            name1 = 'conv1D_{}_{}'.format(name_suffix,k)
            with tf.name_scope(name1):
                conv_x = tf.keras.layers.Conv1D(
                    filters=nb_filters,kernel_size=kernel_size,
                    padding='causal', dilation_rate = dilation[k],
                    activation='relu',name=name1
                )(conv_x)
        name2 = 'identity_{}'.format(name_suffix)
        with tf.name_scope(name2):
            identity = tf.keras.layers.Conv1D(
            filters=nb_filters, kernel_size=1,
            padding='same', name=name2
            )(identity)
        residual_out = tf.keras.layers.Add()([conv_x,identity])
        name3 = 'residual_out_{}'.format(name_suffix)
        with tf.name_scope(name3):            
            if not return_sequences:
                return tf.keras.layers.Lambda(lambda x: x[:,-1,:], name=name3)(residual_out)
            return residual_out


class TCNModel(object):
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
        self.workers = 2
        self.callbacks = list()
        self.dilation = [1,2,4,8,16,32,64,128,256,512]

        self.path_suffix = '-'.join([train_suffix,valid_suffix])

        self.props = {} # in case I wanna use it


        self.train_data = train_data
        self.valid_data = valid_data
        self.features = self.train_data.shape[1]
        self.info_x = '-'.join([str(self.kernel_size),str(self.filters),str(self.timesteps),str(self.dropout)])

    def add_callback(self,callback):
        """Adds a callback to the list of callbacks

        Argument:
            callback: callback to be added to the current callbacks list
        """

        self.callbacks.append(callback)

    def model(self):
        
        inputs = tf.keras.layers.Input(shape=(self.timesteps,self.features))
        x = ResidualLayer_Dropout(
            inputs,nb_filters=self.filters*2,kernel_size = self.kernel_size,
            dilation=self.dilation[:2],name='residual_block_1', dropout_rate=self.dropout,
            name_suffix = '1',return_sequences=True)
        x = ResidualLayer_Dropout(
            x,nb_filters=self.filters,kernel_size = self.kernel_size,
            dilation=self.dilation[2:4],name='residual_block_2', dropout_rate=self.dropout,
            name_suffix = '2',return_sequences=True)
        x = ResidualLayer_Dropout(
            x,nb_filters=self.filters*2,kernel_size = self.kernel_size,
            dilation=self.dilation[4:6],name='residual_block_3', dropout_rate=self.dropout,
            name_suffix = '3',return_sequences=True)
        x = ResidualLayer_Dropout(
            x,nb_filters=self.filters*4,kernel_size = self.kernel_size,
            dilation=self.dilation[6:8],name='residual_block_4', dropout_rate=self.dropout,
            name_suffix = '4')

        x = tf.keras.layers.RepeatVector(1)(x)
        x = tf.keras.layers.Conv1D(filters=self.features,kernel_size=1,padding = 'same',name='Last_Conv1d')(x)
        x = tf.keras.layers.ReLU()(x)
        outputs = tf.keras.layers.GlobalAveragePooling1D()(x)

        # outputs = tf.keras.layers.Dense(self.features)(x)
        model = tf.keras.models.Model(
            inputs = inputs,
            outputs = outputs,
            name = "PX4_TCN_4Layer"
        )


        # inputs = tf.keras.layers.Input(shape=(self.timesteps,self.features))
        # x = ResidualLayer_Dropout(
        #     inputs,nb_filters=self.filters,kernel_size = self.kernel_size,
        #     dilation=self.dilation[:2],name='residual_block_1',
        #     name_suffix = '1',return_sequences=True,dropout_rate=self.dropout)
        # x = ResidualLayer_Dropout(
        #     x,nb_filters=self.filters,kernel_size = self.kernel_size,
        #     dilation=self.dilation[2:4],name='residual_block_2',
        #     name_suffix = '2',return_sequences=True,dropout_rate=self.dropout)
        # # x2 = tf.keras.layers.Lambda(lambda x: x[:,-1,:], name='Conv1D_2_last')(x)
        # x = ResidualLayer_Dropout(
        #     x,nb_filters=self.filters,kernel_size = self.kernel_size,
        #     dilation=self.dilation[4:6],name='residual_block_3',
        #     name_suffix = '3',return_sequences=True,dropout_rate=self.dropout)
        # # x3 = tf.keras.layers.Lambda(lambda x: x[:,-1,:], name='Conv1D_3_last')(x)
        # x = ResidualLayer_Dropout(
        #     x,nb_filters=self.filters,kernel_size = self.kernel_size,
        #     dilation=self.dilation[6:8],name='residual_block_4',
        #     name_suffix = '4',return_sequences=True,dropout_rate=self.dropout)
        # # x4 = tf.keras.layers.Lambda(lambda x: x[:,-1,:], name='Conv1D_4_last')(x)
        # x = ResidualLayer_Dropout(
        #     x,nb_filters=self.filters,kernel_size = self.kernel_size,
        #     dilation=self.dilation[8:10],name='residual_block_5',
        #     name_suffix = '5',dropout_rate=self.dropout)
        # # x5 = tf.keras.layers.Lambda(lambda x: x, name='Conv1D_5_last')(x)

        # # concat = tf.keras.layers.Concatenate(name='ConcatLayer')([x2,x3,x4,x5])

        # # outputs = tf.keras.layers.Dense(self.features)(concat)

        # outputs = tf.keras.layers.Dense(self.features)(x)

        # model = tf.keras.models.Model(
        #     inputs = inputs,
        #     outputs = outputs,
        #     name = "PX4_TCN_Dropout"
        # )

        # SGD is slow but give great results
        # optimizer = tf.keras.optimizers.SGD(lr=1e-7, momentum=0.9)

        # ADAM is fast but tends to over-fit
        # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,epsilon=1e-08,clipnorm=1.0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5,epsilon=1e-07,clipnorm=1.0)
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

class TCNPooled(object):
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
        self.info_x = '-'.join([str(self.kernel_size),str(self.filters),str(self.timesteps),str(self.dropout)])

    def add_callback(self,callback):
        """Adds a callback to the list of callbacks

        Argument:
            callback: callback to be added to the current callbacks list
        """

        self.callbacks.append(callback)

    def model(self):
        
        inputs = tf.keras.layers.Input(shape=(self.timesteps,self.features))
        x = ResidualLayer_Pooling(
            inputs,nb_filters=self.filters,kernel_size = self.kernel_size,
            dilation=self.dilation[:2],name='residual_block_1',
            name_suffix = '1',return_sequences=True)
        x = ResidualLayer_Pooling(
            x,nb_filters=self.filters,kernel_size = self.kernel_size,
            dilation=self.dilation[2:4],name='residual_block_2',
            name_suffix = '2',return_sequences=True)
        x = ResidualLayer_Pooling(
            x,nb_filters=self.filters,kernel_size = self.kernel_size,
            dilation=self.dilation[4:6],name='residual_block_3',
            name_suffix = '3',return_sequences=True)
        # x3 = tf.keras.layers.Lambda(lambda x: x[:,-1,:], name='Conv1D_3_last')(x)
        x = ResidualLayer_Pooling(
            x,nb_filters=self.filters,kernel_size = self.kernel_size,
            dilation=self.dilation[6:8],name='residual_block_4',
            name_suffix = '4',return_sequences=True)
        # x4 = tf.keras.layers.Lambda(lambda x: x[:,-1,:], name='Conv1D_4_last')(x)
        x = ResidualLayer_Pooling(
            x,nb_filters=self.filters,kernel_size = self.kernel_size,
            dilation=self.dilation[8:10],name='residual_block_5',
            name_suffix = '5')
        # x5 = tf.keras.layers.Lambda(lambda x: x, name='Conv1D_5_last')(x)

        # concat = tf.keras.layers.Concatenate(name='ConcatLayer')([x3,x4,x5])

        # outputs = tf.keras.layers.Dense(self.features)(concat)

        outputs = tf.keras.layers.Dense(self.features)(x)

        model = tf.keras.models.Model(
            inputs = inputs,
            outputs = outputs,
            name = "PX4_TCN_Pooled"
        )

        # SGD is slow but give great results
        # optimizer = tf.keras.optimizers.SGD(lr=1e-7, momentum=0.9)

        # ADAM is fast but tends to over-fit
        # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,epsilon=1e-08,clipnorm=1.0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,epsilon=1e-07,clipnorm=1.0)
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

        model.save_weights(
            os.path.join(
                parentdir,
                "logs/TCN/%s/%s/weights_pooled.hdf5" % (self.path_suffix,self.info_x)
            )
        )

        logging.getLogger().setLevel(logging.INFO)
        logging.info("Training Time : %s secs"%str(time.time() - start))
        return hist

class TCNGrouped(object):
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
        self.workers = 2
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

# Batch before ReLU
def ResidualLayer_Dropout_ex(input_tensor,dilation = [1,2],
                    padding = 'causal', nb_filters=16,
                  return_sequences = False, dropout_rate = 0.5,
                    kernel_size = 3, name = 'residual_', name_suffix= '1'):
    name = name
    identity = input_tensor
    conv_x = input_tensor
    with tf.name_scope(name):
        for k in range(2):
            name1 = 'Conv1D_{}_{}'.format(name_suffix,k)            
            with tf.name_scope(name1):
                conv_x = tf.keras.layers.Conv1D(
                    filters=nb_filters,kernel_size=kernel_size,
                    padding='causal', dilation_rate = dilation[k],
                    name=name1)(conv_x)
                conv_x = tf.keras.layers.BatchNormalization()(conv_x)
                conv_x = tf.keras.layers.LayerNormalization()(conv_x)
                conv_x = tf.keras.layers.LeakyReLU(alpha=0.5)(conv_x)
                conv_x = tf.keras.layers.Dropout(dropout_rate)(conv_x)

        name2 = 'Identity_{}'.format(name_suffix)
        with tf.name_scope(name2):
            identity = tf.keras.layers.Conv1D(
            filters=nb_filters, kernel_size=1,
            padding='same', name=name2
            )(identity)
        residual_out = tf.keras.layers.Add()([conv_x,identity])
        name3 = 'Residual_out_{}'.format(name_suffix)
        with tf.name_scope(name3):            
            if not return_sequences:
                return tf.keras.layers.Lambda(lambda x: x[:,-1,:], name=name3)(residual_out)
            return residual_out

class TCNModel_Ex(object):
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
        self.info_x = '-'.join([str(self.kernel_size),str(self.filters),str(self.timesteps),str(self.dropout)])

    def add_callback(self,callback):
        """Adds a callback to the list of callbacks

        Argument:
            callback: callback to be added to the current callbacks list
        """

        self.callbacks.append(callback)

    def model(self):
        
        inputs = tf.keras.layers.Input(shape=(self.timesteps,self.features))
        x = ResidualLayer_Dropout_ex(
            inputs,nb_filters=self.filters*4,kernel_size = self.kernel_size,
            dilation=self.dilation[:2],name='residual_block_1', dropout_rate=self.dropout,
            name_suffix = '1',return_sequences=True)
        x = ResidualLayer_Dropout_ex(
            x,nb_filters=self.filters*2,kernel_size = self.kernel_size,
            dilation=self.dilation[2:4],name='residual_block_2', dropout_rate=self.dropout,
            name_suffix = '2',return_sequences=True)
        x = ResidualLayer_Dropout_ex(
            x,nb_filters=self.filters*4,kernel_size = self.kernel_size,
            dilation=self.dilation[4:6],name='residual_block_3', dropout_rate=self.dropout,
            name_suffix = '3',return_sequences=True)
        
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.RepeatVector(1)(x)
        x = tf.keras.layers.Conv1D(filters=self.features,kernel_size=1,padding = 'same',name='Last_Conv1d')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.5)(x)
        outputs = tf.keras.layers.GlobalAveragePooling1D()(x)

        model = tf.keras.models.Model(
            inputs = inputs,
            outputs = outputs,
            name = "PX4_TCN_3Layer"
        )


        # inputs = tf.keras.layers.Input(shape=(self.timesteps,self.features))
        # x = ResidualLayer_Dropout(
        #     inputs,nb_filters=self.filters,kernel_size = self.kernel_size,
        #     dilation=self.dilation[:2],name='residual_block_1',
        #     name_suffix = '1',return_sequences=True,dropout_rate=self.dropout)
        # x = ResidualLayer_Dropout(
        #     x,nb_filters=self.filters,kernel_size = self.kernel_size,
        #     dilation=self.dilation[2:4],name='residual_block_2',
        #     name_suffix = '2',return_sequences=True,dropout_rate=self.dropout)
        # # x2 = tf.keras.layers.Lambda(lambda x: x[:,-1,:], name='Conv1D_2_last')(x)
        # x = ResidualLayer_Dropout(
        #     x,nb_filters=self.filters,kernel_size = self.kernel_size,
        #     dilation=self.dilation[4:6],name='residual_block_3',
        #     name_suffix = '3',return_sequences=True,dropout_rate=self.dropout)
        # # x3 = tf.keras.layers.Lambda(lambda x: x[:,-1,:], name='Conv1D_3_last')(x)
        # x = ResidualLayer_Dropout(
        #     x,nb_filters=self.filters,kernel_size = self.kernel_size,
        #     dilation=self.dilation[6:8],name='residual_block_4',
        #     name_suffix = '4',return_sequences=True,dropout_rate=self.dropout)
        # # x4 = tf.keras.layers.Lambda(lambda x: x[:,-1,:], name='Conv1D_4_last')(x)
        # x = ResidualLayer_Dropout(
        #     x,nb_filters=self.filters,kernel_size = self.kernel_size,
        #     dilation=self.dilation[8:10],name='residual_block_5',
        #     name_suffix = '5',dropout_rate=self.dropout)
        # # x5 = tf.keras.layers.Lambda(lambda x: x, name='Conv1D_5_last')(x)

        # # concat = tf.keras.layers.Concatenate(name='ConcatLayer')([x2,x3,x4,x5])

        # # outputs = tf.keras.layers.Dense(self.features)(concat)

        # outputs = tf.keras.layers.Dense(self.features)(x)

        # model = tf.keras.models.Model(
        #     inputs = inputs,
        #     outputs = outputs,
        #     name = "PX4_TCN_Dropout"
        # )

        # SGD is slow but give great results
        # optimizer = tf.keras.optimizers.SGD(lr=1e-7, momentum=0.9)

        # ADAM is fast but tends to over-fit
        # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,epsilon=1e-08,clipnorm=1.0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,epsilon=1e-07,clipnorm=1.0)
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



