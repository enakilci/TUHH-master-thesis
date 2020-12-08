import os
import numpy as np
import pandas as pd
import math

import tensorflow as tf
# import tf.keras as K

filepath = os.path.abspath(__file__)

class PX4Generator(tf.keras.utils.Sequence):

    def __init__(self,data, batch_size = 48, timesteps = 400):

        self.batch_size = batch_size
        self.timesteps = timesteps
        self.dir = os.path.dirname(filepath)
        self.dataset_path = os.path.join(self.dir,'datasets')
        self.data = data
        self.num_cols = self.data.shape[1]
        self.len = math.floor((self.data.shape[0]-self.timesteps)/(self.batch_size))


    def __len__(self):
        """Number of batch in the Sequence.

        Returns:
            The number of batches in the Sequence.
        """
        if ((self.data.shape[0]-self.timesteps) % (self.batch_size) == 0):
            return self.len

        return self.len + 1

    def group_data(self, start_idx, end_idx):
        """trims the values of the data between given thresholds.

        Arguments:
            start_idx: Minimum threshold 
            end_idx: Maximum threshold

        Returns:
            A clipped array       
        """

        return self.data[start_idx:end_idx]


    def __getitem__(self, batch_idx):
        """Gets batch at position `batch_idx`.

        Arguments:
            batch_idx: position of the batch in the Sequence.

        Returns:
            A batch
        """

        x = np.zeros((self.batch_size,self.timesteps,self.num_cols))
        y = np.zeros((self.batch_size,self.num_cols))

        idxi = batch_idx * self.batch_size
        # debug only
        print("batch_idx: ",batch_idx)
        # print("batch_size: ",self.batch_size)
        # print("timesteps: ",self.timesteps)
        # print("idxi: ", idxi)
        
        if(batch_idx == (self.__len__() - 1)):
            last_batch_size = self.data.shape[0] - self.timesteps - idxi
            x = np.zeros((last_batch_size,self.timesteps,self.num_cols))
            y = np.zeros((last_batch_size,self.num_cols))
            # debug only
            print("last_batch_size: ",last_batch_size)
            for i in range(last_batch_size):
                x[i,:,:] = self.group_data(idxi,idxi + self.timesteps)
                y[i,:] = self.data[idxi + self.timesteps]
                idxi +=1

            return x , y

        for i in range(self.batch_size):
            x[i,:,:] = self.group_data(idxi,idxi + self.timesteps)
            y[i,:] = self.data[idxi + self.timesteps]
            idxi +=1

        return x , y
