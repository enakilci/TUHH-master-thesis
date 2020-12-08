import os
import numpy as np 
import pandas as pd
import sys
import errno
import tensorflow as tf

from utils import load_dataset, create_dirs
from data_generator import PX4Generator
from models.LSTMModel import LSTMModel
from models.CNN1D import CNN1DModel
from models.Attmodel import AdditiveAttentionModel, MultiplicativeAttentionModel, TemporalAttentionModel
from models.TCN import TCNModel
from models.TCN_parallel import TCNParallelModel

filepath = os.path.abspath(__file__)

class AnomalyDetector:
    def __init__(self, train_datapath_suffix:str = None,
                    valid_datapath_suffix:str = None, test_datapath_suffix:str = None,
                    model = 'lstm', batch_size = 64, timesteps = 750, epochs = 7,
                    dropout = 0.4, hidden = 20, hidden_att = 300,
                    kernel_size = 5, filters = 20, pool_size = 2):
        
        self.model = model
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.epochs = epochs
        self.dropout = dropout
        self.hidden = hidden
        self.hidden_att = hidden_att
        self.kernel_size = kernel_size
        self.filters = filters
        self.pool_size = pool_size

        if train_datapath_suffix != None and valid_datapath_suffix != None:
            try:
                
                self.train_datapath_suffix = train_datapath_suffix
                self.valid_datapath_suffix = valid_datapath_suffix
                filedir = os.path.dirname(filepath)
                dataset_path = os.path.join(filedir,'datasets')
                print('Loading the training dataset from the date ' + str(train_datapath_suffix))
                dataset_train = load_dataset(os.path.join(dataset_path,str(train_datapath_suffix)))

                print('Loading the validation dataset from the date ' + str(valid_datapath_suffix))
                dataset_valid = load_dataset(os.path.join(dataset_path,str(valid_datapath_suffix)))

            except OSError as err:
                print("OS error: {0}".format(err))
            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise


            self.train_data = dataset_train.values
            self.valid_data = dataset_valid.values
        else:
            print('No dataset was given to train')


        if test_datapath_suffix != None:
            try:
                self.test_datapath_suffix = test_datapath_suffix
                filedir = os.path.dirname(filepath)
                dataset_path = os.path.join(filedir,'datasets')
                print('Loading the training dataset from the date ' + str(self.test_datapath_suffix))
                dataset_test = load_dataset(os.path.join(dataset_path,str(self.test_datapath_suffix)))

            except OSError as err:
                print("OS error: {0}".format(err))
            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise


            self.test_data = dataset_test.values
        else:
            print('No dataset was given for testing')
      
    def run(self):

        path_suffix = self.train_datapath_suffix+'-'+self.valid_datapath_suffix

        if self.model == 'lstm':

            logs_path_lstm = os.path.join(os.path.dirname(filepath),"logs/LSTM/%s/%s" % (path_suffix,info_x_lstm))
            tensorboard_path_lstm = os.path.join(os.path.dirname(filepath),"tensorboard/LSTM/%s/%s" % (path_suffix,info_x_lstm))
            create_dirs(logs_path_lstm)
            create_dirs(tensorboard_path_lstm)

            LSTM = LSTMModel(
                self.train_data,
                self.valid_data,
                self.train_datapath_suffix,
                self.valid_datapath_suffix,  
                batch_size = self.batch_size,
                epochs = self.epochs,              
                timesteps = self.timesteps,
                hidden = self.hidden,
                dropout = self.dropout
                
            )

            LSTM.add_callback(
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(
                        logs_path_lstm,
                        "{epoch:02d}-{val_loss:.4f}.hdf5"
                    ),
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=True,
                    mode='min'
                )
            )
            
            LSTM.add_callback(
                tf.keras.callbacks.TensorBoard(tensorboard_path_lstm)
            )
            LSTM.add_callback(
                tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1)
            )
            LSTM.add_callback(
                tf.keras.callbacks.TerminateOnNaN()
            )

            hist = LSTM.fit()
        
        if self.model == 'cnn1d':

            logs_path_cnn1d = os.path.join(os.path.dirname(filepath),"logs/CNN1D/%s/%s" % (path_suffix,info_x_cnn))
            tensorboard_path_cnn1d = os.path.join(os.path.dirname(filepath),"tensorboard/CNN1D/%s/%s" % (path_suffix,info_x_cnn))
            create_dirs(logs_path_cnn1d)
            create_dirs(tensorboard_path_cnn1d)            

            CNN1D = CNN1DModel(
                self.train_data,
                self.valid_data,
                self.train_datapath_suffix,
                self.valid_datapath_suffix,  
                batch_size = self.batch_size,
                epochs = self.epochs,
                timesteps = self.timesteps,
                kernel_size = self.kernel_size,
                filters = self.filters,
                pool_size = self.pool_size,
                dropout = self.dropout
            )


            CNN1D.add_callback(
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(
                        logs_path_cnn1d,
                        "{epoch:02d}-{val_loss:.4f}.hdf5"
                    ),
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=True,
                    mode='min'
                )
            )
            
            CNN1D.add_callback(
                tf.keras.callbacks.TensorBoard(tensorboard_path_cnn1d)
            )
            CNN1D.add_callback(
                tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1)
            )
            CNN1D.add_callback(
                tf.keras.callbacks.TerminateOnNaN()
            )

            hist = CNN1D.fit()

        if self.model == 'attention_multip':
            logs_path_att = os.path.join(os.path.dirname(filepath),"logs/MultiplicativeAttentionModel/%s/%s" % (path_suffix,info_x_att))
            tensorboard_path_att = os.path.join(os.path.dirname(filepath),"tensorboard/MultiplicativeAttentionModel/%s/%s" % (path_suffix,info_x_att))
            create_dirs(logs_path_att)
            create_dirs(tensorboard_path_att)

            MultipAttModel = MultiplicativeAttentionModel(
                self.train_data,
                self.valid_data,
                self.train_datapath_suffix,
                self.valid_datapath_suffix,  
                batch_size = self.batch_size,
                epochs = self.epochs,              
                timesteps = self.timesteps,
                hidden = self.hidden,
                hidden_att = self.hidden_att,
                dropout = self.dropout                
            )

            MultipAttModel.add_callback(
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(
                        logs_path_att,
                        "{epoch:02d}-{val_loss:.4f}.hdf5"
                    ),
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=True,
                    mode='min'
                )
            )
            
            MultipAttModel.add_callback(
                tf.keras.callbacks.TensorBoard(tensorboard_path_att)
            )
            MultipAttModel.add_callback(
                tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1)
            )
            MultipAttModel.add_callback(
                tf.keras.callbacks.TerminateOnNaN()
            )

            hist = MultipAttModel.fit()

        if self.model == 'attention_add':
            logs_path_att = os.path.join(os.path.dirname(filepath),"logs/AdditiveAttentionModel/%s/%s" % (path_suffix,info_x_att))
            tensorboard_path_att = os.path.join(os.path.dirname(filepath),"tensorboard/AdditiveAttentionModel/%s/%s" % (path_suffix,info_x_att))
            create_dirs(logs_path_att)
            create_dirs(tensorboard_path_att)

            AddAttModel = AdditiveAttentionModel(
                self.train_data,
                self.valid_data,
                self.train_datapath_suffix,
                self.valid_datapath_suffix,  
                batch_size = self.batch_size,
                epochs = self.epochs,              
                timesteps = self.timesteps,
                hidden = self.hidden,
                hidden_att = self.hidden_att,
                dropout = self.dropout                
            )

            AddAttModel.add_callback(
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(
                        logs_path_att,
                        "{epoch:02d}-{val_loss:.4f}.hdf5"
                    ),
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=True,
                    mode='min'
                )
            )
            
            AddAttModel.add_callback(
                tf.keras.callbacks.TensorBoard(tensorboard_path_att)
            )
            AddAttModel.add_callback(
                tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1)
            )
            AddAttModel.add_callback(
                tf.keras.callbacks.TerminateOnNaN()
            )

            hist = AddAttModel.fit()

        if self.model == 'attention_temp':
            logs_path_att = os.path.join(os.path.dirname(filepath),"logs/TemporalAttentionModel/%s/%s" % (path_suffix,info_x_att_temp))
            tensorboard_path_att = os.path.join(os.path.dirname(filepath),"tensorboard/TemporalAttentionModel/%s/%s" % (path_suffix,info_x_att_temp))
            create_dirs(logs_path_att)
            create_dirs(tensorboard_path_att)

            TempAttModel = TemporalAttentionModel(
                self.train_data,
                self.valid_data,
                self.train_datapath_suffix,
                self.valid_datapath_suffix,  
                batch_size = self.batch_size,
                epochs = self.epochs,              
                timesteps = self.timesteps,
                hidden = self.hidden,
                dropout = self.dropout                
            )

            TempAttModel.add_callback(
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(
                        logs_path_att,
                        "{epoch:02d}-{val_loss:.4f}.hdf5"
                    ),
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=True,
                    mode='min'
                )
            )
            
            TempAttModel.add_callback(
                tf.keras.callbacks.TensorBoard(tensorboard_path_att)
            )
            TempAttModel.add_callback(
                tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1)
            )
            TempAttModel.add_callback(
                tf.keras.callbacks.TerminateOnNaN()
            )

            hist = TempAttModel.fit()

        if self.model == 'tcn':


            logs_path_tcn = os.path.join(os.path.dirname(filepath),"logs/TCN/%s/%s" % (path_suffix,info_x_tcn))
            tensorboard_path_tcn = os.path.join(os.path.dirname(filepath),"tensorboard/TCN/%s/%s" % (path_suffix,info_x_tcn))
            create_dirs(logs_path_tcn)
            create_dirs(tensorboard_path_tcn)            

            TCN = TCNModel(
                self.train_data,
                self.valid_data,
                self.train_datapath_suffix,
                self.valid_datapath_suffix,  
                batch_size = self.batch_size,
                epochs = self.epochs,
                timesteps = self.timesteps,
                kernel_size = self.kernel_size,
                filters = self.filters,
                pool_size = self.pool_size,
                dropout = self.dropout
            )

            # TCN = TCNParallelModel(
            #     self.train_data,
            #     self.valid_data,
            #     self.train_datapath_suffix,
            #     self.valid_datapath_suffix,  
            #     batch_size = self.batch_size,
            #     epochs = self.epochs,
            #     timesteps = self.timesteps,
            #     kernel_size = self.kernel_size,
            #     filters = self.filters,
            #     pool_size = self.pool_size,
            #     dropout = self.dropout
            # )

            TCN.add_callback(
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(
                        logs_path_tcn,
                        "{epoch:02d}-{val_loss:.4f}.hdf5"
                    ),
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=True,
                    mode='min'
                )
            )
            
            TCN.add_callback(
                tf.keras.callbacks.TensorBoard(tensorboard_path_tcn)
            )
            TCN.add_callback(
                tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1,patience=1)
            )
            TCN.add_callback(
                tf.keras.callbacks.TerminateOnNaN()
            )

            hist = TCN.fit()

    def predict(self):

        model_path_suffix = self.train_datapath_suffix+'-'+self.valid_datapath_suffix

        if self.model == 'lstm':

            prediction_path_lstm = os.path.join(
                                    os.path.dirname(filepath),
                                    "predictions/LSTM/%s-%s" % (self.test_datapath_suffix,info_x_lstm)
                                    )
            create_dirs(prediction_path_lstm)

            LSTM = tf.keras.models.load_model(
                os.path.join(
                    os.path.dirname(filepath),
                    "logs/LSTM/%s/%s/result.hdf5" % (model_path_suffix,info_x_lstm)    
                )
            )
            LSTM.summary()
            test_batches = PX4Generator(
                    self.test_data,
                    batch_size=self.batch_size,timesteps=self.timesteps
                )

            print('Predicting test dataset...')
            predictions_test = LSTM.predict(test_batches)
            y_true_test = self.test_data[self.timesteps:]
            np.save(os.path.join(prediction_path_lstm,'predictions_test'), predictions_test)
            np.save(os.path.join(prediction_path_lstm,'y_true_test'),y_true_test)
            print('Predictions are saved to ', prediction_path_lstm)

            train_batches = PX4Generator(
                    self.train_data,
                    batch_size=self.batch_size,timesteps=self.timesteps
                )

            print('Predicting training dataset...')
            predictions_train = LSTM.predict(train_batches)
            y_true_train = self.train_data[self.timesteps:]
            np.save(os.path.join(prediction_path_lstm,'predictions_test'), predictions_train)
            np.save(os.path.join(prediction_path_lstm,'y_true_test'),y_true_train)
            print('Predictions are saved to ', prediction_path_lstm)

        if self.model == 'attention_add':

            prediction_path_att = os.path.join(
                                    os.path.dirname(filepath),
                                    "predictions/AdditiveAttentionModel/%s-%s" % (self.test_datapath_suffix,info_x_att)
                                    )
            create_dirs(prediction_path_att)

            AddAttModel = tf.keras.models.load_model(
                os.path.join(
                    os.path.dirname(filepath),
                    "logs/AdditiveAttentionModel/%s/%s/result.hdf5" % (model_path_suffix,info_x_att)    
                )
            )
            AddAttModel.summary()
            test_batches = PX4Generator(
                    self.test_data,
                    batch_size=self.batch_size,timesteps=self.timesteps
                )

            print('Predicting test dataset...')
            predictions_test = AddAttModel.predict(test_batches)
            y_true_test = self.test_data[self.timesteps:]
            np.save(os.path.join(prediction_path_att,'predictions_test'), predictions_test)
            np.save(os.path.join(prediction_path_att,'y_true_test'),y_true_test)
            print('Predictions are saved to ', prediction_path_att)

            train_batches = PX4Generator(
                    self.train_data,
                    batch_size=self.batch_size,timesteps=self.timesteps
                )

            print('Predicting training dataset...')
            predictions_train = AddAttModel.predict(train_batches)
            y_true_train = self.train_data[self.timesteps:]
            np.save(os.path.join(prediction_path_att,'predictions_train'), predictions_train)
            np.save(os.path.join(prediction_path_att,'y_true_train'),y_true_train)
            print('Predictions are saved to ', prediction_path_att)

        if self.model == 'attention_multip':

            prediction_path_att = os.path.join(
                                    os.path.dirname(filepath),
                                    "predictions/MultiplicativeAttentionModel/%s-%s" % (self.test_datapath_suffix,info_x_att)
                                    )
            create_dirs(prediction_path_att)

            MultipAttModel = tf.keras.models.load_model(
                os.path.join(
                    os.path.dirname(filepath),
                    "logs/MultiplicativeAttentionModel/%s/%s/result.hdf5" % (model_path_suffix,info_x_att)    
                )
            )
            MultipAttModel.summary()
            test_batches = PX4Generator(
                    self.test_data,
                    batch_size=self.batch_size,timesteps=self.timesteps
                )

            print('Predicting test dataset...')
            predictions_test = MultipAttModel.predict(test_batches)
            y_true_test = self.test_data[self.timesteps:]
            np.save(os.path.join(prediction_path_att,'predictions_test'), predictions_test)
            np.save(os.path.join(prediction_path_att,'y_true_test'),y_true_test)
            print('Predictions are saved to ', prediction_path_att)

            train_batches = PX4Generator(
                    self.train_data,
                    batch_size=self.batch_size,timesteps=self.timesteps
                )

            print('Predicting training dataset...')
            predictions_train = MultipAttModel.predict(train_batches)
            y_true_train = self.train_data[self.timesteps:]
            np.save(os.path.join(prediction_path_att,'predictions_train'), predictions_train)
            np.save(os.path.join(prediction_path_att,'y_true_train'),y_true_train)
            print('Predictions are saved to ', prediction_path_att)

        if self.model == 'tcn':

            prediction_path_tcn = os.path.join(
                                    os.path.dirname(filepath),
                                    "predictions/TCN/%s-%s" % (self.test_datapath_suffix,info_x_tcn)
                                    )
            create_dirs(prediction_path_tcn)

            TCN = tf.keras.models.load_model(
                os.path.join(
                    os.path.dirname(filepath),
                    "logs/TCN/%s/%s/result.hdf5" % (model_path_suffix,info_x_tcn)    
                )
            )
            TCN.summary()
            test_batches = PX4Generator(
                    self.test_data,
                    batch_size=self.batch_size,timesteps=self.timesteps
                )

            print('Predicting test dataset...')
            predictions_test = TCN.predict(test_batches)
            y_true_test = self.test_data[self.timesteps:]
            np.save(os.path.join(prediction_path_tcn,'predictions_test'), predictions_test)
            np.save(os.path.join(prediction_path_tcn,'y_true_test'),y_true_test)
            print('Predictions are saved to ', prediction_path_tcn)

            train_batches = PX4Generator(
                    self.train_data,
                    batch_size=self.batch_size,timesteps=self.timesteps
                )

            print('Predicting training dataset...')
            predictions_train = TCN.predict(train_batches)
            y_true_train = self.train_data[self.timesteps:]
            np.save(os.path.join(prediction_path_tcn,'predictions_train'), predictions_train)
            np.save(os.path.join(prediction_path_tcn,'y_true_train'),y_true_train)
            print('Predictions are saved to ', prediction_path_tcn)


if __name__ == "__main__":

    global info_x_lstm
    global info_x_cnn
    global info_x_att
    global info_x_att_temp
    global info_x_tcn

    import multiprocessing
    # multiprocessing.set_start_method('spawn',True)


    #Shared parameters
    batch_size = 64    
    epochs = 100  


    tf.keras.backend.clear_session()

    # train_data = '2020-7-4-19-09-29'
    # valid_data = '2020-7-4-18-56-40'
    # test_data = '2020-7-4-19-16-58' 

    train_data = '2020-7-11-00-45-03'
    valid_data = '2020-7-11-00-21-48'
    test_data = '2020-7-11-00-11-26' 


    # #LSTM parameters  
    # timesteps_lstm = 400
    # dropout_lstm = 0.15
    # hidden_lstm = 400

    # info_x_lstm = '-'.join([str(hidden_lstm),str(timesteps_lstm),str(dropout_lstm)])

    # LSTMDetector = AnomalyDetector(
    #     train_datapath_suffix = train_data,
    #     valid_datapath_suffix= valid_data,
    #     test_datapath_suffix= test_data,
    #     model='lstm',
    #     batch_size= batch_size,
    #     timesteps = timesteps_lstm,
    #     epochs=epochs,
    #     dropout=dropout_lstm,
    #     hidden=hidden_lstm
    # )
    
    # LSTMDetector.run()
    # LSTMDetector.predict()


    #TCN parameters
    timesteps_tcn = 400
    kernel_size_tcn = 16
    filters_tcn = 32
    pool_size_tcn = 2
    dropout_tcn = 0.1

    info_x_tcn = '-'.join([str(kernel_size_tcn),str(filters_tcn),str(timesteps_tcn),str(dropout_tcn)])

    TCNDetector = AnomalyDetector(
        train_datapath_suffix=train_data,
        valid_datapath_suffix=valid_data,
        # test_datapath_suffix= test_data,
        model='tcn',
        batch_size = batch_size,
        timesteps = timesteps_tcn,
        epochs = epochs,
        dropout= dropout_tcn,
        kernel_size=kernel_size_tcn,
        filters=filters_tcn,
        pool_size = pool_size_tcn
    )


    TCNDetector.run()
    # TCNDetector.predict()

    # #Attention parameters
    # timesteps_att = 400
    # dropout_att = 0.1
    # hidden_att_lstm = 300    
    # hidden_att = 500 # 500 for multiplicative 450 for additive

    # info_x_att = '-'.join([str(hidden_att),str(hidden_att_lstm),str(timesteps_att),str(dropout_att)])


    # MultiplicativeAttentionDetector = AnomalyDetector(
    #     train_datapath_suffix=train_data,
    #     valid_datapath_suffix=valid_data,
    #     model='attention_multip',
    #     batch_size = batch_size,
    #     timesteps = timesteps_att,
    #     epochs = epochs,
    #     dropout= dropout_att,
    #     hidden= hidden_att_lstm,
    #     hidden_att= hidden_att
    # )


    # MultiplicativeAttentionDetector.run()
    # MultiplicativeAttentionDetector.predict()

    # #Attention parameters
    # timesteps_att = 400
    # dropout_att = 0.1
    # hidden_att_lstm = 300    
    # hidden_att = 450 # 500 for multiplicative 450 for additive

    # info_x_att = '-'.join([str(hidden_att),str(hidden_att_lstm),str(timesteps_att),str(dropout_att)])

    # AddtitiveAttentionDetector = AnomalyDetector(
    #     train_datapath_suffix=train_data,
    #     valid_datapath_suffix=valid_data,
    #     model='attention_add',
    #     batch_size = batch_size,
    #     timesteps = timesteps_att,
    #     epochs = epochs,
    #     dropout= dropout_att,
    #     hidden= hidden_att_lstm,
    #     hidden_att= hidden_att
    # )


    # AddtitiveAttentionDetector.run()
    # AddtitiveAttentionDetector.predict()


    # #Attention parameters
    # timesteps_att = 400
    # dropout_att = 0.2
    # hidden_att_lstm = 250    

    # info_x_att_temp = '-'.join([str(hidden_att_lstm),str(timesteps_att),str(dropout_att)])


    # TemporalAttentionDetector = AnomalyDetector(
    #     train_datapath_suffix=train_data,
    #     valid_datapath_suffix=valid_data,
    #     model='attention_temp',
    #     batch_size = batch_size,
    #     timesteps = timesteps_att,
    #     epochs = epochs,
    #     dropout= dropout_att,
    #     hidden= hidden_att_lstm
    # )
    

    # TemporalAttentionDetector.run()


    # #CNN parameters
    # timesteps_cnn = 400
    # kernel_size_cnn = 16
    # filters_cnn = 32
    # pool_size_cnn = 2
    # dropout_cnn = 0.1

    # info_x_cnn = '-'.join([str(kernel_size_cnn),str(filters_cnn),str(timesteps_cnn),str(dropout_cnn)])

    # CNN1DDetector = AnomalyDetector(
    #     train_datapath_suffix=train_data,
    #     valid_datapath_suffix=valid_data,
    #     model='cnn1d',
    #     batch_size = batch_size,
    #     timesteps = timesteps_cnn,
    #     epochs = epochs,
    #     dropout= dropout_cnn,
    #     kernel_size=kernel_size_cnn,
    #     filters=filters_cnn,
    #     pool_size = pool_size_cnn
    # )


    # CNN1DDetector.run()




    #After the training I should de-normalize the output see the realistic values if I need to. I use reverse scaler for thats