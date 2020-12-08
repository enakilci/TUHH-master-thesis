import csv
import os
import pandas as pd
import errno
import sqlite3
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf

def load_dataset(csv_files_abs_path):
    """Creates a dataset as a pandas.DataFrame object from the csv_files in given 'csv_files_path' and debugs the data in the datasets

    Arguments:
        csv_files_path: os path of the csv files
        drop_percent: percentage value that is used to handle missing data in the dataset

    Returns:
        A Debugged dataset
    """

    path = csv_files_abs_path
    files = os.listdir(path)
    data = pd.DataFrame(columns=["timestamp"])
    data["timestamp"] = data["timestamp"].astype("int")
    for file in files:
        cols = {}            
        temp_data = pd.read_csv(os.path.join(path,file))
        temp_data["timestamp"] = temp_data["timestamp"].astype("int")
        for col in temp_data.columns:
            if not col == 'timestamp':
                cols[col] = ''.join([file[:-4],'-',col])
        temp_data.rename(columns=cols,inplace=True)
        data = data.merge(temp_data,how="outer",on="timestamp")    
   
    
    data["timestamp"] = data["timestamp"].astype("int")

    if not len(list(data["timestamp"][data["timestamp"].duplicated()].index)) == 0:
        print("Handling duplicated rows...")
        data = data.drop(index = list(data["timestamp"][data["timestamp"].duplicated()].index))
        data = data.reset_index(drop=True)
        print("Completed!")

    print('Reorganizing the dataset...') 

    data = data.set_index('timestamp')


    data.sort_index(inplace=True)

    print("Handling remaining missing values...")       
    data = data.replace({np.inf: np.nan, -np.inf: np.nan})
    data.fillna(method="ffill",inplace=True)
    data.fillna(method="bfill",inplace=True)    
    data.fillna(value=0,inplace=True)
    print("Completed!")

    # print("Normalizing the datasets...")
    # scaler = MinMaxScaler()
    # data[data.columns] = scaler.fit_transform(data[data.columns])

    print("Normalizing the datasets...")
    scaler_std = StandardScaler()
    data[data.columns] = scaler_std.fit_transform(data[data.columns])
    for i in range(len(data.columns)):
        print("Scaler mean of %s: %f" % (data.columns[i],scaler_std.mean_[i]))
        print("Scaler variance of %s: %f" % (data.columns[i],scaler_std.var_[i]))
    
    print("Completed!")


    # data = data.replace({np.inf: np.nan, -np.inf: np.nan, -0:0})
    # data.fillna(method="ffill",inplace=True)
    # data.fillna(method="bfill",inplace=True)
    # print("Normalizing the datasets...")
    # scaler_std = StandardScaler()
    # data[data.columns] = scaler_std.fit_transform(data[data.columns])
    # for i in range(len(data.columns)):
    #     print("Scaler mean of %s: %f" % (data.columns[i],scaler_std.mean_[i]))
    #     print("Scaler variance of %s: %f" % (data.columns[i],scaler_std.var_[i]))

    # print("Completed!")
    # print("Handling remaining missing values...")
    # data.fillna(value=0,inplace=True)
    # print("Completed!")

    if not len(data.columns[data.columns.duplicated()].unique()) == 0:
        print("Handling columns with the same name...")
        data.columns = pd.io.parsers.ParserBase({'names':data.columns})._maybe_dedup_names(data.columns)
        print("Completed!")

    data = data.reindex(sorted(data.columns),axis=1)
    #I found a couple of missing value handler. 
    #I'll use one of them for the next step.
    
    return data


def load_dataset2(csv_files_abs_path):
    """Creates a dataset as a pandas.DataFrame object from the csv_files in given 'csv_files_path' and debugs the data in the datasets

    Arguments:
        csv_files_path: os path of the csv files
        drop_percent: percentage value that is used to handle missing data in the dataset

    Returns:
        A Debugged dataset
    """

    path = csv_files_abs_path
    files = os.listdir(path)

    data = pd.DataFrame(columns=["timestamp"])
    for file in files:
        # col_prefix = file[25:-6]
        f = open(os.path.join(path,file))
        reader = csv.reader(f)
        headers = next(reader,None)
        cols = {}
        for h in headers:
            if not h == 'timestamp':
                cols[h] = ''.join([file[:-4],'-',h])
            
        temp_data = pd.read_csv(os.path.join(path,file))
        temp_data.rename(columns=cols,inplace=True)
        data = data.merge(temp_data,how="outer",on="timestamp")    
    
    
    data["timestamp"] = data["timestamp"].astype("int")

    if not len(list(data["timestamp"][data["timestamp"].duplicated()].index)) == 0:
        print("Handling duplicated rows...")
        data = data.drop(index = list(data["timestamp"][data["timestamp"].duplicated()].index))
        data = data.reset_index(drop=True)
        print("Completed!")

    print('Reorganizing the dataset...') 

    data = data.set_index('timestamp')


    data = data.reset_index(drop=True)
    data = data.replace({np.inf: np.nan, -np.inf: np.nan})

    print("Normalizing the datasets...")
    scaler = MinMaxScaler()
    data[data.columns] = scaler.fit_transform(data[data.columns])

    # scaler_std = StandardScaler()
    # data[data.columns] = scaler_std.fit_transform(data[data.columns])
    # for i in range(len(data.columns)):
    #     print("Scaler mean of %s: %f" % (data.columns[i],scaler_std.mean_[i]))
    #     print("Scaler variance of %s: %f" % (data.columns[i],scaler_std.var_[i]))

    print("Completed!")
    print("Handling missing values...")
    data.fillna(value=-1,inplace=True)
    print("Completed!")

    if not len(data.columns[data.columns.duplicated()].unique()) == 0:
        print("Handling columns with the same name...")
        data.columns = pd.io.parsers.ParserBase({'names':data.columns})._maybe_dedup_names(data.columns)
        print("Completed!")

    data = data.reindex(sorted(data.columns),axis=1)
    #I found a couple of missing value handler. 
    #I'll use one of them for the next step.
    
    return data


def create_dirs(path):

    from errno import EEXIST
    import os 

    try:
        os.makedirs(path,0o700)
    except OSError as e: # Python >2.5
        if e.errno == EEXIST and os.path.isdir(path):
            pass
        else: raise

    # alternative code
    # if not os.path.exists(path):
    #     try:
    #         os.makedirs(path,0o700)
    #     except OSError as e:
    #         if e.errno != errno.EEXIST:
    #             raise



