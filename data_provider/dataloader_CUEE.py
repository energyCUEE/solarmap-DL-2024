import pdb

import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import numpy as np
import torch 
import glob
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta 
import os
from utils.timefeatures import time_features

CUEE_ROOT = os.path.join(os.getcwd(),'data')
CUEE_DATA = 'updated_measurement_Iclr_new.csv'

def npdatetime_to_string(numpy_array):   
    list_str = np.datetime_as_string(numpy_array, unit='s').tolist()
    return list_str

# def datetime_to_string_list(Datetime_values):  
#     return [npdatetime_to_string(date_time) for date_time in Datetime_values] 

def choose_daytimehours(data, start=1, end=9):  
    daytimehours  = [ True if (hour <=end) and (hour >= start)  else False for hour in data["Hour"] ]
    data_ = data.iloc[daytimehours].copy()
    return data_
 
class DatasetCUEE(data.Dataset):
    def __init__(self,  root_path = CUEE_ROOT, flag='train', size=None, features='S',  
                data_path=CUEE_DATA, target='I', scale=True, timeenc=0, freq='h', train_only=False):
        super(DatasetCUEE, self).__init__()

        if size == None:
            self.seq_len   = 24 * 4 * 4
            self.label_len = 48
            self.pred_len  = 24 * 4
        else:
            self.seq_len    = size[0]
            self.label_len  = size[1]
            self.pred_len   = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type  = type_map[flag] 

        self.features   = features
        self.target     = target
        self.scale      = scale
        self.timeenc    = timeenc
        self.freq       = freq
        self.train_only = train_only
        self.scaler = StandardScaler()


        self.root_path  = root_path    
        self.data_path  = data_path 
        self.__read_data__()

    def __read_data__(self): 
        
        raw_data = []

        read_data     = pd.read_csv(os.path.join(self.root_path, self.data_path) ) #'updated_measurement_Iclr_new.csv' 
        raw_data      = read_data[['Datetime', "site_name", "I", "Iclr"]].copy()   
        raw_data['Datetime']     = pd.to_datetime(raw_data['Datetime'], utc=True) # Should be false If False == Thailand Local Time (Guessing)
 
        raw_data["Hour"] = [ date.hour for date in raw_data['Datetime']  ]    
        df_raw = choose_daytimehours(raw_data, start=1,end=9) 

        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('Datetime')
        cols.remove('Hour')
 

        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type] 


        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['Datetime'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['Datetime'] + cols + [self.target]]
            df_data = df_raw[[self.target]] 

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

 

        df_stamp = df_raw[['Datetime']][border1:border2]
        df_stamp['Datetime'] = pd.to_datetime(df_stamp.Datetime)  
 
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.Datetime.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.Datetime.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.Datetime.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.Datetime.apply(lambda row: row.hour, 1)  
            self.date_time  = npdatetime_to_string(df_stamp['Datetime'].values.copy()) 
            data_stamp = df_stamp.drop(['Datetime'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Datetime'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2] 
        self.data_stamp = data_stamp 
        

    def __getitem__(self, index):

        s_begin = index
        s_end   = s_begin + self.seq_len
        r_begin = s_end   - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
 
        #seq_x_mark_datetime = self.date_time[s_begin:s_end] 
        #seq_y_mark_datetime = self.date_time[r_begin:r_end] 

        return seq_x, seq_y, seq_x_mark, seq_y_mark 

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
 

if __name__ == "__main__":
 
    dataset    = DatasetCUEE(root_path = CUEE_ROOT, flag='train', size=None, features='S', data_path=CUEE_DATA, target='I', scale=True, timeenc=0, freq='h', train_only=False)
    dataloader = DataLoader(dataset)
    for data_ in dataloader:
        seq_x, seq_y, seq_x_mark, seq_y_mark  = data_ 
 


