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
PMAS_CUEE_TEST  = "pmaps_test_data.csv"
PMAS_CUEE_TRAIN = "pmaps_train_data.csv"

def npdatetime_to_string(numpy_array):   
    list_str = np.datetime_as_string(numpy_array, unit='s').tolist()
    return list_str 

def choose_daytimehours(data, start=1, end=9):  
    daytimehours  = [ True if (hour <=end) and (hour >= start)  else False for hour in data["hour"] ]
    data_ = data.iloc[daytimehours].copy()
    return data_

def choose_stations(data, station_num_list=["ISL052"]):   
    chosen_sitename  = [ True if site in station_num_list else False for site in data["site_name"] ]
    data_ = data.iloc[chosen_sitename].copy()
    return data_
 
class DatasetCUEE(data.Dataset):
    def __init__(self,  root_path = CUEE_ROOT, flag='train', size=None, features='S',  
                test_data_path=PMAS_CUEE_TEST, train_data_path=PMAS_CUEE_TRAIN, data_path=CUEE_DATA,   target='I', scale=True, timeenc=0, freq='h', train_only=False ):
        super(DatasetCUEE, self).__init__()


        file = open('./data_provider/stations.txt', 'r') 
        stations_list = file.readlines()  
        self.stations_list = [ station_.split("\n")[0] for station_ in stations_list] 
        # exclude 032 and 045
 
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
        self.scaler_x = StandardScaler()
        self.scaler_v = StandardScaler()
        self.scaler_y = StandardScaler()
        self.flag = flag
 
        self.root_path  = root_path   

        if  ("CUEE_PMAS" in root_path):

            
            self.train_data_path  = train_data_path   
 
            if self.flag == "train": 
                self.data_path  = train_data_path  
            elif (self.flag == "test") or (self.flag == "val"): 
                self.data_path  = test_data_path  
        else: 
            self.data_path  = data_path 

        self.__read_data__()

    def __read_data__(self): 
        
        raw_data = []


 
        train_data     = pd.read_csv(os.path.join(self.root_path, self.train_data_path) ) 
        read_data      = pd.read_csv(os.path.join(self.root_path, self.data_path) ) 
 
        raw_train_data = train_data[['Datetime', 'site_name', 'I', 'Iclr', 'latt', 'long', 'CI', 'R', 'hour_encode1',  'temperature', 'I_nwp']].copy() 
        raw_train_data['Datetime']     = pd.to_datetime(raw_train_data['Datetime'], utc=True) # Should be false If False == Thailand Local Time (Guessing)
        raw_train_data["hour"]         = [ date.hour   for date in raw_train_data['Datetime'] ]
        raw_train_data['day']          = [ date.day    for date in raw_train_data['Datetime'] ]
        raw_train_data['month']        = [ date.month  for date in raw_train_data['Datetime'] ]
        raw_train_data['minute']       = [ date.minute for date in raw_train_data['Datetime'] ]

        #'updated_measurement_Iclr_new.csv'   
        raw_data       =  read_data[['Datetime', 'site_name', 'I', 'Iclr', 'latt', 'long', 'CI', 'R', 'hour_encode1',  'temperature', 'I_nwp']].copy()  
        raw_data['Datetime']     = pd.to_datetime(raw_data['Datetime'], utc=True) # Should be false If False == Thailand Local Time (Guessing)
        raw_data["hour"]         = [ date.hour   for date in raw_data['Datetime'] ]
        raw_data['day']          = [ date.day    for date in raw_data['Datetime'] ]
        raw_data['month']        = [ date.month  for date in raw_data['Datetime'] ]
        raw_data['minute']       = [ date.minute for date in raw_data['Datetime'] ]

            # Shift Iclr to one step in feature and use it as a feature... 

        df_raw_train  = choose_daytimehours(raw_train_data, start=1, end=9)   
        df_raw_time = choose_daytimehours(raw_data, start=1, end=9) 
        
        data_x_list = []
        data_v_list = []
        data_y_list = []
        data_stamp_list = []

        for station_num in self.stations_list: 

            df_train_raw     = choose_stations(df_raw_train, station_num_list=[station_num])   
            df_raw     = choose_stations(df_raw_time, station_num_list=[station_num])  
  
            
            # The last attribute is also a target attribute ...  
            # cols_data  = df_raw.columns[1:]   
            df_data_x    = df_raw[['CI', 'R', 'temperature', 'I_nwp', 'hour_encode1', 'Iclr', 'latt', 'long', 'day', 'month', 'hour', 'minute'] ] #   ['Iclr', 'latt', 'long', 'day', 'month', 'hour', 'minute']
            df_data_v    = df_raw[['Iclr', 'latt', 'long', 'day', 'month', 'hour', 'minute']] 
            df_data_y    = df_raw[[self.target]]  

            # scaling  
            train_data_x = df_train_raw[['CI', 'R', 'temperature', 'I_nwp', 'hour_encode1', 'Iclr', 'latt', 'long', 'day', 'month', 'hour', 'minute'] ] # ['Iclr', 'latt', 'long', 'day', 'month', 'hour', 'minute']
            train_data_v = df_train_raw[['Iclr', 'latt', 'long', 'day', 'month', 'hour', 'minute']] 
            train_data_y = df_train_raw[[self.target]]  
             
 
            self.scaler_x.fit(train_data_x.values) 
            data_x = self.scaler_x.transform(df_data_x.values)
 
            self.scaler_v.fit(train_data_v.values) 
            data_v = self.scaler_v.transform(df_data_v.values)
 
            self.scaler_y.fit(train_data_y.values.reshape(-1,1)) 
            data_y = self.scaler_y.transform(df_data_y.values.reshape(-1,1)) 
 
            # time stamp 
            df_stamp = df_raw.loc[:,['Datetime']]  
            df_stamp['Datetime'] = pd.to_datetime(df_stamp.Datetime)   

            if self.timeenc == 0:
                df_stamp['month']   = df_stamp.Datetime.apply(lambda row: row.month, 1)
                df_stamp['day']     = df_stamp.Datetime.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.Datetime.apply(lambda row: row.weekday(), 1)
                df_stamp['hour']    = df_stamp.Datetime.apply(lambda row: row.hour, 1)  
                df_stamp['min']     = df_stamp.Datetime.apply(lambda row: row.minute, 1)   

                self.date_time      = npdatetime_to_string(df_stamp['Datetime'].values.copy())  
                data_stamp          = df_stamp.drop(['Datetime'],  axis=1).values 
                
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['Datetime'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            # putting them into x and y    
            data_x_list.append(data_x)  # data[border1:border2, -1] data[border1:border2]
            data_v_list.append(data_v) 
            data_y_list.append(data_y.reshape(-1,1))  # data[border1:border2,-1].reshape(-1,1)   data[border1:border2]
            
 
            data_stamp_list.append(data_stamp) 
 
        self.data_x = np.concatenate(data_x_list) 
        self.data_y = np.concatenate(data_y_list)  

        self.data_v = np.concatenate(data_v_list) 
        self.data_stamp = np.concatenate(data_stamp_list)  

    def __getitem__(self, index):

        s_begin = index
        s_end   = s_begin + self.seq_len
         
        ov_begin = s_end  - self.label_len
        ov_end   = s_end  + self.pred_len
 
        r_begin = s_end  
        r_end   = s_end  + self.pred_len
 
        seq_x = self.data_x[s_begin:s_end] 
        seq_y = self.data_y[r_begin:r_end]

        if self.label_len < 1:
            seq_v = torch.zeros(self.pred_len, seq_x.shape[-1]) # 1 x Feat size
        else:
            seq_v = self.data_v[ov_begin:ov_end] 


        seq_x_mark = self.data_stamp[s_begin:s_end] 
        seq_v_mark = self.data_stamp[ov_begin:ov_end] 
        seq_y_mark = self.data_stamp[r_begin:r_end] 

        return seq_x, seq_y, seq_v, seq_x_mark, seq_y_mark, seq_v_mark

    def __len__(self): 
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform_y(self, data_y):
        return self.scaler_y.inverse_transform(data_y)
    
    def inverse_transform_x(self, data_x):
        return self.scaler_x.inverse_transform(data_x)
    
    def inverse_transform_v(self, data_v):
        return self.scaler_v.inverse_transform(data_v)
 

if __name__ == "__main__":
 
    dataset    = DatasetCUEE(root_path = CUEE_ROOT, flag='test', size=None, features='MS', data_path=CUEE_DATA, target='I', scale=True, timeenc=0, freq='h', train_only=False)
    dataloader = DataLoader(dataset, batch_size=1)

    trues_rev_list = []

    for data_ in dataloader:
        seq_x, seq_y, seq_v,  seq_x_mark, seq_y_mark, seq_v_mark  = data_ 

        for seq_i in range(4):
            trues_rev = dataset.inverse_transform_y(seq_y[:,seq_i,:])
            trues_rev_list.append(trues_rev)
    

    trues_rev_full = np.concatenate(trues_rev_list, dim=1)
    trues_rev_ = trues_rev_full[:100]
    

    time_x = np.arange(len(trues_rev_))  
    time_x_tick = np.arange(0, len(trues_rev_), 36)

    plt.plot(time_x, trues_rev_, label='actual', color="black") 

    plt.show()

    
 


