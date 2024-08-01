import pdb

import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import torch 
import glob
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta 
import os
from utils.timefeatures import time_features
import pickle
import h5py
from utils.tools import scaling_LxF

CUEE_ROOT = os.path.join(os.getcwd(),'data')
CUEE_DATA = 'updated_measurement_Iclr_new.csv'
PMAS_CUEE_TEST  = "pmaps_test_data.csv"
PMAS_CUEE_VALID = "pmaps_validate_data.csv"
PMAS_CUEE_TRAIN = "pmaps_train_data.csv"

SUFFIX_SAVED_FILES_LIST = ["data.h5", "date_time_x_list.npy", "date_time_y_list.npy" ]

def npdatetime_to_string(numpy_array):   
    list_str = np.datetime_as_string(numpy_array, unit='s').tolist()
    return list_str 

def choose_date(df_stamp, start_date='2022-04-02'):
    mask = (df_stamp['date']  >=  start_date)
    filtered_data_df = df_stamp.loc[mask]  
    return filtered_data_df

def choose_datetime(df_stamp, start_time, end_time):
    mask = (df_stamp['Datetime'].dt.time >= start_time) & (df_stamp['Datetime'].dt.time <= end_time)
    filtered_data_df = df_stamp.loc[mask] 
    return filtered_data_df

def choose_daytimehours(data, start=1, end=9):  
    daytimehours  = [ True if (hour <=end) and (hour >= start)  else False for hour in data["hour"] ]
    data_ = data.iloc[daytimehours].copy()
    return data_

def choose_stations(data, station_num_list=["ISL052"]):   
    chosen_sitename  = [ True if site in station_num_list else False for site in data["sitename"] ]
    data_ = data.iloc[chosen_sitename].copy()
    return data_
 
class DatasetCUEE(data.Dataset):
    def __init__(self,  root_path = CUEE_ROOT, flag='train', size=None, features='S',  
                test_data_path=PMAS_CUEE_TEST, valid_data_path=PMAS_CUEE_VALID, train_data_path=PMAS_CUEE_TRAIN, data_path=CUEE_DATA,   
                target='I', scale=True, timeenc=0, freq='h', train_only=False, tag="CUEE_PMAPS", option_Ihat1 = 'I_LGBM', DEBUGMode=False, is_noscaley=False, is_noscalex=False):
        super(DatasetCUEE, self).__init__()

        self.DEBUGMode = DEBUGMode

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
        self.is_noscaley = is_noscaley
        self.is_noscalex = is_noscalex
        if not(self.is_noscalex):
            self.scaler_x = StandardScaler()
            self.scaler_v = StandardScaler()
        if not(self.is_noscaley):
            self.scaler_y = StandardScaler()
        self.flag = flag
        self.tag = tag
        self.option_Ihat1 = option_Ihat1
 
        self.root_path  = root_path   
     
        self.train_data_path  = train_data_path   
 
        if self.flag == "train": 
            self.data_path  = train_data_path  

        elif (self.flag == "val"): 
            self.data_path  = valid_data_path 

        elif (self.flag == "test"):
            self.data_path  = test_data_path 

        self.__read_data__()

    def __read_data__(self): 
        
        raw_data = [] 
 
        train_data     = pd.read_csv(os.path.join(self.root_path, self.train_data_path) ) 
        read_data      = pd.read_csv(os.path.join(self.root_path, self.data_path) ) 
        
        start_prediction_time = datetime(2022, 4,  1,  7, 0) 
        end_time              = datetime(2023, 6, 30, 17, 0)     
        
        start_date = start_prediction_time.strftime("%d-%m-%Y") 
        end_time   = pd.Timestamp(end_time).time()  
        start_time = (start_prediction_time - timedelta(minutes= (self.seq_len - 1)*15) ).time()  

        print("Flag %s => Total: %d" % (self.flag, len(read_data)))
        print("... filtering out night time and concatenating data from each station")

        if not(self.is_noscaley):
            self.folder = os.path.join(self.root_path, "%s-%s-%s-%s-%s"    % (self.option_Ihat1, self.flag, start_date, start_time, end_time))
        else: 
            self.folder = os.path.join(self.root_path, "%s-%s-%s-%s-%s-%s" % (self.option_Ihat1, self.flag,  "NotScaleY", start_date, start_time, end_time))
        os.makedirs(self.folder, exist_ok=True)

        print("===============================================" )
        print("Seq length: %d so, we will start %d [%d-1x15] mins. before 7:00 AM" % (self.seq_len,  (self.seq_len-1)*15, (self.seq_len) ))
        print("Start time: %s" % start_time.strftime("%H:%M:%S"))
        print("End   time: %s" % end_time.strftime("%H:%M:%S"))
        print("Foldername: [%s]" % self.folder)
        print("===============================================" )
        
        DATA_COLUMN = ['Datetime', 'sitename', 'I', 'Iclr', 'latt', 'long', 'CI', 'R', 'hour_encode1',  'Tnwp', 'Inwp', 'k_bar','condition', 'I_LGBM', 'is_target']
        # DATA_COLUMN: contains the list of all the info that will be considered. 
        # Datetime, sitename, is_target: will be used in  df_stamp (indexing information)
        # A subset of the DATA_COLUMN will be selected by features_list and overlap_list for the input and overlap features.
        # The I column will be used as the target.
        df_train_raw = train_data[DATA_COLUMN].copy()
        df_train_raw['Datetime'] = pd.to_datetime(df_train_raw['Datetime'], utc=False).dt.tz_localize(None)
  

        #df_train_raw = train_data[['Datetime', 'sitename', 'I', 'Iclr', 'latt', 'long', 'CI', 'R', 'hour_encode1',  'Tnwp', 'Inwp', 'k_bar','condition', 'I_LGBM', 'Ireg']].copy() 
        df_train_raw["hour"]         = [ date.hour   for date in df_train_raw['Datetime'] ]
        df_train_raw['day']          = [ date.dayofyear    for date in df_train_raw['Datetime'] ]
        df_train_raw['month']        = [ date.month  for date in df_train_raw['Datetime'] ]
        df_train_raw['minute']       = [ date.minute for date in df_train_raw['Datetime'] ]

        #'updated_measurement_Iclr_new.csv'   
        raw_data       =  read_data[DATA_COLUMN].copy()
        #raw_data       =  read_data[['Datetime', 'sitename', 'I', 'Iclr', 'latt', 'long', 'CI', 'R', 'hour_encode1',  'Tnwp', 'Inwp', 'k_bar','condition', 'I_LGBM', 'Ireg']].copy() 
        raw_data['Datetime'] = pd.to_datetime(raw_data['Datetime'], utc=False).dt.tz_localize(None)  
        raw_data["hour"]         = [ date.hour   for date in raw_data['Datetime'] ]
        raw_data['day']          = [ date.dayofyear    for date in raw_data['Datetime'] ]
        raw_data['month']        = [ date.month  for date in raw_data['Datetime'] ]
        raw_data['minute']       = [ date.minute for date in raw_data['Datetime'] ] 
         
        df_train_raw['date'] = pd.to_datetime(df_train_raw['Datetime'], format='%Y-%m-%d')
        raw_data['date']     = pd.to_datetime(raw_data['Datetime'], format='%Y-%m-%d') 
 
        df_train_raw  = choose_date(df_train_raw, start_date=start_date )   
        raw_data      = choose_date(raw_data, start_date=start_date )
 
 
        df_raw_train  = choose_datetime(df_train_raw, start_time=start_time, end_time=end_time)   
        df_raw        = choose_datetime(raw_data, start_time=start_time, end_time=end_time)    
 
        
        ## create option for select the list of features and overlap_list
        ### **note : for features_list in the first values should be the same as option_Ihat1
        
        if self.option_Ihat1  == 'Iclr':
            features_list = ['Iclr', 'Inwp', 'CI', 'R', 'hour_encode1', 'day', 'month', 'minute', 'latt', 'long', 'Tnwp']
            overlap_list = ['Inwp']

        elif self.option_Ihat1 == 'Inwp':
            features_list = ['Inwp', 'Iclr', 'CI', 'R', 'hour_encode1', 'day', 'month', 'minute', 'latt', 'long', 'Tnwp']
            overlap_list = ['Iclr']

        elif self.option_Ihat1 == 'I_LGBM':
            features_list = ['I_LGBM', 'Iclr', 'Inwp', 'CI', 'R', 'hour_encode1', 'day', 'month', 'minute', 'latt', 'long', 'Tnwp']
            overlap_list = ['Iclr', 'Inwp'] 
        
        # elif self.option_Ihat1 == 'I_LGBM_wo_nwp':
        #     features_list = ['I_LGBM', 'Iclr', 'CI', 'R', 'hour_encode1', 'day', 'month', 'minute', 'latt', 'long']
        #     overlap_list = ['Iclr']
        
        elif self.option_Ihat1 == 'Ireg':
            features_list = ['Ireg', 'Iclr', 'Inwp', 'CI', 'R', 'hour_encode1', 'day', 'month', 'minute', 'latt', 'long', 'Tnwp']
            overlap_list = ['Iclr', 'Inwp'] 
        
        # elif self.option_Ihat1 == 'Ireg_wo_nwp':
        #     features_list = ['Ireg', 'Iclr', 'CI', 'R', 'hour_encode1', 'day', 'month', 'minute', 'latt', 'long']
        #     overlap_list = ['Iclr'] 

        elif (self.option_Ihat1 == 'I') or (self.option_Ihat1 == 'I_ltime_doy'):
            features_list = ['Iclr', 'CI', 'R', 'hour_encode1', 'day', 'month', 'minute', 'latt', 'long', 'Tnwp', 'Inwp']
            overlap_list  = ['Iclr', 'hour_encode1',  'day', 'month', 'minute', 'latt', 'long', 'Tnwp', 'Inwp']

        elif self.option_Ihat1 == 'I_R':
            features_list = ['Iclr', 'R', 'hour_encode1', 'day', 'month', 'minute', 'latt', 'long', 'Tnwp', 'Inwp']
            overlap_list  = ['Iclr', 'hour_encode1',  'day', 'month', 'minute', 'latt', 'long', 'Tnwp', 'Inwp']


        elif (self.option_Ihat1 == 'I_wo_nwp') or (self.option_Ihat1 == 'I_wo_nwp_ltime_doy'):
            features_list = ['Iclr', 'CI', 'R', 'hour_encode1', 'day', 'month', 'minute', 'latt', 'long' ]
            overlap_list  = ['Iclr', 'hour_encode1',  'day', 'month', 'minute', 'latt', 'long' ]

        elif self.option_Ihat1 == 'I_R_wo_nwp':
            features_list = ['Iclr', 'R', 'hour_encode1', 'day', 'month', 'minute', 'latt', 'long' ]
            overlap_list  = ['Iclr', 'hour_encode1',  'day', 'month', 'minute', 'latt', 'long' ]

        elif self.option_Ihat1 == 'I_wo_nwp_wo_latlong'  or (self.option_Ihat1 == 'I_wo_nwp_wo_latlong_ltime_doy'):
            features_list = ['Iclr', 'CI', 'R', 'hour_encode1', 'day', 'month', 'minute']
            overlap_list  = ['Iclr', 'hour_encode1',  'day', 'month', 'minute' ]

        elif self.option_Ihat1 == 'I_R_wo_nwp_wo_latlong':
            features_list = ['Iclr', 'R', 'hour_encode1', 'day', 'month', 'minute']
            overlap_list  = ['Iclr', 'hour_encode1',  'day', 'month', 'minute' ]
        
        elif self.option_Ihat1 == 'I_optionA':
            features_list = ['hour_encode1', 'day', 'month', 'minute', 'latt', 'long', 'Tnwp', 'Inwp']
            overlap_list  = ['hour_encode1', 'day', 'month', 'minute', 'latt', 'long', 'Tnwp', 'Inwp']
        
        elif self.option_Ihat1 == 'I_optionB':
            features_list = ['Iclr', 'hour_encode1', 'day', 'month', 'minute', 'latt', 'long', 'Tnwp', 'Inwp']
            overlap_list  = ['Iclr', 'hour_encode1', 'day', 'month', 'minute', 'latt', 'long', 'Tnwp', 'Inwp']
        
        elif self.option_Ihat1 == 'I_optionC':
            features_list = ['CI', 'R','hour_encode1', 'day', 'month', 'minute', 'latt', 'long', 'Tnwp', 'Inwp']
            overlap_list  = ['CI', 'R','hour_encode1', 'day', 'month', 'minute', 'latt', 'long', 'Tnwp', 'Inwp']
        
        # create date  for scaling data
        train_data_x = df_raw_train[features_list].copy()
        train_data_v = df_raw_train[overlap_list].copy()
        train_data_y = df_raw_train[[self.target]].copy()
        
        # select data for normalization
        df_data_x    = df_raw[features_list].copy()
        df_data_v    = df_raw[overlap_list].copy()
        df_data_y    = df_raw[self.target].copy()
        # pdb.set_trace()   

        # convert sky condition from string to float          
        df_raw_sky_condition = df_raw.copy()
        df_raw_sky_condition.loc[:, 'condition'] = df_raw_sky_condition['condition'].apply(
            lambda x: 1 if x == 'clr' else (2 if x == 'partly_cloudy' else 3))
        df_data_sky_condition = df_raw_sky_condition[['k_bar', 'condition']].values

        # get sitename 
        df_sitename = df_raw[['sitename']].values
 
        # normalization      
        data_x = scaling_LxF(df_data_x.values, columns=features_list)  

        # check len of overlap_list
        if len(overlap_list) == 1: 
            data_v = scaling_LxF(df_data_v.values.reshape(-1,1), columns=overlap_list)  
        else: 
            data_v = scaling_LxF(df_data_v.values, columns=overlap_list)   

        if not(self.is_noscaley):
            self.scaler_y.fit(train_data_y.values.reshape(-1,1)) 
            data_y = self.scaler_y.transform(df_data_y.values.reshape(-1,1))        
        else:
            data_y = df_data_y.values.reshape(-1,1) 

            # pickle.dump(self.scaler_x, open(os.path.join(self.folder, "Scaler_x.pkl"), 'wb'))
            # pickle.dump(self.scaler_v, open(os.path.join(self.folder, "Scaler_v.pkl"), 'wb')) 


        isfiles_list = []
        for file_suffix in SUFFIX_SAVED_FILES_LIST:
            isfiles_list.append(os.path.isfile( os.path.join(self.folder, file_suffix) )) 
         
        if all(isfiles_list)  == True: 
            self.__read_file_to_list()
        
        else:
            # time stamp  
            df_stamp = df_raw.loc[:,['Datetime', 'sitename', 'is_target']].copy()   

            if self.timeenc == 0:
                df_stamp['month']   = df_stamp.Datetime.apply(lambda row: row.month, 1)
                df_stamp['day']     = df_stamp.Datetime.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.Datetime.apply(lambda row: row.weekday(), 1)
                df_stamp['hour']    = df_stamp.Datetime.apply(lambda row: row.hour, 1)  
                df_stamp['min']     = df_stamp.Datetime.apply(lambda row: row.minute, 1)    
                data_stamp          = df_stamp.drop(['Datetime'],  axis=1).values 
                
            elif self.timeenc == 1: 
                data_stamp = time_features(pd.to_datetime(df_stamp['Datetime'].values, utc=False), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0) 

            date_time = npdatetime_to_string(df_stamp['Datetime'].values.copy())   
            df_stamp['date'] = pd.to_datetime(df_stamp['Datetime']).dt.date 

            self.__stacked_to_daily_seq(data_x, data_y, data_v, data_stamp, df_stamp, df_data_sky_condition, df_sitename)

            self.__save_list_to_file()

            self.__read_file_to_list() 



    def __read_file_to_list(self):

        print("Read %s" % os.path.join(self.folder , 'data.h5') )
        self.h5file = h5py.File(os.path.join(self.folder , 'data.h5'), 'r')
        
        print("Read %s" % os.path.join(self.folder , "date_time_x_list.npy")) 
        with open(os.path.join(self.folder , "date_time_x_list.npy"), 'rb') as fp:
            self.date_time_x_list  = pickle.load(fp)

        print("Read %s" % os.path.join(self.folder , "date_time_y_list.npy")) 
        with open(os.path.join(self.folder , "date_time_y_list.npy"), 'rb') as fp:
            self.date_time_y_list  = pickle.load(fp)

        # print("Read %s" % os.path.join(self.folder, "sky_condition_list.npy"))
        # with  open(os.path.join(self.folder , "sky_condition_list.npy"), 'rb') as fp:
        #     self.seq_sky_condition_list  = pickle.load(fp)
 
        
    def __save_list_to_file(self): 

        print("save %s" % os.path.join(self.folder , "data.h5")) 
        with h5py.File( os.path.join(self.folder , 'data.h5'),'w') as h5f:

            h5f.create_dataset("seq_x", data=np.asarray(self.seq_x_list) )
            h5f.create_dataset("seq_y", data=np.asarray(self.seq_y_list) )
            h5f.create_dataset("seq_v", data=np.asarray(self.seq_v_list) )
 
            h5f.create_dataset("seq_sky_condition", data=np.asarray(self.seq_sky_condition_list).astype(np.float32))
            h5f.create_dataset("seq_sitename", data=np.asarray(self.seq_sitename_list).astype(np.float32))

            h5f.create_dataset("seq_x_mark", data=np.asarray(self.seq_x_mark_list) )
            h5f.create_dataset("seq_y_mark", data=np.asarray(self.seq_y_mark_list) )
            h5f.create_dataset("seq_v_mark", data=np.asarray(self.seq_v_mark_list) )  


        print("save %s" % os.path.join(self.folder , "date_time_x_list.npy"))         
        with open(os.path.join(self.folder , "date_time_x_list.npy"), 'wb') as fp:
            pickle.dump(self.date_time_x_list, fp) 

        print("save %s" % os.path.join(self.folder , "date_time_y_list.npy"))      
        with open(os.path.join(self.folder , "date_time_y_list.npy"), 'wb') as fp:
            pickle.dump(self.date_time_y_list, fp) 

        if self.DEBUGMode:         
            print("save %s" % os.path.join(self.folder , "sample_stats.npy"))
            with open(os.path.join(self.folder , "sample_stats.npy"), 'wb') as fp:
                pickle.dump(self.sample_stats, fp)

    def __stacked_to_daily_seq(self, data_x, data_y, data_v, data_stamp, df_stamp, df_data_sky_condition, df_sitename):
        
        date_list = (np.unique(df_stamp['date'].values)).tolist() 

        self.seq_x_list = []
        self.seq_y_list = []
        self.seq_v_list = []
        
        self.seq_x_mark_list = []
        self.seq_v_mark_list = []
        self.seq_y_mark_list = []

        self.date_time_x_list = []
        self.date_time_y_list = []

        self.seq_sky_condition_list = []
        self.seq_sitename_list = []
        

        if self.DEBUGMode:
            self.sample_stats = []

        bar = tqdm(self.stations_list) 
        for stations_ in bar:
            for date_ in date_list:      
                target_mask_        = (df_stamp['is_target'] == True).values * (df_stamp['sitename'] == int(stations_)).values  * (df_stamp['date'] == date_).values
                bar.set_description("%s - %s - num %d" % (date_, stations_, sum(target_mask_ == True)))

                if sum(target_mask_ == True)  > 0:
                    True_index       = np.arange(len(target_mask_))[target_mask_] 
                    first_True_index = True_index[0]

                    # Extend masking to (self.seq_len - 1) lags 
                    lag_mask_ = target_mask_.copy()   
                    lag_mask_[ (first_True_index - (self.seq_len - 1)) : (first_True_index+1)] = True  # check if lag_mask_ is all continuous, if yes, you are most likely correct 
 
                    masked_data_x     = data_x[lag_mask_,:]
                    masked_data_y     = data_y[lag_mask_,:]
                    masked_data_v     = data_v[lag_mask_,:] 
                    masked_date_time  = df_stamp['Datetime'].iloc[lag_mask_] 
                    masked_data_stamp = data_stamp[lag_mask_,:]
 
                    masked_data_sky_condition = df_data_sky_condition[lag_mask_,:]
                    masked_data_sitename      = df_sitename[lag_mask_,:] 
                    num_sample_per_day        = sum(target_mask_ == True)     

                    masked_target = df_stamp['Datetime'].iloc[target_mask_].tolist()
 
                    for index in range(num_sample_per_day):

                        s_begin = index
                        s_end   = s_begin + self.seq_len

                        ov_begin = s_end - 1  - self.label_len # label_len = overlap_len, (we want overlap_len = seq_len)
                        ov_end   = ov_begin  + self.pred_len  
                
                        r_begin = s_end  - 1
                        r_end   = r_begin  + self.pred_len  
                        
                        seq_x  = masked_data_x[s_begin: s_end,   :]      # seq_x_ = masked_data_x[s_begin:s_end, :]
                        seq_v  = masked_data_v[ov_begin: ov_end, :]      # seq_v = torch.zeros(self.pred_len, masked_data_x.shape[-1]) # 1 x Feat size   
                        seq_y  = masked_data_y[r_begin:  r_end,  :] 

                        seq_sky_condition = masked_data_sky_condition[r_begin:r_end, :]
                        seq_sitename = masked_data_sitename[r_begin:r_end, :]

                        seq_x_mark = masked_data_stamp[s_begin:s_end]
                        seq_v_mark = masked_data_stamp[ov_begin:ov_end] 
                        seq_y_mark = masked_data_stamp[r_begin:r_end]  
 

                        # For debugging
                        # print("Masked_target")
                        # print(masked_target[index])
                        # 
                        # print("Time Y")
                        # print(date_time_y)  

                        if self.DEBUGMode:
                            
                            date_time_x = masked_date_time[s_begin:s_end].dt.tz_localize(None).values
                            date_time_v = masked_date_time[ov_begin:ov_end].dt.tz_localize(None).values
                            date_time_y = masked_date_time[r_begin:r_end].dt.tz_localize(None).values

                            assert date_time_y[0] == masked_target[index].tz_localize(None)
                            self.sample_stats.append({"date":date_ ,"station" : int(stations_), "num_sample_per_day": num_sample_per_day, "Datetime-X" :date_time_x[0], "Datetime-Y": date_time_y[0], "Masked_target": masked_target[index] }) #  datetime-y should match the datetime of is_target
                        
                        else:

                            date_time_x = masked_date_time[s_begin:s_end] 
                            date_time_v = masked_date_time[ov_begin:ov_end] 
                            date_time_y = masked_date_time[r_begin:r_end] 
                            
                        # print("Time V")
                        # print(date_time_v)
 
                        self.seq_x_list.append(seq_x)
                        self.seq_v_list.append(seq_v)
                        self.seq_y_list.append(seq_y)
                        self.seq_sky_condition_list.append(seq_sky_condition)
                        self.seq_sitename_list.append(seq_sitename)

                        self.seq_x_mark_list.append(seq_x_mark)
                        self.seq_v_mark_list.append(seq_v_mark)
                        self.seq_y_mark_list.append(seq_y_mark)
 
                        self.date_time_x_list.append(date_time_x)
                        self.date_time_y_list.append(date_time_y)  
 
 
 
                    #print(masked_target) # should match date_time_y

    def __getitem__(self, index): 

        seq_x = self.h5file["seq_x"][index] 
        seq_y = self.h5file["seq_y"][index] 
        seq_v = self.h5file["seq_v"][index]
        seq_sky_condition = self.h5file["seq_sky_condition"][index]
        seq_sitename = self.h5file["seq_sitename"][index]
 

        seq_x_mark = self.h5file["seq_x_mark"][index] 
        seq_y_mark = self.h5file["seq_y_mark"][index] 
        seq_v_mark = self.h5file["seq_v_mark"][index] 

        date_time_x = self.date_time_x_list[index]
        date_time_y = self.date_time_y_list[index] 
        date_time_x = np.array([ts.timestamp() for ts in self.date_time_x_list[index]], dtype=np.int64)
        date_time_y = np.array([ts.timestamp() for ts in self.date_time_y_list[index]], dtype=np.int64)
        # pdb.set_trace()

        return seq_x, seq_y, seq_v, seq_x_mark, seq_y_mark, seq_v_mark , date_time_x, date_time_y, seq_sky_condition, seq_sitename


    def __len__(self): 
        return len(self.date_time_y_list) # - self.seq_len - self.pred_len + 1
    
    def inverse_transform_y(self, data_y):
        if not(self.is_noscaley):
            return self.scaler_y.inverse_transform(data_y)
        else:
            return data_y
        
        self.is_noscalex
    
    def inverse_transform_x(self, data_x):
        if not(self.is_noscalex):
            return self.scaler_x.inverse_transform(data_x)
        else:
            return data_x
    
    def inverse_transform_v(self, data_v):
        if not(self.is_noscalex):
            return self.scaler_v.inverse_transform(data_v)
        else:
            return data_v

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

    
 


