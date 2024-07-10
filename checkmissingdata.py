import numpy as np
import pdb
import pandas as pd
from tqdm import tqdm
file = open('./data_provider/stations.txt', 'r') 
stations_list = file.readlines()  
stations_list = [ station_.split("\n")[0] for station_ in stations_list] 

seq_length = 4 
missing_date_time_y_filename =  "dataset/true_cloud_relation_08JUL24/I-test-01-04-2022-06:00:00-17:00:00/sample_stats.npy"

numpy_list_of_dict = np.load(missing_date_time_y_filename, allow_pickle=True) 
checking_df        = pd.DataFrame(numpy_list_of_dict)  
x_timestemp_check = checking_df["Datetime-X"]

read_test    = pd.read_csv('dataset/true_cloud_relation_08JUL24/test_data_true_relation_lgbm_new_arrange.csv') 
target       = read_test.iloc[read_test["is_target"].values == True]

total_index  = np.arange(len(read_test["is_target"].values))
target_index = total_index[read_test["is_target"].values == True]
x_index      = (target_index - seq_length).tolist()  
x_timestamp  = read_test["Datetime"].iloc[x_index] 
x_timestamp  = pd.to_datetime(x_timestamp).dt.tz_localize(None).tolist()
  
assert_overall = bool(set(x_timestemp_check.tolist()) & set(x_timestamp))

