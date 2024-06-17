import pdb
import matplotlib.pyplot as plt
from data_provider.dataloader_regression_CUEE_per_day_h5file import  DatasetCUEE 
from torch.utils.data import DataLoader
import os
import numpy as np

tag = "solarmap"
CUEE_ROOT = os.path.join(os.getcwd(),'dataset/' + tag)
CUEE_DATA       = "train_data.csv"
PMAS_CUEE_TEST  = "test_data.csv"
PMAS_CUEE_VALID = "val_data.csv"
PMAS_CUEE_TRAIN = "train_data.csv"
pdb.set_trace()
size = [4, 0, 1]

dataset    = DatasetCUEE(root_path= CUEE_ROOT,  test_data_path=PMAS_CUEE_TEST, valid_data_path=PMAS_CUEE_VALID, train_data_path=PMAS_CUEE_TRAIN, 
                         data_path=CUEE_DATA, flag='train', size=size, features='MS',  target='I', scale=True, timeenc=1, freq='h', train_only=False, tag=tag)
dataloader = DataLoader(dataset, batch_size=1)

trues_rev_list = []
input_list = []
 
for data_i, data_ in enumerate(dataloader):
    seq_x, seq_y, seq_v, seq_x_mark, seq_y_mark, seq_v_mark , date_time_x, date_time_y, seq_sky_condition = data_

    input_list.append(seq_x[:,-1, 0])  
    trues_rev_list.append(seq_y[:,0,:]) 

    #trues_rev_list.append(dataset.inverse_transform_y(seq_y[:,0,:])) 
        
    if data_i == 200:
 
        trues_rev_full = np.concatenate(trues_rev_list, axis=0) 
        trues_rev_ = trues_rev_full.reshape(-1,1) 

        input_list_full = np.concatenate(input_list, axis=0) 
        input_list_     = input_list_full.reshape(-1,1) 

        plt.plot(input_list_, label='input', color="blue") 
        plt.plot(trues_rev_, label='y-gt', color="black") 
        plt.legend()
        plt.savefig("First200.png")
        plt.show() 
        pdb.set_trace()