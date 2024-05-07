import pdb
import matplotlib.pyplot as plt
from data_provider.dataloader_regression_CUEE_per_day_h5file import  DatasetCUEE 
from torch.utils.data import DataLoader
import os
import numpy as np

tag = "CUEE_PMAPS_NIGHT"
CUEE_ROOT = os.path.join(os.getcwd(),'dataset/' + tag)
CUEE_DATA       = "pmaps_validate_with_nighttime.csv"
PMAS_CUEE_TEST  = "pmaps_test_with_nighttime.csv"
PMAS_CUEE_VALID = "pmaps_validate_with_nighttime.csv"
PMAS_CUEE_TRAIN = "pmaps_train_with_nighttime.csv"

size = [4, 0, 1]

dataset    = DatasetCUEE(root_path= CUEE_ROOT,  test_data_path=PMAS_CUEE_TEST, valid_data_path=PMAS_CUEE_VALID, train_data_path=PMAS_CUEE_TRAIN, 
                         data_path=CUEE_DATA, flag='val', size=size, features='MS',  target='I', scale=True, timeenc=1, freq='h', train_only=False, tag=tag)
dataloader = DataLoader(dataset, batch_size=1)

trues_rev_list = []
 
for data_i, data_ in enumerate(dataloader):
    seq_x, seq_y, seq_v,  seq_x_mark, seq_y_mark, seq_v_mark = data_  
  
    trues_rev_list.append(dataset.inverse_transform_y(seq_y[:,0,:])) 
        
    if data_i == 200:
 
        trues_rev_full = np.concatenate(trues_rev_list, axis=0) 
        trues_rev_ = trues_rev_full.reshape(-1,1) 
 
        plt.plot(trues_rev_, label='actual', color="black") 
        plt.savefig("First200.png")
        plt.show() 
        pdb.set_trace()