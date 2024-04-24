import pdb
import matplotlib.pyplot as plt
from data_provider.dataloader_regression_CUEE import  DatasetCUEE 
from torch.utils.data import DataLoader
import os
import numpy as np

CUEE_ROOT = os.path.join(os.getcwd(),'dataset/CUEE')
CUEE_DATA = 'updated_measurement_Iclr_new.csv'

size = [37, 0, 4]

dataset    = DatasetCUEE(root_path = CUEE_ROOT, flag='test', size=size, features='MS', data_path=CUEE_DATA, target='I', scale=True, timeenc=1, freq='h', train_only=False)
dataloader = DataLoader(dataset, batch_size=1)

trues_rev_list = []

for data_ in dataloader:
    seq_x, seq_y, seq_v,  seq_x_mark, seq_y_mark, seq_v_mark  = data_ 

    trues_rev = dataset.inverse_transform_y(seq_y[:,0,:]) 
    trues_rev_list.append(trues_rev)
 
trues_rev_full = np.concatenate(trues_rev_list, axis=0) 
trues_rev_ = trues_rev_full.reshape(-1)[:200]


time_x = np.arange(len(trues_rev_))  
time_x_tick = np.arange(0, len(trues_rev_), 36)

plt.plot(time_x, trues_rev_, label='actual', color="black") 
plt.savefig("First200.png")
plt.show()