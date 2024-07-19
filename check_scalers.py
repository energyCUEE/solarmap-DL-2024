
import os
import pickle
import pdb
import numpy as np
import h5py 
from tqdm import tqdm 

seq_x_list = []
seq_ov_list = []

path = "stats_no_nwp/output_features" 
# ['Iclr', 'CI', 'R', 'hour_encode1', 'day', 'month', 'minute', 'latt', 'long' ]

seq_id = 0
pbar = tqdm(range(1,13))
for Month_id in pbar:
    h5file = h5py.File(os.path.join(path, "feat_fixedfeature_month_%02d.h5" % Month_id), 'r') 
 
    seq_x  = h5file["seq_x"][:,seq_id,:]
    seq_v  = h5file["seq_v"][:,seq_id,:]  


    seq_x_list.append(seq_x)
    seq_ov_list.append(seq_v) 

    pbar.set_description("Month id %d" % Month_id)


seq_x_list = np.concatenate(seq_x_list, axis=0)
seq_ov_list = np.concatenate(seq_ov_list, axis=0)
  
seq_x_max  = seq_x_list.max(axis=0)
seq_x_min  = seq_x_list.min(axis=0)
seq_x_mean  = seq_x_list.mean(axis=0)
seq_x_std  = seq_x_list.std(axis=0)

print("'Iclr', 'CI', 'R', 'hour_encode1', 'day', 'month', 'minute', 'latt', 'long' ")
print(seq_x_max)
print(seq_x_min) 

seq_ov_ = seq_ov_list.max(axis=0)
print(seq_ov_)  