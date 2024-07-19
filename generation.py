import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import pdb
from utils.tools import set_folder_name
from data_provider.data_factory import data_provider
from utils.tools import get_args, set_folder_name

if __name__ == '__main__':

    # run this file with ./cuee_scripts/Data_generation.sh
    parser = get_args()  
    args = parser.parse_args() 

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
     
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]


    print('Args in experiment:') 
 
    _, _ = data_provider(args, 'test')
    # _, _ = data_provider(args, 'val')
    # _, _ = data_provider(args, 'train')
 
  
        
