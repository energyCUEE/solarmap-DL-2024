import os
import pdb
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from utils.tools import get_args, set_folder_name

if __name__ == '__main__':
    
    
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
    print(args)

    Exp = Exp_Main
 
 
    if args.is_training:

        for ii in range(args.itr):
            setting = set_folder_name(args, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting)) 
            
            exp.train(setting)  
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting) 

            exp.evaluation(setting, condition_spit_sky_condition="k_bar") 
            exp.evaluation(setting, condition_spit_sky_condition="condition")

            torch.cuda.empty_cache()
    else:
        ii = 0
        
        setting = set_folder_name(args, ii)

        exp = Exp(args)  # set experiments
        
        print('>>>>>>> Testing    : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        print('>>>>>>> Evaluation : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.evaluation(setting, condition_spit_sky_condition="k_bar")

        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

        exp.evaluation(setting, condition_spit_sky_condition="condition")
        
        torch.cuda.empty_cache()
        
