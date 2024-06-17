import os
import csv
import pdb
import matplotlib.pyplot as plt

from utils.tools import   get_folders_list, plot_tuning_param_mae,  collecting_tuning_param
 

settings = {}
settings["dataset"]        = "CUEE_PMAPS_NIGHT" # "CUEE_PMAPS_NIGHT" # "CUEE_PMAPS"
settings["seq_length"]     =  5
settings["pred_length"]    =  1
settings["dropout"]        =  0.01
settings["network"]        = "RLSTM" 

settings["feature_mode"]   = "MS"
settings["moving_average"] =  4 
settings["enc_in"]         =  11 
settings["label_len"]      =  0
settings["n_heads"]        =  8
settings["d_model"]        =  16 
settings["d_layers"]       =  1   
settings["d_ff"]           =  2048
settings["e_layers"]       =  5 
settings["factor"]         =  1
settings["embed_type"]     =  0
settings["time_embeding"]  = "F"
settings["loss"]           = "l1"
settings["distil"]         = True
settings["des"]            = "Exp"
settings["train_epochs"]   = 50
settings["learning_rate"]  = 0.001
settings["batch_size"]     = 32
settings["use_Iclr"]       = False
 
 
tuning_param    = "e_layers" 
settings[tuning_param] =  None 
value_list             = [5, 10, 15, 20, 25, 50] # [1, 2, 5, 10, 15, 20] 

folder_list = get_folders_list(settings, tuning_param, value_list)   
value_list, overall_mae_list, n_param = collecting_tuning_param(folder_list, tuning_param, value_list,  which_input_dataset="valid")  
plot_tuning_param_mae(settings, value_list, overall_mae_list, n_param, tuning_param,  which_input_dataset="valid")
