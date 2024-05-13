import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pdb 

from utils.tools import evaluation_skycondition, get_folder_name 

settings = {}
settings["dataset"]        = "CUEE_PMAPS_NIGHT" # "CUEE_PMAPS_NIGHT" # "CUEE_PMAPS"
settings["seq_length"]     =  4
settings["pred_length"]    =  1
settings["dropout"]        =  0.1
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
settings["seq_length"]     =  4  
settings["distil"]         = "True"
settings["des"]            = "Exp"


folder = get_folder_name(settings)
evaluation_skycondition(folder)