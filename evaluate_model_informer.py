from utils.tools import  evaluation_skycondition, get_folder_name


settings = {}
settings["dataset"]        = "CUEE_PMAPS_NIGHT"  
settings["seq_length"]     =  4
settings["pred_length"]    =  1
settings["dropout"]        =  0.05
settings["network"]        = "Informer" 

settings["feature_mode"]   = "MS"
settings["moving_average"] =  4 
settings["enc_in"]         =  11 
settings["label_len"]      =  0
settings["n_heads"]        =  8
settings["d_model"]        =  16 
settings["d_layers"]       =  1   
settings["d_ff"]           =  2048
settings["e_layers"]       =  4 
settings["factor"]         =  3
settings["embed_type"]     =  2

settings["time_embeding"]  = "F"
settings["distil"]         = "True"
settings["des"]            = "Exp"

settings["loss"]           = "mse"
  
folder = get_folder_name(settings)
evaluation_skycondition(folder)