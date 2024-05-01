import os
import csv
import pdb
import matplotlib.pyplot as plt

from utils.tools import   get_folders_list 
checkpoint_folder_path = "checkpoints"
which_set = "valid"



if which_set == "valid": 
    val_folder_path = "valids" 
elif which_set == "test": 
    val_folder_path = "results"

settings = {}
settings["dataset"]        = "CUEE_PMAPS"
settings["seq_length"]     =  4
settings["pred_length"]    =  1
settings["network"]        = "RLSTM" 

settings["feature_mode"]   = "MS"
settings["moving_average"] =  4 
settings["enc_in"]         =  12 
settings["label_len"]      =  0
settings["n_heads"]        =  8
settings["d_model"]        =  50 
settings["d_layers"]       =  1   
settings["d_ff"]           =  2048
settings["e_layers"]       =  2 
settings["factor"]         =  1
settings["time_embeding"]  = "F"
 

meaning_param = {}
meaning_param["dm"] = "#Hidden" 
meaning_param["el"] = "#LSMTCell" 

# tuning_param    = "dm"
# settings[tuning_param] =  None 
# value_list             = [8, 16, 32] 

tuning_param    = "el"
settings[tuning_param] =  None 
value_list             = [1, 2, 10, 25, 50] 


     
folder_list = get_folders_list(settings, tuning_param, value_list) 

el_list = []
d_model = []
n_param = []
overall_mae_list = []
overall_mse_list = []

for folder_ in folder_list:
    setting_path_csv       = os.path.join(checkpoint_folder_path, folder_, "model_setting.csv")
    result_stat_path_csv   = os.path.join(val_folder_path, folder_, "stats.csv")
    
    with open(setting_path_csv) as csv_file:
        d_reader = csv.reader(csv_file)
        d_dict   = dict(d_reader)

    with open(result_stat_path_csv) as csv_file:
        stat_reader = csv.reader(csv_file)
        stat_dict   = dict(stat_reader)


    el_list.append(int(d_dict["e_layers"])) 
    d_model.append(int(d_dict["d_model"]))
    n_param.append(int(d_dict["Num-param"]))
    overall_mae_list.append(float(stat_dict["mae-overall"]))
    overall_mse_list.append(float(stat_dict["mse-overall"]))

    

plt.close("all")  

fig, ax1 = plt.subplots(figsize=(15, 5)) 

ax1.plot(value_list, overall_mae_list, color='red')  
ax1.set_xlabel(meaning_param[tuning_param], fontsize = 'large', color='red')
ax1.set_ylabel('MAE', fontsize = 'large')
ax1.tick_params(axis='x', colors='red')
ax1.grid(which='major', color='red', linewidth=0.8)

ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xlabel('# params', fontsize='large', color='green')   
ax2.tick_params(axis='x', colors='green') 
ax2.set_xticks(value_list)
ax2.set_xticklabels(n_param)
ax2.grid(which='major', color='green', linestyle='--', linewidth=1)

if val_folder_path == "valids":
    plt.title("Validation set")
    plt.tight_layout()
    plt.savefig("%s_sq%d_p%d_validate_tuning-%s-%d-%d.png" % (settings["network"], settings["seq_length"], settings["pred_length"], tuning_param, min(value_list), max(value_list))) 

elif val_folder_path == "results":
    plt.title("Test set")
    plt.tight_layout()
    plt.savefig("%s_sq%d_p%d_test_tuning-%s-%d-%d.png" % (settings["network"], settings["seq_length"], settings["pred_length"], tuning_param, min(value_list), max(value_list))) 