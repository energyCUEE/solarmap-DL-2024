import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
import pdb
import torch
import pickle
SMALL_SIZE = 15
MEDIUM_SIZE = 15
BIGGER_SIZE = 15

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

 
def plot_gradients(model, file_path=None):
    dict_grad = get_gradients(model)
    
    with open( file_path + '.pickle', 'wb') as handle:
        pickle.dump(dict_grad, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
    plot_gradient_mean_and_max_for_each_layer(dict_grad, file_path, which_dim="param") 



def get_gradients(model):
    named_parameters = model.named_parameters()
    layers = []
    abs_grads = [] 
    max_num_param = 0
    max_num_batch = 0
    for name, param in named_parameters:
        if(param.requires_grad) and ("bias" not in name) and (param.grad is not None):
            
            abs_grads.append(param.grad.abs().cpu().numpy())
            layers.append(name)
 
            num_param_0 = len(param.grad.abs().mean(dim=0).reshape(-1))

            if num_param_0 > max_num_param:
                max_num_param = num_param_0

             

    dict_grad = {"abs_grads":abs_grads, "layers": layers, "max_num_param": max_num_param  }  
    return dict_grad



def plot_gradient_mean_and_max_for_each_layer(dict_grad, file_path, which_dim="param"):
    
    abs_grads     = dict_grad["abs_grads"]  
    layers        = dict_grad["layers"] 
    max_num_param = dict_grad["max_num_param"]   
    
    #dimension =  max_num_param 
    dimension =  max_num_param # param
    average_across  = 0  # batch
    upper_bound = 0.2 


    averaged_grads = np.zeros([dimension, len(layers)]) + np.nan

    for ind_, abs_grads_ in enumerate(abs_grads):  
        mean_abs_grad_ = np.mean(abs_grads_, axis=average_across)
        if mean_abs_grad_.ndim >= 1:
            mean_abs_grad_ = mean_abs_grad_.reshape(-1)   
            averaged_grads[:mean_abs_grad_.shape[0], ind_] = mean_abs_grad_
        else: 
            averaged_grads[0, ind_] = mean_abs_grad_ 

    max_grads_numpy = np.zeros([dimension, len(layers)]) + np.nan

    for ind_, abs_grads_ in enumerate(abs_grads):
        max_abs_grads_ = np.max(abs_grads_, axis=average_across)  
        if max_abs_grads_.ndim >= 1:
            max_abs_grads_ = max_abs_grads_.reshape(-1)
            max_grads_numpy[:max_abs_grads_.shape[0], ind_] =max_abs_grads_
        else:
            max_grads_numpy[0, ind_] =max_abs_grads_


    if len(abs_grads) > 8: 
        fig_size_width = int(15 * len(abs_grads)/8) 
    else:
        fig_size_width = 15
    
     

    fig = plt.figure(figsize=(fig_size_width, 8))  
    fig.subplots_adjust(bottom=0.2) 
    plt.plot(max_grads_numpy, alpha=0.2, lw=2, color="r")
    plt.plot(averaged_grads,  alpha=0.2, lw=2, color="b") 

    plt.hlines(0, 0, len(abs_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(abs_grads), 1), layers, rotation=30, ha="right" )
    plt.xlim(left=0, right=len(abs_grads)-1)
    plt.ylim(bottom = -0.001, top=upper_bound) # zoom in on the lower gradient regions 
    plt.ylabel("average gradient") 
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient @ %d %s' % (dimension, which_dim), 'mean-gradient @ %d %s' % (dimension, which_dim), 'zero-gradient @ %d %s'  % (dimension, which_dim)], bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3) 
    
    if file_path is not None:
        file_path_name = file_path.split(".png")[0] + "-dim-%s.png" % which_dim
        plt.savefig(file_path_name)


if __name__ == "__main__":

    file_path = "checkpoints/RLSTM_CUEE_PMAPS_NIGHT_mv4_ftMS_enc11_sl5_ll0_pl1_dm16_nh8_el5_dl1_df2048_fc1_etype0_ebtimeF_dtTrue_dp0p10_Exp_l1loss_0/gradflow-@-ep-000.png"
    loaded_pickle = "checkpoints/RLSTM_CUEE_PMAPS_NIGHT_mv4_ftMS_enc11_sl5_ll0_pl1_dm16_nh8_el5_dl1_df2048_fc1_etype0_ebtimeF_dtTrue_dp0p10_Exp_l1loss_0/gradflow-@-ep-000.png.pickle"

    with open(loaded_pickle, 'rb') as handle:
        dict_grad = pickle.load(handle)
     

    plot_gradient_mean_and_max_for_each_layer(dict_grad, file_path, which_dim="param")
    plot_gradient_mean_and_max_for_each_layer(dict_grad, file_path, which_dim="batch")