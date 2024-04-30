import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 


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
    ave_grads, max_grads, layers = get_gradients(model)
    plot_gradient_mean_and_max_for_each_layer(ave_grads, max_grads, layers, file_path)



def get_gradients(model):
    named_parameters = model.named_parameters()
    layers = []
    ave_grads = []
    max_grads = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().numpy())
            max_grads.append(p.grad.abs().max().cpu().numpy())

    return ave_grads, max_grads, layers



def plot_gradient_mean_and_max_for_each_layer(ave_grads, max_grads, layers, file_path):
    
    if len(ave_grads) > 8:

        fig_size_width = int(15 * len(ave_grads)/8)

    else:
        fig_size_width = 15

    fig = plt.figure(figsize=(fig_size_width, 8))  
    fig.subplots_adjust(bottom=0.2)
    plt.plot(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="r")
    plt.plot(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color="b") 
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation=30, ha="right" )
    plt.xlim(left=0, right=len(ave_grads)-1)
    plt.ylim(bottom = -0.001, top=0.05) # zoom in on the lower gradient regions 
    plt.ylabel("average gradient") 
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'], bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
    if file_path is not None:
        plt.savefig(file_path)