import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os
import pdb
import csv

MEANING_PARAM = {}
MEANING_PARAM["d_model"]    = "#Hidden" 
MEANING_PARAM["e_layers"]   = "#LSMTCell/#Layers" 
MEANING_PARAM["seq_length"] = "#lags" 

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.close("all")  
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def save_settings(args, setting, num_params):
    args_dict = vars(args)
    with open(os.path.join('./checkpoints/' + setting, 'model_setting.txt'), 'w') as file : 
        for key,value in args_dict.items():
            file.write("%s:%s\n" % (key, value))  
        file.write("%s:%d \n" % ("Num-param", num_params )) 


def save_settings_dict(args, setting, num_params, folder='checkpoints'):
    args_dict = vars(args)
    args_dict["Num-param" ] = num_params
    
    with open(os.path.join(folder, setting, 'model_setting.csv'), 'w')as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in args_dict.items():
            writer.writerow([key, value])
 


def set_folder_name(args, ii):
    # setting record of experiments
    dropout_argument = ("%.2f" % args.dropout).replace(".","p")

    if args.model == "PatchTST":
        setting = '{}_{}_mv{}_ft{}_enc{}_sl{}_ll{}_pl{}_ps{}_st{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_etype{}_eb{}_dt{}_dp{}_{}_{}loss_{}'.format( 
        args.model,
        args.data,
        args.moving_avg,
        args.features,
        args.enc_in, 
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.patch_len,
        args.stride,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed_type,
        args.embed,
        args.distil,
        dropout_argument,
        args.des, 
        args.loss, 
        ii)

    else: 
        setting = '{}_{}_mv{}_ft{}_enc{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_etype{}_eb{}_dt{}_dp{}_{}_{}loss_{}'.format( 
        args.model,
        args.data,
        args.moving_avg,
        args.features,
        args.enc_in, 
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed_type,
        args.embed,
        args.distil,
        dropout_argument,
        args.des, 
        args.loss, 
        ii )

    return setting

def get_folder_name(settings): 

    
    model_name    = settings["network"]
    dataset       = settings["dataset"] 

    moving_average = settings["moving_average"] 
    mode           = settings["feature_mode"] 
    enc            = settings["enc_in"]   

    seq_length     = settings["seq_length"]
    ll             = settings["label_len"]  
    pred_length    = settings["pred_length"]
 
    dm            = settings["d_model"] 
    nh            = settings["n_heads"]  
    el            = settings["e_layers"]  
    dl            = settings["d_layers"] 
    d_ff          = settings["d_ff"]
    fc            = settings["factor"]  
    time_embeding = settings["time_embeding"]  
    ebtype        = settings["embed_type"]  
    distill       =  settings["distil"]  
    dp            = ("%.02f" % settings["dropout"]).replace(".","p")
    des           = settings["des"]
    loss          = settings["loss"] 
 

    return "%s_%s_mv%d_ft%s_enc%d_sl%d_ll%d_pl%d_dm%d_nh%d_el%d_dl%d_df%d_fc%d_etype%d_ebtime%s_dt%s_dp%s_%s_%sloss_0"   % (model_name, dataset, moving_average, mode, enc, seq_length, ll, pred_length, dm, nh, el, dl, d_ff, fc, ebtype, time_embeding, distill, dp, des, loss)


def get_folders_list(settings, tuning_param, value_list): 
    folder_list = []
    for index_, value_ in enumerate(value_list):
        settings[tuning_param] = value_
        folder_name_ = get_folder_name(settings)
        folder_list.append(folder_name_) 
        print("[%d]: %s" % (index_, folder_name_) ) 
    return folder_list


def plot_axis(ax, y, ref_y=None, ylabel=None, title=None):

    ax.plot(y, '-', color='red', alpha=0.99, linewidth=2.0) 
    if ref_y is not None:
        ax.plot(ref_y, '--', color='blue', alpha=0.99, linewidth=2.0) 
    
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    if title is not None:
        ax.set_title(title)

    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, len(y), 37)
    minor_ticks = np.arange(0, len(y), 4)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True) 

    ax.grid(which='both')
    
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.8) 

def plotting_seasonal_data(y, ref_y=None, filename=None, title=None): 
    fig, axs = plt.subplots(4, 1, figsize=(12, 10))
    plot_axis(axs[0], y.observed, ref_y=ref_y, ylabel="Observation", title=title)
    plot_axis(axs[1], y.trend, ylabel="Trend")
    plot_axis(axs[2], y.seasonal, ylabel="Seasonal")
    plot_axis(axs[3], y.resid, ylabel="Residual") 
    
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def collecting_tuning_param(folder_list, tuning_param, value_list,  which_input_dataset="valid"):

    checkpoint_folder_path = "checkpoints" 

    if which_input_dataset == "valid": 
        val_folder_path = "valids" 
    elif which_input_dataset == "test": 
        val_folder_path = "results"
        
     
    el_list = []
    d_model = []
    n_param = []
    overall_mae_list = []
    overall_mse_list = []

    for folder_, value_ in zip(folder_list, value_list):
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
        overall_mse_list.append(float(stat_dict["rmse-overall"]))

        print("%s @ %s [%.1f] MAE %f" % (folder_, tuning_param, value_, float(stat_dict["mae-overall"])) )
    
    return value_list, overall_mae_list, n_param



def plot_tuning_param_mae(settings, value_list, overall_mae_list, n_param, tuning_param,  which_input_dataset="valid", meaning_param = MEANING_PARAM): 
    
    
    which_input_dataset = "valid"

    if which_input_dataset == "valid": 
        val_folder_path = "valids" 
    elif which_input_dataset == "test": 
        val_folder_path = "results"
        
    
    plt.close("all")  

    fig, ax1 = plt.subplots(figsize=(15, 5)) 

    ax1.plot(value_list, overall_mae_list, color='red', linewidth=2)  
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

    if tuning_param == "seq_length":
        text = "#Hidden=%d & #LSMTCell/Layers=%d" % (settings["d_model"], settings["e_layers"])

    elif tuning_param == "d_model":
        text = "#Lags=%d & #LSMTCell/Layers=%d" % (settings["seq_length"], settings["e_layers"])

    elif tuning_param == "e_layers":
        text = "#Lags=%d & #Hidden=%d" % (settings["seq_length"], settings["d_model"])

    if val_folder_path == "valids":
        plt.title("%s --- %s @ Validation set" % (settings["network"], text))
        plt.tight_layout()
        plt.savefig("%s_%s_sq%d_p%d_validate_tuning-%s-%d-%d.png" % (settings["network"], settings["dataset"], settings["seq_length"], settings["pred_length"], tuning_param, min(value_list), max(value_list))) 

    elif val_folder_path == "results":
        plt.title("Test set")
        plt.tight_layout()
        plt.savefig("%s_%s_sq%d_p%d_test_tuning-%s-%d-%d.png" % (settings["network"], settings["dataset"], settings["seq_length"], settings["pred_length"], tuning_param, min(value_list), max(value_list))) 