import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os
import pdb
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from decimal import Decimal
import argparse

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
 


def get_args():
    
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    # select for BiasCorrModel
    parser.add_argument('--option_Ihat1', default='I_clr', help='we can selected the option_Ihat1 such as I_clr, I_nwp')
    parser.add_argument('--m2_name', type=str, required=True, default='RLSTM',  help='model name, options: [RLSTM, Transformer]')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
    parser.add_argument('--model_id',  type=str,   required=True, default='test', help='model id')
    parser.add_argument('--model',     type=str,   required=True, default='Autoformer',  help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data',            type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path',       type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path',       type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--test_data_path',  type=str, default='pmaps_test_with_nighttime.csv', help='data file')  
    parser.add_argument('--train_data_path', type=str, default='pmaps_train_with_nighttime.csv', help='data file')
    parser.add_argument('--valid_data_path', type=str, default='pmaps_validate_with_nighttime.csv', help='data file')

    parser.add_argument('--features',        type=str, default='M',  help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target',          type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq',            type=str, default='h',  help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints',     type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--test_result_folder', type=str, default='./test_results/', help='location of test results')

    # forecasting task with I_clr
    parser.add_argument('--use_Iclr', action='store_true', default=False, help='whether using Iclr as a prior for input')
    parser.add_argument('--input_dropout',   type=float, default=0.01, help='dropout for regularizing the input Iclr') 
    
    # forecasting task
    parser.add_argument('--seq_len',   type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len',  type=int, default=96, help='prediction sequence length')
    parser.add_argument('--d_target',  type=int, default=1, help='feature dimension of the target prediction')

    # DLinear
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    parser.add_argument('--fc_dropout',    type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout',  type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len',     type=int,   default=16, help='patch length')
    parser.add_argument('--stride',        type=int,   default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin',         type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine',        type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    #parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in',     type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in',     type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out',      type=int, default=7, help='output size')
    parser.add_argument('--d_model',    type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads',    type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers',   type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers',   type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff',       type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor',     type=int, default=1, help='attn factor')
    parser.add_argument('--distil',     action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    parser.add_argument('--embed',      type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')    
    
    ## RLSMT and Formers
    parser.add_argument('--dropout',    type=float, default=0.05, help='dropout')  
    parser.add_argument('--is_noscaley', action='store_true', help='apply scaling to y', default=False)
    parser.add_argument('--is_noscalex', action='store_true', help='apply scaling to x', default=False)

    # optimization
    parser.add_argument('--num_workers',   type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr',           type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs',  type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size',    type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience',      type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des',        type=str, default='test', help='exp description')
    parser.add_argument('--loss',       type=str, default='mse', help='loss function')
    parser.add_argument('--scheduler',   type=str, default='OneCycleLR', help='adjust learning rate')
    parser.add_argument('--lradj',      type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start',  type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    parser.add_argument('--note', default=None, help='Some note about the experiments')

    return parser


def set_folder_name(args, ii):
    # setting record of experiments
    dropout_argument = ("%.2f" % args.dropout).replace(".","p")
    lr_argument = ('%.1E' % Decimal("%.5f" % args.learning_rate)).replace(".","p") 
 
     
    setting = '{}_{}_{}_{}_ft{}_enc{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_etype{}_eb{}_dt{}_dp{}_loss{}_ep{}_lr{}_bs{}'.format( 
    args.option_Ihat1,
    args.model,
    args.m2_name,
    args.data, 
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
    args.loss,
    args.train_epochs, 
    lr_argument,
    args.batch_size )

    if args.is_noscaley: 
        setting = "%s_%s" % (setting, "NotScaleY") 
            
    return setting

def get_folder_name(settings): 

    
    model_name    = settings["network"]

    dataset       = settings["dataset"] 

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
    
    des           = settings["des"]
    loss          = settings["loss"] 
    
    train_epochs  = settings["train_epochs"] 
    learning_rate = settings["learning_rate"] 
    lr_argument = ('%.2E' % Decimal("%.5f" % learning_rate)).replace(".","p")
    batchsize     = settings["batch_size"] 

    if settings["use_Iclr"]:
        dp = ("%.2f-ICLR-%.3f" % (settings["dropout"], settings["input_dropout"])).replace(".","p")
    else:
        dp = ("%.02f" % settings["dropout"]).replace(".","p")


    if model_name == "PatchTST":

        moving_average = settings["moving_average"] 
        patch_len = settings["patch_len"]
        stride = settings["stride"] 

        return "%s_%s_ft%s_enc%d_sl%d_ll%d_pl%d_ps{}_st{}_dm%d_nh%d_el%d_dl%d_df%d_fc%d_etype%d_ebtime%s_dt%s_dp%s_loss%s_ep%d_lr%s_bs%d"  % (model_name, dataset,  mode, enc, seq_length, ll, pred_length, patch_len, stride, dm, nh, el, dl, d_ff, fc, ebtype, time_embeding, distill, dp, loss, train_epochs, lr_argument, batchsize)

    elif (model_name == "Autoformer") or (model_name == "DLinear"): 

        moving_average = settings["moving_average"] 
        return "%s_%s_mv%d_ft%s_enc%d_sl%d_ll%d_pl%d_dm%d_nh%d_el%d_dl%d_df%d_fc%d_etype%d_ebtime%s_dt%s_dp%s_loss%s_ep%d_lr%s_bs%d" % (model_name, dataset, moving_average, mode, enc, seq_length, ll, pred_length, dm, nh, el, dl, d_ff, fc, ebtype, time_embeding, distill, dp, loss, train_epochs, lr_argument, batchsize)
    else:

        return "%s_%s_ft%s_enc%d_sl%d_ll%d_pl%d_dm%d_nh%d_el%d_dl%d_df%d_fc%d_etype%d_ebtime%s_dt%s_dp%s_loss%s_ep%d_lr%s_bs%d" % (model_name, dataset, mode, enc, seq_length, ll, pred_length, dm, nh, el, dl, d_ff, fc, ebtype, time_embeding, distill, dp, loss, train_epochs, lr_argument, batchsize)

def get_folders_list(settings, tuning_param, value_list): 
    folder_list = []
    for index_, value_ in enumerate(value_list):
        settings[tuning_param] = value_
        folder_name_ = get_folder_name(settings)
        folder_list.append(folder_name_) 
        print("[%d]: %s" % (index_, folder_name_) ) 
    return folder_list


def plot_axis(ax, y, ref_y=None, ylabel=None, title=None, error=None):

    line1 = ax.plot(y, '-', color='red', alpha=0.99, linewidth=2.0) 
    
    if ref_y is not None:
        
        line2 = ax.plot(ref_y, '--', color='black', alpha=0.99, linewidth=2.0)  

        if error is not None:
            text ="MAE:%.2f" % error
            ax.text(800, 0.8*np.max(ref_y[~np.isnan(ref_y)]), text, color='red', bbox=dict(facecolor='white', edgecolor='red'))
    
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



def calculate_residual_decom(y, ref_y):
    residual = {}
    residual["Observation"] = mean_absolute_error(y.observed, ref_y.observed)

    y_trend = y.trend
    y_trend = y_trend[~np.isnan(y_trend)]
    ref_y_trend = ref_y.trend
    ref_y_trend = ref_y_trend[~np.isnan(ref_y_trend)] 
    residual["Trend"] = mean_absolute_error(y_trend, ref_y_trend)

    y_seasonal = y.seasonal
    y_seasonal = y_seasonal[~np.isnan(y_seasonal)]
    ref_y_seasonal = ref_y.seasonal
    ref_y_seasonal = ref_y_seasonal[~np.isnan(ref_y_seasonal)] 
    residual["Seasonal"] = mean_absolute_error(y_seasonal, ref_y_seasonal)

    y_resid = y.resid
    y_resid = y_resid[~np.isnan(y_resid)]
    ref_y_resid = ref_y.resid
    ref_y_resid = ref_y_resid[~np.isnan(ref_y_resid)]    
    residual["Residual"] = mean_absolute_error(y_resid, ref_y_resid)
    return residual



def plotting_seasonal_data_wrt_gt(y, ref_y=None, filename=None, title=None, residual=None): 
    fig, axs = plt.subplots(4, 1, figsize=(12, 10))
    if residual is not None:
        plot_axis(axs[0], y.observed, ref_y.observed, ylabel="Observation", title=title, error = residual["Observation"])
        plot_axis(axs[1], y.trend,    ref_y.trend, ylabel="Trend", error = residual["Trend"])
        plot_axis(axs[2], y.seasonal, ref_y.seasonal, ylabel="Seasonal", error = residual["Seasonal"])
        plot_axis(axs[3], y.resid,    ref_y.resid, ylabel="Residual", error = residual["Residual"]) 
    else:
        plot_axis(axs[0], y.observed, ref_y.observed, ylabel="Observation", title=title   )
        plot_axis(axs[1], y.trend,    ref_y.trend, ylabel="Trend" )
        plot_axis(axs[2], y.seasonal, ref_y.seasonal, ylabel="Seasonal" )
        plot_axis(axs[3], y.resid,    ref_y.resid, ylabel="Residual") 
    
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

 

def evaluation_skycondition(folder, condition_spit_sky_condition="k_bar"):

    
    main_folder_path = "testing_true_cloud_relation_pt1" 
    stats_ckp_folder = os.path.join(main_folder_path, folder)
    stats_df = pd.read_csv(os.path.join(stats_ckp_folder, 'stats_mae_mse.csv')) 
    df       = pd.read_csv(os.path.join(stats_ckp_folder, 'result.csv'))  
 

    df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
    df['datetime'] = df['datetime'].dt.tz_localize(None) 
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour

    # pdb.set_trace()
    if condition_spit_sky_condition == 'k_bar':
        df['sky_condition'] = df['k_bar'].apply(
            lambda x: 'cloudy' if x < 0.6 else ('partly_cloudy' if x < 0.9 else 'clear'))
    else :
        df['sky_condition'] = df['condition'].apply(
            lambda x: 'cloudy' if x == 3 else ('partly_cloudy' if x == 2 else 'clear'))

    # Calculate MAE and RMSE by sky_condition
    sky_condition_mae = df.groupby('sky_condition')[['I', 'Ihat']].apply(lambda x: mean_absolute_error(x['I'], x['Ihat'])).reset_index(name='MAE')
    sky_condition_rmse = df.groupby('sky_condition')[['I', 'Ihat']].apply(lambda x: np.sqrt(mean_squared_error(x['I'], x['Ihat']))).reset_index(name='RMSE')

    # Define the order of sky conditions
    category_order = ["clear", "partly_cloudy", "cloudy"]

    # Convert 'sky_condition' to a categorical type with the specified order
    sky_condition_mae['sky_condition'] = pd.Categorical(sky_condition_mae['sky_condition'], categories=category_order, ordered=True)
    sky_condition_rmse['sky_condition'] = pd.Categorical(sky_condition_rmse['sky_condition'], categories=category_order, ordered=True)

    # Sort the DataFrames
    sky_condition_mae = sky_condition_mae.sort_values('sky_condition')
    sky_condition_rmse = sky_condition_rmse.sort_values('sky_condition')
        
    overall_mae        = mean_absolute_error(df['I'], df['Ihat'])
    overall_rmse       = np.sqrt(mean_squared_error(df['I'], df['Ihat']))
    print('Overall MAE [%s]:  %.2f' % (condition_spit_sky_condition, overall_mae))
    print('Overall RMSE [%s]: %.2f' % (condition_spit_sky_condition, overall_rmse))
    
    print('\033[1mMAE by sky_condition [%s]\033[0m' % condition_spit_sky_condition)
    print(sky_condition_mae.to_string(index=False))

    print('\n\033[1mRMSE by sky_condition [%s]\033[0m' % condition_spit_sky_condition)
    print(sky_condition_rmse.to_string(index=False))

    # Save the metrics to a csv file
    stats_data = pd.concat([sky_condition_mae, sky_condition_rmse], axis=0)
    stats_data.to_csv('%s/stats_mae_mbe_skycondition-%s.csv' % (stats_ckp_folder, condition_spit_sky_condition))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey='row')
    sky_condition_names = ['Clear sky', 'Partly cloudy sky', 'Cloudy sky']

    # Define three different color sets for each subplot
    color_sets = [['#86A789', '#D2E3C8'], ['#22668D', '#8ECDDD'], ['#2D3250', '#7077A1']]

    for i, condition in enumerate(['clear', 'partly_cloudy', 'cloudy']):
        filter_test_df = df[df['sky_condition'] == condition]
        mae = filter_test_df.groupby('hour')[['I', 'Ihat']].apply(
            lambda x: mean_absolute_error(x['I'], x['Ihat'])).reset_index(name='MAE')
        mbe = filter_test_df.groupby('hour')[['I', 'Ihat']].apply(lambda x: x['Ihat'].mean() - x['I'].mean()).reset_index(
            name='MBE')

        # Use different color sets for each subplot
        bar_colors_mae = [color_sets[i][0] if 10 <= hour <= 15 else color_sets[i][1] for hour in mae['hour']]
        bar_colors_mbe = [color_sets[i][0] if 10 <= hour <= 15 else color_sets[i][1] for hour in mbe['hour']]

        axes[0, i].bar(mae['hour'], mae['MAE'], color=bar_colors_mae)
        axes[0, i].set_title(f'{sky_condition_names[i]} (n={len(filter_test_df)})', fontsize=20)
        axes[0, i].set_xlabel('Hour', fontsize=20)
        axes[0, i].set_ylabel('MAE [W/sqm]', fontsize=20)

        axes[1, i].bar(mbe['hour'], mbe['MBE'], color=bar_colors_mbe)
        axes[1, i].set_xlabel('Hour', fontsize=20)
        axes[1, i].set_ylabel('MBE [W/sqm]', fontsize=20)
         
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(stats_ckp_folder, 'test_metric-%s.png' % condition_spit_sky_condition)) 


def scale_latt(xarray, LATT_C= 0.5*(5.572250 + 20.509355), LATT_MAX=20.509355, LATT_MIN=5.572250):
    return (xarray - LATT_C)/(LATT_MAX - LATT_MIN) 

def scale_long(xarray, LONG_C= 0.5*(105.688477 + 97.338867), LONG_MAX=105.688477, LONG_MIN=97.338867):
    return (xarray - LONG_C)/(LONG_MAX - LONG_MIN) 


# def scaling_CI(xarray, mean=59.63, std=60.02):
#     xarray = (xarray - mean)/std


# def scaling_R(xarray, mean=100.95, std=55.78):
#     xarray = (xarray - mean)/std

def scale_temp(xarray, mean=28.49, std=7.49): 
    return  (xarray - mean)/std


def scale_Inwp(xarray, mean=477.29, std=291.91):
    return (xarray - mean)/std


def scaling(xarray, tag):
    if tag == "Iclr":
        xarray = xarray/1200
    elif (tag == "CI" ) or (tag == "R" ): 
        xarray = xarray/255
    elif (tag == "hour_encode1" ) : 
        xarray = xarray/5.5
    elif (tag == "day") : 
        xarray = xarray/366 
    elif (tag == "month") : 
        xarray = xarray/12
    elif (tag == "minute") : 
        xarray = xarray/60    
    elif (tag == "latt") : 
        xarray = scale_latt(xarray)
    elif (tag == "long") : 
        xarray = scale_long(xarray)

    elif (tag == "Tnwp") : 
        xarray = scale_temp(xarray)

    elif (tag == "Inwp") : 
        xarray = scale_Inwp(xarray)

    else:
        raise KeyError

    return xarray

def scaling_LxF(xarray, columns=['Iclr', 'CI', "R", 'hour_encode1', 'day', 'month', 'minute', 'latt', 'long', 'Tnwp', 'Inwp']):

    for tag_indx, tag in enumerate(columns):
        xarray[:,tag_indx] = scaling(xarray[:,tag_indx], tag)

    return xarray