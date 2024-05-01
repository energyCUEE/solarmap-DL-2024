import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os
import pdb
import csv
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
 

def get_folder_name(settings): 

    dataset    = settings["dataset"]
    seq_length = settings["seq_length"]
    pred_length = settings["pred_length"]
    model_name  = settings["network"] 
    dataset     = settings["dataset"] 
    moving_average = settings["moving_average"] 
    mode  = settings["feature_mode"] 
    enc   = settings["enc_in"]   
    ll    = settings["label_len"]   
    dm    = settings["d_model"] 
    nh    = settings["n_heads"]  
    el    = settings["e_layers"]  
    dl    = settings["d_layers"] 
    d_ff    = settings["d_ff"]
    fc    = settings["factor"]  
    time_embeding = settings["time_embeding"]  
    loss = settings["loss"] 

    return "%s_%d_%d_%s_%s_mv%d_ft%s_enc%d_sl%d_ll%d_pl%d_dm%d_nh%d_el%d_dl%d_df%d_fc%d_ebtime%s_dtTrue_Exp_%sloss_0"   % (dataset, seq_length, pred_length, model_name, dataset, moving_average, mode, enc, seq_length, ll, pred_length, dm, nh, el, dl, d_ff, fc, time_embeding, loss)


def get_folders_list(settings, tuning_param, value_list): 
    folder_list = []
    for index_, value_ in enumerate(value_list):
        settings[tuning_param] = value_
        folder_name_ = get_folder_name(settings)
        folder_list.append(folder_name_)
        print("[%d]: %s" % (index_, folder_name_) ) 
    return folder_list