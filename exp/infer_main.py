from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, RLSTM, RLSTM_ver2, Transformer, DLinear, Linear, NLinear, PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pdb
import csv
from utils.tools import save_settings_dict
warnings.filterwarnings('ignore')

class Infer_Main(Exp_Basic):
    def __init__(self, args):
        super(Infer_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'RLSTM': RLSTM,
            'RLSTM_ver2': RLSTM_ver2
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion 
 

    def run(self, setting, mode="test"):
        test_data, test_loader = self._get_data(flag=mode)
         
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Number of parameters: %d" % num_params)

        if mode == "test":
            main_folder_path = "results" 
        elif mode == "val": 
            main_folder_path = "valids"

        folder_path = os.path.join(main_folder_path, setting)
        os.makedirs(main_folder_path, exist_ok=True)    
        os.makedirs(folder_path, exist_ok=True)  
        save_settings_dict(self.args, setting, num_params, folder=main_folder_path) 

        preds = []
        trues = []
        inputx = []
        timestamp_y = []
        if mode == "test":
            folder_path_ = './results_per_sample/' + setting + '/'
            if not os.path.exists(folder_path_):
                os.makedirs(folder_path_)

        self.model.eval()
        MSE_temp_list = []
        pbar = tqdm(test_loader)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_v, batch_x_mark, batch_y_mark, batch_v_mark) in enumerate(pbar):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_v = batch_v.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_v_mark = batch_v_mark.float().to(self.device)

                batch_size, pred_len, pred_feature = batch_y.shape

                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if 'Linear' in self.args.model or 'TST' in self.args.model or "RLSTM" in self.args.model:
                        outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, batch_v, batch_v_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_v, batch_v_mark)


                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                 
                outputs = outputs.view(batch_size, pred_len, pred_feature)   ###### <<<<
                batch_y = batch_y.to(self.device)
                 
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze() 
 
                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                timestamp_y.append(batch_y_mark)
                
                if (mode == "test") and (i % 20 == 0) :
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path_, str(i) + '.png'))
                
                MSE_temp = np.mean((pred - true) ** 2)
                MSE_temp_list.append(MSE_temp)
                pbar.set_description("MSE %f" % (sum(MSE_temp_list)/len(MSE_temp_list)) )

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()


        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save

        mae_list = []
        rmse_list = []
        
        performance_dict = {}

        preds_rev_list = []
        trues_rev_list = []
        
        mae_scaled_list = []
        rmse_scaled_list = []

        for seq_i in range(self.args.pred_len): 
 
            preds_rev = test_data.inverse_transform_y(preds[:,seq_i,:])
            trues_rev = test_data.inverse_transform_y(trues[:,seq_i,:]) 

            preds_rev_list.append(preds_rev)
            trues_rev_list.append(trues_rev)

            mae, _ , rmse, mape, mspe, rse, corr = metric(preds_rev, trues_rev)
            
            mae_list.append(mae)
            rmse_list.append(rmse)
 
            mae_scaled, _, rmse_scaled, _, _, _, _ = metric(preds[:,seq_i,:], trues[:,seq_i,:]) 

            performance_dict["rmse-%d" % seq_i] = rmse
            performance_dict["mae-%d" % seq_i]  = mae
            performance_dict["rse-%d" % seq_i]  = rse
            performance_dict["corr-%d" % seq_i] = corr 

            print('%d:  rmse: %f, mae: %f | rmse-s: %f, mae-s: %f' % (seq_i, rmse, mae, rmse_scaled, mae_scaled))
 
            mae_scaled_list.append(mae_scaled)
            rmse_scaled_list.append(rmse_scaled)
        
        performance_dict["rmse-overall" ] = sum(rmse_list)/self.args.pred_len
        performance_dict["mae-overall" ] = sum(mae_list)/self.args.pred_len

        performance_dict["rmse-s-overall" ] = sum(rmse_scaled_list)/self.args.pred_len
        performance_dict["mae-s-overall" ] = sum(mae_scaled_list)/self.args.pred_len

        print('OVERALL:    rmse:{}, mae:{}'.format( sum(rmse_list)/self.args.pred_len, sum(mae_list)/self.args.pred_len ))
        print('OVERALL-S:  mse:{}, mae:{}'.format( sum(rmse_scaled_list)/self.args.pred_len, sum(mae_scaled_list)/self.args.pred_len ))
        
 
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(os.path.join(folder_path , 'pred.npy'), preds)
        np.save(os.path.join(folder_path , 'gt.npy'), trues)
        
        performance_dict["Num-param"] = num_params

        with open(os.path.join(folder_path, 'stats.csv'), 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in performance_dict.items():
                writer.writerow([key, value])

        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        
        preds_rev_concat = np.concatenate(preds_rev_list, axis=1)
        trues_rev_concat = np.concatenate(trues_rev_list, axis=1) 

        np.save(os.path.join(folder_path , 'pred_rev.npy'), preds_rev_concat)
        np.save(os.path.join(folder_path , 'gt_rev.npy'), trues_rev_concat)
 

        return preds_rev_concat, trues_rev_concat, timestamp_y, mae_list, rmse_list, test_data

