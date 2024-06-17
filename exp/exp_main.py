from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, RLSTM, RLSTM_ver2, Transformer, DLinear, Linear, NLinear, PatchTST, BiasCorrModel


from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric, MAE
import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import pdb
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
from utils.tools import save_settings_dict, evaluation_skycondition
from utils.learning_tools import plot_gradients
from tqdm import tqdm


torch.multiprocessing.set_sharing_strategy('file_system') 
warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

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
            'RLSTM_ver2': RLSTM_ver2,
            'BiasCorrModel': BiasCorrModel
        }
        model = model_dict[self.args.model].Model(self.args).float()

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Model: [%s]  Number of parameters: [%d]" % (self.args.model, num_params))

        self.num_params = num_params

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, which_loss="mse"):
        if which_loss == "mse":
            criterion = nn.MSELoss()
        elif which_loss == "l1":
            criterion = nn.L1Loss()
        return criterion
    
    def __myfeedforward(self, batch_x, batch_v, batch_x_mark, batch_v_mark):

        batch_x = batch_x.float().to(self.device) 
        batch_v = batch_v.float().to(self.device)
 

        batch_size, _, _ = batch_x.shape

        batch_x_mark = batch_x_mark.float().to(self.device) 
        batch_v_mark = batch_v_mark.float().to(self.device) 

        pred_len = self.args.pred_len
        pred_feature = self.args.d_target

        # decoder input
        # dec_inp = batch_v
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
         
        outputs = outputs.view(batch_size, pred_len, pred_feature)   ###### <<<<

        return outputs 


    def train(self, setting):
        
        train_data, train_loader = self._get_data(flag='train')
        
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        
        if not os.path.exists(path):
            os.makedirs(path)

        save_settings_dict(self.args, setting, self.num_params, folder=self.args.checkpoints) 
  
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion   = self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if self.args.scheduler == "ReduceLROnPlateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer = model_optim,
                                                mode='min', factor=0.5, patience=3, verbose=True ) 
        else:    
            scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                                steps_per_epoch = train_steps,
                                                pct_start = self.args.pct_start,
                                                epochs = self.args.train_epochs,
                                                max_lr = self.args.learning_rate)
            
 

        stat_ep_list = []
        
        for epoch in range(self.args.train_epochs):
            
            iter_count = 0
            train_loss = []
            train_MAE  = []

            self.model.train() 

            pbar = tqdm(train_loader)
            for i, (batch_x, batch_y,  batch_v, batch_x_mark, batch_y_mark,  batch_v_mark, batch_datetime_x, batch_datetime_y, batch_sky_condition, batch_sitename) in enumerate(pbar):  ###### <<<<
            
                iter_count += 1
                model_optim.zero_grad()
 
                 
                outputs = self.__myfeedforward(batch_x,  batch_v, batch_x_mark, batch_v_mark)

                batch_y = batch_y.float().to(self.device)  
                loss    = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                 
                output_rev_scale = train_data.inverse_transform_y(outputs.view(-1,1).detach().cpu().numpy())
                target_rev_scale = train_data.inverse_transform_y(batch_y.view(-1,1).detach().cpu().numpy()) 
                 
                train_mae  = MAE(output_rev_scale, target_rev_scale) 
                train_MAE.append(train_mae) 

                loss.backward()
                model_optim.step()
                    
                if self.args.scheduler != "ReduceLROnPlateau":
                    
                    if self.args.lradj == 'TST':
                        adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                        scheduler.step()

                pbar.set_description("iters: {0}, epoch: {1} | loss: {2:.7f}".format(i, epoch, sum(train_loss)/(len(train_loss)+1))) 
             
            plot_gradients(self.model,os.path.join(path, 'gradflow-@-ep-%03d.png' % epoch)) 
            
            train_loss = np.average(train_loss)
            train_MAE  = np.average(train_MAE)

            if not self.args.train_only:
                vali_loss, vali_MAE = self.vali(vali_data, vali_loader, criterion)
                test_loss, test_MAE = self.vali(test_data, test_loader, criterion) 

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss)) 
                print("                       | Train MAE : %3.4f Vali MAE : %3.4f Test MAE : %3.4f" % (train_MAE, vali_MAE, test_MAE))

                if self.args.scheduler == "ReduceLROnPlateau": 
                    scheduler.step(vali_loss) 
                early_stopping(vali_loss, self.model, path)

            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)
            
            stat_ep_list.append({"train_MAE":train_MAE, "valid_MAE":vali_MAE, "test_MAE":test_MAE, "train_loss": train_loss, "vali_loss": vali_loss, "test_loss": test_loss})

            if early_stopping.early_stop:
                print("Early stopping")
                break
            

            if self.args.lradj != 'TST' and self.args.scheduler != "ReduceLROnPlateau":
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        
        Stats_ep = pd.DataFrame(stat_ep_list)
        Stats_keys = stat_ep_list[0].keys()

        Stats_ep.to_csv(os.path.join(path, 'stats_mae_loss_ep.csv' ), header=Stats_keys, index=False)  

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))   
        return self.model
    


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_MAE  = []

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():

            for i, (batch_x, batch_y, batch_v, batch_x_mark, batch_y_mark, batch_v_mark, batch_datetime_x, batch_datetime_y, batch_sky_condition, batch_sitename) in enumerate(vali_loader):  ###### <<<<
                
                outputs = self.__myfeedforward(batch_x,  batch_v, batch_x_mark, batch_v_mark)

                batch_y = batch_y.float().to(self.device)  

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu() 

                loss = criterion(pred, true) 
                total_loss.append(loss)

                preds.append(pred.numpy())
                trues.append(true.numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds_rev_list = []
        trues_rev_list = []

        for seq_i in range(self.args.pred_len): 

            preds_rev = vali_data.inverse_transform_y(preds[:,seq_i,:])
            trues_rev = vali_data.inverse_transform_y(trues[:,seq_i,:])

            preds_rev_list.append(preds_rev)
            trues_rev_list.append(trues_rev)
 
            mae_, _, rmse, mape, mspe, rse, corr = metric(preds_rev, trues_rev) 
            total_MAE.append(mae_)  

        total_loss = np.average(total_loss)
        total_MAE  = np.average(total_MAE) 

        self.model.train()

        return total_loss, total_MAE
    
    def test(self, setting, test=0):
 
        test_data, test_loader = self._get_data(flag='test')
        
        print('Loading model : [%s]' % setting) 
        self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))
         
        preds = []
        trues = []
        inputx = []
        datetimes = []
        sky_conditions = []
        sitenames = []
        folder_path = './run_testing/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
 
        self.model.eval()
        with torch.no_grad():
            
            for i, (batch_x, batch_y, batch_v, batch_x_mark, batch_y_mark, batch_v_mark, batch_datetime_x, batch_datetime_y, batch_sky_condition, batch_sitename) in enumerate(test_loader):
 
                outputs = self.__myfeedforward(batch_x, batch_v, batch_x_mark, batch_v_mark)
 
                outputs = outputs.cpu().numpy()
                batch_y = batch_y.numpy()
                batch_sitename = batch_sitename.numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy()) 
                datetimes.append(batch_datetime_y) 
                sky_conditions.append(batch_sky_condition)
                sitenames.append(batch_sitename)
                if i % 1000 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()

        ################### 
        inputx_np    = np.concatenate(inputx, axis=0) 
        preds        = np.concatenate(preds, axis=0) 
        trues        = np.concatenate(trues, axis=0)

        datetimes      = np.concatenate(datetimes, axis=0)
        sky_conditions = np.concatenate(sky_conditions, axis=0)
        sitenames      = np.concatenate(sitenames, axis=0)

 

        preds_rev = test_data.inverse_transform_y(preds[:,0,:])
        trues_rev = test_data.inverse_transform_y(trues[:,0,:])
 

        total_MAE, _, total_RMSE, mape, mspe, rse, corr = metric(preds[:,0,:], trues[:,0,:]) 
        total_MAE_rev, _, total_RMSE_rev, mape, mspe, rse, corr = metric(preds_rev, trues_rev) 
           
        # result save
        folder_path = './testing_true_cloud_relation/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path) 
        
        print("On testing dataset ....")
        print('scal: rmse:{}, mae:{}'.format(total_RMSE, total_MAE)) 
        print('revs: rmse:{}, mae:{}'.format(total_RMSE_rev, total_MAE_rev))
 
        result_dict = {}
        # result_dict["inputx"] = inputx_np
        # result_dict["preds"] = preds.reshape(-1,1)
        # result_dict["trues"] = trues.reshape(-1,1)
        result_dict['datetime'] = datetimes
        result_dict['sitename'] = [int(site[0]) for site in sitenames]
        result_dict["Ihat"] = preds_rev
        result_dict["I"] = trues_rev
 

        # Initialize lists to store individual sky condition components
        sky_condition_kbar = []
        sky_condition_poc = []

        for batch in sky_conditions:
            # Each batch contains sub-arrays, extract kbar and poc from each and store them
            kbars = [item[0] for item in batch]
            pocs = [item[1] for item in batch]

            sky_condition_kbar.extend(kbars)
            sky_condition_poc.extend(pocs)

        # Convert lists to numpy arrays if further numerical processing is needed
        sky_condition_kbar = np.array(sky_condition_kbar)
        sky_condition_poc  = np.array(sky_condition_poc)

        # Add these to your result_dict
        result_dict['k_bar'] = sky_condition_kbar
        result_dict['condition'] = sky_condition_poc

         

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(os.path.join(folder_path, 'result_dict.npy'), result_dict)        

        stats_dict = {}

        stats_dict["mae"] = total_MAE
        stats_dict["mse"] = total_RMSE 
        stats_dict["mae_rev"] = total_MAE_rev
        stats_dict["mse_rev"] = total_RMSE_rev 

        with open(os.path.join(folder_path, 'stats_mae_mse.csv'), 'w') as f:
            for key in stats_dict.keys():
                f.write("%s,%s\n"%(key,stats_dict[key]))
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)

        with open(os.path.join(folder_path, 'result.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result_dict.keys())

            # Find the maximum length among the values
            max_length = max(len(value) for value in result_dict.values())

            # Iterate over the range of max_length and write each row
            for i in range(max_length):
                row = []
                for value in result_dict.values():
                    if i < len(value):
                        if isinstance(value[i], (list, np.ndarray)):
                            row.append(str(value[i][0]))  # Take the first element of the array
                        else:
                            row.append(str(value[i]))
                    else:
                        row.append('')
                writer.writerow(row)
        
        return total_MAE_rev



    def evaluation(self, foldername, condition_spit_sky_condition="k_bar"):
        
        evaluation_skycondition(foldername, condition_spit_sky_condition=condition_spit_sky_condition)