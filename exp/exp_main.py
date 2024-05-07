from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, RLSTM, RLSTM_ver2, Transformer, DLinear, Linear, NLinear, PatchTST
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
from utils.tools import save_settings_dict
from utils.learning_tools import plot_gradients
from tqdm import tqdm
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
            'RLSTM_ver2': RLSTM_ver2
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

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_MAE  = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_v, batch_x_mark, batch_y_mark, batch_v_mark) in enumerate(vali_loader):  ###### <<<<
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_v = batch_v.float().to(self.device)

                batch_size, pred_len, pred_feature = batch_y.shape

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_v_mark = batch_v_mark.float().to(self.device)

                # decoder input
                dec_inp = batch_v
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
                batch_y = batch_y.to(self.device) 

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                #loss = np.mean((pred.numpy() - true.numpy()) ** 2)   
                total_loss.append(loss)
 
                pred_rev = vali_data.inverse_transform_y(pred.view(-1,1).numpy())
                true_rev = vali_data.inverse_transform_y(true.view(-1,1).numpy())
                MAE_loss  = MAE(pred_rev, true_rev)
                total_MAE.append(MAE_loss)


        total_loss = np.average(total_loss)
        total_MAE  = np.average(total_MAE) 
        self.model.train()
        return total_loss, total_MAE

    def train(self, setting):
        
        train_data, train_loader = self._get_data(flag='train')
        
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        
        if not os.path.exists(path):
            os.makedirs(path)

        save_settings_dict(self.args, setting, self.num_params) 
  
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
            for i, (batch_x, batch_y,  batch_v, batch_x_mark, batch_y_mark,  batch_v_mark) in enumerate(pbar):  ###### <<<<
            
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device) 
                batch_y = batch_y.float().to(self.device)
                batch_v = batch_v.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_v_mark = batch_v_mark.float().to(self.device)

                batch_size, pred_len, pred_feature = batch_y.shape 
 
                # decoder input 
                # dec_inp = torch.zeros(self.args.batch_size, self.args.pred_len, self.args.enc_in ).float().to(self.device)
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device) 
                # encoder - decoder 
                if 'Linear' in self.args.model or 'TST' in self.args.model or "RLSTM" in self.args.model: 
                        outputs = self.model(batch_x)

                else:

                    if self.args.output_attention:
                    
                        outputs = self.model(batch_x, batch_x_mark, batch_v, batch_v_mark)[0] 
                    
                    else: 
                        # dec_inp = Batch x Pred_len x Feat  
                        outputs = self.model(batch_x, batch_x_mark, batch_v, batch_y_mark, batch_y)
                
                # print(outputs.shape,batch_y.shape)
                #f_dim = -1 if self.args.features == 'MS' else 0   
 
                outputs = outputs.view(batch_size, pred_len, pred_feature)   ###### <<<<
                batch_y = batch_y.to(self.device)

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

                pbar.set_description("iters: {0}, epoch: {1} | loss: {2:.7f}".format(i, epoch, loss.item())) 
             
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

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './run_testing/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_v, batch_x_mark, batch_y_mark, batch_v_mark) in enumerate(test_loader):
                
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

                #f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs.view(batch_size, pred_len, pred_feature)   ###### <<<<
                batch_y = batch_y.to(self.device)
                
                outputs = outputs.cpu().numpy()
                batch_y = batch_y.cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy()) 
                

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
  
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
  
        
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues) 

        pred_rev = test_data.inverse_transform_y(preds.reshape(-1,1))
        true_rev = test_data.inverse_transform_y(trues.reshape(-1,1))

        mae_rev, mse_rev, _, _, _, rse_rev, _ = metric(pred_rev, true_rev) 
        
        print("On testing dataset ....")
        print('scal: mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        print('revs: mse:{}, mae:{}, rse:{}'.format(mse_rev, mae_rev, rse_rev))
 

        result_dict = {}
        result_dict["inputx"] = inputx
        result_dict["preds"] = preds
        result_dict["trues"] = trues
        result_dict["preds_rev"] = pred_rev
        result_dict["trues_rev"] = true_rev
         

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(os.path.join(folder_path, 'result_dict.npy'), result_dict)        

        stats_dict = {}

        stats_dict["mae"] = mae
        stats_dict["mse"] = mse
        stats_dict["rse"] = rse
        stats_dict["mae_rev"] = mae_rev
        stats_dict["mse_rev"] = mse_rev
        stats_dict["rse_rev"] = rse_rev

        with open(os.path.join(folder_path, 'stats_mae_mse.csv'), 'w') as f:
            for key in stats_dict.keys():
                f.write("%s,%s\n"%(key,stats_dict[key]))
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_v, batch_x_mark, batch_y_mark, batch_v_mark) in enumerate(pred_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_v = batch_v.float()

                batch_size, pred_len, pred_feature = batch_y.shape

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_v_mark = batch_v_mark.float().to(self.device)

                # decoder input
                # dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
 
                if 'Linear' in self.args.model or 'TST' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, batch_v, batch_v_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_v, batch_v_mark)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

        return
