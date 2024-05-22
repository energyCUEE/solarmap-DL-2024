import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import  pdb

class Model(nn.Module):
    def __init__(self, configs):
        '''
        This function is used to initialize the model
        params
        num_features: number of features (4)
        hidden_units: number of hidden units (256)
        forecast_steps: number of forecast steps (96)
        '''
        super().__init__()
        self.num_features = configs.enc_in
        self.seq_len      = configs.seq_len
        self.hidden_units = configs.d_model
        self.num_layers   = configs.e_layers
        self.forecast_steps = configs.pred_len  # Number of forecast steps
        self.num_features_out = configs.enc_in  
        
        # Initial LSTM layer (multi-layer)
        self.lstm = nn.LSTM(
            input_size  = self.num_features, # determine the number of feature
            hidden_size = self.hidden_units, # size of embedded metrics output
            batch_first = True,
            num_layers  = self.num_layers # layer of LSTM unit of each iteration of sequence (usually set between 1-2)
        )


        # Output layer for multi-step forecasting  
        
        self.use_Iclr    = configs.use_Iclr
        self.activation   = nn.ReLU()
        self.dropout      = nn.Dropout(p=configs.dropout, inplace=False)
        self.fc           = nn.Linear(in_features=self.hidden_units, out_features=self.forecast_steps)

        if self.use_Iclr:
            self.input_dropout = nn.Dropout(p=configs.input_dropout, inplace=False)
            self.fc2           = nn.Linear(self.seq_len, 1)
        
    def forward(self, x):
        '''
            This function is used to define the forward pass for the model
            param x: input data
            return: output of the model
        '''
        # Get the batch size  
        batch_size = x.shape[0] 

        if self.use_Iclr:  
            Iclr_dp = self.input_dropout(x[:,:, 0].view(-1,self.seq_len,1)).permute(0,2,1)  
            Iclr    = self.fc2(Iclr_dp).reshape(-1,1) 
            #Iclr    = self.fc2(Iclr)


        # X needs to be in the shape of B x L x F 
 
        # Initialize the hidden and cell state of the LSTM layer 
        # h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(self.device).requires_grad_()
        # c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(self.device).requires_grad_()
        
 
        # recurrent data embedding by LSTM layer 
        output, (hn, cn) = self.lstm(x) # , (h0, c0)
 
        # Output sequence for multi-step forecasting  
        output = self.activation(output)
 
        # Dropout layer for regularization
        output = self.dropout(output)

        output = self.fc(output[:, -1, :])
        
        if self.use_Iclr:
            output = output*Iclr #  
        
        return output
    


def train_model(data_loader, model, loss_function, optimizer, loss_result,  device):
    num_batches = len(data_loader)
    total_loss = 0
    # Set the model in training mode
    model.train()

    for X, y in data_loader:
        X = X.to(device)
        y = y.to(device)
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    loss_result['train_loss'].append(avg_loss)
    print(f"Train loss: {avg_loss}")


def test_model(data_loader, model, loss_function, loss_result, pred_len, device):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y, _, _ in data_loader:
            X = X.float().to(device)
            y = y.float().to(device).view(-1,pred_len)
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")
    loss_result['val_loss'].append(avg_loss)
    return loss_result
