import torch
import torch.nn as nn 
from tqdm import tqdm  

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        
    def forward(self, x):
        B, L, D = x.shape
        Ihat1 = x[:, -1, 0] # Ihat1 shape: (B, seq_len, feat)
        return Ihat1