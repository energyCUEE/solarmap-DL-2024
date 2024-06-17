import torch
import torch.nn as nn
from models.RLSTM import Model as rLSTM
from models.Transformer import Model as Transformer
from models.Informer import Model as Informer

import torch.nn.functional as F
import pdb


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.m2_name = configs.m2_name
        # self.is_m3 = configs.is_m3

        if self.m2_name == "RLSTM":
            self.M2 = rLSTM(configs)
        elif self.m2_name == "Transformer":
            self.M2 = Transformer(configs)
        elif self.m2_name == "Informer":
            self.M2 = Informer(configs)

        self.linear = nn.Linear(2, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc shape : [batch_size, seq_len, features]
        B = x_enc.size(0)
        Ihat1 = x_enc[:, :, 0]  # Ihat1 shape: (B, seq_len)
        Feature_inp = x_enc[:, :, 1:]  # Feature_inp shape: (B, seq_len, Feat-1)
        
        if self.m2_name == "RLSTM":
            ehat2 = self.M2(Feature_inp)  # ehat2 shape: (B, seq_len)
        else:
            ehat2 = self.M2(Feature_inp, x_mark_enc, x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask)  # ehat2 shape: (B, seq_len, d_model)

        Ihat1 = Ihat1[:, -1].view(B, 1)  # Ihat1 shape: (B, 1)

        if ehat2.dim() == 3:
            # Ensure ehat2 has the correct shape for concatenation
            ehat2 = ehat2[:, -1, :].view(B, -1)  # Assuming you want the last time step's output
        else:
            # ehat2 is already 2D
            ehat2 = ehat2.view(B, -1)

        inp_linear = torch.cat([Ihat1, ehat2], dim=1)  # Concatenate along the second dimension: (B, 1 + d_model)

        yhat = self.linear(inp_linear)  # Apply linear layer: (B, 1)

        return yhat
