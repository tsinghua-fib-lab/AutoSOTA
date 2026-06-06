import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from model.TimePFN import Model as TimePFN_Model

class Model(nn.Module):


    def __init__(self, configs):
        super(Model, self).__init__()
        #deep copy the configs 
        self.configs = configs
        self.copy_configs = copy.deepcopy(configs)
        self.copy_configs.pred_len = 96
        self.copy_configs.seq_len =  96

        self.model = TimePFN_Model(self.copy_configs)

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.seq_len < self.copy_configs.seq_len:
            mean_val = torch.mean(x_enc, dim=1, keepdim=True).repeat(1, self.copy_configs.seq_len - self.seq_len, 1)
            x_enc = torch.cat((mean_val, x_enc), dim=1)
            if x_mark_enc is not None:
                x_mark_enc = F.pad(x_mark_enc, (0, 0, self.copy_configs.seq_len - self.seq_len, 0), "constant", 0)
        else:
            x_enc = x_enc[:, -self.copy_configs.seq_len:, :]
            if x_mark_enc is not None:
                x_mark_enc = x_mark_enc[:, -self.copy_configs.seq_len:, :]

        
        dec_out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        return dec_out[:, :self.pred_len, :]  # [B, L, D]
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)