import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import torch
# class Mamba_block(nn.Module):
#     def __init__(self,state_dim=9,actions_dim=5,d_state=32,d_conv=4,expand=2,num_hidden_mlp=32):
#         super().__init__()
#         self.mamba = Mamba(
#             d_model=state_dim + actions_dim,
#             d_state=d_state,
#             d_conv=d_conv,
#             expand=expand,
#         )
#         self.norm1 = nn.BatchNorm1d(state_dim + actions_dim,affine=False)
#         self.norm2 = nn.BatchNorm1d(14,affine=False)
#         self.ff = nn.Linear(state_dim + actions_dim, )

class DAC_block(nn.Module):
    def __init__(self,state_dim=9,actions_dim=5,action_bins=16,d_state=32,d_conv=4,expand=2,num_hidden_mlp=32,mamba_num = 1):
        super().__init__()
        self.mamba_blocks = nn.Sequential(*(Mamba(
            d_model=state_dim + actions_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        ) for _ in range(mamba_num)))
        
        self.ff1 = nn.Linear(state_dim + actions_dim, num_hidden_mlp)
        self.ff2 = nn.Linear(num_hidden_mlp, action_bins)
        # self.normalization = nn.BatchNorm1d(14,affine=False)

    def forward(self, x): # x.shape: bs * ls * (9 + 5)
        bs, ls, fea_dim = x.shape
        x = x.view(bs,-1,fea_dim)
        # print('before mamba: ', x.min(),x.max(),x.mean(),x.std())
        x = self.mamba_blocks(x)
        # x = self.normalization(x.view(bs,fea_dim,ls)).view(bs, ls, fea_dim)
        # print('after mamba: ', x.min(),x.max(),x.mean(),x.std())
        x = F.leaky_relu(self.ff1(x))
        x = self.ff2(x)
        # print('after linear: ', x.min(),x.max(),x.mean(),x.std())
        # Removed per-sequence min-max normalization (iter-1: raw Q-values improve discrimination)
        return x # out.shape: bs * ls * 16
    
