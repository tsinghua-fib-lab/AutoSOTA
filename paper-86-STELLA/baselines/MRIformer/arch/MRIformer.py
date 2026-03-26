import torch
from torch import nn, optim
import torch.nn.functional as F

from .MRI_block import MRI_block_att
from .revin import RevIN


class MRIformer(nn.Module):
    def __init__(self, Input_len, out_len, num_id, num_hi, muti_head,num_layer,dropout,IF_Chanel):
        """
        Input_len: History length
        out_len: Future length
        num_id: number of variable
        num_hi: Number of hidden
        muti_head: number of muti_head attention
        num_layer: number of MRI block
        dropout: dropout rate
        IF_Chanel: Whether to adopt the channel independent modeling strategy
        """
        super(MRIformer, self).__init__()

        self.RevIN = RevIN(num_id)
        # root_path = "./datasets/global_wind"
        # self.embedding = DataEmbedding(Input_len, Input_len, root_path, False)
        ###encorder
        self.MRI_block_1 = MRI_block_att(Input_len, num_id, num_hi, muti_head, dropout,IF_Chanel)
        self.laynorm_1 = nn.LayerNorm([num_id,Input_len])

        ###decorder
        self.num_layer = num_layer
        self.output = nn.Conv1d(in_channels = Input_len, out_channels=out_len, kernel_size=1)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        # Input [B,H,N,C]: B is batch size. N is the number of variables. H is the history length
        # Output [B,L,N,1]: B is batch size. N is the number of variables. L is the future length

        x = history_data[:, :, :, 0]
        #x_mark = history_data[:,0,0,1:]
        x = self.RevIN(x, 'norm').transpose(-2, -1) # [B, N, L]

        #x = self.embedding(x,x_mark) # [B, N, L]

        for i in range(self.num_layer):
            x = x + self.MRI_block_1(x)
            x = self.laynorm_1(x)

        x = self.output(x.transpose(-2, -1))
        x = self.RevIN(x, 'denorm')
        x = x.unsqueeze(-1)
        return x