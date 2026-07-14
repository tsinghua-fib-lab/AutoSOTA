import torch.nn.functional as F
import torch.nn as nn
from utils.knn_interpolate import knn_interpolate
import torch


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, out_size, ib=False):
        super().__init__()
        self.lin1=nn.Linear(input_size, hidden_size)
        self.lin2=nn.Linear(hidden_size, hidden_size)
        self.lin3=nn.Linear(hidden_size,out_size)

        # track_running_stats=False gives per-sample normalization
        # (instance-norm behavior), suitable for small-data GNN training
        self.norm1=nn.BatchNorm1d(input_size, track_running_stats=False)
        self.norm2=nn.BatchNorm1d(hidden_size, track_running_stats=False)
        self.norm3=nn.BatchNorm1d(hidden_size, track_running_stats=False)

    def forward(self, x):
        x=self.norm1(x)
        x=self.lin1(x)
        x=F.relu(x)

        x=self.norm2(x)
        x=self.lin2(x)
        x=F.relu(x)

        x=self.norm3(x)
        x=self.lin3(x)

        return x

class Decoder_F(torch.nn.Module):
    def __init__(self, hidden_size, output_size, residual):
        super().__init__()
        self.decoder=MLP(hidden_size, hidden_size,output_size)
        self.residual=residual
    def forward(self, emb, l_y, l_pos, h_pos):
        x=emb
        x=self.decoder(x)
        if self.residual:
            return x+knn_interpolate(l_y,l_pos,h_pos)
        else:
            return x


class Decoder_G(torch.nn.Module):
    def __init__(self, hidden_size, output_size, residual):
        super().__init__()
        self.decoder=MLP(hidden_size, hidden_size,output_size)
        self.residual=residual
    def forward(self, emb1, l_y1, l_pos1, h_pos1, emb2, l_y2, l_pos2, h_pos2):
        x1=emb1
        x2=emb2
        x=self.decoder(x1-knn_interpolate(x2,h_pos2,h_pos1))
        if self.residual:
            return x+knn_interpolate(l_y1-knn_interpolate(l_y2,l_pos2,l_pos1),l_pos1,h_pos1)
        else:
            return x
