from torch import nn
from typing import Tuple, Literal
import torch

class MultiLabelEncoder(nn.Module):
    def __init__(self, num_class_per_label: Tuple[int, int], d_latent,interaction: Literal['cat','sum']) -> None:
        super().__init__()
        if interaction not in ['cat','sum']:
            raise ValueError(f"interaction must be either 'cat' or 'sum' not {interaction}")
        if interaction == 'cat':
            if d_latent % len(num_class_per_label) != 0:
                raise ValueError(f"d_latent {d_latent} must be divisible by len(num_class_per_label) {len(num_class_per_label)}")
            d_latent = d_latent//len(num_class_per_label)
        self.emb = nn.ModuleList([nn.Embedding(num_classes+1,d_latent) for num_classes in num_class_per_label])
        self.interaction = interaction

    def forward(self,y):
        if self.interaction == 'cat':
            y = torch.cat([emb(y[:,i]) for i,emb in enumerate(self.emb)],dim=1)
        elif self.interaction == 'sum':
            y = sum([emb(y[:,i]) for i,emb in enumerate(self.emb)])
        return y