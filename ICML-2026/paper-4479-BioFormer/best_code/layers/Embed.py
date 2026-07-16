import math
import random
import torch
import torch.nn as nn
from layers.Augmentation import get_augmentation



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
    
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PyramidConvEmbedding(nn.Module):
    """
    Pyramid conv-based patch embedding with downsampling.
    Input : (B, L, C)
    Output: list of [(B, L/2, D), (B, L/4, D), ...] 
    """
    def __init__(
        self,
        c_in,
        d_model,
        augmentation=["none"],
        downsample_factors=[2, 4, 8],     
    ):
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.augmentation = nn.ModuleList([get_augmentation(aug) for aug in augmentation])
        random.seed(42)
        self.stages = nn.ModuleList()
        for factor in downsample_factors:
            stages = []
            num_down = int(math.log2(factor))
            for _ in range(num_down):
                stages.append(
                    nn.Sequential(
                        nn.Conv1d(
                            d_model, d_model,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                         nn.BatchNorm1d(d_model),
                         nn.GELU(),
                    )
                )
            self.stages.append(nn.Sequential(*stages))

            
    def forward(self, x, x_mark=None):
        emb = self.value_embedding(x) + self.position_embedding(x)
        feat = emb.transpose(1, 2)                                          #   (B, D, L)
        out_list = []
        
        for stage in self.stages:
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            feat_aug = self.augmentation[aug_idx](feat.clone())
            f = stage(feat_aug)                                             # (B, D, L_down)
            f = f.transpose(1, 2)                                           # (B, L_down, D)
            out_list.append(f)

        return out_list 
