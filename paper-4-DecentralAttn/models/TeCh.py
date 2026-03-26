import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder,EncoderLayer
from layers.Augmentation import get_augmentation
import math

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.v_layer= configs.v_layer
        self.t_layer= configs.t_layer
        self.channel_encoder = nn.Sequential(Aug_Channel_Embedding(configs),Encoder([EncoderLayer(configs) \
                                            for _ in range(configs.v_layer)])) if configs.v_layer>0 else nn.Identity()
        self.temporal_encoder = nn.Sequential(Aug_Temporal_Embedding(configs),Encoder([EncoderLayer(configs) \
                                            for _ in range(configs.t_layer)])) if configs.t_layer>0 else nn.Identity()
        self.projector = nn.Linear(configs.d_model,configs.num_class)

    def forward(self, x_enc):
        B,T,N= x_enc.shape
        channel= self.channel_encoder(x_enc).mean(1) if self.v_layer>0 else 0
        temporal= self.temporal_encoder(x_enc).mean(1) if self.t_layer>0 else 0
        
        # B N D -> B D -> B C 
        logits = self.projector(channel+temporal)
        return logits 


class Aug_Channel_Embedding(nn.Module):
    def __init__(self,configs):
        super().__init__()
        aug_idxs = configs.augmentations.split(",")
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in aug_idxs]
        )
        self.Channel_Embedding =nn.Linear(configs.seq_len, configs.d_model)
        self.pos_emb = PositionalEmbedding(d_model=configs.seq_len)

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        x = x.transpose(1,2)  # (batch_size, enc_in, seq_len)
        aug_idx = random.randint(0, len(self.augmentation) - 1)
        x_aug = self.augmentation[aug_idx](x)
        x_aug=x_aug+self.pos_emb(x_aug)
        return self.Channel_Embedding(x_aug)
    
class Aug_Temporal_Embedding(nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.patch_len = configs.patch_len
        aug_idxs = configs.augmentations.split(",")
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in aug_idxs]
        )
        self.Temporal_Embedding =CrossChannelPatching(configs) if self.patch_len >1 else nn.Linear(configs.enc_in, configs.d_model)
        self.pos_emb = PositionalEmbedding(d_model=configs.seq_len)

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        x = x.transpose(1,2)  # (batch_size, enc_in, seq_len)
        aug_idx = random.randint(0, len(self.augmentation) - 1)
        x_aug = self.augmentation[aug_idx](x)
        x_aug=x_aug+self.pos_emb(x_aug)
        if self.patch_len ==1: x_aug=x_aug.transpose(1,2)
        return self.Temporal_Embedding(x_aug)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = True

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
    
class CrossChannelPatching(nn.Module):
    def __init__(self, configs):
        super().__init__()
        patch_len=configs.patch_len
        stride = configs.patch_len
        self.tokenConv = nn.Conv2d(
            in_channels=1,
            out_channels=configs.d_model,
            kernel_size=(configs.enc_in, patch_len),
            stride=(1, stride),
            padding=0,
            padding_mode="circular",
            bias=False,
        )
        self.padding = nn.ReplicationPad1d((0, stride))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x=self.padding(x).unsqueeze(1)
        x = self.tokenConv(x).squeeze(2).transpose(1, 2)
        return x