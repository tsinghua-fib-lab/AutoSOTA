import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from positional_encodings.torch_encodings import PositionalEncoding2D
from utils.magnitude_max_pooling import magnitude_max_pooling_1d

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class ConvDataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, seq_len, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(ConvDataEmbedding_inverted, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=c_in*2, kernel_size=3, stride=1, dilation=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=c_in*2, out_channels=c_in, kernel_size=5, stride=1, dilation=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels=c_in, out_channels=c_in//2, kernel_size=7, stride=1, dilation=1, padding='same')
        self.conv4 = nn.Conv2d(in_channels=c_in//2, out_channels=c_in//2, kernel_size=9, stride=1, dilation=1, padding='same')
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.ln = nn.LayerNorm(seq_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            enc_out = x.unsqueeze(1).permute(0, 2, 3, 1)
            enc_out = F.silu(self.conv1(enc_out))
            enc_out = F.silu(self.conv2(enc_out))
            enc_out = F.silu(self.conv3(enc_out))
            #enc_out = F.silu(self.conv4(enc_out))
            enc_out = enc_out.squeeze(3)
            enc_out = self.ln(enc_out)       
            x = torch.cat((x, enc_out), dim=1)
            x = self.value_embedding(x)
            
            
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            enc_out = x.unsqueeze(1).permute(0, 2, 3, 1)
            enc_out = F.silu(self.conv1(enc_out))
            enc_out = F.silu(self.conv2(enc_out))
            enc_out = F.silu(self.conv3(enc_out))
            enc_out = enc_out.squeeze(3)
            x = torch.cat((x, enc_out), dim=1)
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
            
        # x: [Batch Variate d_model]
        
        return self.dropout(x)
    



class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)






class DataEmbedding_FeaturePatching(nn.Module):
    def __init__(self, seq_len,  patch_size,  embed_dim = 512, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_FeaturePatching, self).__init__()
        self.seq_len = seq_len 
        self.patch_size = patch_size
        self.n_of_patches = (seq_len - patch_size)//(patch_size//2) + 1
        self.inner_dim = patch_size * 10
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv1d(1, 3, kernel_size=5)
        self.conv2 = nn.Conv1d(1, 3, kernel_size=9)
        self.conv3 = nn.Conv1d(1, 3, kernel_size=15)
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        self.fc1 = nn.Linear(self.inner_dim, embed_dim*4)
        self.fc2 = nn.Linear(embed_dim*4, embed_dim)
        self.pe  = PositionalEncoding2D(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.sigm = nn.GELU()

    def forward(self, x, x_mark):
        B, L, N = x.shape
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]

        if x_mark is not None:
            N += x_mark.shape[2]
            x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)

        x = x.reshape(-1, 1, L)
        x_1 = F.pad(x, (4, 0), mode = 'replicate')
        x_1 = self.conv1(x_1)
        x_2 = F.pad(x, (8, 0), mode = 'replicate')
        x_2 = self.conv2(x_2)
        x_3 = F.pad(x, (14, 0), mode = 'replicate')
        x_3 = self.conv3(x_3)
        x_1 = F.pad(x_1, (2, 0), mode = 'constant', value = 0)
        x_2 = F.pad(x_2, (4, 0), mode = 'constant', value = 0)
        x_3 = F.pad(x_3, (6, 0), mode = 'constant', value = 0)

        x_1 = magnitude_max_pooling_1d(x_1, 3, 1)
        x_2 = magnitude_max_pooling_1d(x_2, 5, 1)
        x_3 = magnitude_max_pooling_1d(x_3, 7, 1)

        

        x_1 = x_1.reshape(B, N, 3, L)
        x_2 = x_2.reshape(B, N, 3, L)
        x_3 = x_3.reshape(B, N, 3, L)
        x = x.reshape(B, N, 1, L)
        

        x_1 = x_1.unfold(3, self.patch_size, self.patch_size//2)
        x_2 = x_2.unfold(3, self.patch_size, self.patch_size//2)
        x_3 = x_3.unfold(3, self.patch_size, self.patch_size//2)
        x = x.unfold(3, self.patch_size, self.patch_size//2)


        x_1 = x_1.permute(0, 1, 3, 2, 4)
        x_2 = x_2.permute(0, 1, 3, 2, 4)
        x_3 = x_3.permute(0, 1, 3, 2, 4)
        x = x.permute(0, 1, 3, 2, 4)


        x = torch.cat([x, x_1, x_2, x_3], dim = 3)


        x = x.reshape(B, N, self.n_of_patches, -1)
        x = self.gelu1(self.fc1(x))
        x = self.fc2(x)
        x = self.pe(x) + x #apply 2D positional encodings

        x = x.reshape(B, -1, self.embed_dim)

        return self.dropout(x)


#For ablation
class DataEmbedding_PatchingProjection(nn.Module):
    def __init__(self, seq_len,  patch_size,  embed_dim = 512, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_PatchingProjection, self).__init__()
        self.seq_len = seq_len 
        self.patch_size = patch_size
        self.n_of_patches = (seq_len - patch_size)//(patch_size//2) + 1
        self.inner_dim = patch_size 
        self.embed_dim = embed_dim

        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        self.fc1 = nn.Linear(self.inner_dim, embed_dim)
        self.pe  = PositionalEncoding2D(embed_dim)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x, x_mark):
        B, L, N = x.shape
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]

        if x_mark is not None:
            N += x_mark.shape[2]
            x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)

        #sample random x with these dimensions
        x = x.reshape(B, N, 1, L)
        
        x = x.unfold(3, self.patch_size, self.patch_size//2)

        x = x.permute(0, 1, 3, 2, 4)


        x = x.reshape(B, N, self.n_of_patches, -1)
        x = self.fc1(x)
        x = self.pe(x) + x

        x = x.reshape(B, -1, self.embed_dim)

        return self.dropout(x)