import torch
import torch.nn as nn
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
import os


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


# class DataEmbedding_inverted_APE(nn.Module):
#     def __init__(self, c_in, d_model, root_path, dropout=0.1):
#         super().__init__()
#         self.value_embedding = nn.Linear(c_in, d_model)
#         node_pos = np.load(os.path.join(root_path, "pos_data.npy"), allow_pickle=True)
#         scaler = StandardScaler()
#         scaler.fit(node_pos)
#         node_pos = scaler.transform(node_pos)
#         self.node_pos = torch.from_numpy(node_pos).float()
#         self.pos_embedding = nn.Linear(3, d_model)
#         self.time_embedding = TemporalEncoding(d_model)

#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, x, x_mark):
#         x = x.permute(0, 2, 1)
#         # x: [Batch Variate Time]
#         if x_mark is None:
#             x = self.value_embedding(x)
#         else:
#             B, N, _ = x.shape
            
#             x_emb = self.value_embedding(x)
#             print(x_emb.shape, x_emb.device)
#             # x_mark in [B, f], time_emb in  [B, N, D]
#             # dimension changes as [B, f] -> [B, D] -> [N, B, D] -> [B, N, D]
#             time_emb = self.time_embedding(x_mark[:, 0, :]).repeat(N, 1, 1).transpose(0, 1)
#             print(time_emb.shape, time_emb.device)
#             # node_pos in [N, 3], node_emb in [B, C, N, D]
#             # dimension changes as [N, 3] -> [N, D] -> [B, N, D]
#             #pos = self.node_pos.to(x.device)
#             node_emb = self.pos_embedding(self.node_pos.to(x.device)).repeat(B, 1, 1)
#             print(node_emb.shape, node_emb.device)
#             emb =  x_emb + time_emb + node_emb
            
#         # x: [Batch Variate d_model]
#         return self.dropout(emb)


import torch
import numpy as np
import torch.nn as nn
import os
from sklearn.preprocessing import StandardScaler


class PositionalEncoding(nn.Module):
    r"""
    Attaches spatial and temporal encoding to the data embedding

    - spatial encoding: projects the location of stations to the latent space with dimension d_model.
    - temporal encoding: embeds the temporal information of data to an embedding with dimension d_model.

    The output of this module is the sum of above three embeddings.
    """

    def __init__(self, seq_len, d_model, num_nodes, root_path, if_rel=False):
        super(PositionalEncoding, self).__init__()

        self.randomly_init = if_rel
        node_pos = np.load(os.path.join(root_path, "pos_data.npy"), allow_pickle=True)
        scaler = StandardScaler()
        scaler.fit(node_pos)
        node_pos = scaler.transform(node_pos)
        self.node_pos = torch.from_numpy(node_pos).float()
        self.time_emb = TemporalEncoding(d_model)
        self.node_emb = SpatialEncoding(d_model, num_nodes, False)
        self.value_emb = nn.Linear(seq_len, d_model)

    def forward(self, x, x_mark):
        # x in [B, C, N, L]
        x = x.transpose(1,2)
        B, N, _ = x.shape

        # x_mark in [B, f], time_emb in  [B, C, N, D]
        # dimension changes as [B, f] -> [B, D] -> [C, N, B, D] -> [B, C, N, D]
        # print(x_mark[:,0,:])
        time_emb = self.time_emb(x_mark[:, 0, :]).repeat(N, 1, 1).transpose(0, 1)
        # node_pos in [N, 3], node_emb in [B, C, N, D]
        # dimension changes as [N, 3] -> [N, D] -> [B, C, N, D]
        pos = self.node_pos.to(x.device)
        node_emb = self.node_emb(pos).repeat(B, 1, 1)
        value_emb = self.value_emb(x)
        emb = value_emb + node_emb + time_emb
        return emb


class SpatialEncoding(nn.Module):

    def __init__(self, node_dim, num_nodes, if_rel=False):
        super(SpatialEncoding, self).__init__()
        self.if_rel = if_rel
        self.num_nodes = num_nodes

        if self.if_rel is False:
            self.Embedding = nn.Linear(3, node_dim, bias=False)

        else:
            self.Embedding = nn.Embedding(num_nodes, node_dim)

    def forward(self, x):
        if self.if_rel is False:
            x = self.Embedding(x)
        else:
            node_id = torch.Tensor(np.arange(self.num_nodes)).to(x.device)
            x = self.Embedding(node_id)
        return x

# testing
# class SpatialEncoding(nn.Module):

#     def __init__(self, node_dim, num_nodes, if_rel=False):
#         super(SpatialEncoding, self).__init__()
#         self.if_rel = if_rel
#         self.num_nodes = num_nodes

#         if self.if_rel is False:
#             self.Embedding = nn.Sequential(
#                 nn.Linear(3, num_nodes),
#                 nn.LayerNorm(num_nodes),
#                 nn.ReLU(),
#                 nn.Linear(num_nodes, num_nodes)
#             )

#         else:
#             self.Embedding = nn.Embedding(num_nodes, node_dim)

#     def forward(self, x):
#         if self.if_rel is False:
#             x = self.Embedding(x)
#         else:
#             node_id = torch.Tensor(np.arange(self.num_nodes)).to(x.device)
#             x = self.Embedding(node_id)
#         return x


class TemporalEncoding(nn.Module):

    def __init__(self, time_dim):
        super(TemporalEncoding, self).__init__()

        self.hour_size = 24
        # self.weekday_size = 7
        self.day_size = 32
        self.month_size = 13

        self.hour_embed = nn.Embedding(self.hour_size, time_dim)
        # self.weekday_embed = nn.Embedding(self.weekday_size, time_dim)
        self.day_embed = nn.Embedding(self.day_size, time_dim)
        self.month_embed = nn.Embedding(self.month_size, time_dim)

    def forward(self, x):
        _, f = x.shape
        x = x + 0.5
        hour_x = self.hour_embed((x[..., 2] * self.hour_size).long()) if f > 2 else 0.
        #weekday_x = self.weekday_embed((x[..., 2] * self.weekday_size).long()) if f > 2 else 0.
        day_x = self.day_embed((x[..., 1] * self.day_size).long()) if f > 1 else 0.
        month_x = self.month_embed((x[..., 0] * self.month_size).long())

        #return x
        return hour_x + day_x + month_x