import os
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler


class embed(nn.Module):
    def __init__(self,Input_len, num_id,num_samp,IF_node):
        super(embed, self).__init__()
        self.IF_node = IF_node
        self.num_samp = num_samp
        self.embed_layer = nn.Linear(2*Input_len,Input_len)
        # self.pos_emb = GeoPositionalEmbedding(Input_len)

        # root_path = "./datasets/ChinaWind"
        # node_pos = np.load(os.path.join(root_path, "pos_data.npy"), allow_pickle=True)
        # scaler = StandardScaler()
        # scaler.fit(node_pos)
        # node_pos = scaler.transform(node_pos)
        # self.national_position = torch.from_numpy(node_pos).float()

        self.temporal_emb = TemporalEmbedding(Input_len, False)


        self.node_emb = nn.Parameter(torch.empty(num_id, Input_len))
        nn.init.xavier_uniform_(self.node_emb)

    def forward(self, x, x_mark):

        x = x.unsqueeze(-1)
        batch_size, N, L ,_ = x.shape
        node_emb1 = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(-1)
        # node_emb1 = self.pos_emb(self.national_position.to(x.device)).expand(batch_size,-1,-1).unsqueeze(-1)
        # time_emb = self.temporal_emb(x_mark).repeat(1, N, 1, 1).permute(2,1,3,0)
        # print(time_emb.shape)
        # print(node_emb1.shape)

        # x_1 = embed.down_sampling(x, self.num_samp)
        # if self.IF_node:
        #     x_1 = torch.cat([x_1, embed.down_sampling(node_emb1, self.num_samp), embed.down_sampling(time_emb, self.num_samp)], dim=-1)
        #
        # x_2 = embed.Interval_sample(x, self.num_samp)
        # if self.IF_node:
        #     x_2 = torch.cat([x_2, embed.Interval_sample(node_emb1, self.num_samp),embed.Interval_sample(time_emb, self.num_samp)], dim=-1)

        x_1 = embed.down_sampling(x, self.num_samp)
        if self.IF_node:
            x_1 = torch.cat([x_1, embed.down_sampling(node_emb1, self.num_samp)], dim=-1)

        x_2 = embed.Interval_sample(x, self.num_samp)
        if self.IF_node:
            x_2 = torch.cat([x_2, embed.Interval_sample(node_emb1, self.num_samp)], dim=-1)

        return x_1,x_2

    @staticmethod
    def down_sampling(data,n):
        result = 0.0
        for i in range(n):
            line = data[:,:,i::n,:]
            if i == 0:
                result = line
            else:
                result = torch.cat([result, line], dim=3)
        result = result.transpose(2, 3)
        return result

    @staticmethod
    def Interval_sample(data,n):
        result = 0.0
        data_len = data.shape[2] // n
        for i in range(n):
            line = data[:,:,data_len*i:data_len*(i+1),:]
            if i == 0:
                result = line
            else:
                result = torch.cat([result, line], dim=3)
        result = result.transpose(2, 3)
        return result

class GeoPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(GeoPositionalEmbedding, self).__init__()
        # Compute the national position
        self.location_embedding = nn.Linear(3, d_model)

    def forward(self, x):
        return self.location_embedding(x)

class TemporalEmbedding(nn.Module):

    def __init__(self, time_dim, randomly_init=False):
        super(TemporalEmbedding, self).__init__()

        self.hour_size = 24
        self.day_size = 32
        self.month_size = 13

        embedding = nn.Embedding

        self.hour_embed = embedding(self.hour_size, time_dim)
        self.day_embed = embedding(self.day_size, time_dim)
        self.month_embed = embedding(self.month_size, time_dim)

        nn.init.xavier_uniform_(self.hour_embed.weight)
        nn.init.xavier_uniform_(self.day_embed.weight)
        nn.init.xavier_uniform_(self.month_embed.weight)


    def forward(self, x):
        _, f = x.shape
        #x[:,2] = x[:,2] * 24
        x[:,1] = x[:,1] * 31
        x[:,0] = x[:,0] * 12
        x = x.long()

        #hour_x = self.hour_embed(x[:, 2]) if f > 2 else 0.
        day_x = self.day_embed(x[:, 1]) if f > 1 else 0.
        month_x = self.month_embed(x[:, 0])

        #return hour_x + day_x + month_x
        return day_x + month_x