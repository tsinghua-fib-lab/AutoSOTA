import torch
import torch.nn as nn

class AirEmbedding(nn.Module):
    '''
    Embed catagorical variables.
    '''
    def __init__(self):
        super(AirEmbedding, self).__init__()
        self.embed_wdir=nn.Embedding(11,3)
        self.embed_weather=nn.Embedding(18,4)
        self.embed_day=nn.Embedding(24,3) # a typo here but doesn't affect the results. this layer is actually for embedding 24 hours.
        self.embed_hour=nn.Embedding(7,5) # a typo here but doesn't affect the results. this layer is actually for embedding 7 days.

    def forward(self, x):
        x_wdir = self.embed_wdir(x[...,0])
        x_weather = self.embed_weather(x[...,1])
        x_day = self.embed_day(x[...,2])
        x_hour = self.embed_hour(x[...,3])
        out=torch.cat((x_wdir,x_weather,x_day,x_hour),-1)
        return out