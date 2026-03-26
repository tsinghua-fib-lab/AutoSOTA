import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from Models.autoregressive_diffusion.model_utils import LearnablePositionalEncoding, Conv_MLP,\
                                                       AdaLayerNorm, Transpose, GELU2, series_decomp, RevIN, SinusoidalPosEmb, extract                                                       
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

timesteps = 96

class Linear(nn.Module):
    def __init__(
        self,
        n_feat,
        n_channel,
        w_grad=True,
        **kwargs
    ):
        super().__init__()
        self.linear = nn.Linear(n_channel, n_channel)
        self.betas = linear_beta_schedule(96)
        self.betas_dev = cosine_beta_schedule(96)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_dev = 1. - self.betas_dev
        self.w = torch.nn.Parameter(torch.FloatTensor(self.alphas_cumprod.numpy()), requires_grad=w_grad)
        self.w_dev = torch.nn.Parameter(torch.FloatTensor(self.alphas_dev.numpy()), requires_grad=False)

    def forward(self, input_, t, training=True):
        noise = torch.randn_like(input_)
        if not training:
            noise=0
        input_+= self.w_dev[t[0]]*noise
        x_tmp = self.linear(input_.permute(0,2,1)).permute(0,2,1)
        alpha = self.w[t[0]]
        output = (alpha*input_ + (1-2*alpha)*x_tmp) / (1-1*alpha)**(1/2)
        #if not training:
            #print('alpha:',alpha)
            #print('para:',1-1*alpha)
            #print('dis:',x_tmp.mean())
            #print('loss:',((1-1*alpha)*x_tmp).mean())

        output = output.to(torch.float32)

        return output

