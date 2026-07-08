from typing import Callable, Union

from tabdiff.modules.transformer import Reconstructor, Tokenizer, Transformer
import torch
import torch.nn as nn
import torch.optim

ModuleType = Union[str, Callable[..., nn.Module]]

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t = 512, use_mlp=True):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        ) if use_mlp else nn.Linear(dim_t, d_in)

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
        
        self.use_mlp = use_mlp
    
    def forward(self, x, timesteps):
        emb = self.map_noise(timesteps)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        emb = self.time_embed(emb)
    
        x = self.proj(x) + emb
        return self.mlp(x)
    
class UniModMLP(nn.Module):
    """
        Input:
            x_num: [bs, d_numerical]
            x_cat: [bs, len(categories)]
        Output:
            x_num_pred: [bs, d_numerical], the predicted mean for numerical data
            x_cat_pred: [bs, sum(categories)], the predicted UNORMALIZED logits for categorical data
    """
    def __init__(
            self, d_numerical, categories, num_layers, d_token,
            n_head = 1, factor = 4, bias = True, dim_t=512, use_mlp=True, **kwargs
        ):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = categories

        self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias = bias)
        self.encoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        d_in = d_token * (d_numerical + len(categories))
        self.mlp = MLPDiffusion(d_in, dim_t=dim_t, use_mlp=use_mlp)
        self.decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.detokenizer = Reconstructor(d_numerical, categories, d_token)
        
        self.model = nn.ModuleList([self.tokenizer, self.encoder, self.mlp, self.decoder, self.detokenizer])

    def forward(self, x_num, x_cat, timesteps):
        e = self.tokenizer(x_num, x_cat)
        decoder_input = e[:, 1:, :]        # ignore the first CLS token. 
        y = self.encoder(decoder_input)
        pred_y = self.mlp(y.reshape(y.shape[0], -1), timesteps)
        pred_e = self.decoder(pred_y.reshape(*y.shape))
        x_num_pred, x_cat_pred = self.detokenizer(pred_e)
        x_cat_pred = torch.cat(x_cat_pred, dim=-1) if len(x_cat_pred)>0 else torch.zeros_like(x_cat).to(x_num_pred.dtype)

        return x_num_pred, x_cat_pred


class Precond(nn.Module):
    def __init__(self,
        denoise_fn,
        sigma_data = 0.5,              # Expected standard deviation of the training data.
        net_conditioning = "sigma",
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.net_conditioning = net_conditioning
        self.denoise_fn_F = denoise_fn

    def forward(self, x_num, x_cat, t, sigma):

        x_num = x_num.to(torch.float32)

        sigma = sigma.to(torch.float32)
        assert sigma.ndim == 2
        if sigma.dim() > 1: # if learnable column-wise noise schedule, sigma conditioning is set to the defaults schedule of rho=7
            sigma_cond = (0.002 ** (1/7) + t * (80 ** (1/7) - 0.002 ** (1/7))).pow(7)
        else:
            sigma_cond = sigma 
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma_cond.log() / 4

        x_in = c_in * x_num
        if self.net_conditioning == "sigma":
            F_x, x_cat_pred = self.denoise_fn_F(x_in, x_cat, c_noise.flatten())
        elif self.net_conditioning == "t":
            F_x, x_cat_pred = self.denoise_fn_F(x_in, x_cat, t)

        assert F_x.dtype == dtype
        D_x = c_skip * x_num + c_out * F_x.to(torch.float32)
        
        return D_x, x_cat_pred
    

class Model(nn.Module):
    def __init__(
            self, denoise_fn,
            sigma_data=0.5, 
            precond=False, 
            net_conditioning="sigma",
            **kwargs
        ):
        super().__init__()
        self.precond = precond
        if precond:
            self.denoise_fn_D = Precond(
                denoise_fn,
                sigma_data=sigma_data,
                net_conditioning=net_conditioning
            )
        else:
            self.denoise_fn_D = denoise_fn

    def forward(self, x_num, x_cat, t, sigma=None):
        if self.precond:
            return self.denoise_fn_D(x_num, x_cat, t, sigma)
        else:
            return self.denoise_fn_D(x_num, x_cat, t)


