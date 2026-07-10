import torch
import torch.nn as nn

from layers.attention_layers import StandardAttention, PrimeFilterAttention, DifferentialAttention
from layers.ffn_layers import SwiGLU

from prime import *

# iTransformer implementation
class Transformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.configs = configs
        self.d_input = configs.d_input
        self.d_model = configs.d_model
        self.num_layers = configs.num_layers
        self.num_heads = configs.num_heads
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.dropout = configs.dropout
        self.use_norm = configs.use_norm
        self.use_prime = configs.use_prime
        self.frequency_bins = configs.frequency_bins
        self.filter_type = configs.filter_type
        
        # self.encoder = nn.Linear(self.bands_to_keep * 2, self.d_model)
        # self.encoder = nn.Linear((self.seq_len // 2) * 2, self.d_model)
        self.encoder = nn.Linear(self.seq_len, self.d_model)
        self.decoder = nn.Linear(self.d_model, self.pred_len)
        
        self.attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'layer_norm': nn.LayerNorm(self.d_model),
                'attn': PrimeFilterAttention(self.d_model, num_heads=self.num_heads, dropout=self.dropout) if self.use_prime else StandardAttention(self.d_model, num_heads=self.num_heads, dropout=self.dropout),
                'ffn': SwiGLU(self.d_model),
                'layer_norm_ffn': nn.LayerNorm(self.d_model),
                'dropout': nn.Dropout(self.dropout)
            }) for i in range(self.num_layers)
        ])
        
        self.bands_to_keep = int((self.seq_len // 2 + 1) * self.frequency_bins) - 1
        
        self.apply_filter = None
        if self.filter_type == 1:
            self.apply_filter = PrimeFilters(configs, bands_to_keep=self.bands_to_keep)
        elif self.filter_type == 2:
            self.apply_filter = FullFrequencyLeadLagFilters(configs, bands_to_keep=self.bands_to_keep)
        
    def forward(self, x, x_mark=None, save_attn=False):
        """
        args:
            x: (B, L, N) L is for lookback window
            x_mark: (B, L, C) C is for covariates
        returns:
            output: (B, F, N) F is for forecast horizon
        """
        B, L, N = x.shape
        
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        
        x = x.permute(0, 2, 1)
        
        prime_filters = None
        if self.use_prime:
            prime_filters = self.apply_filter(x)
        
        x = self.encoder(x)
        
        attn = None
        for i, layer in enumerate(self.attention_layers):
            residual = x
            x = layer['layer_norm'](x)
            x, attn = layer['attn'](
                q=x,
                k=x,
                v=x,
                prime_filters=prime_filters,
            )
            x = residual + layer['dropout'](x)
            
            residual = x
            x = layer['layer_norm_ffn'](x)
            x = layer['ffn'](x)
            x = residual + layer['dropout'](x)
            
        x = self.decoder(x).permute(0, 2, 1)
        
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return x
        