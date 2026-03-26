import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.nets.components.itransformer.Transformer_EncDec import Encoder, EncoderLayer
from src.models.nets.components.itransformer.SelfAttention_Family import FullAttention, AttentionLayer
from src.models.nets.components.itransformer.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, input_size, output_size, dim, d_ff, dropout, num_layers, num_heads, with_revin, if_snr, **kwargs):
        super(Model, self).__init__()
        self.seq_len = input_size
        self.pred_len = output_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_attention = False
        self.use_norm = with_revin
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(input_size, dim, 'timeF', 'h',
                                                    0.1, if_snr)
        self.class_strategy = 'projection'
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=dropout,
                                    output_attention=self.output_attention), dim, num_heads),
                    dim,
                    d_ff,
                    dropout=dropout,
                    activation='gelu'
                ) for l in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(dim)
        )
        from src.models.modules.snrlinear import SNLinear
        if kwargs.get('if_snr_last', False):
            self.projector = nn.Linear(dim, output_size, bias=True)
        else:
            self.projector = SNLinear(dim, output_size, bias=True)

    def predict(self, x_enc, *args, **kwargs):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(-1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, N, _ = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        
        # Embedding
        x_enc = x_enc.permute(0, 2, 1)
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, None) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out)[:,:N, :] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * stdev
            dec_out = dec_out + means

        return dec_out

    def infer(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mode='eval', **kwargs):
        return self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec, **kwargs)


    def forward(self, x, y, x_mark, y_mark, mask=None, mode='trian', **kwargs):
        result = {}
        y_hat = self.predict(x)
        result['y_hat'] = y_hat
        result['loss_tar'] = F.mse_loss(y_hat, y)
        result['loss_total'] = result['loss_tar']
        return result