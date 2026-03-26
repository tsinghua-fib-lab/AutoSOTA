import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_FeaturePatching
import numpy as np


class Model(nn.Module):


    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_FeaturePatching(configs.seq_len, configs.patch_size, embed_dim = configs.embed_dim, embed_type='fixed', freq='h', dropout=0.1)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.embed_dim, configs.n_heads),
                    configs.embed_dim,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.embed_dim),
        )

        self.head = nn.Linear(configs.embed_dim, configs.patch_size, bias=True)
        self.projector1 = nn.Linear(((configs.seq_len-configs.patch_size)//(configs.patch_size//2)+1)*(configs.embed_dim), configs.d_model, bias=True)
        self.non_linear = nn.GELU()
        self.projector_mid = nn.Linear(configs.d_model, configs.d_model, bias=True)
        self.non_linear_mid = nn.GELU()
        self.projector2 = nn.Linear(configs.d_model, configs.pred_len, bias=True)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-4)
            x_enc /= stdev


        B, _, N = x_enc.shape # B L N
        if x_mark_enc is not None:
            N_ = N +  x_mark_enc.shape[2]
        else :
            N_ = N


        enc_out = self.enc_embedding(x_enc, x_mark_enc) 
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out.reshape(B, N_, -1)

        dec_out = self.projector1(enc_out) 
        dec_out = self.non_linear(dec_out)
        dec_out = self.projector2(dec_out).permute(0, 2, 1)[:, :, :N]


        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
    
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]