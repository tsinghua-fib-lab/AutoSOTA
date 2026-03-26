'''
* @author: EmpyreanMoon
*
* @create: 2025-01-02 14:42
*
* @description: The auxiliary network helps integrate the residual part from koopman into kalman filter
'''

import torch
from einops import rearrange
from torch import nn
import numpy as np
from probts.model.nn.prob.k2VAE.SelfAttention_Family import FullAttention, AttentionLayer
from probts.model.nn.prob.k2VAE.Transformer_EncDec import Encoder, EncoderLayer


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=False,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )
        # Adaption Head
        self.adaptation = nn.Linear(config.dynamic_dim, config.d_model)

        # Prediction Head
        self.head = nn.Linear((self.seq_len // config.patch_len) * config.d_model,
                              (self.pred_len // config.patch_len) * config.dynamic_dim)

        self.config = config

    def forward(self, x_enc):
        # Encoder
        # z: [B L C]
        x_enc = self.adaptation(x_enc)
        enc_out, attns = self.encoder(x_enc)

        # Decoder
        enc_out = rearrange(enc_out, "b l c -> b (l c)")
        dec_out = self.head(enc_out)  # z: [B (C P)]
        dec_out = rearrange(dec_out, "b (p c) -> b p c", c=self.config.dynamic_dim)

        return dec_out


