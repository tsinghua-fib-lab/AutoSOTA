import torch
import torch.nn as nn
import numpy as np
from layers.BiMamba4TS_layers import Encoder, EncoderLayer
from layers.Embed import PositionalEmbedding
from einops import rearrange
from mamba_ssm import Mamba
# from mamba_plus import Mamba
# from layers.Attention import AttentionLayer, FullAttention

class PatchEmbedding(nn.Module):
    def __init__(self, seq_len, d_model, patch_len, stride, dropout,
                 process_layer=None, 
                 pos_embed_type=None, learnable=False, r_layers=1,
                 ch_ind=0):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.ch_ind = ch_ind
        self.process_layer = process_layer
        self.pos_embed_type = pos_embed_type

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model)

        # Positional embedding
        if self.pos_embed_type is not None:
            self.position_embedding = PositionalEmbedding(seq_len, d_model, pos_embed_type, learnable, r_layers, patch_len)
        
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        # x: [B, M, L], x_mark: [B, 4, L]
        n_vars = x.shape[1]
        x = self.process_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        if x_mark is not None and not self.ch_ind:
            x_mark = x_mark.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            x = torch.cat([x, x_mark], dim=1) # [B, (M+4), N, P]
        
        # Input value embedding
        x = self.value_embedding(x) # [B, M, N, D]
        
        x = torch.reshape(x, (-1, x.shape[2], x.shape[3])) # [B*M, N, P]
        
        if self.pos_embed_type is not None:
            x = x + self.position_embedding(x)
        
        return self.dropout(x), n_vars


class TruncateModule(nn.Module):
    def __init__(self, target_length):
        super(TruncateModule, self).__init__()
        self.target_length = target_length

    def forward(self, x, truncate_length):
        return x[: ,: ,:truncate_length]
    
class Model(nn.Module):
    def __init__(self, configs, corr=None):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.revin = configs.revin
        self.embed_type = configs.embed_type
        if configs.SRA:
            self.ch_ind = self.SRA(corr, configs.threshold)
        else:
            self.ch_ind = configs.ch_ind

        # patching
        # being able to divide indicates that there is no need to forcefully padding
        if configs.seq_len % configs.stride==0:
            self.patch_num = int((configs.seq_len - configs.patch_len)/configs.stride + 1)
            process_layer = nn.Identity()
        else:
            # padding at tail
            if configs.padding_patch=="end":
                padding_length = configs.stride - (configs.seq_len % configs.stride)
                self.patch_num = int((configs.seq_len - configs.patch_len)/configs.stride + 2)
                process_layer = nn.ReplicationPad1d((0, padding_length))
            # if not padding, then execute cutting
            else:
                truncated_length = configs.seq_len - (configs.seq_len % configs.stride)
                self.patch_num = int((configs.seq_len - configs.patch_len)/configs.stride + 1)
                process_layer = TruncateModule(truncated_length)
        self.local_token_layer = PatchEmbedding(configs.seq_len, configs.d_model, configs.patch_len, configs.stride, configs.dropout, process_layer,
                                                pos_embed_type=None if configs.embed_type in [0, 2] else configs.pos_embed_type,
                                                learnable=configs.pos_learnable,
                                                ch_ind=configs.ch_ind)
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(d_model=configs.d_model, 
                        #   batch_size=configs.batch_size,
                        #   seq_len=self.patch_num,
                          d_state=configs.d_state, 
                          d_conv=configs.d_conv, 
                          expand=configs.e_fact,
                          use_fast_path=True),
                    Mamba(d_model=configs.d_model, 
                        #   batch_size=configs.batch_size,
                        #   seq_len=self.patch_num,
                          d_state=configs.d_state, 
                          d_conv=configs.d_conv, 
                          expand=configs.e_fact,
                          use_fast_path=True),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    bi_dir=configs.bi_dir,
                    residual=configs.residual==1
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        head_nf = configs.d_model * self.patch_num
        self.head = Flatten_Head(False, configs.enc_in, head_nf, self.pred_len)

    def SRA(self, corr, threshold):
        high_corr_matrix = corr >= threshold
        num_high_corr = np.maximum(high_corr_matrix.sum(axis=1)-1, 0)

        positive_corr_matrix = corr >= 0
        num_positive_corr = np.maximum(positive_corr_matrix.sum(axis=1)-1, 0)
        max_high_corr = num_high_corr.max()
        max_positive_corr = num_positive_corr.max()
        r = max_high_corr/max_positive_corr
        print('SRA -> channel mixing' if r >= 1 - threshold else 'channel independent')
        return 0 if r >= 1 - threshold else 1

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.revin:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # B L M
        B, _, M = x_enc.shape
    
        # [B, M, L] -> [B*M, N, D] or [B*(M+4), N, D]
        enc_out, _ = self.local_token_layer(x_enc.permute(0, 2, 1),
                                            x_mark_enc.permute(0, 2, 1) if self.embed_type==2 else None)
        if not self.ch_ind:
            enc_out = rearrange(enc_out, '(B M) N D -> (B N) M D', B=B)

        enc_out = self.encoder(enc_out)
        # output: [B*M, N, D] or [B*N, M, D] -> [B x M x H]
        if not self.ch_ind:
            dec_out = rearrange(enc_out, '(B N) M D -> B M N D', B=B)
        else:
            dec_out = rearrange(enc_out, '(B M) N D -> B M N D', B=B)
        dec_out = self.head(dec_out).permute(0, 2, 1)[:, :, :M]

        if self.revin:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out



class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0.):
        super().__init__()
        
        self.ch_ind = individual
        self.n_vars = n_vars
        
        if self.ch_ind:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                           # x: [B, C, D, N]
        if self.ch_ind:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])    # z: [B, D * N]
                z = self.linears[i](z)              # z: C x [B, H]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)           # x: [B, C, H]
        else:
            x = self.flatten(x)                     # x: [B, C, D * N]
            x = self.linear(x)                      # x: [B, C, H]
            # x = self.dropout(x)
        return x