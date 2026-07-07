__all__ = ['GoodModel']

from ast import Return
import torch
import torch.nn as nn
import copy
import numpy as np
from utils.tools import ema_update, mean_filter
from layers.SelfAttention_Family import FullAttention, AttentionLayer, MoAttention
from layers.PatchTST_layers import *
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs, d_inputs, d_folds, d_compress, pe='zeros', learn_pe=True):
        """

        """
        super(Model, self).__init__()
        self.d_folds = d_folds
        self.rev_in = configs.revin
        self.pred_len = configs.pred_len
        self.L = configs.level_dim
        if self.rev_in:
            self.revin_layer = RevIN(d_inputs[-1], affine=True, subtract_last=False)
        self.jepa = configs.jepa

        dropout = configs.dropout
        d_model = configs.d_model
        d_ff = configs.d_ff
        n_heads = configs.n_heads
        output_attn = configs.output_attention
        # self.x_encoder = CapEncoder(d_inputs, d_folds, d_compress, d_ff=d_ff, n_block=configs.n_block, dropout=dropout,
        #                             d_model=d_model, n_heads=n_heads, pe=pe, learn_pe=learn_pe, output_attn=output_attn)

        self.decoder = CapDecoder(d_compress, d_inputs, self.pred_len, d_ff, dropout=dropout)

    def forward(self, x):
        if self.rev_in:
            x = self.revin_layer(x, 'norm')
        ys = self.decoder(x) # (B, L, V, T)
        if self.rev_in:
            if isinstance(ys, list):
                ys = [self.revin_layer(y, 'denorm') for y in ys]
            else:
                ys = self.revin_layer(ys, 'denorm')
        return ys


class CapDecoder(nn.Module):
    def __init__(self, d_compress, d_inputs, d_pre, d_ff=256, dropout=0.1):
        super(CapDecoder, self).__init__()
        self.receiver1 = DecoderBlock(d_ff, d_inputs[0], d_pre, dropout=dropout)
        self.d_inputs = d_inputs
        self.pred_len = d_pre

    def forward(self, x):
        ys = self.receiver1(x.transpose(1, 2))
        if isinstance(ys, list):
            ys = [y.transpose(1, 2) for y in ys]
        else:
            ys = ys.transpose(1, 2)
        return ys


class DecoderBlock(nn.Module):
    def __init__(self, d_ff, d_in, d_input, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.linear = nn.Linear(d_in, d_ff)
        self.MLP = nn.Sequential(
                                 nn.Linear(d_ff, d_ff),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 )
        
        self.head1 = nn.Linear(d_ff, d_input//4) # just predic the half length
        self.head2 = nn.Linear(d_ff, d_input//4)
        self.head3 = nn.Linear(d_ff, d_input//4)
        self.head4 = nn.Linear(d_ff, d_input//4)
        self.head = nn.Linear(d_ff, d_input)

    def forward(self, x):
        x1 = self.linear(x)
        x2 = self.MLP(x1)

        out1 = self.head1(x2)
        out2 = self.head2(x2)
        out3 = self.head3(x2)
        out4 = self.head4(x2)
        out = torch.cat([out1, out2, out3, out4], dim=-1)

        # out = self.head(x2)
        return out
    
    
class DecoderBlock2(nn.Module):
    def __init__(self, d_ff, d_in, d_out, dropout=0.1):
        super(DecoderBlock2, self).__init__()
        self.linear = nn.Linear(d_in, d_ff)  # proj from data space into representation space
        self.MLP = nn.Sequential(
                                nn.Dropout(0.1),
                                 nn.Linear(d_ff, d_ff),
                                 nn.Dropout(dropout),
                                 nn.GELU(),
                                 )
        
        self.head1 = nn.Sequential(
                                    # nn.LayerNorm(d_ff),
                                    nn.Linear(d_ff, d_out // 3) #  predict short length
        )
        self.head2 = nn.Sequential(
                                nn.LayerNorm(d_in + d_out // 3),  # post-norm
                                nn.Dropout(0.3),
                                nn.Linear(d_in + d_out // 3, d_ff),
                                nn.Linear(d_ff, d_ff),
                                nn.Dropout(dropout),
                                #    nn.GELU(),
                                nn.Linear(d_ff, d_out))
        self.drop = nn.Dropout(0.)
        self.len = d_out
        # self.bn = nn.BatchNorm1d() 

    def forward(self, x):
        out = []
        x1 = self.linear(x)
        # short term part, this is an implementation of multi-head linear projection 
        # TODO: does this output needs to be projected into the data space?
        x2 = self.MLP(x1)
        out1 = self.head1(x2)
        out.append(out1)
        # entire result
        input_ar = self.drop(torch.cat([x, out1.detach()], dim=-1)) 
        out2 = self.head2(input_ar)
        out.append(out2)
        # out2[..., : self.len // 3] = out1.detach()
        # out = torch.cat([out1, out2], dim=-1)
        return out   

class DecoderBlock4(nn.Module):
    def __init__(self, d_ff, d_in, d_out, n_c=7, dropout=0.1):
        super(DecoderBlock4, self).__init__()
        # block_1
        self.MLP1 = nn.Sequential(
                                    nn.Dropout(0.1),
                                    nn.Linear(d_in, d_ff),  # proj from data space into representation space
                                    nn.Linear(d_ff, d_ff),
                                    nn.Dropout(dropout),
                                    nn.GELU(),
                                 )
        self.head1 = nn.Linear(d_ff, d_out // n_c) #  predict short length
        # block_2
        self.MLP2 = nn.Sequential(
                                    nn.LayerNorm(d_in + d_out // n_c),  # post-norm
                                    nn.Dropout(0.3),
                                    nn.Linear(d_in + d_out // n_c, d_ff),
                                    nn.Linear(d_ff, d_ff),
                                   nn.Dropout(dropout),
                                   nn.GELU(),
                                   )
        self.head2 = nn.Linear(d_ff, d_out // n_c * 2)
        # self.ln1 = nn.LayerNorm(d_ff)

        self.MLP3 = nn.Sequential(
                                    nn.LayerNorm(d_in + d_out // n_c * 3),  # post-norm
                                    nn.Dropout(0.6),
                                    nn.Linear(d_in + d_out // n_c * 3, d_ff),
                                    nn.Linear(d_ff, d_ff),
                                    nn.Dropout(dropout),
                                #    nn.GELU(),
                                   )
        self.head3 = nn.Linear(d_ff, d_out)
        

    def forward(self, x):
        out = []
        x1 = self.MLP1(x)
        out1 = self.head1(x1)
        out.append(out1)

        x2 = self.MLP2(torch.cat([x, out1.detach()], dim=-1))
        out2 = self.head2(x2)
        out.append(out2)

        mask1 = torch.zeros_like(out1) # this mask should be set in the buffer, self.mask...
        mask2 = torch.zeros_like(out2)

        x3 = self.MLP3(torch.cat([x, out1.detach(), out2.detach()], dim=-1))

        out3 = self.head3(x3)
        out.append(out3)
        # out2 = self.head2(self.drop(torch.cat([x1, x2], dim=-1)))
        # out = torch.cat([out1, out2], dim=-1)
        return out
    
    # def forward(self, x):
    #     out = []
    #     x1 = self.MLP1(x)
    #     out1 = self.head1(x1)
    #     out.append(out1)

    #     x2 = self.MLP2(torch.cat([x, out1.detach()], dim=-1))
    #     out2 = torch.cat([out1, self.head2(x2)], dim=-1)
    #     out.append(out2)

    #     x3 = self.MLP3(torch.cat([x, out2.detach()], dim=-1))

    #     out3 = torch.cat([out2, self.head3(x3)], dim=-1)
    #     out.append(out3)
    #     # out2 = self.head2(self.drop(torch.cat([x1, x2], dim=-1)))
    #     # out = torch.cat([out1, out2], dim=-1)
    #     return out
    
    @torch.no_grad()
    def predict(self, x):
        out = self.MLP(x)

        return out

# class DecoderBlock3(nn.Module):
#     def __init__(self, d_ff, d_in, d_out, dropout=0.1):
#         super(DecoderBlock3, self).__init__()
#         self.linear = nn.Linear(d_in, d_ff)  # proj from data space into representation space
#         self.MLP = nn.Sequential(
#                                  nn.Linear(d_ff, d_ff),
#                                  nn.Dropout(dropout),
#                                  nn.GELU(),
#                                  )
        
#         self.head1 = nn.Linear(d_ff, d_out // 3) #  predict short length
#         self.head2 = nn.Sequential(
#                                     # nn.Linear(d_in, d_ff),
#                                     # nn.LayerNorm(d_in),
#                                     # nn.Linear(d_in, d_ff),
#                                     nn.Linear(d_ff, d_ff),
#                                 #    nn.LayerNorm(d_ff),
#                                    nn.Dropout(dropout),
#                                    nn.GELU(),
#                                    nn.Linear(d_ff, d_out))
#         # self.drop = nn.Dropout(0.)
#         # self.bn = nn.BatchNorm1d() 

#     def forward(self, x):
#         out = []
#         x1 = self.linear(x)

#         x2 = self.MLP(x1)
#         out1 = self.head1(x2)
#         out.append(out1)

#         out2 = self.head2(x1)
#         out.append(out2)
#         # out2 = self.head2(self.drop(torch.cat([x1, x2], dim=-1)))
#         # out = torch.cat([out1, out2], dim=-1)
#         return out