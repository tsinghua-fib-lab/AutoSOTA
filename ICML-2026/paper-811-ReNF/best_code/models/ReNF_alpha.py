__all__ = ['GoodModel']

# from ast import Return
import torch
import torch.nn as nn
import torch.nn.init as init
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs, d_inputs, pe='zeros', learn_pe=True):
        """

        """
        super(Model, self).__init__()
        self.rev_in = configs.revin
        self.pred_len = configs.pred_len
        self.L = configs.level_dim
        if self.rev_in:
            self.revin_layer = RevIN(d_inputs[-1], affine=True, subtract_last=False)
        self.decoder = Decoder(d_inputs, self.pred_len, n_c=configs.d_layers, d_ff=configs.d_ff, dropout=configs.dropout, pe=configs.pe)


    def forward(self, x, return_rep=False):
        # B, T, V = x.shape  # (B, T, V)
        # Y_tilde = None
        if self.rev_in:
            x = self.revin_layer(x, 'norm')
        result = self.decoder(x, return_rep=return_rep) # (B, L, V, T)
        if return_rep:
            ys, reps = result
        else:
            ys = result
        if self.rev_in:
            ys = [self.revin_layer(y, 'denorm') for y in ys]
        if return_rep:
            return ys, reps
        else:
            return ys


class Decoder(nn.Module):
    def __init__(self, d_inputs, d_pred, n_c=3, d_ff=256, dropout=0., pe=False, exv_use=0):
        super(Decoder, self).__init__()
        self.receiver = DecoderBlock(d_ff, d_inputs[0], d_pred, n_c, drop_ff=dropout)
        self.d_inputs = d_inputs
        self.pred_len = d_pred
        self.pe = pe
        W_pos = torch.empty((d_inputs[0], d_inputs[-1]))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.embedding = nn.Parameter(W_pos)
        self.exv_use = exv_use
        if exv_use:
            self.fea_proj = nn.Sequential(
                                            nn.Linear(4, 128, bias=False),
                                            nn.GELU(),
                                            nn.Dropout(dropout),
                                            nn.Linear(128, d_inputs[-1]),
                                        )

    def forward(self, x, x_mark=None, return_rep=False):
        if self.pe:
            x = x + self.embedding
        if x_mark is not None and self.exv_use:
            x = x + self.fea_proj(x_mark)
        result = self.receiver(x.transpose(1, 2), return_rep)
        if return_rep:
            ys, reps = result
        else:
            ys = result
        ys = [y.transpose(1, 2) for y in ys]
        if return_rep:
            return ys, reps
        else:
            return ys

class ForeBlock(nn.Module):
    def __init__(self, d_in, d_out, d_ff, drop_ff=0.):
        super(ForeBlock, self).__init__()
        self.bottom =  nn.Sequential(
                                    nn.Dropout(0.1),
                                    nn.Linear(d_in, d_ff),  # proj from data space into representation space
                                 )
        self.MLP = nn.Sequential(
                                    nn.Linear(d_ff, d_ff, bias=False),
                                    nn.GELU(),
                                    nn.Dropout(drop_ff),
                                 )
        self.head = nn.Linear(d_ff, d_out)

    def forward(self, input):
        rep = self.bottom(input)
        rep = self.MLP(rep)
        output = self.head(rep)
        return output, rep

class Activation(nn.Module):
    def __init__(self, act='nonlinear') -> None:
        super().__init__()
        self.act = nn.GELU() if act == 'nonlinear' else nn.Identity()

    def forward(self, x):
        return self.act(x)

class MidBlock(nn.Module):
    def __init__(self, d_in, d_out, d_ff, drop_ff=0., drop_in=0.1, act='nonlinear'):
        super(MidBlock, self).__init__()
        self.bottom = nn.Sequential(
                                    nn.LayerNorm(d_in),
                                    nn.Dropout(drop_in),
                                    nn.Linear(d_in, d_ff),
                                    # nn.Dropout(drop_ff),
        )
        self.MLP = nn.Sequential(   
                                    nn.Linear(d_ff, d_ff),
                                    nn.Dropout(drop_ff),
                                    Activation(act),
                                 )
        self.head = nn.Linear(d_ff, d_out)

    def forward(self, input):
        rep = self.bottom(input)
        rep = self.MLP(rep)
        output = self.head(rep)
        return output, rep

class DecoderBlock(nn.Module):
    def __init__(self, d_ff, d_in, d_out, n_c=3, drop_ff=None, drop_in=0.3):
        super(DecoderBlock, self).__init__()
        self.pred_base = d_out // n_c 
        self.cascade = nn.ModuleList([ForeBlock(d_in, self.pred_base, d_ff, drop_ff)])
        self.n_c = n_c
        for i in range(1, n_c):
            n_acc = i * (i + 1) // 2
            d_cat = d_in + self.pred_base * n_acc
            act='nonlinear' if i < n_c - 1 else 'linear'
            drop_in_layer = min(drop_in * (i + 1), 0.6)
            # drop_in_layer = max(drop_in * (0.8 ** i), 0.05)
            self.cascade.append(MidBlock(d_cat, self.pred_base * (i + 1), d_ff, drop_ff, drop_in_layer, act))
        
    def forward(self, x0, return_rep=False):
        out = []
        x_k = x0
        reps = []
        for k, layer in enumerate(self.cascade):
            y_k, rep = layer(x_k)
            out.append(y_k)
            if return_rep:
                reps.append(rep)
            y_cat = torch.cat(out, dim=-1).detach() # disable deep representation
            mask = 1.0
            x_k = torch.cat([x0, y_cat * mask], dim=-1)
        if return_rep:
            return out, reps
        else:
            return out
    
    # @torch.no_grad()
    # def predict(self, x):
    #     return self.forward(x)

