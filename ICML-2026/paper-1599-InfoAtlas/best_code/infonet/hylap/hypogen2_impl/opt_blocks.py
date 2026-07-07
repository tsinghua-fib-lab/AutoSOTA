"""
Vendored from HyPoGen2: Opt layer blocks for pseudo forward/backward.
Simplified to keep footprint small while retaining behavior.
"""

import einops
import torch
import torch.nn.functional as F
from torch import nn
import math
from .transformer import CrossAttn, SelfAttn
from .modules import MLP


class FIN_FOUT(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.in_dim = args[1]
        self.out_dim = args[1]
        self.hidden_dim = kwargs.get("hidden_dim", 1024)
        self.num_layers = kwargs.get("num_layers", 1)
        self.nhead = kwargs.get("nhead", 4)

        self.next_blocks = []
        self.prev_blocks = []
        self.net = SelfAttn(self.in_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, nhead=self.nhead)

    def add2in_dim(self, dim):
        pass

    def pseudo_forward(self, x):
        res = self.net(x)
        return res

    def pseudo_backward(self, x, z):
        x = torch.cat([x, z], dim=1)
        res = self.net(x)
        return res

    def get_zin(self):
        return self.z_in

    def get_dldin(self, prev_blk):
        return self.dldin


class OptBlock(nn.Module):
    def __init__(self, module, ftask_dim, weight_dim, hidden_dim, num_layers, weight_split_dim, **kwargs):
        super().__init__()
        self.in_features = module.in_features
        self.out_features = module.out_features

        self.ftask_dim = ftask_dim
        self.in_dim = 0
        self.weight_dim = weight_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.weight_split_dim = weight_split_dim
        self.nhead = kwargs.get("nhead")

        self.next_blocks = []
        self.prev_blocks = []

    def setup(self):
        self.num_tokens = self.out_features
        if self.in_features + 1 > self.weight_split_dim:
            self.num_tokens *= int(math.ceil((self.in_features + 1) / self.weight_split_dim))

        self.ftask_zin = CrossAttn(
            self.weight_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, nhead=self.nhead
        )
        self.token_task = CrossAttn(
            self.weight_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, nhead=self.nhead
        )
        self.learn_p_forward = nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty(1, self.num_tokens, self.weight_dim))
        )

        self.weightemb_zprime = CrossAttn(
            self.weight_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, nhead=self.nhead
        )
        self.task_grad = CrossAttn(
            self.weight_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, nhead=self.nhead
        )
        self.token_grad = CrossAttn(
            self.weight_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, nhead=self.nhead
        )
        self.dl_dout_prime_net = CrossAttn(
            self.weight_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, nhead=self.nhead
        )
        self.learn_p_backward_z_grad = nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty(1, self.num_tokens, self.weight_dim))
        )
        self.learn_p_backward_dl_dout = nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty(1, self.num_tokens, self.weight_dim))
        )

        self.lr_attn = CrossAttn(
            self.weight_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, nhead=self.nhead
        )
        lr_token_dim = 8
        self.lr_pre_mlp = MLP(self.weight_dim, lr_token_dim, self.weight_dim // 2, num_layers=2)
        self.lr_mlp = MLP(lr_token_dim * self.num_tokens, 1, 16, num_layers=2)

        self.out_din = CrossAttn(
            self.weight_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, nhead=self.nhead
        )
        self.dout_dw = CrossAttn(
            self.weight_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, nhead=self.nhead
        )
        self.dl_din_net = CrossAttn(
            self.weight_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, nhead=self.nhead
        )
        self.dl_dw_net = CrossAttn(
            self.weight_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, nhead=self.nhead
        )

    def pseudo_forward(self, ftask, weight_emb, z_in):
        if z_in.shape[0] != weight_emb.shape[0]:
            weight_emb = einops.repeat(weight_emb, "1 L D -> n L D", n=z_in.shape[0])

        task_context = self.ftask_zin(ftask, z_in)
        z_prime = self.token_task(self.learn_p_forward.expand(ftask.shape[0], -1, -1), task_context)
        out = self.weightemb_zprime(weight_emb, z_prime)

        lr_attn = self.lr_attn(z_prime, weight_emb)
        lr_reduced = self.lr_pre_mlp(lr_attn)
        lr_flat = lr_reduced.view(lr_reduced.shape[0], -1)
        self.lr = F.sigmoid(self.lr_mlp(lr_flat)).reshape(lr_flat.shape[0])

        return out

    def get_zin(self):
        return self.nxt_z_in

    def get_lrs(self):
        return self.lr

    def pseudo_backward(self, ftask, weight_emb, dl_dout, z_in):
        if z_in.shape[0] != weight_emb.shape[0]:
            weight_emb = einops.repeat(weight_emb, "1 L D -> n L D", n=z_in.shape[0])

        task_context = self.task_grad(ftask, z_in)
        z_grad = self.token_grad(self.learn_p_backward_z_grad.expand(ftask.shape[0], -1, -1), task_context)
        dout_din = self.out_din(weight_emb, z_grad)
        dout_dw = self.dout_dw(z_grad, weight_emb)

        dl_dout_prime = self.dl_dout_prime_net(self.learn_p_backward_dl_dout.expand(ftask.shape[0], -1, -1), dl_dout)
        dl_din = self.dl_din_net(dl_dout_prime, dout_din)
        dl_dw = self.dl_dw_net(dl_dout_prime, dout_dw)

        return dl_dw, dl_din


class OptLayer(nn.Module):
    def __init__(self, target_net, ftask_dim, weight_dim, deriv_hidden_dim, driv_num_layers, *args, **kwargs):
        super().__init__()
        self.ftask_dim = ftask_dim
        self.weight_dim = weight_dim
        self.opt_blocks, self.forward_in, self.dloss_dout = target_net.construct_opt_blocks(
            ftask_dim, weight_dim, deriv_hidden_dim, driv_num_layers, **kwargs
        )
        for opt_block in self.opt_blocks:
            opt_block.setup()

    def pseudo_forward(self, ftask, weight_embs):
        z_ins = []
        z_in = self.forward_in[0].pseudo_forward(ftask)
        z_ins.append(z_in)

        for idx, (opt_block, weight_emb) in enumerate(zip(self.opt_blocks, weight_embs)):
            z_in = opt_block.pseudo_forward(ftask, weight_emb, z_ins[-1])
            z_ins.append(z_in)
        return z_ins

    def pseudo_backward(self, ftask, weight_embs, z_ins):
        dl_dz = self.dloss_dout[0].pseudo_backward(ftask, z_ins[-1])

        dw_dicts = []
        for opt_block, weight_emb, z_in, idx in reversed(
            list(zip(self.opt_blocks, weight_embs, z_ins[:-1], range(len(self.opt_blocks))))
        ):
            dl_dw, dl_dz = opt_block.pseudo_backward(ftask, weight_emb, dl_dz, z_in)
            dw_dicts.append(dl_dw)

        dw_dicts = list(reversed(dw_dicts))
        return dw_dicts

    def forward(self, ftask, weight_embs):
        z_ins = self.pseudo_forward(ftask, weight_embs)
        dw_dicts = self.pseudo_backward(ftask, weight_embs, z_ins)
        return dw_dicts

    def get_lrs(self):
        return [opt_block.get_lrs() for opt_block in self.opt_blocks]


def get_opt_layer(use_compile: bool = True):
    """Return OptLayer class, optionally torch.compile-wrapped."""
    if use_compile:
        try:
            return torch.compile(OptLayer, mode="reduce-overhead")
        except Exception:
            return OptLayer
    return OptLayer
