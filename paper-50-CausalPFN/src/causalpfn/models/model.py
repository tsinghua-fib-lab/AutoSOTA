from enum import Enum

import numpy as np
import torch
import torch.nn as nn

from .transformer_layer import TransformerEncoderLayer


class Task(Enum):
    REG = 1
    CLS = 2


def maskmean(x, mask, dim):
    x = torch.where(mask, x, 0)
    return x.sum(dim=dim, keepdim=True) / mask.sum(dim=dim, keepdim=True)


def maskstd(x, mask, dim=0):
    num = mask.sum(dim=dim, keepdim=True)
    mean = maskmean(x, mask, dim=0)
    diffs = torch.where(mask, mean - x, 0)
    return ((diffs**2).sum(dim=0, keepdim=True) / (num - 1)) ** 0.5


def normalize_data(data, eval_pos):
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=0)
    std = maskstd(X, mask, dim=0) + 1e-6
    data = (data - mean) / std
    return data


def clip_outliers(data, eval_pos, n_sigma=4):
    assert len(data.shape) == 3, "X must be T,B,H"
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=0)
    cutoff = n_sigma * maskstd(X, mask, dim=0)
    mask &= cutoff >= torch.abs(X - mean)
    cutoff = n_sigma * maskstd(X, mask, dim=0)
    return torch.clip(data, mean - cutoff, mean + cutoff)


def convert_to_torch_tensor(input):
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    elif torch.is_tensor(input):
        return input
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")


class TabDPTLongContextModel(nn.Module):
    def __init__(
        self,
        dropout: float,
        n_out: int,
        nhead: int,
        nhid: int,
        ninp: int,
        nlayers: int,
        num_features: int,
        nbins=int,
    ):
        super().__init__()
        self.n_out = n_out
        self.ninp = ninp
        self.nbins = nbins
        self.nhead = nhead
        self.nhid = nhid
        self.nlayers = nlayers
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=ninp,
                    num_heads=nhead,
                    ff_dim=nhid,
                )
                for _ in range(nlayers)
            ]
        )
        self.num_features = num_features
        self.encoder = nn.Linear(num_features, ninp, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.y_encoder = nn.Linear(1, ninp, bias=True)
        self.head = nn.Sequential(
            nn.Linear(ninp, nhid, bias=False), nn.GELU(), nn.Linear(nhid, n_out + nbins, bias=False)
        )
        self.xnorm = nn.LayerNorm(ninp, bias=False)
        self.ynorm = nn.LayerNorm(ninp, bias=False)

    def forward(
        self,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        return_log_act_norms: bool = False,
    ) -> torch.Tensor:
        context_length = y_src.shape[0]
        B = y_src.shape[1]
        x_src = normalize_data(x_src, -1 if self.training else context_length)
        x_src = clip_outliers(x_src, -1 if self.training else context_length, n_sigma=10)
        x_src = torch.nan_to_num(x_src, nan=0)

        x_src = self.xnorm(self.encoder(x_src))
        y_src = self.ynorm(self.y_encoder(y_src.unsqueeze(-1)))
        train_x = x_src[:context_length] + y_src
        src = torch.cat([train_x, x_src[context_length:]], 0)

        log_act_norms = {}
        log_act_norms["y"] = torch.norm(y_src, dim=-1).mean()
        log_act_norms["x"] = torch.norm(x_src, dim=-1).mean()

        for l, layer in enumerate(self.transformer_encoder):
            if l in [0, 1, 3, 6, 9]:
                log_act_norms[f"layer_{l}"] = torch.norm(src, dim=-1).mean()
            src = layer(src, context_length)

        pred = self.head(src)

        pred = 30 * torch.tanh(pred / (7.5 * src.size(-1) ** 0.5))
        if return_log_act_norms:
            return pred[context_length:], log_act_norms
        else:
            return pred[context_length:]

    @classmethod
    def load(cls, model_state, config):
        assert config.model.max_num_classes > 2
        model = TabDPTLongContextModel(
            dropout=config.training.dropout,
            n_out=config.model.max_num_classes,
            nhead=config.model.nhead,
            nhid=config.model.emsize * config.model.nhid_factor,
            ninp=config.model.emsize,
            nlayers=config.model.nlayers,
            num_features=config.model.max_num_features,
            nbins=config.model.nbins,
        )

        module_prefix = "_orig_mod."
        model_state = {k.replace(module_prefix, ""): v for k, v in model_state.items()}
        model.load_state_dict(model_state)
        model.eval()
        return model
