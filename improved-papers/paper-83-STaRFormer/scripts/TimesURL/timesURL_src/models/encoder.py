import torch
import copy
from torch import nn
import numpy as np
from .dilated_conv import DilatedConvEncoder


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t:t + l] = False
    return res


class BertInterpHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.activation = nn.ReLU()
        self.project = nn.Linear(4 * hidden_dim, input_dim)

    def forward(self, first_token_tensor):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.project(pooled_output)
        return pooled_output


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)

        self.feature_extractor = DilatedConvEncoder(
            # input_dims,
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.interphead = BertInterpHead(input_dims, output_dims)

    def forward(self, x, mask=None, dataset: str='dkt'):  # x: B x T x input_dims
        #if dataset == 'dkt':
        #    if isinstance(x, dict):
        #        input_all = copy.deepcopy(x)
        #        m = x['mask']
        #        x = x['data'] if 'data' in x.keys() else x['x']
        #    else:
        #        raise RuntimeError
        #else:

        if isinstance(x, dict):
            input_all = copy.deepcopy(x)
            m = x['mask']
            x = x['data'] if 'data' in x.keys() else x['x']
        else:
            input_all = copy.deepcopy(x)
            m = x[..., -(x.shape[-1] // 2):]
            x = x[..., :-(x.shape[-1] // 2)]

        t = x[..., -1]
        if self.training:
            x = x[..., :-1]

        if mask == 'mask_last':
            nan_mask = ~x.isnan().any(axis=-1)

        x[torch.isnan(x)], m[torch.isnan(m)] = 0, 0

        # whole series without missing
        if self.training:
            x_whole = self.input_fc(x * input_all['mask_origin'])
            x_whole = x_whole.transpose(1, 2)
            x_whole = self.feature_extractor(x_whole)  # B x Ch x T
            x_whole = x_whole.transpose(1, 2)  # B x T x Co
            x_whole = self.repr_dropout(x_whole)

        # recon mask part
        if self.training:
            x_interp = self.input_fc(x * input_all['mask'])
            x_interp = x_interp.transpose(1, 2)
            x_interp = self.feature_extractor(x_interp)  # B x Ch x T
            x_interp = x_interp.transpose(1, 2)  # B x T x Co
            x_interp = self.repr_dropout(x_interp)

        if mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
            mask &= nan_mask
            x[~mask] = 0

        x = self.input_fc(x * m)
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)  # B x Ch x T
        x = x.transpose(1, 2)  # B x T x Co
        x = self.repr_dropout(x)

        if self.training:
            return x_whole, self.interphead(x_interp)
        else:
            return x
