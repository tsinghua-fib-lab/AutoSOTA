import math
import torch
from einops import rearrange


class Performer_Att:
    def __init__(self, num_feats, dim, dev, dtype):
        self.num_feats = num_feats
        self.dim = dim
        self.weights = torch.randn(self.dim[0], self.dim[1], self.dim[2], self.num_feats, dtype=dtype, device=dev)

    def calc_feats(self, K):
        proj_K = torch.einsum('bhnd,bhdr -> bhnr', K, self.weights)
        K_norm = torch.sum(K.pow(2), axis=-1, keepdims=True)
        feats = torch.exp(proj_K - 0.5 * K_norm) / math.sqrt(self.num_feats)
        return feats


class PerformerAttention(torch.nn.Module):
    def __init__(self, rep=10, num_feats=10):
        super().__init__()
        self.rep = rep
        self.num_feats = num_feats

    def forward(self, query, key, value):
        query = rearrange(query, 'b t h e -> b h t e')
        key = rearrange(key, 'b s h e -> b h s e')
        value = rearrange(value, 'b s h d -> b h s d')

        performer = Performer_Att(self.num_feats, (query.shape[0], query.shape[1], query.shape[3]), query.device, query.dtype)
        K_feats = performer.calc_feats(key)
        Q_feats = performer.calc_feats(query)

        D_tilde = torch.einsum('bhnd,bhd -> bhn', Q_feats, torch.sum(K_feats, dim=2))
        D_inverse = 1.0 / D_tilde

        kv = torch.einsum('bhnm,bhnd->bhmd', K_feats, value)
        output = torch.einsum('bhnm,bhmd->bhnd', Q_feats, kv)
        output *= D_inverse.unsqueeze(-1)

        return output