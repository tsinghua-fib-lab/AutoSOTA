import math
import torch
from einops import rearrange


class Performer_Att:
    def __init__(self, num_feats, dim, dev):
        self.num_feats = num_feats
        self.dim = dim
        self.weights = torch.randn(self.dim[0], self.dim[1], self.dim[2], self.num_feats, dtype=torch.float64,
                                   device=dev)

    def calc_feats(self, K):
        proj_K = torch.einsum('bhnd,bhdr -> bhnr', K, self.weights)
        feats = torch.exp(proj_K - 0.5 * (torch.linalg.norm(K, dim=3) ** 2).unsqueeze(-1)) / math.sqrt(self.num_feats)
        return feats


class Performer:
    def __init__(self, rep=10, num_feats=10):
        self.rep = rep
        self.num_feats = num_feats

    def forward(self, query, key):
        query = rearrange(query, 'b t h e -> b h t e')
        key = rearrange(key, 'b s h e -> b h s e')

        performer = Performer_Att(self.num_feats, (query.shape[0], query.shape[1], query.shape[3]), query.device)
        K_feats = performer.calc_feats(key)
        Q_feats = performer.calc_feats(query)

        D_tilde = torch.einsum('bhnd,bhd -> bhn', Q_feats, torch.sum(K_feats, dim=2))
        return D_tilde, query.shape[2]
