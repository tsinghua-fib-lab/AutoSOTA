"""
Vendored from HyPoGen2: initial weight token generation via cross-attention.
"""

import torch
from torch import nn
from .transformer import CrossAttn


class InitWeightHypernet(nn.Module):
    def __init__(self, target_net, embed_dim, weight_split_dim, hidden_dim=128, num_layers=4):
        super(InitWeightHypernet, self).__init__()
        self.embed_dim = embed_dim
        self.num_tokens = []
        self.token_queries = nn.ParameterList()
        self.cross_attn_layers = nn.ModuleList()
        self.weight_split_dim = weight_split_dim

        for module in target_net.get_submodules():
            if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
                raise ValueError("Target module must have `in_features` and `out_features` attributes")
            in_feat = module.in_features + 1
            out_feat = module.out_features

            if in_feat > weight_split_dim:
                L = out_feat * int((in_feat + weight_split_dim - 1) // weight_split_dim)
            else:
                L = out_feat
            self.num_tokens.append(L)

            token = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, L, embed_dim)))
            self.token_queries.append(token)

            cross_attn = CrossAttn(self.embed_dim, hidden_dim=hidden_dim, num_layers=num_layers)
            self.cross_attn_layers.append(cross_attn)

    def forward(self, ftask, ablation=None):
        B, N, D = ftask.shape
        assert D == self.embed_dim, "Task embedding dimension D must match hypernet embed_dim"

        if ablation == "opt_layer_only":
            return [
                token.expand(B, -1, -1) for token in self.token_queries
            ]

        weight_embeddings = []
        for token, cross_attn in zip(self.token_queries, self.cross_attn_layers):
            attn_output = cross_attn(token.expand(B, -1, -1), ftask)
            weight_embeddings.append(attn_output)
        return weight_embeddings
