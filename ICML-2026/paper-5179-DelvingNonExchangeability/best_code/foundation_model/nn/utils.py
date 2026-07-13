from typing import Optional
import torch
from torch import Tensor

def maybe_cat_emb(x: Tensor, emb: Optional[Tensor]):
    if emb is None:
        return x
    if emb.ndim < x.ndim:
        if emb.ndim == 3 and x.ndim == 4:
            emb = emb.unsqueeze(1)
        else:
            emb = emb[[None] * (x.ndim - emb.ndim)]
    emb = emb.expand(*x.shape[:-1], -1)
    return torch.cat([x, emb], dim=-1)


def adj_to_fc_edge_index(adjs):
    num_nodes = adjs.shape[-1]
    adjs = adjs.transpose(-2, -1)
    edge_weight = adjs.flatten()
    idx = torch.arange(num_nodes, device=adjs.device)
    edge_index = torch.cartesian_prod(idx, idx).T
    if adjs.dim() == 3:
        edge_index = [edge_index + num_nodes * i for i in range(adjs.size(0))]
        edge_index = torch.cat(edge_index, dim=-1)
    return edge_index, edge_weight
