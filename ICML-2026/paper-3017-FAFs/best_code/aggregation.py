import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Iterable

from torch_scatter import scatter
from torch_geometric.utils import degree

@torch.no_grad()
def _norm_mean_step(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    N = x.size(0)
    deg = degree(src, N, dtype=x.dtype).clamp(min=1)
    w = (deg[src] * deg[dst]).sqrt().reciprocal()
    return scatter(x[src] * w.unsqueeze(-1), dst, dim=0, dim_size=N, reduce='sum')

def _std_step(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    mean = scatter(x[src], dst, dim=0, dim_size=x.size(0), reduce='mean')
    mean_sq = scatter(x[src]**2, dst, dim=0, dim_size=x.size(0), reduce='mean')
    return (mean_sq - mean**2).clamp(min=0).sqrt()

@torch.no_grad()
def _reduce_step(x: torch.Tensor, edge_index: torch.Tensor, reduce: str) -> torch.Tensor:
    src, dst = edge_index
    return scatter(x[src], dst, dim=0, dim_size=x.size(0), reduce=reduce)

@torch.no_grad()
def multi_hop_reductions(x: torch.Tensor, edge_index: torch.Tensor, K: int,):
    feats = []
    cur_sum = cur_max = cur_min = cur_mean = x
    for t in range(K):
        cur_mean = _norm_mean_step(cur_mean, edge_index)
        cur_sum  = _reduce_step(cur_sum,  edge_index, 'sum')
        cur_max  = _reduce_step(cur_max,  edge_index, 'max')
        cur_min  = _reduce_step(cur_min,  edge_index, 'min')
        feats.extend([cur_sum, cur_mean, cur_max, cur_min])
    return feats

@torch.no_grad()
def aggregate_faf_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    K: int,
    extra_args: Optional[dict] = None
) -> torch.Tensor:
    feats = [x]
    added = [("input", x.size(1))]

    # Multi-hop reductions
    if extra_args.get('multi_agg', False):
        feats_hops = multi_hop_reductions(x, edge_index, K)
        feats.extend(feats_hops)
        added.append((f"multiagg_{K}hop", sum(t.size(1) for t in feats_hops)))
    if extra_args.get('sum_agg', False):
        cur_sum = x
        for t in range(K):
            cur_sum = _reduce_step(cur_sum, edge_index, 'sum')
            feats.append(cur_sum)
        added.append((f"sumagg_{K}hop", K * x.size(1)))
    if extra_args.get('mean_agg', False):
        cur_mean = x
        for t in range(K):
            cur_mean = _norm_mean_step(cur_mean, edge_index)
            feats.append(cur_mean)
        added.append((f"meanagg_{K}hop", K * x.size(1)))
    if extra_args.get('max_agg', False):
        cur_max = x
        for t in range(K):
            cur_max = _reduce_step(cur_max, edge_index, 'max')
            feats.append(cur_max)
        added.append((f"maxagg_{K}hop", K * x.size(1)))
    if extra_args.get('std_agg', False):
        cur_std = x
        for t in range(K):
            cur_std = _std_step(cur_std, edge_index)
            feats.append(cur_std)
        added.append((f"stdagg_{K}hop", K * x.size(1)))
    if extra_args.get('last_agg', False):
        multi_hop_feats = multi_hop_reductions(x, edge_index, K)
        last_hop_feats = multi_hop_feats[-4:]  # last hop
        if extra_args.get('last_agg_only', False):
            return torch.cat(last_hop_feats, dim=-1)
        return torch.cat([x] + last_hop_feats, dim=-1)

    # KA-based reductions
    if extra_args.get('ka_agg', False):
        from aggregation_other import ka_multihop_feats
        argska = extra_args.get('ka_args', {})
        ka_seq = ka_multihop_feats(x,edge_index, K=K, argska=argska)
        feats.extend(ka_seq)
        added.append((f"ka_{K}hop", sum(t.size(1) for t in ka_seq)))
    if extra_args.get('bin_agg', False):
        from aggregation_other import binned_multihop_feats
        argsbin = extra_args.get('bin_args', {})
        bin_feat = binned_multihop_feats(x, edge_index, K=K, argsbin=argsbin)
        feats.extend(bin_feat)
        added.append((f"binned_{K}hop", sum(t.size(1) for t in bin_feat)))

    # Similarity-based reductions
    if extra_args.get('sim_agg', False):
        from aggregation_other import sim_multihop_feats
        argsim = extra_args.get('sim_args', {})
        sim_hops = sim_multihop_feats(x, edge_index, K=K, argsim=argsim)
        feats.extend(sim_hops)
        added.append((f"sim_{K}hop", sum(t.size(1) for t in sim_hops)))

    if extra_args.get('rewire', False):
        argsim = extra_args.get('sim_args', {})
        from aggregation_other import rew_multihop_feats
        rew_hops = rew_multihop_feats(x, edge_index, K=K, argsim=argsim)
        feats.extend(rew_hops)
        added.append((f"rew_{K}hop", sum(t.size(1) for t in rew_hops)))

    if extra_args.get('split_comp', False):
        argsim = extra_args.get('sim_args', {})
        from aggregation_other import split_multihop_feats
        split_hops = split_multihop_feats(x, edge_index, K=K, argsim=argsim)
        feats.extend(split_hops)
        added.append((f"split_{K}hop", sum(t.size(1) for t in split_hops)))

    # Quantile-based reductions
    if extra_args.get('q_agg', False):
        from aggregation_other import neighbor_quantiles
        argsq = extra_args.get('q_args', {})
        q_feat = neighbor_quantiles(x, edge_index, argsq=argsq)
        feats.append(q_feat)
        added.append((f"q", q_feat.size(1)))

    # Network science features
    if extra_args.get('ns_agg', False):
        from aggregation_other import network_science_feats
        argsns = extra_args.get('ns_args', {})
        ns_feat = network_science_feats(x, edge_index, argsns=argsns)
        feats.append(ns_feat)
        added.append((f"ns", ns_feat.size(1)))

    # All features concatenated
    if extra_args.get('exp_agg', False):
        from aggregation_other import exp_multihop_feats
        all_seq = exp_multihop_feats(x, edge_index, K=K)
        feats.extend(all_seq)
        added.append((f"exp_{K}hop", sum(t.size(1) for t in all_seq)))

    if extra_args.get('all_agg', False):
        from aggregation_other import all_multihop_feats
        all_seq = all_multihop_feats(x, edge_index, K=K)
        feats.extend(all_seq)
        added.append((f"all_{K}hop", sum(t.size(1) for t in all_seq)))

    if extra_args.get('meansumall_agg', False):
        from aggregation_other import meansum_multihop_feats
        all_seq = meansum_multihop_feats(x, edge_index, K=K)
        feats.extend(all_seq)
        added.append((f"meansum_{K}hop", sum(t.size(1) for t in all_seq)))

    if extra_args.get('mmask_agg', False):
        from aggregation_other import mmask_multihop_feats
        all_seq = mmask_multihop_feats(x, edge_index, K=K)
        feats.extend(all_seq)
        added.append((f"mmask_{K}hop", sum(t.size(1) for t in all_seq)))

    print("FAF feature sizes (total={}):".format(sum(t.size(1) for t in feats)))
    for name, size in added:
        print(f"\t{name}: {size}")
        
    return torch.cat(feats, dim=-1)
    


## PCA feature reduction (not used currently) function

@torch.no_grad()
def pca_reduce(x: torch.Tensor) -> torch.Tensor:
    from sklearn.decomposition import PCA
    import numpy as np

    x_np = x.cpu().numpy()
    o_d = x_np.shape[1]
    pca = PCA(n_components=0.99999999,copy=False)  # retain 99.999999% variance
    x_reduced = pca.fit_transform(x_np)
    d = x_reduced.shape[1]
    print(f"PCA reduced from {o_d} to {d} dimensions")
    # explained_variance = np.sum(pca.explained_variance_ratio_)
    # print(f"PCA reduced to {d} dimensions, explained variance: {explained_variance:.4f}")
    return torch.tensor(x_reduced, device=x.device, dtype=x.dtype)



