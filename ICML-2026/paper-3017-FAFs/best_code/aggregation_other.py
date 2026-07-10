import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Set, Iterable, Dict
from torch_scatter import scatter
from torch_geometric.utils import degree


#### KA aggregation 
@torch.no_grad()
def ka_multihop_feats(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    K: int,
    argska: dict
) -> list[torch.Tensor]:
    """
    Multi-hop KA aggregation (not permutation invariant).
    Requires argska with keys:
        ka_order: 'by_src_asc' | 'by_src_desc' | 'as_is' | 'by_edge_id'
        ka_D_max: int | None
        ka_truncate: bool
        ka_pad_value: float
        ka_transform: 'identity' | 'sigmoid' | 'softsign' | 'log1p'
        ka_temperature: float > 0
        ka_n_bits: int >= 1

    Returns: list of length K, each element is [N, Fe] (hop-1 .. hop-K).
    """
    outs = []
    cur = x
    for _ in range(K):
        cur = _aggregate_ka_sequence_per_feature(cur, edge_index,argska)
        outs.append(cur)
    return outs

@torch.no_grad()
def _aggregate_ka_sequence_per_feature(x: torch.Tensor,
                                       edge_index: torch.Tensor,
                                       argska: dict,
                                       ka_feat_chunk: Optional[int] = None
                                       ) -> torch.Tensor:
    """
    One-hop KA aggregation per feature with low peak memory.
    Returns [N, Fe].
    """
    device, dtype = x.device, x.dtype
    N, Fe = x.shape

    pos = _ka_positions(edge_index, N, argska)
    if pos is None:
        D = int(argska["ka_D_max"] or 0)
        return torch.zeros(N, Fe, device=device, dtype=dtype)
    src_s, dst_s, idx_in_dst, D, _ = pos

    # Decide chunk size (safe default ≈ 32–128 feats per chunk).
    if ka_feat_chunk is None:
        ka_feat_chunk = 64

    out = torch.empty(N, Fe, device=device, dtype=dtype)

    for s in range(0, Fe, ka_feat_chunk):
        e = min(s + ka_feat_chunk, Fe)
        xch = x[:, s:e]                        # [N, Fch]
        Fch = xch.shape[1]

        # --- Build U for just this chunk ---
        U = xch.new_full((N, D, Fch), argska["ka_pad_value"])
        # fill positions (optionally skip mask when no truncation happened)
        if (idx_in_dst.max().item() + 1) <= D:
            U[dst_s, idx_in_dst] = xch[src_s]
        else:
            m = idx_in_dst < D
            if m.any():
                U[dst_s[m], idx_in_dst[m]] = xch[src_s[m]]

        # --- Transform IN-PLACE to avoid an extra [N,D,Fch] ---
        t = max(float(argska["ka_temperature"]), 1e-12)
        tr = argska["ka_transform"]

        if tr == "identity":
            pass
        elif tr == "sigmoid":
            U.div_(t).sigmoid_()
        elif tr == "softsign":
            U.div_(t)
            tmp = U.abs_().add_(1.0)  # abs_() + 1
            U.div_(tmp).mul_(0.5).add_(0.5)
        elif tr == "log1p":
            sgn = U.sign()
            U.abs_().div_(t).log1p_().mul_(sgn)
            zmin = U.amin(dim=1, keepdim=True)
            zmax = U.amax(dim=1, keepdim=True)
            U.sub_(zmin).div_(zmax.sub(zmin).add_(1e-12))
        else:
            raise ValueError(f"Unknown transform='{tr}'.")

        # --- Scalarize (avoid broadcasted temps) ---
        out[:, s:e] = _ka_scalarize_matrix(U, n_bits=argska["ka_n_bits"], acc_dtype=dtype)

        # free U early
        del U

    return out

@torch.no_grad()
def _ka_scalarize_matrix(
    U: torch.Tensor, *,
    n_bits: int = 16,
    acc_dtype: Optional[torch.dtype] = None,
    d_chunk: int = 128,
) -> torch.Tensor:
    """
    Memory-lean scalarization:
      - streams over D in blocks of size d_chunk
      - reuses a single [N, d_chunk, Fe] workspace per block
      - no broadcasted [N, D, Fe] multiplies
    """
    device = U.device
    N, D, Fe = U.shape
    if acc_dtype is None:
        acc_dtype = U.dtype

    u = U if U.dtype == acc_dtype else U.to(acc_dtype)

    # clamp to just-below 1 (to avoid 1.000... ambiguity), IN-PLACE
    one = torch.tensor(1.0, dtype=acc_dtype, device=device)
    eps1 = torch.nextafter(one, torch.tensor(0.0, dtype=acc_dtype, device=device))
    u.clamp_max_(eps1)

    total = torch.zeros(N, Fe, dtype=acc_dtype, device=device)
    p_idx = torch.arange(1, D + 1, dtype=acc_dtype, device=device)  # 1..D
    base_w = 2.0 * (3.0 ** (-p_idx))  # [D]
    ybuf = torch.empty((N, d_chunk, Fe), dtype=acc_dtype, device=device)

    for k in range(n_bits):
        bit_scale = 3.0 ** (-(k * D))
        for d0 in range(0, D, d_chunk):
            d1 = min(d0 + d_chunk, D)
            B = d1 - d0

            u_blk = u[:, d0:d1, :]                          # [N, B, Fe] (view)
            w_blk = (base_w[d0:d1] * bit_scale)             # [B]

            # y = floor(u*2), u = u*2 - y  (but done blockwise)
            u_blk.mul_(2.0)
            torch.floor(u_blk, out=ybuf[:, :B, :])          # ybuf: [N, B, Fe]
            u_blk.sub_(ybuf[:, :B, :])

            # accumulate: (N,Fe,B) @ (B,) -> (N,Fe)
            total.add_(torch.matmul(ybuf[:, :B, :].transpose(1, 2), w_blk))

    return total.to(U.dtype)

@torch.no_grad()
def _ka_positions(edge_index: torch.Tensor, N: int, argska: dict):
    src, dst = edge_index
    E = src.numel()
    if E == 0:
        return None  # signals empty

    order = argska["ka_order"]
    if order in ("as_is", "by_edge_id"):
        perm = dst.argsort(stable=True)
    elif order in ("by_src_asc", "by_src_desc"):
        Np = int(N) + 1
        sec = src if order == "by_src_asc" else (Np - 1 - src)
        idx = torch.arange(E, device=src.device)
        idx = idx[sec.argsort(stable=True)]
        perm = idx[dst[idx].argsort(stable=True)]
    else:
        raise ValueError(f"Unknown order={order}")

    src_s, dst_s = src[perm], dst[perm]
    deg = torch.bincount(dst, minlength=N)
    ends = deg.cumsum(0)
    starts = ends - deg
    idx_in_dst = torch.arange(E, device=src.device) - starts[dst_s]

    obs_max_deg = int(deg.max().item()) if N > 0 else 0
    if argska["ka_D_max"] is not None:
        D = int(argska["ka_D_max"])
        if not argska["ka_truncate"] and obs_max_deg > D:
            raise ValueError(f"Found degree {obs_max_deg} > D_max={argska['ka_D_max']} with truncate=False.")
        D = min(D, obs_max_deg) if argska["ka_truncate"] else D
    else:
        D = obs_max_deg

    return (src_s, dst_s, idx_in_dst, D, obs_max_deg)


#### Binned aggregation (permutation invariant)
@torch.no_grad()
def binned_multihop_feats(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    K: int,
    argsbin: dict,
) -> list[torch.Tensor]:
    """
    Multi-hop binned aggregation (permutation invariant), no chunking.

    Returns:
        list[Tensor] of length K
        hop h (1-based) has shape [N, Fe * (n_bins ** h)]
    """
    outs = []
    cur = x
    for _ in range(K):
        cur = _binned_aggregate_onehop(cur, edge_index, argsbin)
        outs.append(cur)
    return outs


@torch.no_grad()
def _binned_aggregate_onehop(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    argsbin: dict,
) -> torch.Tensor:
    """
    One-hop binned aggregation (permutation invariant), no chunking.
    For each node v and feature f:
      - Assign each neighbor's value x[u,f] to a bin (global per-feature edges).
      - One-hot the bin id and sum over neighbors.

    Returns:
      [N, Fe * n_bins]
    """
    device = x.device
    N, Fe = x.shape
    src, dst = edge_index
    E = src.numel()
    n_bins = int(argsbin["bin_num"])
    dtype_out = x.dtype

    # Resolve edges for the *current* feature dimension.
    bins_obj = argsbin.get("bin_edges", None)
    if bins_obj is None:
        edges = _compute_bin_edges_global(x, n_bins, prefer_rtdl=True)
    else:
        try:
            edges = _coerce_bin_edges(bins_obj, Fe=Fe, n_bins=n_bins, device=device, dtype=x.dtype)
        except Exception:
            # If provided edges don't match current Fe, recompute from x.
            edges = _compute_bin_edges_global(x, n_bins, prefer_rtdl=True)

    if E == 0:
        out = torch.zeros(N, Fe, n_bins, device=device, dtype=dtype_out)
        return out.reshape(N, Fe * n_bins)

    hist = torch.zeros(N, Fe, n_bins, device=device, dtype=dtype_out)
    x_src = x[src]  # [E, Fe]
    
    if argsbin.get("bin_cdf", False):
        bins_row = torch.arange(n_bins, device=x_src.device).view(1, -1)
        row_ix = torch.arange(E, device=x_src.device)

    for j in range(Fe):
        if argsbin.get("bin_cdf", False):
            b = edges[j]                               # [n_bins+1], ascending
            x = x_src[:, j]                            # [E]
            idx = torch.bucketize(x, b[1:-1], right=False)        # [E] in [0, n_bins-1]
            left, right = b[idx], b[idx + 1]
            frac = ((x - left) / (right - left)).clamp_(0, 1)     # [E]

            # CDF-style encoding: 1 before idx, frac at idx, 0 after
            enc = (bins_row < idx[:, None]).to(dtype_out)         # [E, n_bins]
            enc[row_ix, idx] = frac                               # set the hit bin value
            hist[:, j] = scatter(enc, dst, dim=0, dim_size=N, reduce='mean')
        else:
            idx = torch.bucketize(x_src[:, j], edges[j], right=False)
            idx = idx.clamp_(0, n_bins - 1)
            enc = F.one_hot(idx.to(torch.int64), num_classes=n_bins).to(dtype_out)
            hist[:, j] = scatter(enc, dst, dim=0, dim_size=N, reduce='sum')

    # Zero out isolated nodes
    ones = torch.ones(E, 1, device=device, dtype=x.dtype)
    deg = scatter(ones, dst, dim=0, dim_size=N, reduce='sum')  # [N,1]
    zero_nodes = (deg.squeeze(-1) == 0)
    if zero_nodes.any():
        hist[zero_nodes] = 0

    return hist.reshape(N, Fe * n_bins)

@torch.no_grad()
def _coerce_bin_edges(bins_obj, *, Fe: int, n_bins: int, device, dtype) -> torch.Tensor:
    """
    Try to extract per-feature bin edges of shape [Fe, n_bins-1] from various objects:
    - torch.Tensor of shape [Fe, n_bins-1] or [n_bins-1, Fe]
    - dict / object with attributes like 'bin_edges', 'edges', 'boundaries'
    """
    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)

    cand = None
    if isinstance(bins_obj, torch.Tensor):
        cand = bins_obj
    elif isinstance(bins_obj, dict):
        for k in ('bin_edges', 'edges', 'boundaries', 'cuts'):
            if k in bins_obj:
                cand = _to_tensor(bins_obj[k]); break
    else:
        # try common attribute names
        for k in ('bin_edges', 'edges', 'boundaries', 'cuts'):
            if hasattr(bins_obj, k):
                cand = _to_tensor(getattr(bins_obj, k)); break

    if cand is None:
        raise ValueError("Could not extract bin edges from bins_obj.")

    if cand.dim() != 2:
        raise ValueError(f"Bin edges must be 2D; got shape {tuple(cand.shape)}.")
    if cand.shape == (Fe, n_bins - 1):
        edges = cand
    elif cand.shape == (n_bins - 1, Fe):
        edges = cand.transpose(0, 1)
    else:
        raise ValueError(f"Unexpected bin edges shape {tuple(cand.shape)}; "
                         f"want [Fe,{n_bins-1}] or [{n_bins-1},Fe].")

    # Ensure ascending order per feature (bucketize requirement)
    edges, _ = torch.sort(edges, dim=1)
    return edges


@torch.no_grad()
def _compute_bin_edges_global(
    x: torch.Tensor,
    n_bins: int,
    prefer_rtdl: bool = True,
) -> torch.Tensor:
    """
    Global per-feature bin edges (same bins used for all nodes).
    Tries rtdl-num-embeddings; falls back to global quantiles.
    Returns: edges [Fe, n_bins-1], ascending per row.
    """
    device, dtype = x.device, x.dtype
    Fe = x.size(1)
    if prefer_rtdl:
        try:
            import rtdl_num_embeddings as embs  # PyPI: rtdl-num-embeddings
            bins_obj = embs.compute_bins(x, n_bins)  # API: returns structure with edges
            return _coerce_bin_edges(bins_obj, Fe=Fe, n_bins=n_bins, device=device, dtype=dtype)
        except Exception:
            print("Warning: rtdl-num-embeddings not available or failed; falling back to quantiles.")
            pass  # fall back to quantiles

    # Fallback: global quantile cuts per feature (ignore NaNs)
    qs = torch.linspace(0, 1, n_bins + 1, device=device, dtype=torch.float64)[1:-1]  # (n_bins-1,)
    # Handle NaNs by temporarily masking
    X = x.to(torch.float64)
    X_mask = torch.isfinite(X)
    edges = torch.empty(Fe, n_bins - 1, device=device, dtype=torch.float64)
    for j in range(Fe):
        col = X[:, j]
        msk = X_mask[:, j]
        if msk.any():
            edges[j] = torch.quantile(col[msk], qs, interpolation='linear')
        else:
            edges[j] = torch.linspace(0, 1, n_bins + 1, device=device, dtype=torch.float64)[1:-1]
    edges = edges.to(dtype)
    edges, _ = torch.sort(edges, dim=1)
    return edges


### Similarity-based aggregation

@torch.no_grad()
def sim_multihop_feats(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    K: int,
    argsim: Dict
) -> List[torch.Tensor]:
    """
    Fixed-weight multi-hop stack using the same per-edge weights at every hop.
    Returns [H1, H2, ..., HK], each [N, D] where D = selected feature dim.

    argsim:
        sim_mode: 'cosine' | 'dot' | 'rbf'
        sim_slice: None | slice
        sim_clamp_negatives: bool (optional)
        sim_clamp_positives: bool (optional)
        sim_normalize: 'softmax' | 'l1' | 'none' (default 'none')
        sim_temperature: float > 0 (default 1.0; softmax only)
        sim_eps: float > 0 (default 1e-9)
        sim_sigma: float > 0 (rbf bandwidth; default sqrt(D))
        sim_type: scatter reduce op, e.g. 'mean' | 'sum' (default 'mean')
    """
    N = x.size(0)
    src, dst = edge_index
    x_sim = x if argsim.get("sim_slice") is None else x[:, argsim["sim_slice"]]

    w_edge = _edge_similarity_weights(x_sim, edge_index, argsim)
    reduce = argsim.get("sim_type", "mean")

    outs: List[torch.Tensor] = []
    h = x_sim
    for _ in range(K):
        h = scatter(w_edge.unsqueeze(-1) * h[src],
                    dst, dim=0, dim_size=N, reduce=reduce)
        outs.append(h)
    return outs


@torch.no_grad()
def rew_multihop_feats(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    K: int,
    argsim: Dict
) -> List[torch.Tensor]:
    """
    Rewire-based multi-hop: keep edges with positive raw similarity only (unit weight).
    Returns [H1, ..., HK], each [N, D] where D = selected feature dim.
    """
    N = x.size(0)
    src, dst = edge_index
    x_sim = x if argsim.get("sim_slice") is None else x[:, argsim["sim_slice"]]

    sim_raw = _edge_similarity_raw(x_sim, edge_index, argsim)
    w_edge = (sim_raw > 0).to(x.dtype)

    reduce = argsim.get("sim_type", "mean")
    outs: List[torch.Tensor] = []
    h = x_sim
    for _ in range(K):
        h = scatter(w_edge.unsqueeze(-1) * h[src],
                    dst, dim=0, dim_size=N, reduce=reduce)
        outs.append(h)
    return outs


@torch.no_grad()
def split_multihop_feats(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    K: int,
    argsim: Dict,
) -> List[torch.Tensor]:
    """
    Split positive/negative channels. Returns [H1, ..., HK], each [N, 2*D].
    First D dims aggregate over positively similar edges; second D over negatively similar edges.
    """
    N = x.size(0)
    src, dst = edge_index
    x_sel = x if argsim.get("sim_slice") is None else x[:, argsim["sim_slice"]]
    D = x_sel.size(1)

    # start from [x | x]
    h = torch.cat([x_sel, x_sel], dim=1)

    sim_raw = _edge_similarity_raw(x_sel, edge_index, argsim)
    w_pos = (sim_raw > 0).to(x.dtype)
    w_neg = (sim_raw < 0).to(x.dtype)

    reduce = argsim.get("sim_type", "mean")
    outs: List[torch.Tensor] = []
    for _ in range(K):
        h_pos_prev = h[:, :D]
        h_neg_prev = h[:, D:]

        h_pos = scatter(w_pos.unsqueeze(-1) * h_pos_prev[src],
                        dst, dim=0, dim_size=N, reduce=reduce)
        h_neg = scatter(w_neg.unsqueeze(-1) * h_neg_prev[src],
                        dst, dim=0, dim_size=N, reduce=reduce)
        h = torch.cat([h_pos, h_neg], dim=1)
        outs.append(h)
    return outs



@torch.no_grad()
def _edge_similarity_raw(
    x_sim: torch.Tensor,
    edge_index: torch.Tensor,
    argsim: Dict
) -> torch.Tensor:
    """RAW per-edge similarity (no clamping/normalization)."""
    src, dst = edge_index
    eps = float(argsim.get("sim_eps", 1e-9))
    mode = argsim["sim_mode"]

    if mode == "cosine":
        a = F.normalize(x_sim[src], p=2, dim=1, eps=eps)
        b = F.normalize(x_sim[dst], p=2, dim=1, eps=eps)
        sim = (a * b).sum(dim=1)
    elif mode == "dot":
        sim = (x_sim[src] * x_sim[dst]).sum(dim=1)
    elif mode == "rbf":
        sigma = float(argsim.get("sim_sigma", x_sim.size(1) ** 0.5))
        diff = x_sim[src] - x_sim[dst]
        dist2 = (diff * diff).sum(dim=1)
        sim = torch.exp(-dist2 / (2.0 * (sigma ** 2)))
    else:
        raise ValueError(f"Unknown sim_mode: {mode}")
    return sim


@torch.no_grad()
def _edge_similarity_weights(
    x_sim: torch.Tensor,
    edge_index: torch.Tensor,
    argsim: Dict
) -> torch.Tensor:
    """
    Turn raw similarities into nonnegative per-edge weights, optionally
    normalized per destination node.
    """
    src, dst = edge_index
    N = x_sim.size(0)
    eps = float(argsim.get("sim_eps", 1e-9))
    normalize = argsim.get("sim_normalize", "none")
    mode = argsim["sim_mode"]

    sim = _edge_similarity_raw(x_sim, edge_index, argsim)

    # Optional sign handling (not applied to RBF which is already ≥0)
    if mode != "rbf":
        if argsim.get("sim_clamp_positives", False):
            sim = torch.where(sim < 0, -sim, torch.zeros_like(sim))
        elif argsim.get("sim_clamp_negatives", False):
            sim = sim.clamp_min(0.0)

    if normalize == "softmax":
        temp = float(max(argsim.get("sim_temperature", 1.0), eps))
        z = sim / temp
        max_per_dst = scatter(z, dst, dim=0, dim_size=N, reduce="max")
        z = z - max_per_dst[dst]
        ez = torch.exp(z)
        denom = scatter(ez, dst, dim=0, dim_size=N, reduce="sum").clamp_min_(eps)
        w = ez / denom[dst]

    elif normalize == "l1":
        if mode == "rbf":
            num = sim
        else:
            if argsim.get("sim_clamp_positives", False) or argsim.get("sim_clamp_negatives", False):
                num = sim.clamp_min(0.0)
            else:
                num = sim.abs()
        denom = scatter(num, dst, dim=0, dim_size=N, reduce="sum").clamp_min_(eps)
        w = num / denom[dst]

    elif normalize == "none":
        w = sim if mode == "rbf" else sim.clamp_min(0.0)
    else:
        raise ValueError(f"Unknown sim_normalize: {normalize}")

    return w  # [E]


### Network-science features (degree/log_degree/clustering/centralities)

@torch.no_grad()
def network_science_feats(
    x : Optional[torch.Tensor],
    edge_index: torch.Tensor,
    argsns: dict
) -> Tuple[torch.Tensor, List[str]]:
    """
    Compute common network-science node features from a (possibly directed) graph.

    Returns:
        H: [N, D] stacked feature matrix

    Requires argsns with keys:
        ns_include: Iterable of str (see below)
        ns_cc_k: int >= 1 (for closeness; number of source nodes to sample)
        ns_ev_max_iter: int >= 1 (for eigenvector centrality)
        ns_ev_tol: float > 0 (for eigenvector centrality)
        ns_betweenness_cpu: bool (if True, use NetworkX on CPU for betweenness; else zeros)
        ns_bc_k: int >= 1 (for betweenness; number of source nodes to sample if not using CPU)
        

    include may contain any of:
        'degree', 'log_degree', 'clustering', 'closeness', 'eigenvector', 'betweenness'

    Notes:
      • Graph is treated as undirected for clustering & centralities (same as source code).
      • Betweenness uses NetworkX on CPU when argsns["ns_betweenness_cpu"]=True; otherwise zeros.
      • Closeness here is *harmonic* closeness estimated by sampling up to cc_k sources.
    """
    argsns["ns_include"] = set(argsns["ns_include"])
    src, dst = edge_index
    N = int(torch.max(edge_index).item()) + 1 if edge_index.numel() > 0 else 0

    dtype = torch.float32
    device = edge_index.device

    outs: List[torch.Tensor] = []
    names: List[str] = []

    # ---- degree / log_degree (in-degree on directed; degree on undirected) ----
    if 'degree' in argsns["ns_include"] or 'log_degree' in argsns["ns_include"]:
        deg = degree(src, N, dtype=dtype).to(device).unsqueeze(-1)  # [N,1]
        if 'degree' in argsns["ns_include"]:
            outs.append(deg); names.append('degree')
        if 'log_degree' in argsns["ns_include"]:
            outs.append(deg.clamp_min(1).log()); names.append('log_degree')

    # ---- build undirected sparse adjacency (values=1) for the rest ----
    vals = torch.ones(src.numel(), device=device, dtype=dtype)
    A = torch.sparse_coo_tensor(torch.stack([src, dst], 0), vals, (N, N), device=device, dtype=dtype)
    AT = torch.sparse_coo_tensor(torch.stack([dst, src], 0), vals, (N, N), device=device, dtype=dtype)
    A = (A + AT).coalesce()

    # ---- clustering coefficient (triangle-based, dense path like original) ----
    if 'clustering' in argsns["ns_include"]:
        # Convert to dense (same approach as your original; OK for mid-sized graphs)
        adj = A.to_dense()
        d   = adj.sum(1)
        d1  = d.clamp_min(1)
        A2  = adj @ adj
        tri = (A2 @ adj).diagonal()                        # 2-paths that close into triangles
        cl  = (tri / (d1 * (d1 - 1))).nan_to_num(0.0).unsqueeze(-1)
        outs.append(cl.to(dtype)); names.append('clustering')

    # ---- centralities ----
    need_cents: Set[str] = argsns["ns_include"] & {'closeness','eigenvector','betweenness'}
    if need_cents:
        # closeness (harmonic, sampled BFS)
        if 'closeness' in need_cents:
            # BFS using boolean frontiers on the undirected graph
            harmonic = torch.zeros(N, device=device, dtype=torch.float32)
            # sample up to cc_k distinct sources
            if N <= argsns["ns_cc_k"]:
                sources = torch.arange(N, device=device)
            else:
                sources = torch.randperm(N, device=device)[:argsns["ns_cc_k"]]
            # prepare masks once
            for s in sources.tolist():
                dist = torch.full((N,), -1, device=device, dtype=torch.int32)
                frontier = torch.zeros(N, device=device, dtype=torch.bool)
                visited  = torch.zeros(N, device=device, dtype=torch.bool)
                frontier[s] = True; visited[s] = True; dist[s] = 0
                dstep = 0
                # using dense adjacency multiply like original code for speed/simplicity
                adj = A.to_dense()  # (cache per call)
                while frontier.any():
                    neigh = (torch.mv(adj, frontier.float()) > 0)
                    new_frontier = neigh & (~visited)
                    dstep += 1
                    if new_frontier.any():
                        dist[new_frontier] = dstep
                    visited |= new_frontier
                    frontier = new_frontier
                mask = dist > 0
                harmonic[mask] += 1.0 / dist[mask].to(torch.float32)
            harmonic = harmonic / max(1, sources.numel())
            outs.append(harmonic.to(dtype).unsqueeze(-1)); names.append('closeness')

        # eigenvector centrality (power iteration)
        if 'eigenvector' in need_cents:
            v = torch.rand(N, 1, device=device, dtype=dtype)
            v = v / (v.norm() + 1e-12)
            last_lambda = None
            for _ in range(argsns["ns_ev_max_iter"]):
                Av  = torch.sparse.mm(A, v)
                nrm = Av.norm() + 1e-12
                v_new = Av / nrm
                Avn = torch.sparse.mm(A, v_new)
                lam = float((v_new.t() @ Avn).item())
                if last_lambda is not None and abs(lam - last_lambda) < argsns["ns_ev_tol"] * max(1.0, abs(last_lambda)):
                    v = v_new
                    break
                v, last_lambda = v_new, lam
            outs.append(v); names.append('eigenvector')

        # betweenness (NetworkX on CPU, optional)
        if 'betweenness' in need_cents:
            if argsns["ns_betweenness_cpu"]:
                try:
                    import networkx as nx  # type: ignore
                    # Build undirected graph for NetworkX
                    Gx = nx.Graph(); Gx.add_nodes_from(range(N))
                    # move to CPU ints to avoid large transfers
                    s_cpu = src.detach().cpu().numpy().tolist()
                    d_cpu = dst.detach().cpu().numpy().tolist()
                    Gx.add_edges_from(zip(s_cpu, d_cpu))
                    k = int(min(argsns["ns_bc_k"], N)) if argsns["ns_bc_k"] is not None else None
                    bc = nx.betweenness_centrality(Gx, k=k, normalized=True, seed=42)
                    vec = torch.tensor([bc[v] for v in range(N)], device=device, dtype=dtype).unsqueeze(-1)
                except Exception:
                    vec = torch.zeros(N, 1, device=device, dtype=dtype)
            else:
                vec = torch.zeros(N, 1, device=device, dtype=dtype)
            outs.append(vec); names.append('betweenness')

    if outs:
        H = torch.cat(outs, dim=-1)
    else:
        H = torch.zeros(N, 0, device=device, dtype=dtype)

    return H


# ============================================================
# Neighbor quantiles per feature (25/50/75 like in FAF extras)
# ============================================================

@torch.no_grad()
def neighbor_quantiles(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    argsq: dict
) -> Tuple[torch.Tensor, List[str]]:
    """
    Per-node, per-feature neighbor quantiles.

    Args:
      x:            [N, Fe] node features.
      edge_index:   [2, E] (src, dst). For each node v, aggregates x[u] for u -> v.
      include:      subset of {'quantile_25','quantile_50','quantile_75'}.
      interpolation: passed to torch.quantile.

    Returns:
      Q: [N, Fe * len(include)] stacked quantile features (order respects 'include')
      names: ["q25:f0", ..., "q50:f0", ..., "q75:f(Fe-1)"]
    """
    device, dtype = x.device, x.dtype
    N, Fe = x.shape
    src, dst = edge_index
    tag_to_q = {"quantile_25": 0.25, "quantile_50": 0.50, "quantile_75": 0.75}
    wanted = [(tag_to_q[t], t) for t in list(argsq["q_include"])if t in tag_to_q]
    if not wanted:
        return torch.zeros(N, 0, device=device, dtype=dtype)

    E = src.numel()
    deg_dst = torch.bincount(dst, minlength=N)

    # Quick exits
    if E == 0 or (deg_dst == 0).all():
        # All zeros if no incoming edges
        zeros = torch.zeros(N, Fe * len(wanted), device=device, dtype=dtype)
        names = [f"{t}:f{j}" for t in [w[1] for w in wanted] for j in range(Fe)]
        return zeros

    # Sort by dst to build contiguous segments per destination
    order = dst.argsort()
    dst_s = dst[order]
    X_s   = x[src[order]]  # [E, Fe]
    ends   = deg_dst.cumsum(0)
    starts = ends - deg_dst

    qs_tensor = torch.tensor([q for q, _ in wanted], device=device, dtype=torch.float32)
    out_tensors = [torch.zeros(N, Fe, device=device, dtype=dtype) for _ in wanted]

    # Only iterate nodes that have neighbors
    nz_nodes = (deg_dst > 0).nonzero(as_tuple=False).flatten()
    for v in nz_nodes.tolist():
        s = int(starts[v].item()); e = int(ends[v].item())
        seg = X_s[s:e]  # [deg(v), Fe]
        if seg.size(0) == 1:
            # All desired quantiles equal to the single value
            for i in range(len(wanted)):
                out_tensors[i][v] = seg[0]
        else:
            vals = torch.quantile(seg, qs_tensor, dim=0, interpolation=argsq["q_interpolation"])  # [Q, Fe]
            for i in range(len(wanted)):
                out_tensors[i][v] = vals[i]

    # Stack in the same order as q_include
    Q = torch.cat(out_tensors, dim=-1)  # [N, Fe * Q]
    names = [f"{tag}:f{j}" for _, tag in wanted for j in range(Fe)]
    return Q

# all neighbor feature concated as aggregation, in multihop fashion
@torch.no_grad()
def exp_multihop_feats(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    K: int,
) -> list[torch.Tensor]:
    """
    Multi-hop all-neighbor feature concatenation (order-sensitive).
    Returns: list of length K with shapes:
        hop 1: [N, Fe * D]
        hop 2: [N, Fe * D^2]
        ...
        hop k: [N, Fe * D^k]
    where D = max in-degree in the (directed) graph.
    """
    outs = []
    cur = x
    for k in range(K):
        print(f"Adding hop {k+1} all-neighbor features")
        cur = _aggregate_exp_sequence_per_feature(cur, edge_index)
        outs.append(cur)
        print(f"  -> {k}th shape {tuple(cur.shape)}")
    return outs


@torch.no_grad()
def _aggregate_exp_sequence_per_feature(
    x: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    One-hop all-neighbor feature aggregation per feature with low peak memory.
    Order-sensitive: neighbors are placed in their original (stable) edge order.
    Returns [N, Fe * D], where D is the max in-degree.
    """
    device, dtype = x.device, x.dtype
    N, Fe = x.shape

    # Ensure index tensors are on the same device as x for advanced indexing
    src = edge_index[0].to(device)
    dst = edge_index[1].to(device)

    E = src.numel()

    # Degrees and block starts (independent of edge order)
    deg = torch.bincount(dst, minlength=N)
    D = int(deg.max().item()) if N > 0 else 0

    # Allocate output; if D==0 this is [N, 0] which is consistent and safe
    out = x.new_zeros((N, Fe * D))

    if E == 0 or D == 0:
        return out  # [N, 0]

    ends = deg.cumsum(0)
    starts = ends - deg  # [N]

    # Compute stable rank of each edge within its dst group
    # rank[e] = position of edge e in the list sorted by dst (stable)
    perm = dst.argsort(stable=True)
    rank = torch.empty_like(perm)
    rank[perm] = torch.arange(E, device=device)
    idx_in_dst = rank - starts[dst]  # in [0, deg[dst)-1] ⊆ [0, D-1]

    # Fill per-feature blocks
    for j in range(Fe):
        xj = x[:, j]                     # [N]
        U = xj.new_zeros((N, D))         # [N, D]
        # Advanced indexing write
        U.index_put_((dst, idx_in_dst), xj[src], accumulate=False)
        out[:, j * D : (j + 1) * D] = U
        del U  # free early

    return out

import torch

# all neighbor feature concatenation as aggregation, with degree-mean carryover between hops
import torch

@torch.no_grad()
def all_multihop_feats(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    K: int,
) -> list[torch.Tensor]:
    """
    Multi-hop all-neighbor feature aggregation (order-sensitive).
    Each hop output: [N, Fe * D] where D = max in-degree of this hop.
    Next-hop input = per-feature mean over true degree (not padded zeros): [N, Fe].
    """
    outs = []
    device, dtype = x.device, x.dtype
    N, Fe = x.shape
    # normalize edge_index to correct device/dtype once
    src0 = edge_index[0].to(device=device, dtype=torch.long)
    dst0 = edge_index[1].to(device=device, dtype=torch.long)

    # quick validity checks (catch data issues early, on CPU side)
    assert N >= 0 and Fe >= 0
    if src0.numel() > 0:
        assert int(src0.min()) >= 0 and int(dst0.min()) >= 0, "edge_index has negative node ids"
        assert int(src0.max()) < N and int(dst0.max()) < N, f"edge_index has ids >= N ({N})"

    cur = x  # [N, Fe]
    for hop in range(K):
        print(f"Adding hop {hop+1} all-neighbor features")
        agg, deg, D = _aggregate_all_sequence_per_feature(cur, src0, dst0)  # [N, Fe*D]
        outs.append(agg)

        # Prepare next-hop input: degree-correct mean (avoid padding bias)
        if D == 0:
            cur = torch.zeros(N, Fe, device=device, dtype=dtype)
        else:
            cur = agg.reshape(N, Fe, D).sum(-1) / deg.clamp_min(1).to(dtype).unsqueeze(1)

    return outs


@torch.no_grad()
def _aggregate_all_sequence_per_feature(
    x: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
):
    """
    One-hop all neighbor *sequence* aggregation per feature with low peak memory.
    Returns (out, deg, D) where:
      - out: [N, Fe * D]
      - deg: [N] in-degree per node
      - D:   int, max in-degree
    """
    device, dtype = x.device, x.dtype
    N, Fe = x.shape
    E = src.numel()

    if E == 0 or N == 0 or Fe == 0:
        deg = x.new_zeros(N, dtype=torch.long)
        return x.new_zeros((N, 0)), deg, 0

    # ---- crucial: group edges by destination (stable to preserve intra-node order) ----
    # If already sorted, this is a no-op.
    if not torch.all(dst[:-1] <= dst[1:]):
        perm = dst.argsort(stable=True)
        src = src[perm]
        dst = dst[perm]

    # degrees and offsets
    deg = torch.bincount(dst, minlength=N)                  # [N]
    D = int(deg.max().item())
    if D == 0:
        return x.new_zeros((N, 0)), deg, 0

    starts = deg.cumsum(0) - deg                            # [N]
    arng = torch.arange(E, device=device, dtype=torch.long) # [E]
    idx_in_dst = arng - starts[dst]                         # [E] in [0, deg[v)-1]

    # extra safety checks (sync once, worth it to avoid silent GPU asserts)
    imin = int(idx_in_dst.min().item())
    imax = int(idx_in_dst.max().item())
    assert 0 <= imin and imax < D, f"idx_in_dst out of bounds: [{imin}, {imax}] vs D={D}"

    out = x.new_zeros((N, Fe * D))
    for j in range(Fe):
        U = x.new_zeros((N, D))
        # write each edge's source feature into its slot for its destination row
        U[dst, idx_in_dst] = x[:, j][src]
        out[:, j*D:(j+1)*D] = U
        del U

    return out, deg, D


import torch

@torch.no_grad()
def meansum_multihop_feats(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    K: int,
) -> list[torch.Tensor]:
    """
    Multi-hop all-neighbor feature aggregation (order-sensitive).
    Each hop output: [N, Fe_cur * D] where D = max in-degree of this hop,
    Fe_cur is the *current* feature count (doubles each hop with mean+sum carryover).
    Next-hop input = concat([mean, sum]) over true degree: [N, 2*Fe_cur].
    """
    outs = []
    device, dtype = x.device, x.dtype
    N, Fe0 = x.shape

    # normalize edge_index to correct device/dtype once
    src0 = edge_index[0].to(device=device, dtype=torch.long)
    dst0 = edge_index[1].to(device=device, dtype=torch.long)

    # quick validity checks to catch data issues early
    if src0.numel() > 0:
        assert int(src0.min()) >= 0 and int(dst0.min()) >= 0, "edge_index has negative node ids"
        assert int(src0.max()) < N and int(dst0.max()) < N, f"edge_index has ids >= N ({N})"

    cur = x  # [N, Fe_cur]
    for hop in range(K):
        print(f"Adding hop {hop+1} meansum-neighbor features")
        agg, deg, D = _aggregate_meansum_sequence_per_feature(cur, src0, dst0)  # agg: [N, Fe_cur * D]
        outs.append(agg)

        # Prepare next-hop input: per-feature MEAN and SUM (concat)
        Fe_cur = cur.shape[1]
        if D == 0:
            # no edges — propagate zeros (feature count doubles)
            cur = torch.zeros(N, 2 * Fe_cur, device=device, dtype=dtype)
        else:
            agg_3d = agg.reshape(N, Fe_cur, D)                 # [N, Fe_cur, D]
            sums = agg_3d.sum(dim=-1)                          # [N, Fe_cur]
            means = sums / deg.clamp_min(1).to(dtype).unsqueeze(1)  # [N, Fe_cur]
            cur = torch.cat([means, sums], dim=1)              # [N, 2*Fe_cur]

    return outs


@torch.no_grad()
def _aggregate_meansum_sequence_per_feature(
    x: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
):
    """
    One-hop all neighbor *sequence* aggregation per feature with low peak memory.
    Returns (out, deg, D) where:
      - out: [N, Fe * D]
      - deg: [N] in-degree per node
      - D:   int, max in-degree
    CUDA-safe: stable-sorts by dst so idx_in_dst ∈ [0, deg[v)-1].
    """
    device, dtype = x.device, x.dtype
    N, Fe = x.shape
    E = src.numel()

    if E == 0 or N == 0 or Fe == 0:
        deg = x.new_zeros(N, dtype=torch.long)
        return x.new_zeros((N, 0)), deg, 0

    # ---- group edges by destination (stable to preserve intra-node order) ----
    if not torch.all(dst[:-1] <= dst[1:]):
        perm = dst.argsort(stable=True)
        src = src[perm]
        dst = dst[perm]

    # degrees and offsets
    deg = torch.bincount(dst, minlength=N)                  # [N]
    D = int(deg.max().item())
    if D == 0:
        return x.new_zeros((N, 0)), deg, 0

    starts = deg.cumsum(0) - deg                            # [N]
    arng = torch.arange(E, device=device, dtype=torch.long) # [E]
    idx_in_dst = arng - starts[dst]                         # [E] in [0, deg[v)-1]

    # safety checks (synchronize once to avoid silent GPU asserts)
    imin = int(idx_in_dst.min().item())
    imax = int(idx_in_dst.max().item())
    assert 0 <= imin and imax < D, f"idx_in_dst out of bounds: [{imin}, {imax}] vs D={D}"

    out = x.new_zeros((N, Fe * D))
    for j in range(Fe):
        U = x.new_zeros((N, D))
        U[dst, idx_in_dst] = x[:, j][src]                  # place each edge's source feature
        out[:, j*D:(j+1)*D] = U
        del U

    return out, deg, D


import torch

@torch.no_grad()
def mmask_multihop_feats(
    x: torch.Tensor,           # original features [N, Fe0]
    edge_index: torch.Tensor,  # [2, E]
    K: int,
) -> list[torch.Tensor]:
    """
    Hop 1:
      • Filter edges to neighbors with original x[u,0] == 0
      • Aggregate only features 1: (skip channel 0)
    Hops 2..K:
      • Use full (unfiltered) edges
      • Aggregate all current features
    Padding in the per-hop concatenation is -1 for slots beyond a node's degree.
    Next-hop input after each hop = concat([mean, sum]) over TRUE neighbors
    (padded slots are ignored via a mask).
    Returns: list of hop outputs; hop k has shape [N, Fe_used_k * D_k].
    """
    outs = []
    device, dtype = x.device, x.dtype
    N, Fe0 = x.shape
    if N == 0 or Fe0 == 0:
        return [x.new_zeros((0, 0)) for _ in range(K)]

    # normalize edge_index
    src0 = edge_index[0].to(device=device, dtype=torch.long)
    dst0 = edge_index[1].to(device=device, dtype=torch.long)
    if src0.numel() > 0:
        assert int(src0.min()) >= 0 and int(dst0.min()) >= 0, "edge_index has negative ids"
        assert int(src0.max()) < N and int(dst0.max()) < N, f"edge_index has ids >= N ({N})"

    # full edges (sorted by dst once)
    src_full, dst_full = src0, dst0
    if src_full.numel() > 1 and not torch.all(dst_full[:-1] <= dst_full[1:]):
        perm = dst_full.argsort(stable=True)
        src_full, dst_full = src_full[perm], dst_full[perm]

    # hop-1 filtered edges: keep only neighbors whose ORIGINAL first feature == 0
    mask = x[src0, 0].eq(0)  # use a tolerance if needed for floats
    src_f, dst_f = src0[mask], dst0[mask]
    if src_f.numel() > 1 and not torch.all(dst_f[:-1] <= dst_f[1:]):
        perm = dst_f.argsort(stable=True)
        src_f, dst_f = src_f[perm], dst_f[perm]

    cur = x  # features carried between hops
    for hop in range(K):
        if hop == 0:
            # first hop: filtered edges, skip original channel
            start_feature = 1
            src_use, dst_use = src_f, dst_f
            print(f"Hop 1: filtered edges = {src_use.numel()}, aggregating features 1:")
        else:
            # later hops: full edges, use all current features
            start_feature = 0
            src_use, dst_use = src_full, dst_full
            print(f"Hop {hop+1}: full edges = {src_use.numel()}, aggregating all features")

        agg, deg, D = _aggregate_seq_per_feature_pad_neg1(cur, src_use, dst_use, start_feature)  # [N, Fe_used*D]
        outs.append(agg)

        # Next-hop input: concat([mean, sum]) over TRUE neighbors only (ignore -1 padding)
        Fe_used = cur.shape[1] - start_feature
        if D == 0 or Fe_used == 0:
            cur = torch.zeros(N, 2 * Fe_used, device=device, dtype=dtype)
        else:
            agg_3d = agg.reshape(N, Fe_used, D)                    # [N, Fe_used, D]
            # mask: for each node v, first deg[v] positions are True, others False
            mask_nd = (torch.arange(D, device=device).unsqueeze(0) < deg.unsqueeze(1)).unsqueeze(1)  # [N,1,D]
            sums = (agg_3d * mask_nd.to(dtype)).sum(dim=-1)        # [N, Fe_used], padded slots ignored
            means = sums / deg.clamp_min(1).to(dtype).unsqueeze(1) # [N, Fe_used]
            cur = torch.cat([means, sums], dim=1)                  # [N, 2*Fe_used]

    return outs


@torch.no_grad()
def _aggregate_seq_per_feature_pad_neg1(
    x: torch.Tensor,           # current features [N, Fe_cur]
    src: torch.Tensor,         # edges' src (filtered for hop1, full for later)
    dst: torch.Tensor,         # edges' dst
    start_feature: int = 0,    # 0 = include all; 1 = skip first channel
):
    """
    Order-sensitive per-feature neighbor sequence concatenation over the provided edges.
    Padding value is -1 for positions beyond each node's degree.
    Returns (out, deg, D) where:
      out: [N, Fe_used * D], padded with -1
      deg: [N] in-degree w.r.t. these edges
      D:   int, max in-degree
    CUDA-safe: stable-sorts by dst so idx_in_dst ∈ [0, deg[v)-1].
    """
    device, dtype = x.device, x.dtype
    N, Fe_cur = x.shape
    E = src.numel()

    Fe_used = max(0, Fe_cur - start_feature)
    if N == 0 or Fe_cur == 0 or E == 0 or Fe_used == 0:
        deg = x.new_zeros(N, dtype=torch.long)
        # width is zero when Fe_used == 0 or D==0 → return empty feature dimension
        return x.new_zeros((N, 0)), deg, 0

    # ensure sorted by dst (idempotent if already sorted)
    if not torch.all(dst[:-1] <= dst[1:]):
        perm = dst.argsort(stable=True)
        src, dst = src[perm], dst[perm]

    # degrees & positions
    deg = torch.bincount(dst, minlength=N)                 # [N]
    D = int(deg.max().item())
    if D == 0:
        return x.new_zeros((N, 0)), deg, 0

    starts = deg.cumsum(0) - deg                           # [N]
    arng = torch.arange(E, device=device, dtype=torch.long)
    idx_in_dst = arng - starts[dst]                        # [E] in [0, deg[v)-1]

    # safety check
    imin = int(idx_in_dst.min().item())
    imax = int(idx_in_dst.max().item())
    assert 0 <= imin and imax < D, f"idx_in_dst OOB: [{imin}, {imax}] vs D={D}"

    # build output with -1 padding
    out = x.new_full((N, Fe_used * D), fill_value=-1, dtype=dtype)
    for j in range(start_feature, Fe_cur):
        U = x.new_full((N, D), fill_value=-1, dtype=dtype)  # <- padding is -1
        U[dst, idx_in_dst] = x[:, j][src]                   # place neighbors
        out[:, (j - start_feature) * D : (j - start_feature + 1) * D] = U
        del U

    return out, deg, D


