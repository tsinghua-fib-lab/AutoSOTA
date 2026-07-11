# graph_classification/augmentation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Set, Tuple

import torch
from torch_geometric.utils import coalesce, remove_self_loops

AugMode = Literal["none", "drop", "add", "both"]


# -------------------------
# Helpers: undirected canon
# -------------------------

def _unique_undirected_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_attr: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Set[int], Optional[torch.Tensor]]:
    """
    Canonicalize to unique undirected edges (u < v), removing self-loops.

    Returns:
      u_undir: [M]  CPU long
      v_undir: [M]  CPU long
      key_set: Python set of keys (CPU-side membership)
      attr_undir: [M, F] CPU (if edge_attr provided), representative per undirected pair

    Notes:
      - If the input graph contains both directions, we keep one representative edge_attr.
      - If multiple parallel edges exist for the same undirected pair, we keep the first (after sorting by key).
    """
    edge_index = edge_index.to(torch.long)
    if edge_attr is not None:
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    else:
        edge_index, _ = remove_self_loops(edge_index)

    row, col = edge_index[0].cpu(), edge_index[1].cpu()
    u = torch.minimum(row, col)
    v = torch.maximum(row, col)
    keys = u * int(num_nodes) + v  # [E], CPU

    perm = torch.argsort(keys)
    keys_s = keys[perm]

    # first occurrence of each key
    first = torch.ones(keys_s.numel(), dtype=torch.bool)
    if keys_s.numel() > 1:
        first[1:] = keys_s[1:] != keys_s[:-1]

    uniq_keys = keys_s[first]
    u_undir = (uniq_keys // int(num_nodes)).to(torch.long)
    v_undir = (uniq_keys % int(num_nodes)).to(torch.long)

    attr_undir = None
    if edge_attr is not None:
        ea = edge_attr.detach().cpu()
        ea_s = ea[perm]
        attr_undir = ea_s[first].contiguous()

    key_set = set(uniq_keys.tolist())
    return u_undir, v_undir, key_set, attr_undir


def _sample_non_edges_uniform(
    num_nodes: int,
    k: int,
    existing_keys: Set[int],
    generator: torch.Generator,
    *,
    batch_size: int = 200_000,
    max_batches: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniformly sample k undirected non-edges (u < v) without replacement, rejecting:
      - self-loops
      - existing undirected edges (existing_keys)
      - duplicates within the sampled set

    Returns:
      (u_add, v_add) as CPU long tensors.
    """
    if k <= 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    added_keys: Set[int] = set()
    us: List[int] = []
    vs: List[int] = []

    n = int(num_nodes)

    for _ in range(max_batches):
        if len(added_keys) >= k:
            break

        need = k - len(added_keys)
        bsz = min(batch_size, max(10_000, 20 * need))

        a = torch.randint(0, n, (bsz,), generator=generator, device="cpu")
        b = torch.randint(0, n, (bsz,), generator=generator, device="cpu")

        mask = a != b
        a = a[mask]
        b = b[mask]

        u = torch.minimum(a, b)
        v = torch.maximum(a, b)
        keys = (u * n + v).tolist()

        u_list = u.tolist()
        v_list = v.tolist()

        for key, uu, vv in zip(keys, u_list, v_list):
            if key in existing_keys:
                continue
            if key in added_keys:
                continue
            added_keys.add(key)
            us.append(uu)
            vs.append(vv)
            if len(added_keys) >= k:
                break

    if len(added_keys) < k:
        raise RuntimeError(
            f"Failed to sample {k} non-edges after {max_batches} batches. "
            f"Try reducing q or increasing max_batches/batch_size."
        )

    return (
        torch.tensor(us, dtype=torch.long, device="cpu"),
        torch.tensor(vs, dtype=torch.long, device="cpu"),
    )


def _undir_to_dir(
    u_undir: torch.Tensor,
    v_undir: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    attr_undir: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Expand undirected pairs (u<v) into directed edges both ways.

    Returns:
      edge_index_dir: [2, 2M] on device
      edge_attr_dir:  [2M, F] on device (if attr_undir provided)
    """
    if u_undir.numel() == 0:
        ei = torch.empty((2, 0), dtype=torch.long, device=device)
        return ei, None

    u_undir = u_undir.to(torch.long)
    v_undir = v_undir.to(torch.long)

    row = torch.cat([u_undir, v_undir], dim=0).to(device)
    col = torch.cat([v_undir, u_undir], dim=0).to(device)
    ei = torch.stack([row, col], dim=0)

    # coalesce protects against any accidental duplicates
    ei = coalesce(ei, num_nodes=int(num_nodes))

    if attr_undir is None:
        return ei, None

    ea = attr_undir.to(device)
    ea_dir = torch.cat([ea, ea], dim=0)
    return ei, ea_dir


# -------------------------
# RADE sampler
# -------------------------

@dataclass
class BernoulliEdgeAugmentor:
    """
    RADE-consistent augmentation for a SINGLE graph.

    Clean undirected edge set E is cached as unique (u<v).
    Per call, we sample:
      - keep edges by dropping each clean edge w.p. p
      - add edges by sampling k_add ≈ round(q * |Ebar|) undirected non-edges from the CLEAN complement

    Outputs are directed edge_index tensors (both directions), matching RADE conv interfaces.
    """
    edge_u: torch.Tensor                 # [M] CPU long (u<v)
    edge_v: torch.Tensor                 # [M] CPU long (u<v)
    edge_keys: Set[int]                  # membership over CLEAN undirected edges
    num_nodes: int
    edge_attr_undir: Optional[torch.Tensor] = None  # [M,F] CPU, representative per undirected pair

    @classmethod
    def from_graph(
        cls,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> "BernoulliEdgeAugmentor":
        u, v, key_set, attr_undir = _unique_undirected_edges(edge_index, int(num_nodes), edge_attr=edge_attr)
        return cls(edge_u=u, edge_v=v, edge_keys=key_set, num_nodes=int(num_nodes), edge_attr_undir=attr_undir)

    def augment_edge_indices(
        self,
        *,
        p: float,
        q: float,
        mode: AugMode,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        safety_max_additions: int = 2_000_000,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Returns (edge_index_aug, edge_index_keep, edge_index_add, stats).
        Edge attributes are ignored (topology-only). Use augment_edge_indices_with_attr() if needed.
        """
        ei_aug, ei_k, ei_a, _, _, _, stats = self.augment_edge_indices_with_attr(
            p=p, q=q, mode=mode, seed=seed, device=device, safety_max_additions=safety_max_additions
        )
        return ei_aug, ei_k, ei_a, stats

    def augment_edge_indices_with_attr(
        self,
        *,
        p: float,
        q: float,
        mode: AugMode,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        safety_max_additions: int = 2_000_000,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
        dict
    ]:
        """
        Returns:
          edge_index_aug:  [2, E_aug_dir]
          edge_index_keep: [2, E_keep_dir]
          edge_index_add:  [2, E_add_dir]
          edge_attr_aug:   [E_aug_dir, F] or None
          edge_attr_keep:  [E_keep_dir, F] or None
          edge_attr_add:   [E_add_dir, F] or None
          stats: dict

        Policy for added edges when edge_attr exists:
          - edge_attr_add is set to zeros with the same feature dimension/dtype as cached edge_attr_undir.
        """
        mode = str(mode).lower().strip()
        if mode not in {"none", "drop", "add", "both"}:
            raise ValueError(f"mode must be one of {{none,drop,add,both}}. Got mode={mode}")
        if not (0.0 <= float(p) < 1.0):
            raise ValueError(f"p must be in [0,1). Got p={p}")
        if not (0.0 <= float(q) <= 1.0):
            raise ValueError(f"q must be in [0,1]. Got q={q}")

        if device is None:
            device = torch.device("cpu")

        n = int(self.num_nodes)
        u = self.edge_u
        v = self.edge_v
        m_clean = int(u.numel())

        total_pairs = n * (n - 1) // 2
        num_non_edges = int(total_pairs - m_clean)

        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(int(seed))

        # -----------------------
        # DROP from clean edges
        # -----------------------
        u_keep, v_keep = u, v
        attr_keep = self.edge_attr_undir
        dropped_undir = 0

        if mode in {"drop", "both"} and float(p) > 0.0 and m_clean > 0:
            keep_mask = (torch.rand(m_clean, generator=g) < (1.0 - float(p)))
            u_keep = u_keep[keep_mask]
            v_keep = v_keep[keep_mask]
            if attr_keep is not None:
                attr_keep = attr_keep[keep_mask]
            dropped_undir = m_clean - int(u_keep.numel())

        kept_undir = int(u_keep.numel())

        # -----------------------
        # ADD from clean complement
        # -----------------------
        u_add = torch.empty(0, dtype=torch.long, device="cpu")
        v_add = torch.empty(0, dtype=torch.long, device="cpu")
        attr_add = None
        added_undir = 0

        if mode in {"add", "both"} and float(q) > 0.0 and num_non_edges > 0:
            k_add = int(round(float(q) * float(num_non_edges)))
            if k_add > safety_max_additions:
                raise ValueError(
                    f"Requested expected additions k_add={k_add} exceeds safety_max_additions={safety_max_additions}. "
                    f"Choose smaller q. (Here |Ebar|={num_non_edges}.)"
                )
            if k_add > 0:
                u_add, v_add = _sample_non_edges_uniform(
                    num_nodes=n,
                    k=k_add,
                    existing_keys=self.edge_keys,  # exclude CLEAN edges (RADE-correct)
                    generator=g,
                )
                added_undir = int(u_add.numel())

                if self.edge_attr_undir is not None:
                    Fdim = int(self.edge_attr_undir.size(-1))
                    attr_add = torch.zeros((added_undir, Fdim), dtype=self.edge_attr_undir.dtype, device="cpu")

        # -----------------------
        # Build directed tensors
        # -----------------------
        ei_keep, ea_keep = _undir_to_dir(u_keep, v_keep, n, device, attr_undir=attr_keep)
        ei_add, ea_add = _undir_to_dir(u_add, v_add, n, device, attr_undir=attr_add)

        ei_aug = torch.cat([ei_keep, ei_add], dim=1) if ei_add.numel() else ei_keep
        ei_aug = coalesce(ei_aug, num_nodes=n)

        ea_aug = None
        if ea_keep is not None or ea_add is not None:
            # If we have attrs, concatenate in the same way we concatenated edges.
            # coalesce() above does not merge opposite directions; it only removes exact duplicates.
            if ea_keep is None:
                ea_keep = torch.zeros((int(ei_keep.size(1)), int(ea_add.size(-1))), device=device, dtype=ea_add.dtype)
            if ea_add is None:
                ea_add = torch.zeros((int(ei_add.size(1)), int(ea_keep.size(-1))), device=device, dtype=ea_keep.dtype)
            ea_aug = torch.cat([ea_keep, ea_add], dim=0) if ea_add.numel() else ea_keep

        stats = {
            "clean_undir": int(m_clean),
            "kept_undir": int(kept_undir),
            "dropped_undir": int(dropped_undir),
            "added_undir": int(added_undir),
            "final_undir": int(kept_undir + added_undir),
            "final_dir": int(ei_aug.size(1)),
            "p": float(p),
            "q": float(q),
            "num_non_edges": int(num_non_edges),
        }

        return ei_aug, ei_keep, ei_add, ea_aug, ea_keep, ea_add, stats
