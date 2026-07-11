# nc_minibatch/mb_augmentation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Set, List, Dict

import torch
from torch_geometric.utils import coalesce, remove_self_loops

AugMode = Literal["none", "drop", "add", "both"]


def _unique_undirected_edges(edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor, Set[int]]:
    """
    Convert possibly-directed edge_index into a unique undirected edge list (u < v),
    returning (u, v, key_set), where key = u * num_nodes + v.
    Self-loops are removed.

    This is batch-local: num_nodes is the mini_batch node count (after relabeling).
    """
    edge_index = edge_index.to(torch.long)

    # remove self-loops
    edge_index, _ = remove_self_loops(edge_index)

    if edge_index.numel() == 0:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            set(),
        )

    row, col = edge_index[0], edge_index[1]
    u = torch.minimum(row, col)
    v = torch.maximum(row, col)

    # u != v already due to remove_self_loops
    keys = u * num_nodes + v
    keys = torch.unique(keys, sorted=True)

    u = keys // num_nodes
    v = keys % num_nodes
    key_set = set(keys.tolist())  # CPU-side membership structure
    return u, v, key_set


def _to_dir(u_undir: torch.Tensor, v_undir: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    """
    Expand undirected unique edges (u<v) into directed (both directions), coalesced.
    """
    if u_undir.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    row = torch.cat([u_undir, v_undir], dim=0).to(device)
    col = torch.cat([v_undir, u_undir], dim=0).to(device)
    ei = torch.stack([row, col], dim=0)
    return coalesce(ei, num_nodes=num_nodes)


def _sample_non_edges_uniform(
    num_nodes: int,
    k: int,
    existing_keys: Set[int],
    generator: torch.Generator,
    *,
    batch_size: int = 100_000,
    max_batches: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniformly sample k undirected non-edges (u < v) without replacement, rejecting:
      - self-loops
      - existing edges (existing_keys)
      - duplicates within sampled set

    Returns u_add, v_add on CPU (long).

    Batch-local: num_nodes is batch node count.
    """
    if k <= 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    added_keys: Set[int] = set()
    us: List[int] = []
    vs: List[int] = []

    # Rejection sampling in batches
    for _ in range(max_batches):
        if len(added_keys) >= k:
            break

        need = k - len(added_keys)
        bsz = min(batch_size, max(10_000, 20 * need))

        a = torch.randint(0, num_nodes, (bsz,), generator=generator, device="cpu")
        b = torch.randint(0, num_nodes, (bsz,), generator=generator, device="cpu")

        mask = a != b
        a = a[mask]
        b = b[mask]

        u = torch.minimum(a, b)
        v = torch.maximum(a, b)

        keys = (u * num_nodes + v).tolist()
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
            f"[mb_augmentation] Failed to sample {k} non-edges after {max_batches} batches. "
            f"Try reducing q, reducing mb_batch_size, or increasing max_batches/batch_size."
        )

    u_add = torch.tensor(us, dtype=torch.long, device="cpu")
    v_add = torch.tensor(vs, dtype=torch.long, device="cpu")
    return u_add, v_add


@dataclass
class BatchBernoulliEdgeAugmentor:
    """
    Mini-batch (batch-local) Bernoulli edge augmentation.

    This class operates on a *batch-induced* subgraph whose nodes have already been relabeled
    to [0, ..., N_batch-1] (as done by NeighborLoader / subgraph extraction).

    It does NOT cache across epochs/batches because each batch has different node sets.
    """

    @staticmethod
    def augment_edge_indices(
        edge_index_clean: torch.Tensor,
        num_nodes: int,
        *,
        p: float,
        q: float,
        mode: AugMode,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        safety_max_additions: int = 2000000,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Returns:
          edge_index_aug:  [2, E_aug_dir]
          edge_index_keep: [2, E_keep_dir]
          edge_index_add:  [2, E_add_dir]
          stats: dict

        Notes:
          - Drops are sampled only from CLEAN batch edges.
          - Adds are sampled only from CLEAN batch non-edges, i.e., never add a clean edge even if dropped.
          - All sampling is batch-local (over pairs within the batch node set).
        """
        if device is None:
            device = edge_index_clean.device

        mode = str(mode).lower().strip()
        if mode not in {"none", "drop", "add", "both"}:
            raise ValueError(f"mode must be one of {{none,drop,add,both}}. Got mode={mode}")

        if not (0.0 <= float(p) < 1.0):
            raise ValueError(f"p must be in [0,1). Got p={p}")
        if not (0.0 <= float(q) <= 1.0):
            raise ValueError(f"q must be in [0,1]. Got q={q}")

        # Build unique undirected clean edge list + membership set (CPU-side)
        u, v, edge_keys = _unique_undirected_edges(edge_index_clean.detach().cpu(), int(num_nodes))
        m_clean_undir = int(u.numel())

        # RNG on CPU (sampling uses python set membership)
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(int(seed))

        # -----------------------
        # DROP
        # -----------------------
        u_keep, v_keep = u, v
        dropped_undir = 0
        if mode in {"drop", "both"} and float(p) > 0.0 and m_clean_undir > 0:
            keep_mask = (torch.rand(m_clean_undir, generator=g) < (1.0 - float(p)))
            u_keep = u_keep[keep_mask]
            v_keep = v_keep[keep_mask]
            dropped_undir = m_clean_undir - int(u_keep.numel())
        kept_undir = int(u_keep.numel())

        # -----------------------
        # ADD (sample from clean complement)
        # -----------------------
        total_pairs = int(num_nodes) * (int(num_nodes) - 1) // 2
        num_non_edges = int(total_pairs - m_clean_undir)

        u_add = torch.empty(0, dtype=torch.long, device="cpu")
        v_add = torch.empty(0, dtype=torch.long, device="cpu")
        added_undir = 0

        if mode in {"add", "both"} and float(q) > 0.0 and num_non_edges > 0:
            k_add = int(round(float(q) * float(num_non_edges)))
            if k_add > safety_max_additions:
                raise ValueError(
                    f"[mb_augmentation] Requested expected additions k_add={k_add} exceeds "
                    f"safety_max_additions={safety_max_additions}. Choose smaller q. "
                    f"(Here |Ebar_batch|={num_non_edges}.)"
                )
            if k_add > 0:
                u_add, v_add = _sample_non_edges_uniform(
                    num_nodes=int(num_nodes),
                    k=k_add,
                    existing_keys=edge_keys,  # excludes CLEAN edges (RADE rule)
                    generator=g,
                )
                added_undir = int(u_add.numel())

        # -----------------------
        # Build directed tensors (on target device)
        # -----------------------
        edge_index_keep = _to_dir(u_keep, v_keep, int(num_nodes), device)
        edge_index_add = _to_dir(u_add, v_add, int(num_nodes), device)

        edge_index_aug = torch.cat([edge_index_keep, edge_index_add], dim=1)
        edge_index_aug = coalesce(edge_index_aug, num_nodes=int(num_nodes))

        stats: Dict[str, float] = {
            "clean_undir": float(m_clean_undir),
            "kept_undir": float(kept_undir),
            "dropped_undir": float(dropped_undir),
            "added_undir": float(added_undir),
            "final_undir": float(kept_undir + added_undir),
            "final_dir": float(edge_index_aug.size(1)),
            "p": float(p),
            "q": float(q),
            "num_non_edges": float(num_non_edges),
        }

        return edge_index_aug, edge_index_keep, edge_index_add, stats
