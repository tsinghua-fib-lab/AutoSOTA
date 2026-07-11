from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Set, List, Union

import torch
from torch_geometric.utils import coalesce, remove_self_loops

AugMode = Literal["none", "drop", "add", "both"]


def _unique_undirected_edges(edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor, Set[int]]:
    """
    Convert a possibly-directed edge_index into a unique undirected clean edge list (u < v),
    returning (u, v, key_set), where key = u * num_nodes + v.

    Self-loops are removed.

    This corresponds to the clean undirected edge set E used by RADE.
    """
    edge_index = edge_index.to(torch.long)

    # Remove self-loops.
    edge_index, _ = remove_self_loops(edge_index)

    row, col = edge_index[0], edge_index[1]
    u = torch.minimum(row, col)
    v = torch.maximum(row, col)

    # u != v already due to remove_self_loops
    keys = u * num_nodes + v
    keys = torch.unique(keys, sorted=True)

    u = keys // num_nodes
    v = keys % num_nodes

    # Cached CPU-side membership structure for clean-edge lookup.
    key_set = set(keys.tolist())
    return u, v, key_set


def _build_undirected_edge_index(
    u: torch.Tensor,
    v: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a symmetric directed edge_index from undirected pairs (u < v).
    """
    u = u.to(torch.long)
    v = v.to(torch.long)
    ei = torch.stack([torch.cat([u, v], dim=0), torch.cat([v, u], dim=0)], dim=0).to(device)
    ei = coalesce(ei, num_nodes=num_nodes)
    return ei


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
    Uniformly sample k undirected clean non-edges (u < v) without replacement, rejecting:
      - self-loops
      - clean edges
      - duplicates within the sampled set

    Returns:
        u_add, v_add on CPU (long)

    This is the practical fixed-size sampling scheme used in code in place of
    independent Bernoulli(q) sampling per non-edge. It matches the paper's
    implementation note that uses a fixed expected number of additions.
    """
    if k <= 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    added_keys: Set[int] = set()
    us: List[int] = []
    vs: List[int] = []

    # Batched rejection sampling.
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
            f"Failed to sample {k} non-edges after {max_batches} batches. "
            f"Try reducing q (expected adds too large) or increasing max_batches/batch_size."
        )

    u_add = torch.tensor(us, dtype=torch.long, device="cpu")
    v_add = torch.tensor(vs, dtype=torch.long, device="cpu")
    return u_add, v_add


@dataclass
class BernoulliEdgeAugmentor:
    """
    Cache the clean undirected edge set and support per-epoch RADE perturbations:
      - drop clean edges with probability p
      - add clean non-edges with rate q

    Important paper-consistency note:
      - The idealized paper model uses independent Bernoulli(q) sampling over clean non-edges.
      - This implementation instead uses a fixed expected number of additions
            k_add = round(q * |non-edges|)
        sampled uniformly without replacement from the clean complement.
      - This matches the paper's practical implementation note.

    The class is designed for undirected graphs.
    """
    edge_u: torch.Tensor              # clean undirected unique edges u < v (CPU)
    edge_v: torch.Tensor              # clean undirected unique edges u < v (CPU)
    edge_keys: Set[int]               # python set of clean undirected keys (CPU)
    num_nodes: int

    @classmethod
    def from_edge_index(cls, edge_index: torch.Tensor, num_nodes: int) -> "BernoulliEdgeAugmentor":
        u, v, key_set = _unique_undirected_edges(edge_index.cpu(), num_nodes)
        return cls(edge_u=u.cpu(), edge_v=v.cpu(), edge_keys=key_set, num_nodes=num_nodes)

    def augment_edge_index(
        self,
        p: float,
        q: float,
        mode: AugMode,
        *,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        safety_max_additions: int = 2_000_000,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Return an augmented undirected edge_index.

        Returns:
          - edge_index                          if return_stats=False
          - (edge_index, stats_dict)            if return_stats=True

        Notes:
          - Drops are sampled only from the clean edge set E.
          - Adds are sampled only from the clean complement Ebar.
          - A dropped clean edge is NOT eligible to be re-added in the same epoch.
          - Added pairs are sampled uniformly from the clean complement using
                k_add = round(q * |Ebar|),
            i.e. the fixed-size approximation used in the paper's implementation note.
          - Reported stats are in UNDIRECTED unique edges (u < v).
        """
        if device is None:
            device = torch.device("cpu")

        m_clean_undir = int(self.edge_u.numel())

        dropped_undir = 0
        added_undir = 0
        kept_undir = m_clean_undir

        if mode == "none" or (p <= 0.0 and q <= 0.0):
            ei = _build_undirected_edge_index(self.edge_u, self.edge_v, self.num_nodes, device)
            if return_stats:
                stats = {
                    "clean_undir": m_clean_undir,
                    "kept_undir": m_clean_undir,
                    "dropped_undir": 0,
                    "added_undir": 0,
                    "final_undir": m_clean_undir,
                    "final_dir": int(ei.size(1)),
                }
                return ei, stats
            return ei

        if not (0.0 <= p < 1.0):
            raise ValueError(f"p must be in [0,1). Got p={p}")
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"q must be in [0,1]. Got q={q}")

        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(int(seed))

        # -----------------------
        # DROP from the clean undirected edge set E
        # -----------------------
        u_keep = self.edge_u
        v_keep = self.edge_v
        if mode in ("drop", "both") and p > 0.0:
            keep_prob = 1.0 - p
            keep_mask = (torch.rand(u_keep.numel(), generator=g) < keep_prob)
            u_keep = u_keep[keep_mask]
            v_keep = v_keep[keep_mask]
            kept_undir = int(u_keep.numel())
            dropped_undir = m_clean_undir - kept_undir

        # -----------------------
        # ADD from the clean complement edges
        # -----------------------
        u_add = torch.empty(0, dtype=torch.long)
        v_add = torch.empty(0, dtype=torch.long)
        if mode in ("add", "both") and q > 0.0:
            n = self.num_nodes
            total_pairs = n * (n - 1) // 2
            num_non_edges = int(total_pairs - m_clean_undir)

            k_add = int(round(q * num_non_edges))
            if k_add > safety_max_additions:
                raise ValueError(
                    f"Requested expected additions k_add={k_add} exceeds safety_max_additions={safety_max_additions}. "
                    f"Choose smaller q. (Here |Ebar|={num_non_edges}.)"
                )

            if k_add > 0:
                u_add, v_add = _sample_non_edges_uniform(
                    num_nodes=n,
                    k=k_add,
                    existing_keys=self.edge_keys,  # exclude clean edges
                    generator=g,
                )
                added_undir = int(u_add.numel())

        # -----------------------
        # Build final symmetric edge_index
        # -----------------------
        u_final = torch.cat([u_keep, u_add], dim=0)
        v_final = torch.cat([v_keep, v_add], dim=0)
        ei = _build_undirected_edge_index(u_final, v_final, self.num_nodes, device)

        if return_stats:
            final_undir = int(u_final.numel())
            stats = {
                "clean_undir": m_clean_undir,
                "kept_undir": kept_undir,
                "dropped_undir": dropped_undir,
                "added_undir": added_undir,
                "final_undir": final_undir,
                "final_dir": int(ei.size(1)),
            }
            return ei, stats

        return ei

    def augment_edge_indices(
        self,
        *,
        p: float,
        q: float,
        mode: str,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        safety_max_additions: int = 2_000_000,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Return:
          edge_index_aug:  [2, E_aug_dir]
          edge_index_keep: [2, E_keep_dir]  retained clean edges, directed both ways
          edge_index_add:  [2, E_add_dir]   sampled clean non-edges, directed both ways
          stats: dict

        This is the representation needed by the corrected RADE-OF / RADE-OFS
        convolution layers.

        Notes:
          - Drops are sampled only from clean edges E.
          - Adds are sampled only from clean non-edges Ebar.
          - A clean edge that is dropped is never re-added in the same epoch.
          - Adds use the fixed-size sampling approximation
                k_add = round(q * |Ebar|).
        """
        mode = mode.lower().strip()
        if device is None:
            device = torch.device("cpu")

        if not (0.0 <= p < 1.0):
            raise ValueError(f"p must be in [0,1). Got p={p}")
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"q must be in [0,1]. Got q={q}")
        if mode not in {"none", "drop", "add", "both"}:
            raise ValueError(f"mode must be one of {{none,drop,add,both}}. Got mode={mode}")

        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(int(seed))

        n = int(self.num_nodes)
        u = self.edge_u  # clean undirected unique edges u<v (CPU)
        v = self.edge_v  # clean undirected unique edges u<v (CPU)
        m_clean_undir = int(u.numel())

        # -----------------------
        # DROP from clean undirected edges E
        # -----------------------
        u_keep, v_keep = u, v
        dropped_undir = 0
        if mode in {"drop", "both"} and p > 0.0:
            keep_mask = (torch.rand(m_clean_undir, generator=g) < (1.0 - p))
            u_keep = u_keep[keep_mask]
            v_keep = v_keep[keep_mask]
            dropped_undir = m_clean_undir - int(u_keep.numel())

        kept_undir = int(u_keep.numel())

        # -----------------------
        # ADD from clean complement edges
        # -----------------------
        u_add = torch.empty(0, dtype=torch.long, device="cpu")
        v_add = torch.empty(0, dtype=torch.long, device="cpu")
        added_undir = 0

        total_pairs = n * (n - 1) // 2
        num_non_edges = int(total_pairs - m_clean_undir)

        if mode in {"add", "both"} and q > 0.0:
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
                    existing_keys=self.edge_keys,  # exclude clean edges (RADE-consistent)
                    generator=g,
                )
                added_undir = int(u_add.numel())

        # -----------------------
        # Build directed edge_index tensors
        # -----------------------
        def to_dir(u_undir: torch.Tensor, v_undir: torch.Tensor) -> torch.Tensor:
            if u_undir.numel() == 0:
                return torch.empty((2, 0), dtype=torch.long, device=device)
            row = torch.cat([u_undir, v_undir], dim=0).to(device)
            col = torch.cat([v_undir, u_undir], dim=0).to(device)
            ei = torch.stack([row, col], dim=0)
            return coalesce(ei, num_nodes=n)

        edge_index_keep = to_dir(u_keep, v_keep)
        edge_index_add = to_dir(u_add, v_add)

        edge_index_aug = torch.cat([edge_index_keep, edge_index_add], dim=1)
        edge_index_aug = coalesce(edge_index_aug, num_nodes=n)

        stats = {
            "clean_undir": m_clean_undir,
            "kept_undir": kept_undir,
            "dropped_undir": dropped_undir,
            "added_undir": added_undir,
            "final_undir": kept_undir + added_undir,
            "final_dir": int(edge_index_aug.size(1)),
            "p": float(p),
            "q": float(q),
            "num_non_edges": int(num_non_edges),
        }

        return edge_index_aug, edge_index_keep, edge_index_add, stats