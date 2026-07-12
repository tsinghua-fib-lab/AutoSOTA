"""
Graph construction utilities for GAT-FM.

This module provides:
1. Loading directed PPI edges from TSV files
2. Mask-aware edge pruning to prevent leakage to masked targets
3. Batch graph helpers for PyTorch Geometric
"""

from typing import Dict, List, Optional, Tuple
import os
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, remove_self_loops


def load_ppi_network(
    ppi_path: str,
    protein_names: List[str],
    score_threshold: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Load a directed PPI graph from a TSV file.

    Supported column names:
    - Preferred: '#node1', 'node2' (STRING exports)
    - Fallback: 'protein1', 'protein2'

    If 'combined_score' exists, rows are filtered by score_threshold.
    """
    if not os.path.exists(ppi_path):
        print(f"Warning: PPI file not found: {ppi_path}")
        return torch.zeros((2, 0), dtype=torch.long), {}

    protein_to_idx = {name: idx for idx, name in enumerate(protein_names)}

    try:
        ppi_df = pd.read_csv(ppi_path, sep='\t')
    except Exception as e:
        print(f"Warning: failed to read PPI file {ppi_path}: {e}")
        return torch.zeros((2, 0), dtype=torch.long), protein_to_idx

    if '#node1' in ppi_df.columns and 'node2' in ppi_df.columns:
        p1_col, p2_col = '#node1', 'node2'
    elif 'protein1' in ppi_df.columns and 'protein2' in ppi_df.columns:
        p1_col, p2_col = 'protein1', 'protein2'
    else:
        print("Warning: required PPI columns are missing. Expected (#node1,node2) or (protein1,protein2).")
        return torch.zeros((2, 0), dtype=torch.long), protein_to_idx

    if 'combined_score' in ppi_df.columns and score_threshold > 0:
        # Support either 0-1000 or 0-1 score scales.
        if ppi_df['combined_score'].max() <= 1.0 and score_threshold > 1.0:
            threshold = score_threshold / 1000.0
        else:
            threshold = score_threshold
        ppi_df = ppi_df[ppi_df['combined_score'] >= threshold]

    edges = []
    for _, row in ppi_df.iterrows():
        p1 = str(row[p1_col]).strip()
        p2 = str(row[p2_col]).strip()
        if p1 in protein_to_idx and p2 in protein_to_idx:
            edges.append([protein_to_idx[p1], protein_to_idx[p2]])

    if not edges:
        print("Warning: no valid PPI edges matched the provided protein names.")
        return torch.zeros((2, 0), dtype=torch.long), protein_to_idx

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, protein_to_idx


def build_protein_graph(
    num_proteins: int,
    edge_index: torch.Tensor,
    add_self_loops_flag: bool = True,
) -> Data:
    """Build the base protein graph as a PyG Data object."""
    edge_index, _ = remove_self_loops(edge_index)
    if add_self_loops_flag:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_proteins)
    return Data(edge_index=edge_index, num_nodes=num_proteins)


def apply_mask_aware_edge_pruning(
    edge_index: torch.Tensor,
    mask: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Prune edges to prevent leakage into masked target proteins.

    Keep an edge if:
    1) It is a self-loop, or
    2) Its target node is observed (mask[target] == 1).
    """
    del num_nodes  # API compatibility

    src, tgt = edge_index[0], edge_index[1]
    is_self_loop = src == tgt
    is_masked_target = mask[tgt] == 0
    keep_mask = is_self_loop | (~is_masked_target)
    return edge_index[:, keep_mask]


def batch_graphs_with_masks(
    base_edge_index: torch.Tensor,
    masks: torch.Tensor,
    num_proteins: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a batched graph with per-sample mask-aware pruning.

    Returns:
        batched_edge_index: (2, E_total)
        batch: (B * P,) node-to-sample assignment
    """
    bsz = masks.shape[0]
    device = masks.device

    all_edges = []
    batch_indices = []

    for b in range(bsz):
        pruned_edges = apply_mask_aware_edge_pruning(
            base_edge_index.to(device), masks[b], num_proteins
        )
        all_edges.append(pruned_edges + b * num_proteins)
        batch_indices.append(torch.full((num_proteins,), b, dtype=torch.long, device=device))

    batched_edge_index = (
        torch.cat(all_edges, dim=1)
        if all_edges else
        torch.zeros((2, 0), dtype=torch.long, device=device)
    )
    batch = torch.cat(batch_indices)
    return batched_edge_index, batch


class ProteinGraph:
    """Manager for protein-protein interaction graph utilities."""

    def __init__(
        self,
        protein_names: List[str],
        ppi_path: Optional[str] = None,
        score_threshold: float = 400.0,
        add_self_loops_flag: bool = True,
    ):
        self.protein_names = protein_names
        self.num_proteins = len(protein_names)
        self.protein_to_idx = {name: idx for idx, name in enumerate(protein_names)}

        if ppi_path is not None and ppi_path.strip() != '':
            self.edge_index, _ = load_ppi_network(
                ppi_path, protein_names, score_threshold
            )
        else:
            print("Warning: no PPI path provided, using self-loops only.")
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)

        if add_self_loops_flag:
            self.edge_index, _ = remove_self_loops(self.edge_index)
            self.edge_index, _ = add_self_loops(
                self.edge_index, num_nodes=self.num_proteins
            )

        self._base_data = Data(
            edge_index=self.edge_index,
            num_nodes=self.num_proteins,
        )

    @property
    def base_edge_index(self) -> torch.Tensor:
        """Return base edge_index without mask-aware pruning."""
        return self.edge_index

    def get_edge_index_for_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Return edge_index pruned for a single sample mask."""
        return apply_mask_aware_edge_pruning(
            self.edge_index.to(mask.device), mask, self.num_proteins
        )

    def batch_for_masks(
        self,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return batched edge_index and node batch assignment."""
        return batch_graphs_with_masks(
            self.edge_index, masks, self.num_proteins
        )

    def to(self, device: torch.device) -> 'ProteinGraph':
        """Move graph tensors to the target device."""
        self.edge_index = self.edge_index.to(device)
        self._base_data = self._base_data.to(device)
        return self

    def get_stats(self) -> Dict[str, int]:
        """Get basic graph statistics."""
        num_self_loops = (self.edge_index[0] == self.edge_index[1]).sum().item()
        return {
            'num_nodes': self.num_proteins,
            'num_edges': self.edge_index.shape[1],
            'num_self_loops': num_self_loops,
            'num_ppi_edges': self.edge_index.shape[1] - num_self_loops,
        }


def create_dense_graph(num_nodes: int, add_self_loops: bool = True) -> torch.Tensor:
    """Create a fully-connected graph for ablation experiments."""
    src = torch.arange(num_nodes).repeat_interleave(num_nodes)
    tgt = torch.arange(num_nodes).repeat(num_nodes)
    edge_index = torch.stack([src, tgt], dim=0)

    if not add_self_loops:
        edge_index, _ = remove_self_loops(edge_index)

    return edge_index


def create_sparse_identity_graph(num_nodes: int) -> torch.Tensor:
    """Create a graph with only self-loops (no message passing)."""
    nodes = torch.arange(num_nodes)
    return torch.stack([nodes, nodes], dim=0)
