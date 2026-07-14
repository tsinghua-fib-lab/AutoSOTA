"""Data loader module."""

import torch
from torch_cluster import random_walk
from torch_geometric.loader import NeighborSampler as NS


def loaders(data, par):
    """Loaders."""
    nb = [par["n_sampled_nb"]] * par["order"]

    train_loader = NeighborSampler(
        data.edge_index,
        sizes=nb,
        batch_size=par["batch_size"],
        shuffle=True,
        num_nodes=data.num_nodes,
        node_idx=data.train_mask,
    )

    val_loader = NeighborSampler(
        data.edge_index,
        sizes=nb,
        batch_size=par["batch_size"],
        shuffle=False,
        num_nodes=data.num_nodes,
        node_idx=data.val_mask,
    )

    test_loader = NeighborSampler(
        data.edge_index,
        sizes=nb,
        batch_size=par["batch_size"],
        shuffle=False,
        num_nodes=data.num_nodes,
        node_idx=data.test_mask,
    )

    return train_loader, val_loader, test_loader


class NeighborSampler(NS):
    """Neighbor Sampler."""

    def sample(self, batch):
        """Sample."""
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # sample) and a random node (as negative sample):
        batch = torch.tensor(batch)
        pos_batch = random_walk(row, col, batch, walk_length=1, coalesced=False)
        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(),))
        batch = torch.cat([batch, pos_batch[:, 1], neg_batch], dim=0)

        return super().sample(batch)
