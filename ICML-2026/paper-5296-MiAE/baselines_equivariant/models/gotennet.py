import torch.nn as nn
from gotennet import GotenNetWrapper
from gotennet.models.components.layers import CosineCutoff
from torch_geometric.nn import global_mean_pool


class GotenNetProtein(nn.Module):
    def __init__(self, n_atom_basis, n_classes, cutoff, n_interactions=4):
        super().__init__()
        self.model = GotenNetWrapper(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            cutoff_fn=CosineCutoff(cutoff),
        )
        self.unpack_data = False

        self.mlp = nn.Sequential(
            nn.Linear(n_atom_basis, n_classes),
            nn.ReLU(),
            nn.Linear(n_classes, n_classes),
        )

    def forward(self, data):
        data.x = data.x.long().squeeze(-1)
        data.z = data.z.long().squeeze(-1)
        h, X = self.model(data)
        pooled = global_mean_pool(h, data.batch)
        out = self.mlp(pooled)
        return out, X
