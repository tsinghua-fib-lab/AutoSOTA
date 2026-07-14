import torch.nn as nn
from .message_passing import InteractionNetworkBlock, EquivariantNetworkBlock

class MessagePassingNetwork(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim,
        num_layers=3,
        n_message_layers=2,
        n_update_layers=2,
        activation=nn.SiLU,
        regression_head=None,
        recurrent=True,
        dropout=None,
    ):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_dim, hidden_dim)
        self.num_layers = num_layers
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = InteractionNetworkBlock(
                node_dim=hidden_dim,
                edge_dim=hidden_dim,
                hidden_dim=hidden_dim,
                n_message_layers=n_message_layers,
                n_update_layers=n_update_layers,
                activation=activation,
                dropout=dropout,
                recurrent=recurrent
            )
            self.blocks.append(block)

        if regression_head is not None:
            self.regression_head = regression_head

    def forward(self, batch):
        edge_index, x, edge_attr = batch.edge_index, batch.x, batch.edge_attr
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        for block in self.blocks:
            x = block(edge_index, x, edge_attr)
        if hasattr(self, "regression_head"):
            x = self.regression_head(x, batch.batch)
        return x


class EquivariantGraphNeuralNetwork(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim,
        num_layers=3,
        n_message_layers=2,
        n_update_layers=2,
        num_pos_layers=2,
        activation=nn.SiLU,
        regression_head=None,
        dropout=None,
        recurrent=True,
        normalize=False,
        return_only_pos=True,
    ):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, hidden_dim)
        self.num_layers = num_layers
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = EquivariantNetworkBlock(
                node_dim=hidden_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                n_message_layers=n_message_layers,
                n_update_layers=n_update_layers,
                activation=activation,
                recurrent=recurrent,
                normalize=normalize,
                dropout=dropout
            )
            self.blocks.append(block)
        self.return_only_pos = return_only_pos
        if regression_head is not None:
            self.regression_head = regression_head

    def forward(self, batch):
        edge_index, x, edge_attr, pos, vel = batch.edge_index, batch.x, batch.edge_attr, batch.pos, batch.vel
        x = self.node_emb(x)
        for block in self.blocks:
            x, pos = block(x, batch.edge_index, pos, vel, edge_attr)
        if hasattr(self, "regression_head"):
            x = self.regression_head(x, batch.batch)
        if self.return_only_pos:
            return pos
        return x, pos