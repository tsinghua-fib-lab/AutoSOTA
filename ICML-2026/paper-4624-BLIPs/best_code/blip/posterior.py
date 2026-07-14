from functools import partial

import torch
import torch.nn as nn

from orb_models.forcefield.angular import UnitVector
from orb_models.forcefield.featurization_utilities import gaussian_basis_function

from .message_passing import MLPBlock

class PosteriorModel(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        num_alphas_edge,
        num_alphas_node,
        hidden_dim=32,
        n_node_layers=2,
        n_edge_layers=2,
        activation=nn.SiLU,
        clamp_max_p=.8,
    ):
        super().__init__()
        self.edge_alphas = MLPBlock(
            input_dimension=2 * node_dim + edge_dim,
            output_dimension=num_alphas_edge,
            n_layers=n_edge_layers,
            activation=activation,
            hidden_dimension=hidden_dim,
            dropout=None
        )
        self.node_alphas = MLPBlock(
            input_dimension=node_dim,
            output_dimension=num_alphas_node,
            n_layers=n_node_layers,
            activation=activation,
            hidden_dimension=hidden_dim,
            dropout=None
        )
        self.clamp_max_p=clamp_max_p


    def forward(self, batch):
        edge_index, x, edge_attr = batch.edge_index, batch.x, batch.edge_attr
        row, col = edge_index
        # Edge-level alphas
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        p_edge = self.edge_alphas(edge_input).sigmoid().clamp(max=self.clamp_max_p)
        # Node-level alphas
        p_node = self.node_alphas(x).sigmoid().clamp(max=self.clamp_max_p)
        # Convert probabilities to alpha 
        alphas_edge = p_edge  / (1. - p_edge)
        alphas_node = p_node / (1. - p_node)
        return alphas_node, alphas_edge


class PosteriorModelPaiNN(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        num_alphas_edge,
        num_alphas_node,
        hidden_dim=32,
        n_node_layers=2,
        n_edge_layers=2,
        activation=nn.SiLU,
        clamp_max_p=.8,
    ):
        super().__init__()
        self.edge_alphas = MLPBlock(
            input_dimension=edge_dim,
            output_dimension=num_alphas_edge,
            n_layers=n_edge_layers,
            activation=activation,
            hidden_dimension=hidden_dim,
            dropout=None
        )
        self.node_alphas = MLPBlock(
            input_dimension=hidden_dim // 2,
            output_dimension=num_alphas_node,
            n_layers=n_node_layers,
            activation=activation,
            hidden_dimension=hidden_dim,
            dropout=None,
        )
        self.clamp_max_p=clamp_max_p
        self.emb = torch.nn.Embedding(100, hidden_dim // 2, padding_idx=0)

    def make_directed(self, nbr_list):
        gtr_ij = (nbr_list[:, 0] > nbr_list[:, 1]).any().item()
        gtr_ji = (nbr_list[:, 1] > nbr_list[:, 0]).any().item()
        directed = gtr_ij and gtr_ji

        if directed:
            return nbr_list, directed

        new_nbrs = torch.cat([nbr_list, nbr_list.flip(1)], dim=0)
        return new_nbrs, directed

    def forward(self, batch):
        nxyz = batch["nxyz"]
        nbrs, _ = self.make_directed(batch["nbr_list"])
        Z = nxyz[:, 0]
        emb = self.emb(Z.long())
        pos = nxyz[:, 1:].detach()
        r_ij = (pos[nbrs[:, 1]] - pos[nbrs[:, 0]]).pow(2).sum(dim=-1, keepdim=True)
        # Edge-level alphas
        p_edge = self.edge_alphas(r_ij).sigmoid().clamp(max=self.clamp_max_p)
        # Node-level alphas
        p_node = self.node_alphas(emb).sigmoid().clamp(max=self.clamp_max_p)
        # Convert probabilities to alpha
        alphas_edge = p_edge  / (1. - p_edge)
        alphas_node = p_node / (1. - p_node)
        return alphas_node, alphas_edge


class PosteriorModelORB(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        num_alphas_edge,
        num_alphas_node,
        hidden_dim=32,
        n_node_layers=2,
        n_edge_layers=2,
        activation=nn.SiLU,
        clamp_max_p=0.1,
    ):
        super().__init__()
        self.edge_alphas = MLPBlock(
            input_dimension=edge_dim,
            output_dimension=num_alphas_edge,
            n_layers=n_edge_layers,
            activation=activation,
            hidden_dimension=hidden_dim,
            dropout=None,
        )
        self.node_alphas = MLPBlock(
            input_dimension=node_dim,
            output_dimension=num_alphas_node,
            n_layers=n_node_layers,
            activation=activation,
            hidden_dimension=hidden_dim,
            dropout=None,
        )
        self.clamp_max_p = clamp_max_p
        self.rbf_transform = partial(gaussian_basis_function, num_bases=20, radius=5.0)
        self.angular_transform = UnitVector()
        self.node_offset = nn.Parameter(torch.tensor(10.0))
        self.edge_offset = nn.Parameter(torch.tensor(10.0))

    def forward(self, batch):
        # node embedding
        node_emb = batch.node_features["atomic_numbers_embedding"]
        # edge embedding
        vectors = batch.edge_features["vectors"]
        lengths = vectors.norm(dim=1)
        angular_embedding = self.angular_transform(vectors)  # (nedges, x)
        rbfs = self.rbf_transform(lengths)
        edge_features = torch.cat([rbfs, angular_embedding], dim=1)
        # Node-level alphas
        p_node = torch.sigmoid(self.node_alphas(node_emb) - self.node_offset).clamp(
            min=0.00001, max=self.clamp_max_p
        )  # 0.0005
        # Edge-level alphas
        p_edge = torch.sigmoid(
            self.edge_alphas(edge_features) - self.edge_offset
        ).clamp(min=0.00001, max=self.clamp_max_p)
        # Convert probabilities to alpha
        alphas_edge = p_edge / (1.0 - p_edge)
        alphas_node = p_node / (1.0 - p_node)
        return alphas_node, alphas_edge
