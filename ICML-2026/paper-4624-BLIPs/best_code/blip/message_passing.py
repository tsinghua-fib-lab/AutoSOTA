import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool

class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dimension,
        hidden_dimension,
        output_dimension,
        n_layers,
        activation,
        dropout=None,
    ):
        super().__init__()
        # build networks
        layers = []
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(input_dimension, hidden_dimension))
            input_dimension = hidden_dimension
        layers.append(nn.Linear(input_dimension, output_dimension))
        self.mlp_block = nn.ModuleList(layers)
        self.activation = activation()
        if dropout is None or dropout == 0.:
            self.local_dropout = nn.Identity()
        else:
            self.local_dropout = nn.Dropout(dropout)

    def forward(self, x):
        for idx, block in enumerate(self.mlp_block):
            x = block(x)
            x = self.local_dropout(x)
            if idx < len(self.mlp_block) - 1:
                x = self.activation(x)
        return x

class InteractionNetworkBlock(MessagePassing):
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim,
        n_message_layers=2,
        n_update_layers=2,
        activation=torch.nn.SiLU,
        dropout=None,
        recurrent=True,
    ):
        super().__init__(aggr="sum")
        # message network
        self.message_net = MLPBlock(
            input_dimension=2 * node_dim + edge_dim,
            hidden_dimension=hidden_dim,
            output_dimension=hidden_dim,
            n_layers=n_message_layers,
            activation=activation,
            dropout=dropout,
        )
        # update network
        self.update_net = MLPBlock(
            input_dimension=node_dim + hidden_dim,
            hidden_dimension=hidden_dim,
            output_dimension=node_dim,
            n_layers=n_update_layers,
            activation=activation,
            dropout=dropout,
        )
        # store activation
        self.activation = activation()
        self.recurrent = recurrent

    def forward(self, edge_index, x, edge_attr):
        return self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
        )

    def message(self, x_i, x_j, edge_attr):
        input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.activation(self.message_net(input))

    def update(self, message, x):
        input = torch.cat((x, message), dim=-1)
        return self.update_net(input) + x * bool(self.recurrent)

class EquivariantNetworkBlock(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim,
        n_message_layers=2,
        n_update_layers=2,
        num_pos_layers=2,
        activation=torch.nn.SiLU,
        recurrent=True,
        normalize=False,
        dropout=None,
    ):
        super().__init__()
        self.residual = recurrent
        self.normalize = normalize
        self.pos_agg = 'mean'
        self.epsilon = 1e-8

        self.edge_mlp = MLPBlock(
            input_dimension=2 * node_dim + 1 + edge_dim,
            hidden_dimension=hidden_dim,
            output_dimension=hidden_dim,
            n_layers=n_message_layers,
            activation=activation,
            dropout=dropout,
        )
        self.activation = activation()
        self.node_mlp = MLPBlock(
            input_dimension=node_dim + hidden_dim,
            hidden_dimension=hidden_dim,
            output_dimension=node_dim,
            n_layers=n_update_layers,
            activation=activation,
            dropout=dropout,
        )
        layer = nn.Linear(hidden_dim, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = []
        for _ in range(num_pos_layers-1):
            coord_mlp.append(nn.Linear(hidden_dim, hidden_dim))
            if dropout is not None:
                coord_mlp.append(nn.Dropout(dropout))
            coord_mlp.append(activation())
        coord_mlp.append(layer)
        self.pos_mlp = nn.Sequential(*coord_mlp)
        self.pos_mlp_vel = MLPBlock(
            input_dimension=hidden_dim,
            hidden_dimension=hidden_dim,
            output_dimension=1,
            n_layers=num_pos_layers,
            activation=activation,
            dropout=dropout,
        )

    def edge_model(self, source, target, radial, edge_attr):
        out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.activation(self.edge_mlp(out))
        return out

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = global_add_pool(edge_attr, row, size=x.size(0))
        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out

    def pos_model(self, pos, edge_index, pos_diff, edge_feat):
        row, col = edge_index
        trans = pos_diff * self.pos_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100)
        if self.pos_agg == 'sum':
            agg = global_add_pool(trans, row, size=pos.size(0))
        elif self.pos_agg == 'mean':
            agg = global_mean_pool(trans, row, size=pos.size(0))
        else:
            raise Exception('Wrong pos_agg parameter' % self.pos_agg)
        pos = pos + agg
        return pos

    def pos2radial(self, edge_index, pos):
        row, col = edge_index
        pos_diff = pos[row] - pos[col]
        radial = torch.sum(pos_diff**2, 1).unsqueeze(1)
        if self.normalize:
            norm = torch.sqrt(radial).detach() + 1e-8
            pos_diff = pos_diff / norm
        return radial, pos_diff

    def forward(self, x, edge_index, pos, vel, edge_attr=None):
        row, col = edge_index
        radial, pos_diff = self.pos2radial(edge_index, pos)
        edge_feat = self.edge_model(x[row], x[col], radial, edge_attr)
        pos = self.pos_model(pos, edge_index, pos_diff, edge_feat) + self.pos_mlp_vel(x) * vel
        x = self.node_model(x, edge_index, edge_feat)
        return x, pos