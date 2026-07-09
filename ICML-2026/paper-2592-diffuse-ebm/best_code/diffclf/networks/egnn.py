# Implementation of EGNN

# Libraries
import torch
import torch.nn as nn
from ..utils.se3_utils import remove_mean
from .mlp import TimeNet
from .utils import PositionalEmbedding
from typing import List

"""
 - EGNN_dynamics is used for modeling LJ-n systems, which are associated with
    a fully connected graph and identical elements
 - EGNN_atom is used for ALDP, which is associated with bonds (i.e. edge_index)
    and atom types (i.e. node_attrs)
"""

class EGNN_EnergyWrapper(nn.Module):
    def __init__(self, egnn: nn.Module, per_particle_energy=False, repeat_dim=True):
        super().__init__()
        self.egnn = egnn
        self.per_particle_energy = per_particle_energy
        self.repeat_dim = repeat_dim

    def forward(self, t, xs):
        out = self.egnn(t, xs) * xs
        if self.per_particle_energy:
            out = torch.sum(out.view(-1, self.egnn._n_particles, self.egnn._n_dimension), dim=-1)
            if self.repeat_dim:
                return out.repeat_interleave(self.egnn._n_dimension, dim=-1)
            else:
                return out
        else:
            return torch.sum(out, dim=-1, keepdim=True)


class EGNN_dynamics(nn.Module):
    def __init__(
        self,
        n_particles,
        n_dimension,
        hidden_nf=64,
        t_emb_dims=64,
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        tanh=False,
        agg="sum",
        use_pos_embedding=False,
    ):
        super().__init__()
        # Store the various dimensions
        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.t_emb_dims = t_emb_dims
        # Initialize the _edges_dict cache
        self.edges = self._create_edges()
        self._edges_dict = {}
        # Build the EGNN
        self.egnn = EGNN(
            in_node_nf=t_emb_dims,
            in_edge_nf=1,
            hidden_nf=hidden_nf,
            act_fn=act_fn,
            n_layers=n_layers,
            recurrent=recurrent,
            attention=attention,
            tanh=tanh,
            agg=agg
        )
        # Build the positional embedding
        if use_pos_embedding:
            self.timestep_embed = PositionalEmbedding(t_emb_dims)
        else:
            self.timestep_embed = TimeNet(
                dim_out=t_emb_dims,
                activation=torch.nn.GELU,
                num_layers=1,
                channels=t_emb_dims,
            )

    def forward(self, t, xs):
        n_batch = xs.shape[0]
        # Remove input mean
        xs = remove_mean(xs)
        # Build the edges
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles, xs.device)
        # Reshape the particles
        x = xs.view(n_batch * self._n_particles, self._n_dimension)
        # Embed time
        h = self.timestep_embed(t.squeeze(-1)).unsqueeze(1)
        h = h.expand((-1, self._n_particles, -1)).reshape((-1, self.t_emb_dims))
        # Run the EGNN
        edge_attr = torch.sum(torch.square(x[edges[0]] - x[edges[1]]), dim=1, keepdim=True)
        # Build the output
        _, x_final = self.egnn(h, x.clone(), edges, edge_attr=edge_attr)
        vel = x_final - x
        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        return vel

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes, device):
        if n_batch not in self._edges_dict:
            rows, cols = edges
            arr = torch.arange(n_batch, device=device)
            rows_total = rows.unsqueeze(0) + arr.unsqueeze(1) * n_nodes
            cols_total = cols.unsqueeze(0) + arr.unsqueeze(1) * n_nodes
            self._edges_dict[n_batch] = (rows_total.flatten(), cols_total.flatten())
        return self._edges_dict[n_batch]

    def _apply(self, fn):
        """Move the _edges_dict with the device"""
        new_self = super(EGNN_dynamics, self)._apply(fn)
        new_self.edges = (fn(new_self.edges[0]), fn(new_self.edges[1]))
        new_self._edges_dict = {
            k : (fn(v[0]), fn(v[1])) for k,v in new_self._edges_dict.items()
        }
        return new_self

class EGNN_atom(EGNN_dynamics):
    def __init__(
        self,
        n_particles,
        n_dimension,
        atom_type_labels: List,
        bonds: List,
        hidden_nf=128,
        act_fn=torch.nn.SiLU(),
        n_layers=6,
        recurrent=False,
        attention=True,
        tanh=False,
        agg="sum",
        time_embedding_dim=128,
        use_pos_embedding=False,
        atom_type_embedding_dim=64,
    ):
        super(EGNN_dynamics, self).__init__()
        # Store the various dimensions
        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.time_embedding_dim = time_embedding_dim
        self.atom_type_embedding_dim = atom_type_embedding_dim
        # Initialize the _edges_dict cache
        self.edges = self._create_edges()
        self._edges_dict = {}
        # Build the EGNN
        self.egnn = EGNN(
            in_node_nf=time_embedding_dim + atom_type_embedding_dim,
            in_edge_nf=2,
            hidden_nf=hidden_nf,
            act_fn=act_fn,
            n_layers=n_layers,
            recurrent=recurrent,
            attention=attention,
            tanh=tanh,
            agg=agg,
            norm_constant=1
        )
        # Register the atom types
        self.num_atom_types = len(set(atom_type_labels))
        self.register_buffer('atom_type_labels', torch.LongTensor(atom_type_labels))
        self.atom_type_embedding_layer = nn.Embedding(self.num_atom_types, atom_type_embedding_dim)
        # Create adjacency matrix for bond information
        adj = torch.zeros((self._n_particles, self._n_particles), dtype=torch.bool)
        for i, j in bonds:
            adj[i, j] = True
            adj[j, i] = True  # Assuming undirected bonds
        self.register_buffer('adj', adj)
        # Build the positional embedding
        if use_pos_embedding:
            self.timestep_embed = PositionalEmbedding(time_embedding_dim)
        else:
            self.timestep_embed = TimeNet(
                dim_out=time_embedding_dim,
                activation=torch.nn.GELU,
                num_layers=1,
                channels=time_embedding_dim,
            )

    def forward(self, t, xs):
        n_batch = xs.shape[0]
        # Remove input mean
        xs = remove_mean(xs)
        # Build the edges
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles, xs.device)
        # Embed the time
        time_emb = self.timestep_embed(t.squeeze(-1)).unsqueeze(1)
        time_emb = time_emb.expand((-1, self._n_particles, -1)).reshape((-1, self.time_embedding_dim))
        # Embed the atoms
        atom_type_emb = self.atom_type_embedding_layer(self.atom_type_labels)
        atom_type_emb = atom_type_emb.unsqueeze(0).expand((n_batch, -1, -1))
        atom_type_emb = atom_type_emb.reshape(n_batch * self._n_particles, -1)
        # Build the total embedding
        h = torch.cat([time_emb, atom_type_emb], dim=-1)
        # Compute edge attributes including bond information
        node_indices0 = edges[0] % self._n_particles
        node_indices1 = edges[1] % self._n_particles
        # Retrieve bond information
        bond_mask = self.adj[node_indices0, node_indices1].unsqueeze(1)
        # Reshape the input
        x = xs.view(n_batch * self._n_particles, self._n_dimension)
        # Compute squared distances
        distance_sq = torch.sum(torch.square(x[edges[0]] - x[edges[1]]), dim=1, keepdim=True)
        # Combine distance and bond information
        edge_attr = torch.cat([distance_sq, bond_mask], dim=1)
        _, x_final = self.egnn(h, x.clone(), edges, edge_attr=edge_attr)
        vel = x_final - x
        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        return vel

class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        act_fn=nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        norm_diff=True,
        out_node_nf=None,
        tanh=False,
        coords_range=15,
        agg="sum",
        norm_constant=1
    ):
        super().__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range) / self.n_layers
        if agg == "mean":
            self.coords_range_layer = self.coords_range_layer * 19
        # Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    attention=attention,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    coords_range=self.coords_range_layer,
                    agg=agg,
                    norm_constant=norm_constant
                ),
            )

    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](
                h,
                edges,
                x,
                edge_attr=edge_attr,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.

    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.SiLU(),
        recurrent=True,
        attention=False,
        clamp=False,
        norm_diff=True,
        tanh=False,
        coords_range=1,
        agg="sum",
        norm_constant=1,
    ):
        super().__init__()
        input_edge = input_nf * 2
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.agg_type = agg
        self.tanh = tanh
        self.norm_constant = norm_constant
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = coords_range

        self.coord_mlp = nn.Sequential(*coord_mlp)
        self.clamp = clamp

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        # if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)

    def edge_model(self, source, target, radial, edge_attr, edge_mask):
        # print("edge_model", radial, edge_attr)
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        if edge_mask is not None:
            out = out * edge_mask
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        # print("node_model", edge_attr)
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask):
        # print("coord_model", coord_diff, radial, edge_feat)
        row, col = edge_index
        if self.tanh:
            trans = coord_diff * self.coord_mlp(edge_feat) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(edge_feat)
        # trans = torch.clamp(trans, min=-100, max=100)
        if edge_mask is not None:
            trans = trans * edge_mask

        if self.agg_type == "sum":
            unsorted_segment_sum(trans, row, num_segments=coord.size(0), out=coord)
        elif self.agg_type == "mean":
            if node_mask is not None:
                # raise Exception('This part must be debugged before use')
                agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
                M = unsorted_segment_sum(node_mask[col], row, num_segments=coord.size(0))
                coord += agg / (M - 1)
            else:
                unsorted_segment_mean(trans, row, num_segments=coord.size(0), out=coord)
        else:
            raise Exception("Wrong coordinates aggregation type")
        return coord

    def forward(
        self,
        h,
        edge_index,
        coord,
        edge_attr=None,
        node_attr=None,
        node_mask=None,
        edge_mask=None,
    ):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, edge_mask)
        coord = self.coord_model(
            coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask
        )

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        # print("h", h)
        if node_mask is not None:
            h = h * node_mask
            coord = coord * node_mask
        return h, coord, edge_attr

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)

        norm = torch.sqrt(radial + 1e-8)
        coord_diff = coord_diff / (norm + self.norm_constant)

        return radial, coord_diff


def broadcast(src, other, dim):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
    """Imported from torch_scatter"""
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        # return out.scatter_add_(dim, index, src)
        return torch.scatter_add(out, dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    """Imported from torch_scatter"""
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)
    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1
    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out

def unsorted_segment_sum(data, segment_ids, num_segments, out=None):
    return scatter_sum(data, segment_ids, dim=0, dim_size=num_segments, out=out)

def unsorted_segment_mean(data, segment_ids, num_segments, out=None):
    return scatter_mean(data, segment_ids, dim=0, dim_size=num_segments, out=out)