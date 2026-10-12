import torch
from torch import nn
import torch.nn.functional as F

from torchdrug import layers, core
from torchdrug.core import Registry as R

from torch_cluster import knn_graph

from s3f import gvp_layer as layer
from s3f import surface


def rbf(d, d_min=0.0, d_max=20.0, dim=16):
    d_mu = torch.linspace(d_min, d_max, dim, device=d.device)
    d_mu = d_mu.view([1, -1])
    d_sigma = (d_max - d_min) / dim
    d_expand = torch.unsqueeze(d, -1)

    rbf = torch.exp(-((d_expand - d_mu) / d_sigma) ** 2)
    return rbf


@R.register("models.SurfGVP")
class SurfGVP(nn.Module, core.Configurable):

    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim, surf_in_dim, surf_edge_in_dim,
                 num_surf_graph_neighbor=16,
                 num_surf_res_neighbor=3, readout="sum",
                 num_layers=3, drop_rate=0.1,
                 activations=(F.relu, None), vector_gate=True):

        super().__init__()
        self.output_dim = node_h_dim[0]
        self.rbf_dim = edge_in_dim[0]
        self.surf_in_dim = surf_in_dim[0]
        self.surf_rbf_dim = surf_edge_in_dim[0]
        self.k = num_surf_res_neighbor
        self.num_surf_graph_neighbor = num_surf_graph_neighbor

        self.residue_embdding = nn.Linear(node_in_dim[0], node_in_dim[0], bias=False)
        self.W_v = nn.Sequential(
            layer.GVPLayerNorm(node_in_dim),
            layer.GVP(node_in_dim, node_h_dim, activations=(None, None), vector_gate=vector_gate)
        )
        self.W_e = nn.Sequential(
            layer.GVPLayerNorm(edge_in_dim),
            layer.GVP(edge_in_dim, edge_h_dim, activations=(None, None), vector_gate=vector_gate)
        )

        self.layers = nn.ModuleList(
                layer.GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate,
                                 activations=activations, vector_gate=vector_gate)
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            layer.GVPLayerNorm(node_h_dim),
            layer.GVP(node_h_dim, (ns, 0), activations=activations, vector_gate=vector_gate)
        )

        # Similar layer for surface graphs
        self.surf_in_linear = nn.Linear(node_in_dim[0]+1, node_in_dim[0], bias=False)
        self.surf_in_mlp = nn.Sequential(
                        nn.Linear(node_in_dim[0] + surf_in_dim[0], node_in_dim[0] * 2),
                        nn.Dropout(drop_rate),
                        nn.LayerNorm(node_in_dim[0] * 2),
                        nn.ReLU(),
                        nn.Linear(node_in_dim[0] * 2, node_in_dim[0]),
                    )
        self.surf_W_v = nn.Sequential(
            layer.GVPLayerNorm(node_in_dim),
            layer.GVP(node_in_dim, node_h_dim, activations=(None, None), vector_gate=vector_gate)
        )
        self.surf_W_e = nn.Sequential(
            layer.GVPLayerNorm(edge_in_dim),
            layer.GVP(edge_in_dim, edge_h_dim, activations=(None, None), vector_gate=vector_gate)
        )
        self.surf_layers = nn.ModuleList(
                layer.GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate,
                                 activations=activations, vector_gate=vector_gate)
            for _ in range(num_layers))
        ns, _ = node_h_dim
        self.surf_W_out = nn.Sequential(
            layer.GVPLayerNorm(node_h_dim),
            layer.GVP(node_h_dim, (ns, 0), activations=activations, vector_gate=vector_gate)
        )

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def residue2surface(self, graph, surf_graph):
        res2surf = graph.res2surf.flatten(1, 2)
        # Re-index after batching surface graphs
        res2surf = res2surf + (surf_graph.num_cum_nodes - surf_graph.num_nodes)[graph.residue2graph].unsqueeze(-1)  

    def surface_feature_init(self, graph, surf_graph, input):
        # Residue -> surface graph correspondence
        surf2res, dist = surface.knn_atoms(surf_graph.node_position, graph.node_position, k=self.k, batch_x=surf_graph.node2graph, batch_y=graph.node2graph)
        surf2res = surf2res[:, :self.k]
        dist = dist[:, :self.k].sqrt()
        
        # Knn graph for surface points
        surf_edge_index = knn_graph(surf_graph.node_position, k=self.num_surf_graph_neighbor, batch=surf_graph.node2graph)
        surf_node_in, surf_node_out = surf_edge_index
        surf_pos_in, surf_pos_out = surf_graph.node_position[surf_node_in], surf_graph.node_position[surf_node_out]
        surf_vec_edge = (surf_pos_in - surf_pos_out).unsqueeze(-2)  # [n_edge, 1, 3]
        h_surf_edge = rbf((surf_pos_out - surf_pos_in).norm(dim=-1), dim=self.surf_rbf_dim), surf_vec_edge

        # Inherit node features from residues with distance
        h_surf_node = torch.cat((input[surf2res.flatten()], dist.view(-1, 1)), dim=-1)
        h_surf_node = h_surf_node.view(surf_graph.num_node, self.k, -1)
        h_surf_node = self.surf_in_linear(h_surf_node)
        h_surf_node = h_surf_node.mean(dim=1)   # Average pooling of K neighbors on the residue graph
        h_surf_node = self.surf_in_mlp(torch.cat((h_surf_node, surf_graph.node_feature), dim=-1))

        return surf_edge_index, h_surf_node, h_surf_edge

    def forward(self, graph, input, surf_graph, all_loss=None, metric=None):      
        # Input features
        h_node = self.residue_embdding(input)

        edge_index = graph.edge_list.t()[:2]
        node_in, node_out = edge_index
        pos_in, pos_out = graph.node_position[node_in], graph.node_position[node_out]
        vec_edge = (pos_out - pos_in).unsqueeze(-2)  # [n_edge, 1, 3]
        h_edge = rbf((pos_out - pos_in).norm(dim=-1), dim=self.rbf_dim), vec_edge
        
        h_node = self.W_v(h_node)
        h_edge = self.W_e(h_edge)

        # Surface -> residue graph correspondence
        res2surf = self.residue2surface(graph, surf_graph)

        # Initialize surface graph, node and edge features
        surf_edge_index, h_surf_node, h_surf_edge = self.surface_feature_init(graph, surf_graph, input)
        h_surf_node = self.surf_W_v(h_surf_node)
        h_surf_edge = self.surf_W_e(h_surf_edge)

        for layer, surf_layer in zip(self.layers, self.surf_layers):
            h_node = layer(h_node, edge_index, h_edge)
            h_surf_node = surf_layer(h_surf_node, surf_edge_index, h_surf_edge)

        bb_node_feature = self.W_out(h_node)
        surf_node_feature = self.surf_W_out(h_surf_node)
        surf_node_feature = surf_node_feature[res2surf].mean(dim=1)     # Mean readout from surface to residue graph
        node_feature = bb_node_feature + surf_node_feature
        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }
    

@R.register("models.GVPGNN")
class GVPGNN(nn.Module, core.Configurable):
    '''
    Modified based on https://github.com/drorlab/gvp-pytorch/blob/main/gvp/models.py
    GVP-GNN for Model Quality Assessment as described in manuscript.
    
    Takes in protein structure graphs of type `torchdrug.data.Graph` 
    or `torchdrug.data.PackedGraph` and returns a scalar representation for
    each graph and node in the batch in a `torch.Tensor` of shapes [n_nodes, d]
    and [batch_size, d]
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim, readout="sum",
                 num_layers=3, drop_rate=0.1,
                 activations=(F.relu, None), vector_gate=True):

        super().__init__()
        self.output_dim = node_h_dim[0]
        self.rbf_dim = edge_in_dim[0]

        self.residue_embdding = nn.Linear(node_in_dim[0], node_in_dim[0], bias=False)
        self.W_v = nn.Sequential(
            layer.GVPLayerNorm(node_in_dim),
            layer.GVP(node_in_dim, node_h_dim, activations=(None, None), vector_gate=vector_gate)
        )
        self.W_e = nn.Sequential(
            layer.GVPLayerNorm(edge_in_dim),
            layer.GVP(edge_in_dim, edge_h_dim, activations=(None, None), vector_gate=vector_gate)
        )

        self.layers = nn.ModuleList(
                layer.GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate,
                                 activations=activations, vector_gate=vector_gate)
            for _ in range(num_layers))

        # Learnable weights for multi-layer feature aggregation (JK-Net style)
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            layer.GVPLayerNorm(node_h_dim),
            layer.GVP(node_h_dim, (ns, 0), activations=activations, vector_gate=vector_gate)
        )

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):      
        h_node = self.residue_embdding(input)

        edge_index = graph.edge_list.t()[:2]
        node_in, node_out = edge_index
        pos_in, pos_out = graph.node_position[node_in], graph.node_position[node_out]
        vec_edge = (pos_out - pos_in).unsqueeze(-2)  # [n_edge, 1, 3]
        h_edge = rbf((pos_out - pos_in).norm(dim=-1), dim=self.rbf_dim), vec_edge
        
        h_node = self.W_v(h_node)
        h_edge = self.W_e(h_edge)
        # Multi-layer feature aggregation (JK-Net style)
        layer_features = []
        for layer in self.layers:
            h_node = layer(h_node, edge_index, h_edge)
            layer_features.append(h_node)
        # Weighted aggregation via softmax over layer weights
        weights = torch.softmax(self.layer_weights, dim=0)
        h_agg_s = weights[0] * layer_features[0][0]
        h_agg_v = weights[0] * layer_features[0][1]
        for i in range(1, len(layer_features)):
            h_agg_s = h_agg_s + weights[i] * layer_features[i][0]
            h_agg_v = h_agg_v + weights[i] * layer_features[i][1]
        h_node = (h_agg_s, h_agg_v)
        node_feature = self.W_out(h_node)

        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }