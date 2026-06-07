from torch_geometric.nn.pool import *
from fgl.model.gcn import GCN


def load_node_edge_level_default_model(args, input_dim, output_dim, client_id=None):
    return GCN(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
