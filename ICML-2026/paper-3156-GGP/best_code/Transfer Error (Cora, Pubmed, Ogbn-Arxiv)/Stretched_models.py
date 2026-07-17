import torch
import torch.nn.functional as F
import os
from torch_geometric.nn import MessagePassing

# A custom sparse aggregation layer for specific adjacency matrix logic
class SpecificAggrLayer(MessagePassing):
    def __init__(self):
        # aggr='add' corresponds to "summation" (A * X) in adjacency matrix multiplication
        super(SpecificAggrLayer, self).__init__(aggr='add', flow='source_to_target')

    def forward(self, x, edge_index):
        num_2edges = edge_index.size(1)

        # 1. Compute A * X (sum over all neighbor features)
        ax = self.propagate(edge_index, x=x)

        # 2. Apply formula: (1 / sqrt(2|E|)) * AX + X
        norm_factor = 1.0 / (num_2edges ** 0.5)

        out = ax * norm_factor + x
        return out

    def message(self, x_j):
        return x_j


# 1. Define the GCN models (Direct output before log_softmax)
class GCN_direct_2layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_direct_2layer, self).__init__()
        self.aggr_layer = SpecificAggrLayer()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)
        # ALGO-05: LayerNorm after each ReLU to normalize feature magnitudes
        # Independent of graph-size-dependent aggregation scale
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.norm2 = torch.nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        x = self.aggr_layer(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.norm1(x)    # ALGO-05: normalize post-aggregation features
        x = self.aggr_layer(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.norm2(x)    # ALGO-05: normalize output features
        return x


class GCN_direct_3layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_direct_3layer, self).__init__()
        self.aggr_layer = SpecificAggrLayer()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = torch.nn.Linear(hidden_channels, out_channels)
        # ALGO-05: LayerNorm after each ReLU
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        self.norm3 = torch.nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        x = self.aggr_layer(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.norm1(x)
        x = self.aggr_layer(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.norm2(x)
        x = self.aggr_layer(x, edge_index)
        x = self.fc3(x)
        x = self.norm3(x)
        return x


# 2. Initialize or load pretrained weights
def load_model(num_layers, model_path, in_channels, hidden_channels, out_channels, device):
    if num_layers == 2:
        model = GCN_direct_2layer(in_channels, hidden_channels, out_channels).to(device)
    elif num_layers == 3:
        model = GCN_direct_3layer(in_channels, hidden_channels, out_channels).to(device)
    else:
        raise ValueError(f"Unsupported num_layers={num_layers}")

    if model_path is not None and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded pretrained model from {model_path}")
    else:
        print("No pretrained model found. Initialized with random weights.")

    return model


# Returns the feature tensor of all nodes with shape (NumNodes, OutChannels)
def get_standard(data, model, device):
    """Compute the standard output features for all nodes on the full graph."""
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
    return out
