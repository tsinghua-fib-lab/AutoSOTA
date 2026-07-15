"""
Graph Convolutional Network (GCN) model for node classification using PyTorch Geometric.

This module implements a 2-layer GCN for semi-supervised node classification on citation
networks like Cora. The model is designed to work with APPFL's federated learning framework.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) for node classification.

    Architecture:
        - Input: Node features
        - Layer 1: GCN convolution -> ReLU -> Dropout
        - Layer 2: GCN convolution -> Log Softmax
        - Output: Log probabilities for each class

    The model uses graph convolutions to aggregate information from neighboring nodes,
    enabling it to learn from the graph structure for node classification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 7,
        dropout: float = 0.5,
    ):
        """
        Initialize the GCN model.

        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layer
            output_dim: Number of output classes
            dropout: Dropout rate for regularization
        """
        super().__init__()

        # First graph convolution layer
        self.conv1 = GCNConv(input_dim, hidden_dim)

        # Second graph convolution layer
        self.conv2 = GCNConv(hidden_dim, output_dim)

        # Dropout for regularization
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass of the GCN.

        Args:
            x: Node feature matrix of shape [num_nodes, input_dim]
            edge_index: Graph connectivity in format of shape [2, num_edges]
            edge_weight: Optional edge weights of shape [num_edges]

        Returns:
            Log probabilities for each node of shape [num_nodes, output_dim]
        """
        # First layer: GCN -> ReLU -> Dropout
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second layer: GCN -> Log Softmax
        x = self.conv2(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    """
    Graph Attention Network (GAT) for node classification.

    This is an alternative to GCN that uses attention mechanisms to weight
    the importance of neighboring nodes differently.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 7,
        dropout: float = 0.5,
        heads: int = 8,
    ):
        """
        Initialize the GAT model.

        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layer (per attention head)
            output_dim: Number of output classes
            dropout: Dropout rate for regularization
            heads: Number of attention heads in the first layer
        """
        super().__init__()

        try:
            from torch_geometric.nn import GATConv
        except ImportError:
            raise ImportError(
                "GATConv requires a newer version of PyTorch Geometric. "
                "Please install with: pip install torch-geometric"
            )

        # First GAT layer with multi-head attention
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)

        # Second GAT layer
        self.conv2 = GATConv(
            hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout
        )

        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass of the GAT.

        Args:
            x: Node feature matrix of shape [num_nodes, input_dim]
            edge_index: Graph connectivity in format of shape [2, num_edges]

        Returns:
            Log probabilities for each node of shape [num_nodes, output_dim]
        """
        # First layer with multi-head attention
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        # Second layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE model for node classification.

    GraphSAGE uses sampling and aggregation to scale to large graphs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 7,
        dropout: float = 0.5,
    ):
        """
        Initialize the GraphSAGE model.

        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layer
            output_dim: Number of output classes
            dropout: Dropout rate for regularization
        """
        super().__init__()

        try:
            from torch_geometric.nn import SAGEConv
        except ImportError:
            raise ImportError(
                "SAGEConv requires PyTorch Geometric. "
                "Please install with: pip install torch-geometric"
            )

        # First SAGE layer
        self.conv1 = SAGEConv(input_dim, hidden_dim)

        # Second SAGE layer
        self.conv2 = SAGEConv(hidden_dim, output_dim)

        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass of GraphSAGE.

        Args:
            x: Node feature matrix of shape [num_nodes, input_dim]
            edge_index: Graph connectivity in format of shape [2, num_edges]

        Returns:
            Log probabilities for each node of shape [num_nodes, output_dim]
        """
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second layer
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
