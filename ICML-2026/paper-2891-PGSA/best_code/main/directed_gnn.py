import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch import nn, optim
from torch_sparse import sum as sparsesum
from torch_sparse import mul
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear
from torch_geometric.nn import (
    SAGEConv,
    GCNConv,
    GATConv,
    JumpingKnowledge,
)



def row_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=1)

    return mul(adj, 1 / row_sum.view(-1, 1))


def directed_norm(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))
    return adj


def get_norm_adj(adj, norm):
    if norm == "sym":
        return gcn_norm(adj, add_self_loops=False)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "dir":
        return directed_norm(adj)
    else:
        raise ValueError(f"{norm} normalization is not supported")
def get_conv(conv_type, input_dim, output_dim, alpha):
    if conv_type == "gcn":
        return GCNConv(input_dim, output_dim, add_self_loops=False)
    elif conv_type == "sage":
        return SAGEConv(input_dim, output_dim)
    elif conv_type == "gat":
        return GATConv(input_dim, output_dim, heads=1)
    elif conv_type == "dir-gcn":
        return DirGCNConv(input_dim, output_dim, alpha)
    elif conv_type == "dir-gcn-weighted":
        return DirGCNConv_weighted(input_dim, output_dim, alpha)
    elif conv_type == "1dir-gcn":
        return DirGCNConv_onedirected(input_dim, output_dim, alpha)
    elif conv_type == "dir-sage":
        return DirSageConv(input_dim, output_dim, alpha)
    elif conv_type == "dir-gat":
        return DirGATConv(input_dim, output_dim, heads=1, alpha=alpha)
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")


class DirGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirGCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        return self.alpha * self.lin_src_to_dst(self.adj_norm @ x) + (1 - self.alpha) * self.lin_dst_to_src(
            self.adj_t_norm @ x
        )
class DirGCNConv_weighted(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirGCNConv_weighted, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha = alpha
        #self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index, edge_weight=None):
        col, row = edge_index
        num_nodes = x.size(0)

        adj = SparseTensor(row=row, col=col, value=edge_weight,
                           sparse_sizes=(num_nodes, num_nodes)).coalesce()
        adj_norm = get_norm_adj(adj, norm="dir")


        adj_t = SparseTensor(row=col, col=row, value=edge_weight,
                             sparse_sizes=(num_nodes, num_nodes)).coalesce()
        adj_t_norm = get_norm_adj(adj_t, norm="dir")

        out = self.alpha * self.lin_src_to_dst(adj_norm @ x) + (1 - self.alpha) * self.lin_dst_to_src(adj_t_norm @ x)
        return out



class DirGCNConv_onedirected(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirGCNConv_onedirected, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        #self.lin_dst_to_src = Linear(input_dim, output_dim)
        #self.alpha = alpha
        #self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index, edge_weight=None):
        row, col = edge_index
        num_nodes = x.size(0)

        adj = SparseTensor(row=row, col=col, value=edge_weight,
                           sparse_sizes=(num_nodes, num_nodes)).coalesce()
        adj_norm = get_norm_adj(adj, norm="row")


        # adj_t = SparseTensor(row=col, col=row, value=edge_weight,
        #                      sparse_sizes=(num_nodes, num_nodes)).coalesce()
        # adj_t_norm = get_norm_adj(adj_t, norm="dir")

        out = self.lin_src_to_dst(adj_norm @ x)
        return out

class DirSageConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirSageConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = SAGEConv(input_dim, output_dim, flow="source_to_target", root_weight=False)
        self.conv_dst_to_src = SAGEConv(input_dim, output_dim, flow="target_to_source", root_weight=False)
        self.lin_self = Linear(input_dim, output_dim)
        self.alpha = alpha

    def forward(self, x, edge_index):
        return (
            self.lin_self(x)
            + (1 - self.alpha) * self.conv_src_to_dst(x, edge_index)
            + self.alpha * self.conv_dst_to_src(x, edge_index)
        )


class DirGATConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heads, alpha):
        super(DirGATConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = GATConv(input_dim, output_dim, heads=heads)
        self.conv_dst_to_src = GATConv(input_dim, output_dim, heads=heads)
        self.alpha = alpha

    def forward(self, x, edge_index):
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)

        return (1 - self.alpha) * self.conv_src_to_dst(x, edge_index) + self.alpha * self.conv_dst_to_src(
            x, edge_index_t
        )


class GNN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        num_layers=2,
        dropout=0,
        conv_type="dir-gcn",
        jumping_knowledge=False,
        normalize=False,
        alpha=1 / 2,
        learn_alpha=False,
    ):
        super(GNN, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)
        output_dim = hidden_dim if jumping_knowledge else num_classes
        if num_layers == 1:
            self.convs = ModuleList([get_conv(conv_type, num_features, output_dim, self.alpha)])
        else:
            self.convs = ModuleList([get_conv(conv_type, num_features, hidden_dim, self.alpha)])
            for _ in range(num_layers - 2):
                self.convs.append(get_conv(conv_type, hidden_dim, hidden_dim, self.alpha))
            self.convs.append(get_conv(conv_type, hidden_dim, output_dim, self.alpha))

        if jumping_knowledge is not None:
            input_dim = hidden_dim * num_layers if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, num_classes)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize

    def forward(self, x, edge_index,edge_weight=None):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index,edge_weight)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]

        if self.jumping_knowledge is not None:
            x = self.jump(xs)
            x = self.lin(x)

        return x


def get_model(input_dim,output_dim,args):
    return GNN(
        num_features=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.K,
        num_classes=args.conv_dim,
        dropout=args.dropout,
        conv_type=args.conv_type,
        jumping_knowledge=args.jk,
        normalize=args.bn,
        alpha=args.alpha,
        learn_alpha=args.learn_alpha,
    )