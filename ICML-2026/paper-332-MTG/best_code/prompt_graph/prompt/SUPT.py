from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.init import orthogonal_ as orthogonal

from torch import Tensor
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import GATConv, GINConv, GCNConv, SGConv
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
    to_dense_adj,
    to_dense_batch,
    select,
    softmax,
    scatter,
)


class SAGPoolPrompt(nn.Module):
    def __init__(self, in_channels: int, num_clusters=1, ratio=0.5, orth_loss=False, softmax=False,
                 nonlinearity='tanh'):
        super(SAGPoolPrompt, self).__init__()
        # self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))

        self.pools = nn.ModuleList()
        self.pools_emb = nn.Parameter(torch.Tensor(num_clusters, in_channels))
        self.num_clusters = num_clusters

        # self.gnn = SGConv(in_channels, 4, K=2)
        # self.linear = nn.Linear(4, num_clusters)

        self.gnn = GCNConv(in_channels, num_clusters)

        if isinstance(nonlinearity, str):
            self.nonlinearity = getattr(torch, nonlinearity)
        self.ratio = ratio
        self.softmax = softmax

        self.orth_loss = orth_loss
        if orth_loss:
            self.eye = torch.eye(num_clusters).cuda()
            self.eye.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()
        # self.linear.reset_parameters()
        if self.num_clusters > 1 and self.orth_loss:
            orthogonal(self.pools_emb)
        else:
            # normal(self.pools_emb)
            glorot(self.pools_emb)
        # glorot(self.global_emb)
        # normal(self.global_emb)

    def orthogonal_loss(self):
        return torch.norm(torch.mm(self.pools_emb, self.pools_emb.t()) - self.eye, p=2)

    def add(self, x: Tensor, edge_index: Tensor, batch: Tensor):
        p = torch.zeros_like(x)
        # x = x+self.global_emb
        # score = self.linear(torch.relu(self.gnn(x+self.pools_emb.sum(dim=0), edge_index)))
        score = self.gnn(x + self.pools_emb.sum(dim=0), edge_index)

        if self.softmax:
            score = softmax(score, batch)
        else:
            score = self.nonlinearity(score)

        idx_count = torch.ones_like(batch)
        for i in range(self.num_clusters):
            perm = topk(score[:, i], self.ratio, batch, None)
            p_ = score[:, i][perm].unsqueeze(1) * self.pools_emb[i]
            if self.orth_loss: p_ = p_ / self.num_clusters
            idx_count[perm] += 1
            p[perm] += p_
            # pdb.set_trace()
        # print((p/idx_count.unsqueeze(1)).max())
        return x + p / idx_count.unsqueeze(1)  # + self.global_emb


class DiffPoolPrompt(nn.Module):
    def __init__(self, in_channels: int, num_clusters=10, orth_loss=False):
        super(DiffPoolPrompt, self).__init__()
        # if num_clusters==1:
        #     assert False, "DiffPoolPrompt should have num_clusters>1"

        # self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.cluster_emb = nn.Parameter(torch.Tensor(num_clusters, in_channels))

        self.num_clusters = num_clusters

        # self.gnn_pool = SGConv(in_channels, 4, K=2)
        # self.linear = nn.Linear(4, num_clusters)

        self.gnn_pool = GCNConv(in_channels, num_clusters)

        # self.gnn_pool = DenseGCNConv(in_channels, num_clusters)
        # self.gnn_pool = DenseSAGEConv(in_channels, num_clusters)
        self.link_loss = 0
        self.ent_loss = 0
        self.orth_loss = orth_loss
        if orth_loss:
            self.eye = torch.eye(num_clusters).cuda()
            self.eye.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn_pool.reset_parameters()
        # self.linear.reset_parameters()
        # glorot(self.global_emb)
        if self.num_clusters > 1 and self.orth_loss:
            orthogonal(self.cluster_emb)
        else:
            # normal(self.pools_emb)
            glorot(self.cluster_emb)

    def orthogonal_loss(self):
        return torch.norm(torch.mm(self.cluster_emb, self.cluster_emb.t()) - self.eye, p=2)

    def add(self, x: Tensor, edge_index: Tensor, batch: Tensor):
        # x = x*self.global_emb

        # s = softmax(self.gnn_pool(x+self.cluster_emb.sum(dim=0), edge_index),batch)

        # s = self.linear(torch.relu(torch.softmax(self.gnn_pool(x+self.cluster_emb.sum(dim=0), edge_index), dim=1)))
        s = torch.softmax(self.gnn_pool(x + self.cluster_emb.sum(dim=0), edge_index), dim=1)

        # pdb.set_trace()

        # top_s_score, top_s_idx = torch.topk(s, 2, dim=1)
        # selected = self.cluster_emb[top_s_idx]
        # p = (selected*top_s_score.unsqueeze(-1)).sum(dim=1)

        p = torch.matmul(s, self.cluster_emb)

        # if self.training:
        #     self.ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()
        # s_out, _ = to_dense_batch(s, batch)
        # adj = to_dense_adj(edge_index, batch)
        # self.link_loss = torch.norm(adj-torch.matmul(s_out, s_out.transpose(1, 2)), p=2) / adj.numel()

        return x + p  # + self.global_emb


def topk(
        x: Tensor,
        ratio: Optional[Union[float, int]],
        batch: Tensor,
        min_score: Optional[float] = None,
        tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)

    elif ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')
        batch_size, max_num_nodes = num_nodes.size(0), int(num_nodes.max())

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes,), -60000.0)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0),), int(ratio))
            k = torch.min(k, num_nodes)
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        if isinstance(ratio, int) and (k == ratio).all():
            # If all graphs have exactly `ratio` or more than `ratio` entries,
            # we can just pick the first entries in `perm` batch-wise:
            index = torch.arange(batch_size, device=x.device) * max_num_nodes
            index = index.view(-1, 1).repeat(1, ratio).view(-1)
            index += torch.arange(ratio, device=x.device).repeat(batch_size)
        else:
            # Otherwise, compute indices per graph:
            index = torch.cat([
                torch.arange(k[i], device=x.device) + i * max_num_nodes
                for i in range(batch_size)
            ], dim=0)

        perm = perm[index]

    else:
        raise ValueError("At least one of 'min_score' and 'ratio' parameters "
                         "must be specified")

    return perm