import math

import torch
from e3nn.o3 import spherical_harmonics
from torch import nn
from torch_geometric.nn.pool import global_mean_pool


class BigE3NN(nn.Module):
    def __init__(
        self,
        F_in: int,
        n_classes: int,
        cutoff: float = 8.0,
        K: int = 32,
        hidden: int = 256,
        num_layers: int = 10,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.hidden = hidden
        self.num_layers = num_layers

        self.register_buffer("rbf_centers", torch.linspace(0.0, cutoff, K))
        self.rbf_gamma = 10.0

        self.init_mlp = nn.Sequential(
            nn.Linear(F_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.rbf_mlp = nn.Sequential(
            nn.Linear(K, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_classes),
        )

    def rbf(self, d):
        return torch.exp(-self.rbf_gamma * (d[:, None] - self.rbf_centers[None, :]) ** 2)

    def forward(self, pos, x, edge_index, edge_vec, dist, batch_idx):
        src, dst = edge_index
        h = self.init_mlp(x)
        y1 = spherical_harmonics(1, edge_vec, normalize=True)
        r_feat = self.rbf_mlp(self.rbf(dist))

        for _ in range(self.num_layers):
            g_x = self.node_mlp(h[src])
            coeff = r_feat * g_x
            m_vec = coeff.unsqueeze(-1) * y1.unsqueeze(-2)
            n_nodes = h.size(0)
            agg_vec = torch.zeros(n_nodes, self.hidden, 3, device=h.device).index_add_(0, dst, m_vec)
            h = h + self.update_mlp(agg_vec.norm(dim=-1))

        s_graph = global_mean_pool(h, batch_idx)
        logits = self.readout(s_graph)
        node_vec = agg_vec.mean(dim=1)
        return logits, node_vec


class IntermediateAttnPooling(nn.Module):
    def __init__(self, hidden: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.key = nn.Linear(hidden, hidden * heads, bias=False)
        self.val = nn.Linear(hidden, hidden * heads, bias=False)
        self.q = nn.Parameter(torch.randn(heads, hidden))
        self.out = nn.Linear(hidden * heads, hidden)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, seq_list):
        stacked = torch.stack(seq_list, dim=1)
        batch_size, seq_len, hidden = stacked.shape

        keys = self.key(stacked).view(batch_size, seq_len, self.heads, self.hidden)
        values = self.val(stacked).view(batch_size, seq_len, self.heads, self.hidden)

        scores = (keys * self.q.unsqueeze(0).unsqueeze(0)).sum(-1) / math.sqrt(self.hidden)
        scores = scores.permute(0, 2, 1)
        alpha = self.drop(torch.softmax(scores, dim=-1))

        values = values.permute(0, 2, 1, 3).contiguous()
        flat_batch = batch_size * self.heads
        alpha_flat = alpha.reshape(flat_batch, 1, seq_len)
        values_flat = values.reshape(flat_batch, seq_len, hidden)
        pooled_flat = torch.bmm(alpha_flat, values_flat).squeeze(1)
        pooled = pooled_flat.view(batch_size, self.heads * self.hidden)

        return self.out(pooled)


class StatefulVN(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(hidden, 2 * hidden),
            nn.LayerNorm(2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, 2 * hidden),
            nn.LayerNorm(2 * hidden),
        )
        self.gru = nn.GRUCell(2 * hidden, hidden)
        self.post = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

    def forward(self, h, vn_state, batch_idx):
        pooled = global_mean_pool(self.pre(h), batch_idx)
        vn_state = self.gru(pooled, vn_state)
        return self.post(vn_state), vn_state


class BigE3NN_NoSH(nn.Module):
    def __init__(
        self,
        F_in: int,
        n_classes: int,
        cutoff: float = 8.0,
        K: int = 64,
        hidden: int = 128,
        num_layers: int = 10,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.hidden = hidden
        self.num_layers = num_layers

        self.register_buffer("rbf_centers", torch.linspace(0.0, cutoff, K))
        self.rbf_gamma = 10.0

        self.stateful_vn = StatefulVN(hidden)
        self.stateful_vn2 = StatefulVN(hidden)

        self.init_mlp = nn.Sequential(
            nn.Linear(F_in, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.rbf_mlp = nn.Sequential(
            nn.Linear(K, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

        self.node_mlps = nn.ModuleList()
        self.update_mlps = nn.ModuleList()
        for _ in range(num_layers):
            self.node_mlps.append(
                nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden),
                )
            )
            self.update_mlps.append(
                nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden),
                )
            )

        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * hidden),
            nn.LayerNorm(2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, n_classes),
        )
        self.virtual_node_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.int_attn = IntermediateAttnPooling(hidden=hidden, heads=4, dropout=0.0)

    def rbf(self, d):
        return torch.exp(-self.rbf_gamma * (d[:, None] - self.rbf_centers[None, :]) ** 2)

    def forward(self, pos, x, edge_index, edge_vec, dist, batch_idx, vn=True, int_embs=False):
        src, dst = edge_index
        h = self.init_mlp(x)
        u = edge_vec / (dist.unsqueeze(-1) + 1e-8)
        r_feat = self.rbf_mlp(self.rbf(dist))
        last_agg_vec = None

        if int_embs:
            int_graph_embs = []

        n_graphs = int(batch_idx.max()) + 1
        vn_state = torch.zeros(n_graphs, self.hidden, device=h.device, dtype=h.dtype) if vn else None
        vn_state2 = torch.zeros(n_graphs, self.hidden, device=h.device, dtype=h.dtype) if vn else None

        for i in range(self.num_layers):
            g_x = self.node_mlps[i](h[src])
            coeff = r_feat * g_x
            m_vec = coeff.unsqueeze(-1) * u.unsqueeze(-2)

            agg_vec = torch.zeros(h.size(0), self.hidden, 3, device=h.device, dtype=h.dtype)
            agg_vec.index_add_(0, dst, m_vec)
            last_agg_vec = agg_vec

            h = h + self.update_mlps[i](agg_vec.norm(dim=-1))

            if int_embs:
                int_graph_embs.append(global_mean_pool(h, batch_idx))

            if vn:
                vn_msg, vn_state = self.stateful_vn(h, vn_state, batch_idx)
                vn_msg2, vn_state2 = self.stateful_vn2(h, vn_state2, batch_idx)
                vn_msgs = self.virtual_node_mlp(torch.cat([vn_msg, vn_msg2], dim=-1))
                h = h + vn_msgs[batch_idx]

        s_graph = self.int_attn(int_graph_embs) if int_embs else global_mean_pool(h, batch_idx)
        logits = self.readout(s_graph)
        node_vec = last_agg_vec.mean(dim=1) if last_agg_vec is not None else None
        return logits, node_vec
