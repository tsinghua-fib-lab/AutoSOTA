import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

def interval_mse_loss(pred, target, tau,alpha_tau, reduction='mean'):
    """
    pred: (B, 1)
    target: (B, 1)
    tau: (B, 1), 
    """
    pred = pred.view(-1)
    target = target.view(-1)
    mse = (pred - target)**2
    modulated_loss = mse / (1.0 +alpha_tau* tau)
    
    if reduction == 'mean':
        return modulated_loss.mean()
    elif reduction == 'sum':
        return modulated_loss.sum()
    else:
        return modulated_loss

def rank_loss_comp(y_true, y_pred, k=1.0):
    y_true = y_true.view(-1, 1)
    y_pred = y_pred.view(-1, 1)
    n = y_true.size(0)
    diff_true = y_true - y_true.T        # (N, N)
    diff_pred = y_pred - y_pred.T        # (N, N)
    mask = torch.triu(
        torch.ones(n, n, device=y_true.device),
        diagonal=1
    ).bool()

    concordance = torch.sigmoid(k * diff_true[mask] * diff_pred[mask])
    return 1.0 - concordance.mean()

class RankAwareEncoder(nn.Module):
    """
    排序可信编码器：专注于学习样本间的相对排序关系
    设计理念：通过排序扰动增强对排序信息的敏感性
    """
    def __init__(self, input_dim, hidden_dim, rank_dim, state_embed_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False)
        
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, rank_dim),
            nn.Sigmoid() 
        )
        self.fidelity_proj = nn.Linear(state_embed_dim, rank_dim)
        self.rank_attention = nn.Sequential(
            nn.Linear(rank_dim, 1),
            nn.Sigmoid()
        )
        
    
    def forward(self, x, s):
        x_norm = self.bn(x)
        rank_features = self.feature_net(x_norm)
        fidelity_cond = self.fidelity_proj(s)
        rank_features = rank_features * fidelity_cond
        attention_weights = self.rank_attention(rank_features)
        rank_features = rank_features * attention_weights
        
        return rank_features


class IntervalAwareEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_embed_dim):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        self.tau_head = nn.Sequential(
            nn.Linear(hidden_dim + state_embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()   # τ >= 0
        )

    def forward(self, x, s):
        feat = self.feature_extractor(x)
        tau = self.tau_head(torch.cat([feat, s], dim=-1))
        return tau  # (B, 1)

class GINRegressorModes(nn.Module):
    def __init__(
        self,
        node_feat_dim=16,
        edge_feat_dim=100,
        hidden_dim=64,
        num_layers=3,
        mode='state_emb',
        n_states=3,
        alpha_rank=1.0,
        alpha_tau=1e-10,
    ):
        super().__init__()
        assert mode in ['state_emb']
        self.mode = mode
        self.alpha_rank = alpha_rank
        self.alpha_tau = alpha_tau

        # -------- GIN backbone --------
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_dim = node_feat_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # -------- state embedding --------
        self.state_emb = nn.Embedding(n_states, hidden_dim)

        # -------- readout --------
        self.readout = global_add_pool

        self.rank_encoder = RankAwareEncoder(input_dim=hidden_dim, hidden_dim=hidden_dim, rank_dim=hidden_dim,state_embed_dim=hidden_dim)

        self.rank_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.interval_head = IntervalAwareEncoder(hidden_dim,hidden_dim,hidden_dim)

        # -------- evidential head --------
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    # --------------------------------------------------
    def forward(self, data, fidelity):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        state = fidelity
        y = data.y.view(-1,1)
        s_emb = self.state_emb(state)
        
        # -------- GIN --------
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
        g = self.readout(x, batch)
        
        # -------- rank encoding --------
        rank_feat = self.rank_encoder(g, s_emb)
        y_rank = self.rank_head(rank_feat)
        
        # -------- interval τ --------
        tau = self.interval_head(g, s_emb)
        
        # -------- MSE prediction --------
        enhanced = torch.concat((rank_feat, g, s_emb), dim=-1)
        mu = self.mlp(enhanced) 
        # tau modulation
        loss = interval_mse_loss(mu, y, tau,self.alpha_tau)
        
        # -------- rank loss (optional) --------
        rank_loss = 0.0
        for s in state.unique():
            mask = (state == s)
            if mask.sum() > 1:
                rank_loss += rank_loss_comp(y[mask], y_rank[mask])
        rank_loss /= len(state.unique())
        
        total_loss = loss  + self.alpha_rank * rank_loss
        
        return {
            "loss": total_loss,
            "mu": mu.detach(),
            "tau": tau.detach(),
            "mse_loss": loss.item(),
            "rank_loss": rank_loss.item(),
            "mae": torch.mean(torch.abs(mu - y)).item()}

