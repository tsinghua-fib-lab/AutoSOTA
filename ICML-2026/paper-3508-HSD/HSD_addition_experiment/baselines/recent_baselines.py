"""
Recent neural operator baselines (2023-2024).

GNOT: General Neural Operator Transformer (Hao et al., 2023)
ONO:  Orthogonal Neural Operator (Wang et al., 2023)
HAMLET: Hierarchical Attention for Mesh Learning (Bryutkin et al., 2024)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GNOTLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, chunk_size=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.chunk_size = chunk_size

    def forward(self, x, context=None):
        B, N, D = x.shape
        C = self.chunk_size
        if N <= C:
            h = self.norm1(x + self.attn(x, x, x)[0])
        else:
            pad = (C - N % C) % C
            x_pad = F.pad(x, (0, 0, 0, pad)) if pad > 0 else x
            N_pad = x_pad.shape[1]
            n_chunks = N_pad // C
            x_chunks = x_pad.reshape(B * n_chunks, C, D)
            attn_out = self.attn(x_chunks, x_chunks, x_chunks)[0]
            attn_out = attn_out.reshape(B, N_pad, D)[:, :N, :]
            h = self.norm1(x + attn_out)
        return self.norm2(h + self.ffn(h))


class GNOT(nn.Module):
    """General Neural Operator Transformer (Hao et al., 2023)."""
    def __init__(self, n_nodes, out_dim=3, hidden=128, n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        self.global_branch = nn.Sequential(
            nn.Linear(n_nodes, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.input_proj = nn.Linear(1, hidden)
        self.pos_enc = nn.Linear(3, hidden)
        self.layers = nn.ModuleList([GNOTLayer(hidden, n_heads, dropout) for _ in range(n_layers)])
        self.output_proj = nn.Linear(hidden, out_dim)

    def forward(self, x_scalar, pts):
        B = x_scalar.shape[0]
        global_feat = self.global_branch(x_scalar).unsqueeze(1)
        local_feat = self.input_proj(x_scalar.unsqueeze(-1)) + self.pos_enc(pts).unsqueeze(0)
        tokens = local_feat + global_feat
        for layer in self.layers:
            tokens = layer(tokens)
        return self.output_proj(tokens)


class ONOLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_basis=32, dropout=0.1):
        super().__init__()
        self.n_basis = n_basis
        self.basis_q = nn.Parameter(torch.randn(n_basis, d_model) * 0.02)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.scale = d_model ** -0.5

    def forward(self, x):
        B, N, D = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        basis = F.normalize(self.basis_q, dim=-1).unsqueeze(0)
        attn_scores = torch.matmul(q, basis.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        v_basis = torch.matmul(attn_scores.transpose(-1, -2), v) / (N + 1e-6)
        out = torch.matmul(attn_weights, v_basis)
        h = self.norm1(x + self.W_out(out))
        return self.norm2(h + self.ffn(h))


class ONO(nn.Module):
    """Orthogonal Neural Operator (Wang et al., 2023)."""
    def __init__(self, n_nodes, out_dim=3, hidden=128, n_heads=4, n_layers=4, n_basis=32, dropout=0.1):
        super().__init__()
        self.global_branch = nn.Sequential(
            nn.Linear(n_nodes, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.input_proj = nn.Linear(4, hidden)
        self.layers = nn.ModuleList([ONOLayer(hidden, n_heads, n_basis, dropout) for _ in range(n_layers)])
        self.output_proj = nn.Linear(hidden, out_dim)

    def forward(self, x_scalar, pts):
        B = x_scalar.shape[0]
        global_feat = self.global_branch(x_scalar).unsqueeze(1)
        inp = torch.cat([x_scalar.unsqueeze(-1), pts.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        h = self.input_proj(inp) + global_feat
        for layer in self.layers:
            h = layer(h)
        return self.output_proj(h)


class HAMLETBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.layer = GNOTLayer(d_model, n_heads, dropout)

    def forward(self, x):
        return self.layer(x)


class HAMLET(nn.Module):
    """Hierarchical Attention Mesh Learning Transformer (Bryutkin et al., 2024)."""
    def __init__(self, n_nodes, out_dim=3, hidden=96, n_heads=4, n_layers=3, pool_ratio=4, dropout=0.1):
        super().__init__()
        self.global_branch = nn.Sequential(
            nn.Linear(n_nodes, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.input_proj = nn.Linear(4, hidden)
        n_coarse = max(n_nodes // pool_ratio, 32)
        self.n_coarse = n_coarse
        self.pool_proj = nn.Linear(hidden, hidden)
        self.unpool_proj = nn.Linear(hidden, hidden)
        self.fine_blocks = nn.ModuleList([HAMLETBlock(hidden, n_heads, dropout) for _ in range(n_layers // 2 + 1)])
        self.coarse_blocks = nn.ModuleList([HAMLETBlock(hidden, n_heads, dropout) for _ in range(n_layers // 2 + 1)])
        self.output_proj = nn.Linear(hidden, out_dim)
        self.n_nodes = n_nodes

    def forward(self, x_scalar, pts):
        B = x_scalar.shape[0]
        global_feat = self.global_branch(x_scalar).unsqueeze(1)
        inp = torch.cat([x_scalar.unsqueeze(-1), pts.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        h = self.input_proj(inp) + global_feat

        for blk in self.fine_blocks[:len(self.fine_blocks) // 2]:
            h = blk(h)

        h_coarse = self.pool_proj(h[:, ::h.shape[1] // self.n_coarse, :])
        for blk in self.coarse_blocks:
            h_coarse = blk(h_coarse)

        ratio = (h.shape[1] + h_coarse.shape[1] - 1) // h_coarse.shape[1]
        h_up = h_coarse.repeat(1, ratio, 1)[:, :h.shape[1], :]
        h = h + self.unpool_proj(h_up)

        for blk in self.fine_blocks[len(self.fine_blocks) // 2:]:
            h = blk(h)

        return self.output_proj(h)


def train_recent_baseline(model, X_train, Y_train, X_val, Y_val, pts, epochs, cfg, device):
    """Train GNOT/ONO/HAMLET. Input: (x_scalar, pts) → (B, N, out_dim)."""
    from hodge_spectral.utils import LpLoss, EarlyStopping

    pts_t = torch.from_numpy(pts.astype(np.float32)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', 3e-3),
                                   weight_decay=cfg.get('weight_decay', 1e-5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_criterion = torch.nn.MSELoss()
    rel_criterion = LpLoss()
    early_stop = EarlyStopping(patience=cfg.get('patience', 30))

    bs = cfg.get('batch_size', 64)
    best_val, best_state = float('inf'), None
    train_losses, val_losses = [], []

    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(X_train))
        total_loss, n_batch = 0, 0
        for i in range(0, len(X_train), bs):
            batch_idx = idx[i:i + bs]
            xb = X_train[batch_idx].to(device)
            yb = Y_train[batch_idx].to(device)
            optimizer.zero_grad()
            pred = model(xb, pts_t)
            if pred.dim() == 3 and pred.shape[-1] == 1 and yb.dim() == 2:
                pred = pred.squeeze(-1)
            criterion = mse_criterion if ep < epochs // 2 else rel_criterion
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        scheduler.step()
        train_losses.append(total_loss / n_batch)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device), pts_t)
            val_target = Y_val.to(device)
            if val_pred.dim() == 3 and val_pred.shape[-1] == 1 and val_target.dim() == 2:
                val_pred = val_pred.squeeze(-1)
            val_loss = rel_criterion(val_pred, val_target).item()
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if early_stop(val_loss):
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses
