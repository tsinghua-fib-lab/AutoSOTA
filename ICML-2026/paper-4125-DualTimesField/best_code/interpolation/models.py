import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TimeEncoding(nn.Module):
    def __init__(self, embed_dim, num_frequencies=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frequencies = num_frequencies
        freqs = torch.exp(torch.linspace(0, np.log(1000), num_frequencies))
        self.register_buffer('freqs', freqs)
        self.linear = nn.Linear(num_frequencies * 2, embed_dim)
    
    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        angles = t * self.freqs
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.linear(encoding)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        B, T_q, _ = query.shape
        T_k = key.shape[1]
        
        q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None and mask.any():
            mask_expanded = mask.view(1, 1, 1, T_k).expand(B, self.num_heads, T_q, T_k)
            scores = scores.masked_fill(~mask_expanded, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        return self.out_proj(out)


class ContinuousTimeField(nn.Module):
    def __init__(self, num_variables, embed_dim=64, num_frequencies=16):
        super().__init__()
        self.num_variables = num_variables
        self.embed_dim = embed_dim
        
        self.time_enc = TimeEncoding(embed_dim, num_frequencies)
        self.value_enc = nn.Linear(num_variables, embed_dim)
        
        self.cross_attn = CrossAttention(embed_dim, num_heads=4)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.output = nn.Linear(embed_dim, num_variables)
    
    def forward(self, query_times, obs_times, obs_values, obs_mask):
        query_emb = self.time_enc(query_times)
        
        obs_time_emb = self.time_enc(obs_times)
        obs_value_emb = self.value_enc(obs_values)
        obs_emb = obs_time_emb + obs_value_emb
        
        if query_emb.dim() == 2:
            query_emb = query_emb.unsqueeze(0)
        if obs_emb.dim() == 2:
            obs_emb = obs_emb.unsqueeze(0)
        
        if obs_mask.dim() > 1:
            time_mask = obs_mask.any(dim=-1)
        else:
            time_mask = obs_mask
        
        attn_out = self.cross_attn(query_emb, obs_emb, obs_emb, time_mask)
        query_emb = self.norm1(query_emb + attn_out)
        
        ffn_out = self.ffn(query_emb)
        query_emb = self.norm2(query_emb + ffn_out)
        
        return self.output(query_emb.squeeze(0))


class DiscreteGeometricField(nn.Module):
    def __init__(self, num_variables, num_atoms=16, embed_dim=64):
        super().__init__()
        self.num_variables = num_variables
        self.num_atoms = num_atoms
        
        self.tau = nn.Parameter(torch.linspace(0.05, 0.95, num_atoms))
        self.log_sigma = nn.Parameter(torch.zeros(num_atoms) - 1.0)
        self.omega = nn.Parameter(torch.linspace(5, 30, num_atoms))
        self.phi = nn.Parameter(torch.zeros(num_atoms))
        
        self.amplitude_net = nn.Sequential(
            nn.Linear(num_variables, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_atoms * num_variables)
        )
    
    def forward(self, query_times, obs_values, obs_mask):
        if query_times.dim() == 1:
            t = query_times.unsqueeze(-1)
        else:
            t = query_times
        
        obs_mean = (obs_values * obs_mask.float()).sum(dim=0) / (obs_mask.float().sum(dim=0) + 1e-8)
        amplitude = self.amplitude_net(obs_mean).view(self.num_atoms, self.num_variables)
        
        tau = self.tau.view(1, -1)
        sigma = F.softplus(self.log_sigma).view(1, -1) + 0.02
        omega = F.softplus(self.omega).view(1, -1)
        phi = self.phi.view(1, -1)
        
        envelope = torch.exp(-((t - tau) ** 2) / (2 * sigma ** 2))
        oscillation = torch.cos(omega * (t - tau) + phi)
        basis = envelope * oscillation
        
        return torch.matmul(basis, amplitude)


class DualTimesFieldInterpolator(nn.Module):
    def __init__(
        self,
        num_variables,
        embed_dim=64,
        num_frequencies=16,
        num_atoms=16
    ):
        super().__init__()
        self.num_variables = num_variables
        
        self.ctf = ContinuousTimeField(
            num_variables=num_variables,
            embed_dim=embed_dim,
            num_frequencies=num_frequencies
        )
        
        self.dgf = DiscreteGeometricField(
            num_variables=num_variables,
            num_atoms=num_atoms,
            embed_dim=embed_dim
        )
        
        self.combine = nn.Linear(num_variables * 2, num_variables)
        
        self.t_min = None
        self.t_max = None
        self.obs_times = None
        self.obs_values = None
        self.obs_mask = None
    
    def forward(self, query_times):
        t_norm = (query_times - self.t_min) / (self.t_max - self.t_min + 1e-8)
        obs_t_norm = (self.obs_times - self.t_min) / (self.t_max - self.t_min + 1e-8)
        
        ctf_out = self.ctf(t_norm, obs_t_norm, self.obs_values, self.obs_mask)
        dgf_out = self.dgf(t_norm, self.obs_values, self.obs_mask)
        
        combined = torch.cat([ctf_out, dgf_out], dim=-1)
        return self.combine(combined)
    
    def fit(self, times, values, obs_mask, target_mask=None, num_epochs=1000, lr=1e-3, verbose=False):
        self.t_min = times.min()
        self.t_max = times.max()
        self.obs_times = times
        
        obs_values = values.clone()
        obs_values[~obs_mask] = 0.0
        self.obs_values = obs_values
        self.obs_mask = obs_mask
        
        if target_mask is None:
            target_mask = obs_mask
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            pred = self.forward(times)
            
            loss = F.mse_loss(pred[target_mask], values[target_mask])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        return loss.item()
    
    def predict(self, times, values=None):
        with torch.no_grad():
            return self.forward(times)
