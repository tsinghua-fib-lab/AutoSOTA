import torch
import torch.nn as nn
import torch.nn.functional as F


class TropicalLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TropicalLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.randn(output_dim, input_dim))
    
    def forward(self, x):
        x_expanded = x.unsqueeze(-2)
        W_expanded = self.W.unsqueeze(0)
        Wx = x_expanded + W_expanded  
        y, _ = torch.max(Wx, dim=-1)
        return y
    
class TropicalAttention(nn.Module):
    def __init__(self, d_model, n_heads, device, tropical_proj=True, tropical_norm=False, symmetric=True):
        super(TropicalAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.tropical_proj = tropical_proj
        self.tropical_norm = tropical_norm
        self.symmetric = symmetric
        
        # Linear layers without bias
        self.out = nn.Linear(d_model, d_model, bias=False)

        if self.tropical_proj:
            # Multi-head attention tropical linear map
            self.query_trop = TropicalLinear(self.d_k, self.d_k)
            self.key_trop = TropicalLinear(self.d_k, self.d_k)
            self.value_trop = TropicalLinear(self.d_k, self.d_k)

        if self.tropical_norm:
            self.lambda_param = nn.Parameter(torch.ones(1, 1, d_model, device=device))

    def normalize_tropical(self, x):
        return x - self.lambda_param

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        if self.tropical_norm:
            # Apply ReLU and log1p in a single pass before linear transformation
            q = self.normalize_tropical(torch.log1p(F.relu(x)))
            k = self.normalize_tropical(torch.log1p(F.relu(x)))
            v = self.normalize_tropical(torch.log1p(F.relu(x)))
        else:
            q = torch.log1p(F.relu(x))
            k = torch.log1p(F.relu(x))
            v = torch.log1p(F.relu(x))
        
        # Reshape and permute for multi-head attention
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # [B, H, S, D]
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        
        # Merge batch and heads for parallel computation
        B = batch_size * self.n_heads
        q = q.reshape(B, seq_len, self.d_k)  # [B, S, D]
        k = k.reshape(B, seq_len, self.d_k)
        v = v.reshape(B, seq_len, self.d_k)

        # Tropical linear map
        if self.tropical_proj:
            q = self.query_trop(q)
            k = self.key_trop(k)
            v = self.value_trop(v)

        # Compute Hilbert Projective Metric
        if self.symmetric:
            diff = q.unsqueeze(2) - k.unsqueeze(1)  # [B, S, S, D]
            # Calculate tropical distance
            max_diff, _ = diff.max(dim=-1)  # [B, S, S]
            min_diff, _ = diff.min(dim=-1)  # [B, S, S]
            d_trop = max_diff - min_diff    # [B, S, S]
            attn_scores = - d_trop           # Higher scores for closer queries and keys
        else:
            diff = q.unsqueeze(2) - k.unsqueeze(3)   # [B, S, 1, D] - [B, 1, S, D] -> [B, S, S, D]
            sum_diff = diff.sum(dim=-1)              # [B, S, S]
            min_diff = diff.amin(dim=-1)             # [B, S, S]
            n = q.size(-1)
            attn_scores = - (sum_diff - n * min_diff)
        
        # Compute context using tropical multiplication and aggregation
        sum_sv = attn_scores.unsqueeze(-1) + v.unsqueeze(1)  # [B, S, S, D]
        context = sum_sv.max(dim=2).values  # [B, S, D]
        
        # Reshape context back to [batch_size, seq_len, d_model]
        context = context.reshape(batch_size, self.n_heads, seq_len, self.d_k).permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        # Apply the output linear layer after exponentiation
        context = torch.expm1(context)
        output = self.out(context)
        
        return output, attn_scores
