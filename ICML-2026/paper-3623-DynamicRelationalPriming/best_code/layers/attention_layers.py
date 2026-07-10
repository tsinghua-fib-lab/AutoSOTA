import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class StandardAttention(nn.Module):
    def __init__(self, d_model, num_heads=1, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.q = nn.Linear(d_model, d_model * num_heads)
        self.k = nn.Linear(d_model, d_model * num_heads)
        self.v = nn.Linear(d_model, d_model * num_heads)
        
        self.out = nn.Linear(d_model * num_heads, d_model)
        
        # Per-head learnable temperature for attention entropy control
        self.temp = nn.Parameter(torch.ones(num_heads))
    
    def forward(self, q, k, v, prime_filters=None, attn_mask=None, is_causal=False):        
        # Project queries, keys, and values
        q = self.q(q)  # (batch_size, n_tokens, d_model * num_heads)
        k = self.k(k)  # (batch_size, n_tokens, d_model * num_heads)
        v = self.v(v)  # (batch_size, n_tokens, d_model * num_heads)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=is_causal)
        
        attn = q @ k.mT / math.sqrt(self.d_model)
        
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout)
        attn = attn_weights @ v
        
        attn = rearrange(attn, 'b h n d -> b n (h d)')
        
        return self.out(attn), attn_weights.mean(dim=1)

class PrimeFilterAttention(nn.Module):
    def __init__(self, d_model, num_heads=1, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.q = nn.Linear(d_model, d_model * num_heads)
        self.k = nn.Linear(d_model, d_model * num_heads)
        self.v = nn.Linear(d_model, d_model * num_heads)
        
        self.out = nn.Linear(d_model * num_heads, d_model)
        
        # Per-head learnable temperature for attention entropy control
        self.temp = nn.Parameter(torch.ones(num_heads))
    
    def forward(self, q, k, v, prime_filters=None, attn_mask=None, is_causal=False):
        B, N, d_model = q.shape
        
        # expand prime_filters for num_heads
        if len(prime_filters.shape) < 5:
            prime_filters = prime_filters.unsqueeze(1).expand(B, self.num_heads, N, N, d_model)
        
        # Project queries, keys, and values
        q = self.q(q)  # (batch_size, n_tokens, d_model * num_heads)
        k = self.k(k)  # (batch_size, n_tokens, d_model * num_heads)
        v = self.v(v)  # (batch_size, n_tokens, d_model * num_heads)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Apply filters to K and V instead of Q
        k_expanded = k.unsqueeze(2).expand(B, self.num_heads, N, N, d_model)
        v_expanded = v.unsqueeze(2).expand(B, self.num_heads, N, N, d_model)
        
        # Filter K and V based on lead-lag relationships
        # prime_filters[i,j] modulates how key j and value j appear to query i
        filtered_k = k_expanded * prime_filters  # (B, H, N, N, d_model)
        filtered_v = v_expanded * prime_filters  # (B, H, N, N, d_model)
        
        # Compute attention scores: each query i attends to filtered keys
        scores = torch.einsum('bhid,bhijd->bhij', q, filtered_k) / math.sqrt(d_model)

        attn = F.softmax(scores / self.temp.view(1, -1, 1, 1).abs().clamp(min=0.1), dim=-1)
        attn = F.dropout(attn, p=self.dropout)
        
        # Apply attention weights to filtered values
        attn_output = torch.einsum('bhij,bhijd->bhid', attn, filtered_v)
        
        attn_output = rearrange(attn_output, 'b h n d -> b n (h d)')
        
        return self.out(attn_output), attn.mean(dim=1)

def lambda_init_fn(depth):
    """Initialize lambda parameter based on layer depth"""
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

def repeat_kv(hidden_states, n_rep):
    """Repeat key-value pairs for grouped query attention"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight is not None:
            hidden_states = self.weight * hidden_states
        return hidden_states.to(input_dtype)

class DifferentialAttention(nn.Module):
    """
    Differential Attention implementation based on the DIFF Transformer paper.
    Expands StandardAttention to use differential attention mechanism.
    """
    def __init__(self, d_model, num_heads=1, depth=0, num_kv_heads=None, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model * num_heads
        # For differential attention, we split each head into two sub-heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.dropout = dropout
        
        # Head dimension is halved because we use two sub-heads per head
        self.head_dim = self.d_model // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        # Linear projections - note the dimensions for differential attention
        self.q_proj = nn.Linear(d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_model // self.n_rep, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_model // self.n_rep, bias=False)
        self.out_proj = nn.Linear(self.d_model, d_model, bias=False)
        
        # Differential attention parameters
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        
        # Sub-layer normalization - applied to each head independently
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
    def forward(self, q, k, v, prime_filters=None, attn_mask=None, is_causal=False):
        bsz, tgt_len, embed_dim = q.size()
        src_len = k.size(1)
        
        # Project queries, keys, and values
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # Reshape for differential attention (2 sub-heads per head)
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
        
        # Transpose to get head dimension first
        q = q.transpose(1, 2)  # (bsz, 2*num_heads, tgt_len, head_dim)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)  # (bsz, 2*num_heads, src_len, head_dim)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)  # (bsz, num_heads, src_len, 2*head_dim)
        
        # Scale queries
        q *= self.scaling
        
        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights += attn_mask
        elif is_causal:
            # Create causal mask
            offset = src_len - tgt_len
            causal_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
            attn_weights += causal_mask
        
        # Handle NaN values
        attn_weights = torch.nan_to_num(attn_weights)
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        
        # Apply dropout
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Compute lambda values for differential attention
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        
        # Apply differential attention: split attention weights into two parts and subtract
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        
        # Apply attention to values
        attn = torch.matmul(attn_weights, v)
        # attn shape: (bsz, num_heads, tgt_len, 2*head_dim)
        
        # Apply sub-layer normalization to each head independently
        # Reshape to apply normalization per head
        attn_reshaped = attn.view(bsz * self.num_heads, tgt_len, 2 * self.head_dim)
        attn_normalized = self.subln(attn_reshaped)
        attn = attn_normalized.view(bsz, self.num_heads, tgt_len, 2 * self.head_dim)
        
        # Scale by (1 - lambda_init)
        attn = attn * (1 - self.lambda_init)
        
        # Reshape back to original dimensions
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        
        # Final output projection
        attn = self.out_proj(attn)
        
        return attn, attn_weights.mean(dim=1)