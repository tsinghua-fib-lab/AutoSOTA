import math
from math import comb
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
## 2. MODEL DEFINITIONS (Vanilla & Tropical Transformers)
################################################################################

def adaptive_temperature_softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Applies adaptive temperature softmax per head from softmax is not enough 
    (https://arxiv.org/abs/2410.01104).
    """
    original_probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(original_probs * torch.log(original_probs + 1e-9), dim=-1, keepdim=True)

    # Polynomial coefficients for beta(θ) = 1 / θ
    # poly_fit corresponds to a 4th-degree polynomial: a0*x^4 + a1*x^3 + ... + a4
    poly_fit = torch.tensor([-0.037, 0.481, -2.3, 4.917, -1.791],
                            device=logits.device, dtype=logits.dtype)

    # Evaluate polynomial at entropy values via Horner's method
    beta = poly_fit[0]
    for coef in poly_fit[1:]:
        beta = beta * entropy + coef
    beta = torch.where(entropy > 0.5, torch.clamp(beta, min=1.0), torch.ones_like(entropy))

    # softmax with adaptive temperature
    return F.softmax(logits * beta, dim=-1)

class VanillaAttention(nn.Module):
    """Standard scaled dot-product attention with optional adaptive temperature softmax."""
    def __init__(self, d_model, n_heads, aggregator='softmax'):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.aggregator = aggregator
        
        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear   = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        
        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)
        
        Q = Q.view(bsz, seq_len, self.n_heads, self.d_k).permute(0,2,1,3)
        K = K.view(bsz, seq_len, self.n_heads, self.d_k).permute(0,2,1,3)
        V = V.view(bsz, seq_len, self.n_heads, self.d_k).permute(0,2,1,3)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B,H,S,S]
        
        # aggregator
        if self.aggregator == 'softmax':
            attn_weights = F.softmax(scores, dim=-1)
        elif self.aggregator == 'adaptive':
            attn_weights = adaptive_temperature_softmax(scores)
        else:
            raise ValueError(f"Unknown aggregator {self.aggregator}")
        
        context = torch.matmul(attn_weights, V)  # [B,H,S,Dk]
        context = context.permute(0,2,1,3).reshape(bsz, seq_len, d_model)
        output = self.out(context)
        return output


        
def smooth_max(x: torch.Tensor, dim: int, tau: float = 1.0) -> torch.Tensor:
    """
    Differentiable approximation to max along a given dimension.
    """
    return torch.logsumexp(x * tau, dim=dim) / tau


def smooth_min(x: torch.Tensor, dim: int = -1, tau: float = 1.0) -> torch.Tensor:
    """
    Differentiable approximation to min along a given dimension.
    """
    return -smooth_max(-x, dim=dim, tau=tau)


class PluckerTropicalSpace(nn.Module):
    """
    Implements a fully differentiable embedding of inputs into the tropical linear space
    defined by a valuated matroid (via Plücker coordinates), with both forward mapping
    (tropical convex combination of cocircuits) and a circuit-violation loss to enforce
    tropical Plücker relations.
    """
    def __init__(self, n: int, d: int, tau: float = 10.0):
        super().__init__()
        assert 1 <= d <= n, "d must be between 1 and n"
        self.n = n
        self.d = d
        self.tau = tau

        # Number of Plücker coordinates = comb(n, d)
        self.num_plucker = comb(n, d)
        # Raw (log-)Plücker weights, to be learned
        self.pi_raw = nn.Parameter(torch.randn(self.num_plucker) * 1e-2)

        # Enumerate all d-subsets and build index mapping
        self.all_d_subsets = list(combinations(range(n), d))
        self.d_subset_to_index = {subset: i for i, subset in enumerate(self.all_d_subsets)}

        # Cocircuits correspond to all (d-1)-subsets
        self.all_d_minus_1_subsets = list(combinations(range(n), d - 1))
        self.num_cocircuits = len(self.all_d_minus_1_subsets)

        # Pre-build entries for fast C matrix construction
        circuit_entries = []  # (cocircuit_index, coord, plucker_index)
        for c_idx, sigma in enumerate(self.all_d_minus_1_subsets):
            for i in range(n):
                if i in sigma:
                    continue
                basis = tuple(sorted(sigma + (i,)))
                p_idx = self.d_subset_to_index[basis]
                circuit_entries.append((c_idx, i, p_idx))
        self.register_buffer(
            'circuit_entries',
            torch.tensor(circuit_entries, dtype=torch.long)
        )

        # Alpha network: maps input x -> tropical weights over cocircuits
        self.alpha_net = nn.Linear(n, self.num_cocircuits, bias=True)

        # Precompute all (d+1)-subsets for circuit violation loss
        self.all_d_plus_1_subsets = list(combinations(range(n), d + 1))
        self._T = len(self.all_d_plus_1_subsets)
        self.register_buffer(
            '_all_d_plus1',
            torch.tensor(self.all_d_plus_1_subsets, dtype=torch.long)
        )

        # For each (d+1)-subset, build indices for gathering
        plus1_list = []
        removed_list = []
        for tau in self.all_d_plus_1_subsets:
            plus_row = []
            rem_row = []
            for i in tau:
                sub = tuple(sorted(set(tau) - {i}))
                plus_row.append(self.d_subset_to_index[sub])
                rem_row.append(i)
            plus1_list.append(plus_row)
            removed_list.append(rem_row)

        self.register_buffer('_plus1_indices', torch.tensor(plus1_list, dtype=torch.long))
        self.register_buffer('_removed_coords', torch.tensor(removed_list, dtype=torch.long))

    def build_cocircuits(self) -> torch.Tensor:
        """
        Build the (num_cocircuits x n) matrix C whose rows are the normalized cocircuit vectors:
        C[c, i] = pi_raw[basis] if i not in sigma_c, else -inf; then normalized so max over i is 0.
        """
        device = self.pi_raw.device
        dtype = self.pi_raw.dtype
        C = torch.full((self.num_cocircuits, self.n), float('-inf'), device=device, dtype=dtype)
        entries = self.circuit_entries.to(device)
        c_idx, coord, p_idx = entries[:,0], entries[:,1], entries[:,2]
        C[c_idx, coord] = self.pi_raw[p_idx]
        # Normalize each row so its max is zero
        C = C - C.max(dim=-1, keepdim=True).values
        return C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map input x of shape (..., n) into the tropical linear space via a learned
        tropical convex combination of the cocircuit vectors.
        Returns z of the same shape as x, projectively normalized.
        """
        orig_shape = x.shape
        if x.dim() < 2 or x.shape[-1] != self.n:
            raise ValueError(f"Input must have last dim {self.n}, got {x.shape}")
        # Flatten batch dims for alpha_net
        flat = x.view(-1, self.n)
        alpha = self.alpha_net(flat)  # -> (B*, num_cocircuits)
        # Tropical normalization
        alpha = alpha - alpha.max(dim=-1, keepdim=True).values
        # Expand for combination with C
        C = self.build_cocircuits()  # (num_cocircuits, n)
        # Shape for broadcasting: (B*, num_cocircuits, n)
        alpha_bc = alpha.unsqueeze(-1)
        C_bc = C.unsqueeze(0)
        # Tropical convex combination: z_i = max_c (alpha_c + C_{c,i}) approx.
        z_flat = smooth_max(alpha_bc + C_bc, dim=1, tau=self.tau)
        # Projective normalization: subtract max coordinate
        z_flat = z_flat - z_flat.max(dim=-1, keepdim=True).values
        # Restore original batch shape
        return z_flat.view(*orig_shape)

    def circuit_violation_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        For each (d+1)-subset tau and each example in z, compute T_vals = pi_raw[basis]
        + z[..., i] over i in tau.  The minimal entry should be attained at least twice;
        we penalize the gap between the smallest and second-smallest values.
        Returns a scalar loss.
        """
        # Ensure shape (..., n)
        if z.dim() < 1 or z.shape[-1] != self.n:
            raise ValueError(f"Expected z[..., {self.n}], got {z.shape}")
        flat_z = z.view(-1, self.n)  # (B*, n)
        Bstar = flat_z.shape[0]
        T = self._T

        # Gather pi and z for each circuit
        # pi_vals: (T, d+1)
        pi_vals = self.pi_raw[self._plus1_indices]            
        # shape (B*, T, d+1)
        pi_exp = pi_vals.unsqueeze(0).expand(Bstar, -1, -1)
        # z_vals: gather z at removed coords (B*, T, d+1)
        rem = self._removed_coords.unsqueeze(0).expand(Bstar, -1, -1)
        z_exp = flat_z.unsqueeze(1).expand(-1, T, -1)
        z_g = torch.gather(z_exp, dim=-1, index=rem)

        T_vals = pi_exp + z_g  # (B*, T, d+1)
        # Get two smallest per circuit
        smallest, _ = T_vals.topk(k=2, dim=-1, largest=False)
        gap = (smallest[...,1] - smallest[...,0]).clamp(min=0.0)
        return gap.mean()


class TropicalLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TropicalLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.randn(output_dim, input_dim))
        #self.b = nn.Parameter(torch.randn(output_dim))
    
    def forward(self, x):
        x_expanded = x.unsqueeze(-2)
        W_expanded = self.W.unsqueeze(0)
        Wx = x_expanded + W_expanded  
        y, _ = torch.max(Wx, dim=-1)
        #y = y + self.b  # (..., output_dim)
        return y
    
class TropicalAttention(nn.Module):
    def __init__(self, d_model, n_heads, device, use_logsumexp=False, 
                 use_tropical_metric=True):
        super(TropicalAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.use_logsumexp = use_logsumexp
        self.use_tropical_metric = use_tropical_metric
        
        # Linear layers without bias for efficiency
        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # Multi-head attention tropical linear map
        self.query_trop = TropicalLinear(self.d_k, self.d_k)
        self.key_trop = TropicalLinear(self.d_k, self.d_k)
        self.value_trop = TropicalLinear(self.d_k, self.d_k)

        # Learnable scaling coefficient (lambda) with shape compatible for broadcasting
        #self.lambda_param = nn.Parameter(torch.ones(1, 1, 1, device=device))
        self.lambda_param = nn.Parameter(torch.ones(1, 1, d_model, device=device))

    def normalize_tropical(self, x):
        # Efficient broadcasting subtraction
        return x - self.lambda_param

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Apply ReLU and log1p in a single pass before linear transformation
        q = self.normalize_tropical(torch.log1p(F.relu(self.query_linear(x))))
        k = self.normalize_tropical(torch.log1p(F.relu(self.key_linear(x))))
        v = self.normalize_tropical(torch.log1p(F.relu(self.value_linear(x))))
        
        # Reshape and permute for multi-head attention
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # [B, H, S, D]
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        
        # Merge batch and heads for parallel computation
        B = batch_size * self.n_heads
        q = q.reshape(B, seq_len, self.d_k)  # [B, S, D]
        k = k.reshape(B, seq_len, self.d_k)
        v = v.reshape(B, seq_len, self.d_k)

        # Multi-head attention tropical linear map
        q =  self.query_trop(q)
        k = self.key_trop(k)
        v = self.value_trop(v)

        if self.use_tropical_metric:
            # Compute pairwise differences
            diff = q.unsqueeze(2) - k.unsqueeze(1)  # [B, S, S, D]
            # Calculate tropical distance
            max_diff, _ = diff.max(dim=-1)  # [B, S, S]
            min_diff, _ = diff.min(dim=-1)  # [B, S, S]
            d_trop = max_diff - min_diff    # [B, S, S]
            attn_scores = - d_trop           # Higher scores for closer queries and keys
            
            # Compute context using tropical multiplication and aggregation
            sum_sv = attn_scores.unsqueeze(-1) + v.unsqueeze(1)  # [B, S, S, D]
            context = sum_sv.max(dim=2).values  # [B, S, D]
        else:
            # Compute sum of queries and keys
            sum_qk = q.unsqueeze(2) + k.unsqueeze(1)  # [B, S, S, D]
            
            if self.use_logsumexp:
                attn_scores = torch.logsumexp(sum_qk, dim=-1)  # [B, S, S]
                # Compute context using logsumexp
                context = torch.logsumexp(attn_scores.unsqueeze(-1) + v.unsqueeze(1), dim=2)  # [B, S, D]
            else:
                attn_scores = sum_qk.max(dim=-1).values  # [B, S, S]
                # Compute context using max aggregation
                context = (attn_scores.unsqueeze(-1) + v.unsqueeze(1)).max(dim=2).values  # [B, S, D]
        
        # Reshape context back to [batch_size, seq_len, d_model]
        context = context.reshape(batch_size, self.n_heads, seq_len, self.d_k).permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        # Apply the output linear layer after exponentiation
        context = torch.expm1(context)
        output = self.out(context)
        
        return output, attn_scores

def tropical_sinkhorn_normalization(attn_scores, iterations=5, max_plus=True):
    out = attn_scores.clone()  # shape (B, S, S)
    for _ in range(iterations):
        # Row normalization
        if max_plus:
            row_vals, _ = out.max(dim=-1, keepdim=True)  # (B, S, 1)
            out = out - row_vals
        else:
            row_vals, _ = out.min(dim=-1, keepdim=True)  # (B, S, 1)
            out = out - row_vals
        # Column normalization
        if max_plus:
            col_vals, _ = out.max(dim=-2, keepdim=True)  # (B, 1, S)
            out = out - col_vals
        else:
            col_vals, _ = out.min(dim=-2, keepdim=True)  # (B, 1, S)
            out = out - col_vals

    return out 

def trop_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Projective normalization in tropical projective space:
    subtract the max coordinate so that max_i x_i = 0.
    """
    return x - x.max(dim=-1, keepdim=True).values


class TropicalAttention_(nn.Module):
    """
    Multi-head attention in the tropical (max,+) semiring, with optional smooth softmax
    and per-head tropical Plücker embedding.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        tau: float = 10.0,
        use_tropical_sinkhorn: bool = False,
        sinkhorn_iterations: int = 2,
        max_plus: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.tau = tau
        #self.use_softmax = use_softmax
        self.use_tropical_sinkhorn = use_tropical_sinkhorn
        self.sinkhorn_iterations = sinkhorn_iterations
        self.max_plus = max_plus

        # Linear maps for Q, K, V and final projection
        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear   = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # Per-head tropical Plücker embedding (rank d=2)
        self.plucker = PluckerTropicalSpace(n=self.d_k, d=2, tau=self.tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, d_model)
        returns: (B, S, d_model)
        """
        B, S, _ = x.shape

        # 1) Linear projections
        q = self.query_linear(x)  # (B, S, d_model)
        k = self.key_linear(x)
        v = self.value_linear(x)

        # 2) Split heads
        q = q.view(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # (B, H, S, d_k)
        k = k.view(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v = v.view(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        # 3) Projective normalization into TP^{d_k - 1}
        #q = trop_norm(torch.log1p(F.relu(q)))
        #k = trop_norm(torch.log1p(F.relu(k)))
        #v = trop_norm(torch.log1p(F.relu(v)))
        q = torch.log1p(trop_norm(q))
        k = torch.log1p(trop_norm(k))
        v = torch.log1p(trop_norm(v))

        # 4) Per-head tropical Plücker map
        last_q = self.plucker(q.reshape(-1, S, self.d_k)).view(B, self.n_heads, S, self.d_k)
        last_k = self.plucker(k.reshape(-1, S, self.d_k)).view(B, self.n_heads, S, self.d_k)
        last_v = self.plucker(v.reshape(-1, S, self.d_k)).view(B, self.n_heads, S, self.d_k)
        self.last_q = last_q.clone()
        self.last_k = last_k.clone()
        self.last_v = last_v.clone()


        # 5) Compute tropical distances / similarities
        #   d_trop(q, k) = max_i(q_i - k_i) - min_i(q_i - k_i)
        diff = q.unsqueeze(3) - k.unsqueeze(2)  # (B, H, S_q, S_k, d_k)
        max_diff = diff.max(dim=-1).values  # (B, H, S_q, S_k)
        min_diff = diff.min(dim=-1).values  # (B, H, S_q, S_k)
        scores = - (max_diff - min_diff)

        if self.use_tropical_sinkhorn:
            scores = tropical_sinkhorn_normalization(
                scores, iterations=self.sinkhorn_iterations, max_plus=self.max_plus
            )

        # 7) Aggregate values: context[i] = max_j(scores[i,j] + v[j])
        sum_sv = scores.unsqueeze(-1) + v.unsqueeze(2)  # (B,H,S_q,S_k,d_k)
        if self.max_plus:
            context = sum_sv.max(dim=3).values  # (B,H,S_q,d_k)
        else:
            context = sum_sv.min(dim=3).values

        # 8) Merge heads and final linear
        context = torch.expm1(context)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, S, self.d_model)
        output = self.out(context)
        return output, scores

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int = 128,
        dropout: float = 0.0,
        vanilla: bool = True,
        tropical_attention=None,
        pre_norm: bool = False,
        aggregator: str = 'softmax'
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.vanilla = vanilla
        # Attention layer
        if vanilla:
            self.attn = VanillaAttention(d_model, n_heads, aggregator=aggregator)
        else:
            self.attn = tropical_attention
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sublayer
        if self.pre_norm:
            x_norm = self.norm1(x)
            if self.vanilla:
                attn_out = self.attn(x_norm)
            else:
                attn_out, _ = self.attn(x_norm)
            x = x + self.dropout(attn_out)
        else:
            if self.vanilla:
                attn_out = self.attn(x)
            else:
                attn_out, _ = self.attn(x)
            x = x + self.dropout(attn_out)
            x = self.norm1(x)

        # Feed-forward sublayer
        if self.pre_norm:
            x_norm = self.norm2(x)
            ff_out = self.ff(x_norm)
            x = x + self.dropout(ff_out)
        else:
            ff_out = self.ff(x)
            x = x + self.dropout(ff_out)
            x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.0,
        tropical: bool = False,
        tropical_attention_cls=None,
        pre_norm: bool = False,
        aggregator: str = 'softmax',
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            block = TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dim_ff=4*d_model,
                dropout=dropout,
                vanilla=not tropical,
                tropical_attention=tropical_attention_cls,
                pre_norm=pre_norm,
                aggregator=aggregator
            )
            self.layers.append(block)

        self.last_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        if self.pre_norm:
            x = self.last_norm(x)
        return x

class SimpleTransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 32,
        n_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.0,
        tropical: bool = False,
        tropical_attention_cls=None,
        classification: bool = False,
        pool: bool = False,
        pre_norm: bool = False,
        aggregator: str = 'softmax',
    ):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.classification = classification

        # Choose attention class
        attn_cls = VanillaAttention
        if tropical:
            attn_cls = tropical_attention_cls

        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
            tropical=tropical,
            tropical_attention_cls=attn_cls,
            pre_norm=pre_norm,
            aggregator=aggregator,
        )       
        self.pool = pool
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_linear(x) # [B, S]
        x = self.encoder(x) # [B, S, d_model]  
        if self.pool:
            pooled = x.mean(dim=1) # [B, d_model]
            out = self.output_linear(pooled) # [B, 1]
        else:
            out = self.output_linear(x) # [B, S, 1] 
        if not self.classification:
            out = out.squeeze(-1) # [B, S] or [B]
        return out
