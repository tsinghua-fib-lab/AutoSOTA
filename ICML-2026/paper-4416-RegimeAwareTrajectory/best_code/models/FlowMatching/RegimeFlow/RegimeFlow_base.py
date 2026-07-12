import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: tensor of shape (B,) or (B, 1)
        Returns:
            embeddings: tensor of shape (B, dim)
        """
        device = time.device
        
        # Ensure time is 1D
        if len(time.shape) == 0:
            time = time.unsqueeze(0)
        elif len(time.shape) > 1:
            time = time.squeeze(-1)
        
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ===================== Normalization Layers =====================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    More efficient than LayerNorm, commonly used in modern LLMs (LLaMA, etc.)
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RMSNormNoAffine(nn.Module):
    """RMSNorm without learnable affine parameters (for AdaLN)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


# ===================== Positional Encoding =====================
NUM_FREQS = 16

def positional_encoding_tensor(time_tensor: torch.Tensor, num_freqs: int = NUM_FREQS) -> torch.Tensor:
    """
    Sinusoidal positional encoding for continuous time values.
    
    Args:
        time_tensor: [B] or [B, 1] tensor of time values
        num_freqs: number of frequency components
        
    Returns:
        [B, num_freqs * 2] tensor of positional encodings
    """
    if time_tensor.dim() == 1:
        time_tensor = time_tensor.unsqueeze(-1)
    
    freqs = torch.exp(
        torch.arange(0, num_freqs, dtype=torch.float32, device=time_tensor.device) * 
        (-math.log(10000.0) / num_freqs)
    )
    
    args = time_tensor * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    return embedding


# ===================== Feed-Forward Network (FFN) =====================
class FeedForward(nn.Module):
    """
    Feed-Forward Network with SwiGLU or GELU activation.
    
    SwiGLU (Shazeer 2020) provides better performance on language modeling tasks.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        out_dim: int = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: float = None,
        dropout: float = 0.1,
        activation: str = 'swiglu',
    ):
        super().__init__()
        out_dim = out_dim or dim
        
        # Calculate hidden dimension
        hidden_dim = hidden_dim or int(2 * 4 * dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.activation = activation.lower()
        self.dropout = nn.Dropout(dropout)
        
        if self.activation == 'swiglu':
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        else:
            self.w1 = nn.Linear(dim, hidden_dim)
            self.w2 = nn.Linear(hidden_dim, out_dim)
            self.w3 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == 'swiglu':
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        else:
            return self.dropout(self.w2(F.gelu(self.w1(x))))


# ===================== Fourier Feature Embedder =====================
class FourierEmbedder(nn.Module):
    """
    Fourier feature embedder for continuous values.
    Uses learnable frequencies and a simple MLP to project into a higher-dimensional space.
    """
    def __init__(self, hidden_dim: int, num_frequencies: int = 64):
        super().__init__()
        self.num_frequencies = num_frequencies
        
        # Learnable frequencies
        self.frequencies = nn.Parameter(torch.randn(num_frequencies) * 0.1)
        
        self.mlp = nn.Sequential(
            nn.Linear(num_frequencies * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B] continuous input tensor
        Returns: [B, hidden_dim] tensor of Fourier features
        """
        # Fourier features
        x = x.unsqueeze(-1)  # [B, 1]
        freqs = self.frequencies.unsqueeze(0)  # [1, num_freq]
        
        fourier = torch.cat([
            torch.sin(x * freqs * 2 * math.pi),
            torch.cos(x * freqs * 2 * math.pi),
        ], dim=-1)  # [B, num_freq*2]
        
        return self.mlp(fourier)
