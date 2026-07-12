import math, sys

import torch
from torch import nn
import torch.nn.functional as F
from torchtyping import TensorType
from typeguard import typechecked

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not installed. Please install with: pip install mamba-ssm")

sys.path.extend(["./", "../", "../../"])
from models.FlowMatching.RegimeFlow.RegimeFlow_base import SinusoidalPositionEmbeddings, FeedForward, RMSNorm


# ===================== Block Types =====================
BLOCK_TYPES = ['mamba', 'attention']


# ===================== Adaptive Layer Normalization =====================
class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (DiT-style, optimized).
    
    Note: No SiLU activation to avoid consecutive activations.
    time_emb is already non-linear from time_init.
    """
    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        
        # Direct linear projection (no SiLU)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * dim, bias=True)
        
        # Zero initialization for stable training
        nn.init.constant_(self.adaLN_modulation.weight, 0)
        nn.init.constant_(self.adaLN_modulation.bias, 0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, C)
            cond: (B, cond_dim)
        Returns:
            (B, L, C)
        """
        scale, shift = self.adaLN_modulation(cond).chunk(2, dim=-1)
        # RMSNorm
        x_norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SelfAttentionLayer(nn.Module):
    """
    Multi-head Self-Attention layer using PyTorch's native nn.MultiheadAttention.
    
    Designed to have the same interface as MambaLayer for easy swapping.
    Supports both PyTorch 2.0+ (with Flash Attention) and earlier versions.
    """
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        num_heads: int = 8,
        use_adaLN: bool = False,
        cond_dim: int = None,
        causal: bool = False,
    ):
        super().__init__()
        
        self.use_adaLN = use_adaLN
        self.causal = causal
        self.d_model = d_model
        
        # PyTorch native MultiheadAttention
        # batch_first=True: input format is (B, L, D)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Normalization
        if use_adaLN:
            if cond_dim is None:
                raise ValueError("cond_dim must be provided when use_adaLN=True")
            self.norm = AdaLN(d_model, cond_dim)
        else:
            self.norm = RMSNorm(d_model)
        
        # Check PyTorch version for is_causal support
        self._use_is_causal = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def _get_causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Generate causal mask (upper triangular with -inf)."""
        return torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype),
            diagonal=1
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        z = x.transpose(-1, -2)  # (B, D, L) -> (B, L, D)
        
        z_norm = self.norm(z, cond) if self.use_adaLN else self.norm(z)

        # Prepare causal mask if needed
        attn_mask = None
        if self.causal:
            L = z_norm.size(1)
            attn_mask = self._get_causal_mask(L, z_norm.device, z_norm.dtype)

        attn_output, _ = self.attn(
            z_norm, z_norm, z_norm,
            # is_causal=self.causal,
            attn_mask=attn_mask,
            need_weights=False,
        )
        
        return x + attn_output.transpose(-1, -2)

# ===================== Mamba Layer (Simplified) =====================
class MambaLayer(nn.Module):
    """
    Simplified unidirectional Mamba layer with optional AdaLN.
    
    Simplified for univariate time series (C=1).
    """
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_adaLN: bool = False,
        cond_dim: int = None,
        **kwargs,  # Accept additional kwargs for compatibility
    ):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm is required. Install with: pip install mamba-ssm")
        
        self.use_adaLN = use_adaLN
        
        # Mamba SSM (unidirectional only)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # Normalization
        if use_adaLN:
            if cond_dim is None:
                raise ValueError("cond_dim must be provided when use_adaLN=True")
            self.norm = AdaLN(d_model, cond_dim)
        else:
            self.norm = RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, H, L)
            cond: (B, cond_dim) - optional, required if use_adaLN=True
        Returns:
            (B, H, L)
        """
        # Transpose for normalization and Mamba: (B, H, L) -> (B, L, H)
        z = x.transpose(-1, -2)
        
        # Normalization
        if self.use_adaLN:
            if cond is None:
                raise ValueError("cond must be provided when use_adaLN=True")
            z = self.norm(z, cond)
        else:
            z = self.norm(z)
        
        # Mamba SSM
        z = self.mamba(z)
        
        # Transpose back: (B, L, H) -> (B, H, L)
        z = z.transpose(-1, -2)
        
        # Dropout
        z = self.dropout(z)
        
        # Residual connection
        return x + z


# ===================== Generic Block (Supports Mamba and Attention) =====================
class SequenceBlock(nn.Module):
    """
    Generic sequence modeling block supporting both Mamba and Self-Attention.
    
    Architecture:
    1. Time injection: x + time_linear(time_emb)
    2. Sequence layer (Mamba or Self-Attention with RMSNorm or AdaLN)
    3. FFN with SwiGLU (with RMSNorm or AdaLN)
    4. Skip connection
    """
    def __init__(
        self,
        d_model: int,
        block_type: str = 'mamba',
        dropout: float = 0.0,
        # Mamba-specific
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        # Attention-specific
        num_heads: int = 8,
        causal: bool = False,
        # FFN
        ffn_dim_multiplier: float = 2.5,
        ffn_dropout: float = 0.1,
        # Conditioning
        use_adaLN: bool = False,
        cond_dim: int = None,
    ):
        super().__init__()
        
        assert block_type in BLOCK_TYPES, f"block_type must be one of {BLOCK_TYPES}, got {block_type}"
        self.block_type = block_type
        
        # Sequence layer (Mamba or Self-Attention)
        if block_type == 'mamba':
            self.mamba_layer = MambaLayer(
                d_model=d_model,
                dropout=dropout,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_adaLN=use_adaLN,
                cond_dim=cond_dim,
            )
        elif block_type == 'attention':
            self.seq_layer = SelfAttentionLayer(
                d_model=d_model,
                dropout=dropout,
                num_heads=num_heads,
                use_adaLN=use_adaLN,
                cond_dim=cond_dim,
                causal=causal,
            )
        
        # FFN with SwiGLU
        self.ffn = FeedForward(
            dim=d_model,
            ffn_dim_multiplier=ffn_dim_multiplier,
            dropout=ffn_dropout,
            activation='swiglu',
        )
        
        # FFN normalization
        self.use_adaLN = use_adaLN
        if use_adaLN:
            if cond_dim is None:
                raise ValueError("cond_dim must be provided when use_adaLN=True")
            self.ffn_norm = AdaLN(d_model, cond_dim)
        else:
            self.ffn_norm = RMSNorm(d_model)
        
        # Time projection
        self.time_linear = nn.Linear(d_model, d_model, bias=False)
        
        # Skip projection
        self.skip_proj = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(
        self, 
        x: torch.Tensor, 
        time_emb: torch.Tensor, 
        cond: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, H, L)
            time_emb: (B, H) - non-linear time representation
            cond: (B, cond_dim) - optional conditioning
        Returns:
            out: (B, H, L)
            skip: (B, H, L)
        """
        # Time injection
        t_emb = self.time_linear(time_emb).unsqueeze(-1)  # (B, H, 1)
        seq_in = x + t_emb
        
        # Sequence layer (Mamba or Attention)
        if self.block_type == 'mamba': seq_out = self.mamba_layer(seq_in, cond)
        else: seq_out = self.seq_layer(seq_in, cond)
        
        # FFN
        ffn_in = seq_out.transpose(-1, -2)  # (B, L, H)
        if self.use_adaLN:
            ffn_in = self.ffn_norm(ffn_in, cond)
        else:
            ffn_in = self.ffn_norm(ffn_in)
        
        ffn_out = self.ffn(ffn_in).transpose(-1, -2)  # (B, H, L)
        
        # Residual
        out = seq_out + ffn_out
        
        # Skip
        skip = self.skip_proj(out)
        
        return out + x, skip


# ===================== Legacy MambaBlock (for backward compatibility) =====================
class MambaBlock(SequenceBlock):
    """
    Legacy MambaBlock for backward compatibility.
    Simply wraps SequenceBlock with block_type='mamba'.
    """
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        ffn_dim_multiplier: float = 2.5,
        ffn_dropout: float = 0.1,
        use_adaLN: bool = False,
        cond_dim: int = None,
    ):
        super().__init__(
            d_model=d_model,
            block_type='mamba',
            dropout=dropout,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            ffn_dim_multiplier=ffn_dim_multiplier,
            ffn_dropout=ffn_dropout,
            use_adaLN=use_adaLN,
            cond_dim=cond_dim,
        )


# ===================== Backbone Model (with Ablation Support) =====================
class BackboneModel(nn.Module):
    """
    Backbone model for univariate time series (C=1) with ablation support.
    
    Supports two block types:
    - 'mamba': Mamba SSM blocks (default)
    - 'attention': Self-Attention blocks
    
    Use `block_type` parameter for ablation experiments.
    
    Input: (B, L, C) where C=1
    Output: (B, L, C) where C=1
    """
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 1,
        step_emb: int = 128,
        num_residual_blocks: int = 4,
        dropout: float = 0.0,
        init_skip: bool = True,
        # Block type for ablation
        block_type: str = 'mamba',
        # Mamba-specific
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        # Attention-specific
        num_heads: int = 8,
        causal: bool = False,
        # FFN
        ffn_dim_multiplier: float = 2.5,
        ffn_dropout: float = 0.1,
        # Conditioning
        use_adaLN: bool = False,
        cond_dim: int = 0,
    ):
        super().__init__()
        
        assert block_type in BLOCK_TYPES, f"block_type must be one of {BLOCK_TYPES}, got {block_type}"
        
        if block_type == 'mamba' and not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm is required for block_type='mamba'. Install with: pip install mamba-ssm"
            )
        
        if use_adaLN and cond_dim == 0:
            raise ValueError("cond_dim must be > 0 when use_adaLN=True")
        
        self.block_type = block_type
        self.use_adaLN = use_adaLN
        self.cond_dim = cond_dim
        self.init_skip = init_skip
        
        # Input projection: (B, L, C) -> (B, L, H)
        self.input_init = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # Time embedding: produces non-linear representation
        self.time_init = nn.Sequential(
            nn.Linear(step_emb, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),  # Non-linear output for direct time injection
        )
        
        # Condition processing (if using AdaLN)
        if use_adaLN and cond_dim > 0:
            self.cond_processor = nn.Sequential(
                nn.Linear(hidden_dim + cond_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
            )

        # Sequence blocks (Mamba or Attention)
        self.residual_blocks = nn.ModuleList([
            SequenceBlock(
                d_model=hidden_dim,
                block_type=block_type,
                dropout=dropout,
                # Mamba-specific
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                # Attention-specific
                num_heads=num_heads,
                causal=causal,
                # FFN
                ffn_dim_multiplier=ffn_dim_multiplier,
                ffn_dropout=ffn_dropout,
                # Conditioning
                use_adaLN=use_adaLN,
                cond_dim=hidden_dim if use_adaLN else None,
            )
            for _ in range(num_residual_blocks)
        ])
        
        # Sinusoidal position embeddings
        self.step_embedding = SinusoidalPositionEmbeddings(step_emb)
        
        # Output projection: (B, L, H) -> (B, L, output_dim)
        self.out_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # init_skip projection (if dimensions don't match)
        if init_skip and input_dim != output_dim:
            self.skip_proj_init = nn.Linear(input_dim, output_dim)
        else:
            self.skip_proj_init = None

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  block_type={self.block_type},\n"
            f"  num_blocks={len(self.residual_blocks)},\n"
            f"  use_adaLN={self.use_adaLN},\n"
            f"  cond_dim={self.cond_dim},\n"
            f"  num_params={self.get_num_params():,}\n"
            f")"
        )

    @typechecked
    def forward(
        self,
        t: TensorType[float],
        x_in: TensorType[float, "batch", "length", "channel"],
        cond_emb: TensorType[float, "batch", "cond_dim"] | None = None,
    ) -> TensorType[float, "batch", "length", "channel"]:
        """
        Forward pass.
        
        Args:
            t: Time step(s) - scalar, (B,), or (B, 1)
            x_in: Input (B, L, C) where C=1
            cond_emb: Optional conditioning (B, cond_dim)
        Returns:
            Output (B, L, C) where C=1
        """
        B, L, C = x_in.shape
        assert C == 1, f"This backbone is designed for univariate series (C=1), got C={C}"
        
        # Handle time step shapes
        if len(t.shape) == 0:
            t = t.unsqueeze(0).expand(B)
        elif t.shape[0] == 1 and B > 1:
            t = t.expand(B)
        elif len(t.shape) > 1:
            t = t.squeeze(-1) if t.shape[-1] == 1 else t[..., 0]
        
        # Time embedding (non-linear output)
        t_raw = self.step_embedding(t * 10000)  # (B, step_emb)
        time_emb = self.time_init(t_raw)         # (B, hidden_dim)
        
        # Condition for AdaLN
        cond_for_adaLN = None
        if self.use_adaLN:
            if self.cond_dim > 0 and cond_emb is not None:
                combined = torch.cat([time_emb, cond_emb], dim=-1)
                cond_for_adaLN = self.cond_processor(combined)
            else:
                cond_for_adaLN = time_emb
        
        # Input projection: (B, L, C) -> (B, L, H)
        x = self.input_init(x_in)
        
        # Transpose: (B, L, H) -> (B, H, L)
        x = x.transpose(-1, -2)
        
        # Apply sequence blocks
        skips = []
        for layer in self.residual_blocks:
            if self.use_adaLN:
                x, skip = layer(x, time_emb, cond_for_adaLN)
            else:
                x, skip = layer(x, time_emb, None)
            skips.append(skip)
        
        # Skip aggregation: (B, H, L)
        skip = torch.stack(skips).sum(0)
        
        # Transpose: (B, H, L) -> (B, L, H)
        skip = skip.transpose(-1, -2)
        
        # Output projection: (B, L, H) -> (B, L, output_dim)
        out = self.out_linear(skip)
        
        # Optional init_skip
        if self.init_skip:
            if self.skip_proj_init is not None:
                out = out - self.skip_proj_init(x_in)
            else:
                out = out - x_in
        
        return out


# ===================== Factory function =====================
def create_backbone(
    block_type: str = 'mamba',
    **kwargs
) -> BackboneModel:
    """
    Factory function to create backbone model.
    
    Args:
        block_type: 'mamba' or 'attention'
        **kwargs: Additional arguments for BackboneModel
    
    Returns:
        BackboneModel instance
    
    Example:
        # Create Mamba backbone
        model_mamba = create_backbone(block_type='mamba', hidden_dim=128, num_residual_blocks=4)
        
        # Create Attention backbone for ablation
        model_attn = create_backbone(block_type='attention', hidden_dim=128, num_residual_blocks=4)
    """
    return BackboneModel(block_type=block_type, **kwargs)


# ===================== Test =====================
if __name__ == '__main__':
    # Test both block types
    B, L, C = 2, 100, 1
    x = torch.randn(B, L, C)
    t = torch.rand(B)
    
    print("=" * 60)
    print("Testing BackboneModel with PyTorch native attention")
    print("=" * 60)
    
    # Test Attention backbone (always available)
    print("\n[1] Testing Attention backbone (PyTorch native)...")
    model_attn = BackboneModel(
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        num_residual_blocks=2,
        block_type='attention',
        num_heads=4,
    )
    print(model_attn)
    out_attn = model_attn(t, x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out_attn.shape}")
    
    # Test Mamba backbone (if available)
    if MAMBA_AVAILABLE:
        print("\n[2] Testing Mamba backbone...")
        model_mamba = BackboneModel(
            input_dim=1,
            hidden_dim=64,
            output_dim=1,
            num_residual_blocks=2,
            block_type='mamba',
            d_state=16,
            d_conv=4,
        )
        print(model_mamba)
        out_mamba = model_mamba(t, x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out_mamba.shape}")
    else:
        print("\n[2] Skipping Mamba backbone test (mamba_ssm not installed)")
    
    # Test with AdaLN
    print("\n[3] Testing Attention backbone with AdaLN...")
    cond_dim = 32
    cond = torch.randn(B, cond_dim)
    model_adaLN = BackboneModel(
        input_dim=1,
        hidden_dim=64,
        output_dim=1,
        num_residual_blocks=2,
        block_type='attention',
        num_heads=4,
        use_adaLN=True,
        cond_dim=cond_dim,
    )
    print(model_adaLN)
    out_adaLN = model_adaLN(t, x, cond)
    print(f"Input shape: {x.shape}")
    print(f"Cond shape: {cond.shape}")
    print(f"Output shape: {out_adaLN.shape}")
    
    # Speed comparison
    print("\n[4] Speed comparison...")
    import time
    
    model_attn.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model_attn(t, x)
        
        # Timing
        start = time.time()
        for _ in range(100):
            _ = model_attn(t, x)
        end = time.time()
        print(f"Average forward time (PyTorch native): {(end - start) / 100 * 1000:.3f} ms")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)