
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .nn_utils import ThermodynamicEmbedder


class MLPRes(nn.Module):
    """Residual MLP block."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)
        
        self.residual = (
            nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.residual(x)
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.norm(out + identity)
        return out


class DualOutputMLP(nn.Module):
    """Refactored DualOutputMLP for MDNS.
    
    Takes lattice configuration and thermodynamic conditions, outputs 
    conditional logits (per site) and marginal log-probability (global).
    """

    def __init__(
        self,
        input_dim: int, # Number of sites (L)
        hidden_dim: int = 512, # Hidden dimension
        num_layers: int = 4,   # Number of res blocks
        num_scalars: int = 1,  # Output scalars per site
        num_marginal: int = 1, # Output marginals
        num_atom_types: int = 100,
        embedding_dim: int = 16, # Dimension to embed each atom type before flattening
        dropout: float = 0.0,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_scalars = num_scalars
        self.num_marginal = num_marginal
        
        # Embedding for atoms
        self.atom_embedding = nn.Embedding(num_atom_types, embedding_dim)
        
        # Flattened size after embedding
        self.flat_input_dim = input_dim * embedding_dim

        # Thermodynamic embedding (Temp + Field)
        self.thermo_embedder = ThermodynamicEmbedder(
            hidden_size=64, # Small embedding for params
            frequency_embedding_size=32,
            use_mlp=True,
            num_scalars=2
        )
        
        # Initial projection
        # Input to MLP is: Flattened Lattice + Thermo Embeddings
        mlp_input_dim = self.flat_input_dim + 64 
        
        self.input_proj = nn.Linear(mlp_input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.Sequential(
            *[MLPRes(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        # Heads
        # Scalar head outputs (B, L, num_scalars) -> output size (B, L * num_scalars)
        self.scalar_head = nn.Linear(hidden_dim, input_dim * num_scalars)
        
        # Marginal head outputs (B, num_marginal)
        self.marginal_head = nn.Linear(hidden_dim, num_marginal)

        if use_spectral_norm:
            self.apply(self._add_spectral_norm)

    def _add_spectral_norm(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            torch.nn.utils.spectral_norm(module)

    def forward(
        self, 
        x: Tensor, 
        temp: Tensor, 
        field: Tensor,
        time: Tensor = None
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Occupational indices (B, L)
            temp: Temperatures (B,)
            field: Fields (B,)
            time: Optional time (B,)

        Returns:
            dict containing 'scalars' and 'marginal'.
        """
        B, L = x.shape
        assert L == self.input_dim, f"Input length {L} does not match configured input_dim {self.input_dim}"

        # 1. Embed and flatten atoms
        h_atoms = self.atom_embedding(x) # (B, L, emb)
        h_atoms = h_atoms.view(B, -1)    # (B, L * emb)

        # 2. Embed thermo params
        if temp.dim() == 1:
            temp = temp.unsqueeze(-1)
        if field.dim() == 1:
            field = field.unsqueeze(-1)
        if field.shape[-1] > 1:
            field = field[..., 0:1] # Take first component if vector
            
        thermo_params = torch.cat([temp, field], dim=-1) # (B, 2)
        h_cond = self.thermo_embedder(thermo_params) # (B, 64)

        # 3. Concatenate
        h = torch.cat([h_atoms, h_cond], dim=-1) # (B, input_dim + 64)

        # 4. MLP
        h = self.input_proj(h)
        h = self.blocks(h)
        h = self.output_norm(h)

        # 5. Outputs
        # Scalar head: reshape back to (B, L, num_scalars)
        scalars_flat = self.scalar_head(h)
        scalars = scalars_flat.view(B, L, self.num_scalars)

        # Marginal head: (B, num_marginal)
        marginal = self.marginal_head(h)

        return {
            "scalars": scalars,
            "marginal": marginal
        }
