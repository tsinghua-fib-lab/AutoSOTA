
import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .nn_utils import RotaryPositionalEncoding, ThermodynamicEmbedder
from .transformer_layers import Block, BlockSequential


class MultiOutputTransformer(nn.Module):
    """Refactored MultiOutputTransformer that removes torch_geometric dependency.
    
    Predicts both conditional logits (scalars) and marginal log-probabilities (marginal).
    """

    def __init__(
        self,
        num_scalars: int = 1,  # Number of output scalars per node (e.g., logits per element type)
        num_marginal: int = 1, # Number of global marginal outputs (e.g., total log prob)
        n_layers: int = 12,
        n_heads: int = 12,
        n_embed: int = 768,
        max_src_len: int = 16,
        aggr: str = "mean",
        num_atom_types: int = 100,
        dropout: float = 0.0,
        physical_dim: Literal[1, 2, 3] = 1,
        grid_shape: tuple[int, ...] | None = None,
        periodicity: float | tuple[float, ...] = 1.0,
        fixed_positions: Tensor | None = None,
        use_max_pool_marginal: bool = False,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_scalars = num_scalars
        self.num_marginal = num_marginal
        self.output_dim = num_scalars + num_marginal
        self.n_embed = n_embed
        self.aggr = aggr
        self.use_max_pool_marginal = use_max_pool_marginal

        # Embedding for atomic numbers / occupation indices
        self.atom_embedding = nn.Embedding(num_atom_types, n_embed)

        # Thermodynamic parameter embedding (temperature, field)
        # We embed 2 scalars: Temperature and Field (Chemical Potential)
        self.thermo_embedder = ThermodynamicEmbedder(
            hidden_size=n_embed,
            frequency_embedding_size=256,
            use_mlp=True,
            num_scalars=2,
        )

        # Positional encoding
        self.position_embedder = RotaryPositionalEncoding(
            head_dim=n_embed // n_heads,
            max_src_len=max_src_len,
            physical_dim=physical_dim,
            grid_shape=grid_shape,
            fixed_positions=fixed_positions,
            periodicity=periodicity,
        )

        # Transformer blocks
        self.blocks = BlockSequential(
            *[
                Block(
                    nembed=n_embed,
                    nhead=n_heads,
                    hidden_size=4 * n_embed,
                )
                for _ in range(n_layers)
            ]
        )

        # Output heads
        self.ln_f = nn.LayerNorm(n_embed, eps=1e-6)
        
        # Head for node-level scalars (logits)
        self.scalar_head = nn.Linear(n_embed, num_scalars, bias=False)
        
        # Head for graph-level marginals (log prob)
        self.marginal_head = nn.Linear(n_embed, num_marginal, bias=False)

        self.apply(self._init_weights)
        if use_spectral_norm:
            self.apply(self._add_spectral_norm)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

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
            field: Fields/Chemical potentials (B,) or (B, D)
            time: Optional time (B,) - currently unused in main flow but kept for API

        Returns:
            dict containing:
            - 'scalars': Node-level logits (B, L, num_scalars)
            - 'marginal': Graph-level marginal (B, num_marginal)
        """
        B, L = x.shape
        
        # 1. Embed atoms
        h = self.atom_embedding(x) # (B, L, D)

        # 2. Embed thermodynamic params
        # Ensure temp and field are correct shape
        if temp.dim() == 1:
            temp = temp.unsqueeze(-1)
        if field.dim() == 1:
            field = field.unsqueeze(-1)
            
        # Combine temp and field for embedder (B, 2)
        # Note: field might be vector if multicomponent, but here we assume scalar field per element diff
        # If field is vector, we might need to adjust ThermodynamicEmbedder or project it.
        # For binary Ising/AuCu with single chemical potential difference, we treat it as 1 dim.
        if field.shape[-1] > 1:
             # Just take the first component or norm if it's a vector field for now, 
             # OR if ThermodynamicEmbedder supports >2.
             # The current ThermodynamicEmbedder setup expects 2 scalars based on init.
             # We assume field is scalar (chem pot diff) for now.
             field = field[..., 0:1]

        thermo_params = torch.cat([temp, field], dim=-1) # (B, 2)
        cond_embed = self.thermo_embedder(thermo_params) # (B, D)
        
        # Add conditioning to node embeddings
        h = h + cond_embed.unsqueeze(1) # Broadcast (B, 1, D) -> (B, L, D)

        # 3. Transformer blocks with RoPE
        # RotaryPositionalEncoding is passed to each block's attention
        # We need to pass it as an argument to the blocks
        
        # Access the RoPE module directly; it manages its own cache/state
        # The blocks expect 'rotary_pos_emb' kwarg
        h = self.blocks(h, rotary_pos_emb=self.position_embedder)

        h = self.ln_f(h)

        # 4. Outputs
        # Node-level scalars (logits)
        logits = self.scalar_head(h) # (B, L, num_scalars)
        # Apply log_softmax to convert to log probabilities (consistent with vit_rope)
        # Note: utils_train will strip the last dimension (mask token) before sampling
        log_probs = F.log_softmax(logits, dim=-1) # (B, L, num_scalars)

        # Graph-level marginal
        # Aggregation
        if self.use_max_pool_marginal:
            h_graph = torch.max(h, dim=1)[0] # (B, D)
        else:
            # Mean pooling
            h_graph = torch.mean(h, dim=1) # (B, D)
            
        marginal = self.marginal_head(h_graph) # (B, num_marginal)

        return {
            "scalars": log_probs,   # (B, L, num_scalars) - log probabilities
            "marginal": marginal, # (B, num_marginal)
        }
