from typing import Any, Optional

import torch
from torch import nn

from minesm.layers.blocks import UnifiedTransformerBlock
from minesm.layers.transformer_stack import TransformerStack


class GeometricEncoderStack(TransformerStack):
    """TransformerStack specialised to purely geometric (SE(3)-equivariant) attention blocks.

    Each block uses geometric self-attention and no plain self-attention, matching
    the ESM3 geometric encoder design described in Appendix B.1 of the paper.
    """

    def __init__(self, d_model: int, n_heads: int, v_heads: int, n_layers: int) -> None:
        super().__init__(d_model, n_heads, v_heads, 0)
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    v_heads=v_heads,
                    use_geom_attn=True,
                    use_plain_attn=False,
                    use_flash_attn=True,
                    expansion_ratio=4,
                    bias=True,
                )
                for i in range(n_layers)
            ]
        )
        # self.norm = nn.Identity()


class GeometricEncoder(nn.Module):
    """Geometric encoder that processes visible protein residue frames.

    Applies ``n_layers`` geometric attention blocks to SE(3)-invariant
    residue-frame representations.  No positional embeddings are injected
    here; they are added by the caller after this module (MiAECore design).

    Args:
        d_model: Residue embedding dimension.
        n_heads: Number of attention heads.
        v_heads: Number of value heads for geometric attention.
        n_layers: Number of :class:`GeometricEncoderStack` blocks.
    """

    def __init__(self, d_model: int, n_heads: int, v_heads: int, n_layers: int) -> None:
        super().__init__()
        # We only support fully-geometric structure token encoders for now...
        # setting n_layers_geom to something that's not n_layers won't work because
        # sequence ID isn't supported fully in this repo for plain-old transformers
        self.transformer = GeometricEncoderStack(d_model, n_heads, v_heads, n_layers)

    @property
    def num_layers(self):
        return len(self.transformer.blocks)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        affine: Any,
        affine_mask: Optional[torch.Tensor] = None,
        sequence_id: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        return self.transformer(
            x=x,
            attention_mask=mask,
            affine=affine,
            affine_mask=affine_mask,
            sequence_id=sequence_id,
        )
