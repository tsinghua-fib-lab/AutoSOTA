import torch
from torch import nn

from minesm.layers.transformer_stack import TransformerStack
from minesm.utils.constants import esm3 as C
from minesm.utils.structure.affine3d import build_affine3d_from_coordinates

from ..layer.pos_embed import SinusoidalPositionalEmbedding1D
from .encoder import GeometricEncoder


class MiAEEncoder(nn.Module):
    """MiAE encoder with a protein-level classification head.

    Processes full backbone coordinates through geometric attention blocks
    and a standard Transformer, then pools the representation to a single
    vector for classification.

    For pretraining, the decoder is discarded and this module takes the
    place of the frozen encoder (see Section 4 of the paper).

    Args:
        embed_dim: Residue embedding dimension (512 / 768 / 1024 for S / B / L).
        depth: Number of standard Transformer layers.
        num_heads: Number of attention heads.
        geometric_depth: Number of geometric attention layers prepended.
        v_heads: Value-head count for geometric attention (``v_heads`` vectors).
        use_seq_input: If ``True``, embed amino-acid tokens and add them to
            residue representations (MiAE+seq variant from Table 2).
        num_classes: Output dimension of the classification head.
        avg_pool: If ``True``, pool by averaging over residue representations;
            otherwise use the CLS token.  Average pooling is better for linear
            probing (Table 7 in the paper).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        geometric_depth: int = 2,
        v_heads: int = 96,
        use_seq_input: bool = False,
        num_classes: int = 1195,
        avg_pool: bool = False,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = SinusoidalPositionalEmbedding1D(embed_dim)
        self.input_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.geometric_encoder = GeometricEncoder(
            embed_dim, num_heads, v_heads, geometric_depth
        )
        self.seq_embed = None
        if use_seq_input:
            self.seq_embed = nn.Embedding(
                len(C.SEQUENCE_VOCAB), embed_dim, padding_idx=C.SEQUENCE_PAD_TOKEN
            )
        self.encoder = TransformerStack(
            embed_dim,
            num_heads,
            1,
            depth,
            scale_residue=False,
            n_layers_geom=0,
            use_rotary=False,
        )
        self.register_buffer("cls_mask", torch.ones(1, 1, dtype=torch.bool), False)
        self.avg_pool = avg_pool

        self.head = nn.Linear(embed_dim, num_classes)
        self.init_weights()

    def init_weights(self) -> None:
        if self.seq_embed is not None:
            nn.init.normal_(self.seq_embed.weight, std=0.02)

    @property
    def num_layers(self):
        return self.geometric_encoder.num_layers + len(self.encoder.blocks) + 1

    def get_layer_id_by_param_name(self, name: str) -> int:
        offset = self.geometric_encoder.num_layers
        if name.startswith("input_embed"):
            return 0
        elif name.startswith("geometric_encoder"):
            if "blocks" in name:
                return int(name.split(".")[3])
            return offset
        elif name.startswith("seq_embed"):
            return offset
        elif name in ["cls_token", "pos_embed"]:
            return offset
        elif name.startswith("encoder"):
            if "blocks" in name:
                return int(name.split(".")[2]) + offset
            return self.num_layers
        else:
            return self.num_layers

    def forward(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        residue_index: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
        seq_tokens: torch.Tensor | None = None,
        repr_only: bool = False,
    ):
        coords = coords[..., :3, :]
        pos_embed = self.pos_embed(residue_index)
        affine, affine_mask = build_affine3d_from_coordinates(coords=coords)
        x = self.input_embed.expand(affine.shape[0], affine.shape[1], -1)
        x, _, _ = self.geometric_encoder(x, mask, affine, affine_mask, sequence_id)
        if (self.seq_embed is not None) and (seq_tokens is not None):
            seq_embed = self.seq_embed(seq_tokens)
            x = x + seq_embed
        x = x + pos_embed
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        mask = torch.cat((self.cls_mask.expand(x.shape[0], -1), mask), dim=1)
        x, _, _ = self.encoder(
            x,
            attention_mask=mask,
            affine=None,
            affine_mask=None,
            sequence_id=sequence_id,
        )
        if self.avg_pool:
            x = x[:, 1:]
            mask = mask[:, 1:].float().unsqueeze(-1)
            x = x * mask
            x = x.sum(dim=1) / mask.sum(dim=1)
        else:
            x = x[:, 0]
        if repr_only:
            return x
        x = self.head(x)
        return x


class MiAEEncoderDense(MiAEEncoder):
    """Token-level (dense) variant of :class:`MiAEEncoder` for sequence-level tasks.

    Instead of pooling to a single vector, returns per-residue representations
    and the corresponding mask.  Used internally for inverse-folding pretraining.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        geometric_depth: int = 2,
        v_heads: int = 96,
        use_seq_input: bool = False,
        num_classes: int = len(C.SEQUENCE_VOCAB),
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            geometric_depth=geometric_depth,
            v_heads=v_heads,
            use_seq_input=use_seq_input,
            num_classes=num_classes,
        )

    def forward(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        residue_index: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
        seq_tokens: torch.Tensor | None = None,
        repr_only: bool = False,
    ):
        coords = coords[..., :3, :]
        pos_embed = self.pos_embed(residue_index)
        affine, affine_mask = build_affine3d_from_coordinates(coords=coords)
        x = self.input_embed.expand(affine.shape[0], affine.shape[1], -1)
        x, _, _ = self.geometric_encoder(x, mask, affine, affine_mask, sequence_id)
        if (self.seq_embed is not None) and (seq_tokens is not None):
            seq_embed = self.seq_embed(seq_tokens)
            x = x + seq_embed
        x = x + pos_embed
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        mask = torch.cat((self.cls_mask.expand(x.shape[0], -1), mask), dim=1)
        x, _, _ = self.encoder(
            x,
            attention_mask=mask,
            affine=None,
            affine_mask=None,
            sequence_id=sequence_id,
        )
        x = x[:, 1:]
        mask = mask[:, 1:]
        if repr_only:
            return x, mask
        x = self.head(x)
        return x, mask


def fot_small(model_cls: type, **kwargs) -> "MiAEEncoder":
    return model_cls(
        embed_dim=512,
        depth=6,
        num_heads=8,
        geometric_depth=2,
        v_heads=64,
        **kwargs,
    )


def fot_base(model_cls: type, **kwargs) -> "MiAEEncoder":
    return model_cls(
        embed_dim=768,
        depth=12,
        num_heads=12,
        geometric_depth=2,
        v_heads=96,
        **kwargs,
    )


def fot_large(model_cls: type, **kwargs) -> "MiAEEncoder":
    return model_cls(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        geometric_depth=2,
        v_heads=128,
        **kwargs,
    )


def fot_huge(model_cls: type, **kwargs) -> "MiAEEncoder":
    return model_cls(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        geometric_depth=2,
        v_heads=128,
        **kwargs,
    )


def miae_encoder_model(name: str = "miae_b", dense: bool = False, **kwargs) -> "MiAEEncoder":
    """Instantiate a MiAE encoder (classification head) by name.

    Args:
        name: Model variant — ``"miae_s"`` (29M), ``"miae_b"`` (102M),
            ``"miae_l"`` (339M), or ``"miae_h"`` (huge). Legacy aliases
            ``"fot_small"``, ``"fot_base"``, ``"fot_large"``, ``"fot_huge"``
            are also accepted.
        dense: If ``True``, return a token-level (dense) variant used for
            inverse-folding pretraining instead of the CLS classification head.
        **kwargs: Forwarded to the model constructor (e.g. ``num_classes``).

    Returns:
        An instance of :class:`MiAEEncoder` or :class:`MiAEEncoderDense`.
    """
    model_cls = MiAEEncoderDense if dense else MiAEEncoder
    if name in ("miae_s", "fot_small"):
        return fot_small(model_cls, **kwargs)
    elif name in ("miae_b", "fot_base"):
        return fot_base(model_cls, **kwargs)
    elif name in ("miae_l", "fot_large"):
        return fot_large(model_cls, **kwargs)
    elif name in ("miae_h", "fot_huge"):
        return fot_huge(model_cls, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")
