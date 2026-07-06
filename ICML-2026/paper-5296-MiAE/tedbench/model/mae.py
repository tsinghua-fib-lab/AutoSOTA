import torch
from torch import nn

from minesm.layers.transformer_stack import TransformerStack
from minesm.utils.constants import esm3 as C
from minesm.utils.structure.affine3d import build_affine3d_from_coordinates

from ..layer.head import PredictionHead
from ..layer.pos_embed import SinusoidalPositionalEmbedding1D
from .encoder import GeometricEncoder
from .loss import ESM3Loss


class MiAECore(nn.Module):
    """Core MiAE (Masked Invariant Autoencoder) pretraining model.

    Implements an asymmetric masked autoencoder for protein backbone frames
    following the design in Section 4 of the TEDBench paper:

    * **Encoder**: geometric attention blocks (SE(3)-invariant) followed by a
      standard Transformer stack; processes only the *visible* (unmasked) residues.
    * **Decoder**: lightweight Transformer with rotary positional embeddings that
      operates on all positions (visible encoder outputs + learned mask tokens)
      to reconstruct the full backbone.

    After pretraining the decoder is discarded and the encoder is used as a
    feature extractor in :class:`~tedbench.model.MiAEEncoder`.

    Args:
        embed_dim: Encoder hidden dimension (512 / 768 / 1024 for S / B / L).
        depth: Number of standard Transformer layers in the encoder.
        num_heads: Number of attention heads in the encoder.
        geometric_depth: Number of geometric attention layers prepended to encoder.
        v_heads: Value-head count for geometric attention.
        use_seq_input: Add sequence embeddings to visible-frame encoder inputs
            (MiAE+seq variant).
        decoder_embed_dim: Hidden dimension of the lightweight decoder (default 512).
        decoder_depth: Number of Transformer layers in the decoder (default 8 for
            MiAE-B, recommended to tune jointly with width).
        decoder_num_heads: Number of attention heads in the decoder.
        masking_strategy: One of ``"random"`` (default, best per Table 4e),
            ``"span"``, ``"uniform"``, or ``"alternating"``.
        use_inverse_folding_loss: Include the inverse-folding cross-entropy term
            in the reconstruction loss (strongly recommended, see Table 4d).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        geometric_depth: int = 2,
        v_heads: int = 96,
        use_seq_input: bool = False,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        masking_strategy: str = "fixed",
        use_inverse_folding_loss: bool = True,
    ) -> None:
        super().__init__()
        self.masking_strategy = masking_strategy
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = SinusoidalPositionalEmbedding1D(embed_dim)
        self.input_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.geometric_encoder = GeometricEncoder(
            embed_dim, num_heads, v_heads, geometric_depth
        )
        self.seq_embed = None
        if use_seq_input:
            self.seq_embed = nn.Embedding(len(C.SEQUENCE_VOCAB), embed_dim)
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

        # decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder = TransformerStack(
            decoder_embed_dim,
            decoder_num_heads,
            1,
            decoder_depth,
            scale_residue=False,
            n_layers_geom=0,
            use_rotary=True,
        )
        self.decoder_head = PredictionHead(decoder_embed_dim)

        # loss
        self.loss_fn = ESM3Loss(
            decoder_embed_dim, use_inverse_folding_loss=use_inverse_folding_loss
        )
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        if self.seq_embed is not None:
            nn.init.normal_(self.seq_embed.weight, std=0.02)

    @torch.no_grad()
    def random_masking(self, mask: torch.Tensor, mask_ratio: float):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        mask: [N, L], sequence
        """
        B, N = mask.shape
        device = mask.device
        # valid lengths and per-row keep counts
        valid_lengths = mask.sum(dim=1)  # [B]
        keep_nums = (
            (valid_lengths.float() * (1.0 - mask_ratio)).floor().long().clamp(min=1)
        )  # [B]
        K_max = keep_nums.max()

        # Noise: valid in [0,1), padded in [2,3) so valid sort ahead of pads without ties
        noise = torch.rand(B, N, device=device)
        noise = torch.where(mask, noise, 2.0 + noise)

        # Per-sample shuffle (ascending noise)
        ids_shuffle = noise.argsort(dim=1)  # [B, N]
        ids_restore = ids_shuffle.argsort(dim=1)  # [B, N]
        ids_restore[ids_restore >= keep_nums.unsqueeze(1)] = N - 1

        # The first L_i entries of ids_shuffle are valid tokens; keep first K_i of those
        max_valid = valid_lengths.max()
        ids_valid = ids_shuffle[:, :max_valid]  # [B, max_valid]

        # keep the first subset
        ids_keep_pad = ids_valid[:, :K_max].clone()  # tentative [B, K_max]
        arangeK = torch.arange(K_max, device=device).unsqueeze(0)  # [1, K_max]
        over = arangeK >= keep_nums.unsqueeze(
            1
        )  # [B, K_max] True where padded positions
        fill = ids_shuffle[:, -1:].expand(
            -1, K_max
        )  # [B, K_max] duplicate first valid index
        ids_keep_pad = torch.where(over, fill, ids_keep_pad)  # [B, K_max]
        random_mask = torch.ones(B, N, dtype=mask.dtype, device=device)
        random_mask.scatter_(1, ids_keep_pad, False)
        random_mask = mask & random_mask
        new_mask = ~over

        return ids_keep_pad, new_mask, random_mask, ids_restore

    def span_masking(self, mask: torch.Tensor, mask_ratio: float, span_length: int = 5):
        """
        Fully vectorized span masking with fixed span length, enforced minimum distance
        between span starts, and exact mask count matching floor(valid_length * mask_ratio).

        Args:
            mask: [B, N], boolean mask indicating valid (non-padding) tokens
            mask_ratio: fraction of valid tokens to mask
            span_length: fixed length of each span

        Returns:
            ids_keep_pad: [B, K_max] indices of kept tokens (padded)
            new_mask: [B, K_max] mask for valid kept positions
            span_mask: [B, N] boolean mask (True = masked position)
            ids_restore: [B, N] indices to restore original order
        """
        B, N = mask.shape
        device = mask.device

        valid_lengths = mask.sum(dim=1)  # [B]

        # Exact targets (matching random_masking)
        keep_nums_target = (
            (valid_lengths.float() * (1.0 - mask_ratio)).floor().long().clamp(min=1)
        )  # [B]
        mask_nums_target = (valid_lengths - keep_nums_target).clamp(min=0)  # [B]

        # ----- Chunk-based span start sampling -----

        # Compute number of spans needed
        num_spans = (
            (valid_lengths.float() * mask_ratio / span_length)
            .ceil()
            .long()
            .clamp(min=1)
        )
        max_num_spans = num_spans.max().item()

        span_idx = torch.arange(max_num_spans, device=device).unsqueeze(
            0
        )  # [1, max_num_spans]

        # Chunk boundaries
        chunk_starts = (
            span_idx * valid_lengths.unsqueeze(1) // num_spans.unsqueeze(1).clamp(min=1)
        )
        chunk_ends = (
            (span_idx + 1)
            * valid_lengths.unsqueeze(1)
            // num_spans.unsqueeze(1).clamp(min=1)
        )
        chunk_ends = chunk_ends.clamp(max=valid_lengths.unsqueeze(1))

        # Sample uniformly within each chunk
        chunk_lengths = (chunk_ends - chunk_starts).clamp(min=1)
        offsets = (
            torch.rand(B, max_num_spans, device=device) * chunk_lengths.float()
        ).long()
        span_start_positions = (chunk_starts + offsets).clamp(
            min=0, max=N - 1
        )  # [B, max_num_spans]

        # Valid span slots
        slot_valid = span_idx < num_spans.unsqueeze(1)  # [B, max_num_spans]

        # Scatter span starts
        span_starts = torch.zeros(B, N, dtype=torch.bool, device=device)
        span_start_positions_masked = torch.where(
            slot_valid, span_start_positions, torch.zeros_like(span_start_positions)
        )
        span_starts.scatter_(1, span_start_positions_masked, slot_valid)
        span_starts = span_starts & mask

        # ----- Fixed span length propagation with cummax -----

        positions = torch.arange(N, device=device).unsqueeze(0)  # [1, N]
        span_ends = torch.where(
            span_starts,
            (positions + span_length).clamp(max=N),
            torch.zeros(B, N, dtype=torch.long, device=device),
        )

        cummax_ends, _ = torch.cummax(span_ends, dim=1)
        span_mask = (cummax_ends > positions) & mask  # [B, N]

        # ----- Post-hoc adjustment to hit exact target -----

        noise = torch.rand(B, N, device=device)

        # Scores: currently masked [0,1), unmasked valid [1,2), padding [2,3)
        adjustment_score = torch.where(
            span_mask, noise, torch.where(mask, 1.0 + noise, 2.0 + noise)
        )

        # Sort by score and select exactly mask_nums_target positions
        sorted_indices = adjustment_score.argsort(dim=1)  # [B, N]

        # Positions in sorted order < mask_nums_target should be masked
        should_be_masked_sorted = positions < mask_nums_target.unsqueeze(1)  # [B, N]

        # Scatter back to original positions
        final_span_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        final_span_mask.scatter_(1, sorted_indices, should_be_masked_sorted)
        final_span_mask = final_span_mask & mask

        # ----- Compute outputs (same as random_masking) -----

        keep_nums = keep_nums_target
        K_max = keep_nums.max().item()

        noise = torch.rand(B, N, device=device)
        score = torch.where(
            mask & ~final_span_mask, noise, torch.where(mask, 1.0 + noise, 2.0 + noise)
        )

        ids_shuffle = score.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_restore[ids_restore >= keep_nums.unsqueeze(1)] = N - 1

        ids_keep_pad = ids_shuffle[:, :K_max].clone()
        arange_K = torch.arange(K_max, device=device).unsqueeze(0)
        over = arange_K >= keep_nums.unsqueeze(1)
        fill = ids_shuffle[:, -1:].expand(-1, K_max)
        ids_keep_pad = torch.where(over, fill, ids_keep_pad)

        new_mask = ~over

        return ids_keep_pad, new_mask, final_span_mask, ids_restore

    def forward_encoder(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        residue_index: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
        seq_tokens: torch.Tensor | None = None,
        mask_ratio: float = 0.0,
        noise: float = 0.0,
    ):
        coords = coords[..., :3, :]
        if self.masking_strategy == "uniform" and self.training:
            mask_ratio = (
                torch.rand(1, device=coords.device) * (1 - mask_ratio) + mask_ratio
            )
        if self.masking_strategy == "alternating" and self.training:
            if (self.seq_embed is not None) and (seq_tokens is not None):
                mask_ratio = 0.15 if torch.rand(1) < 0.5 else mask_ratio
        if self.masking_strategy == "span":
            ids_keep, new_mask, random_mask, ids_restore = self.span_masking(
                mask, mask_ratio=mask_ratio
            )
        else:
            ids_keep, new_mask, random_mask, ids_restore = self.random_masking(
                mask, mask_ratio=mask_ratio
            )
        coords_masked = coords.gather(
            1, ids_keep.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, coords.shape[-1])
        )
        residue_index_masked = residue_index.gather(1, ids_keep)
        pos_embed_masked = self.pos_embed(residue_index_masked)
        if noise > 0.0 and self.training:
            coords_masked = coords_masked + noise * torch.randn_like(coords_masked)
        affine, affine_mask = build_affine3d_from_coordinates(coords=coords_masked)
        x = self.input_embed.expand(affine.shape[0], affine.shape[1], -1)
        x, _, _ = self.geometric_encoder(x, new_mask, affine, affine_mask, sequence_id)
        if (self.seq_embed is not None) and (seq_tokens is not None):
            seq_tokens_masked = seq_tokens.gather(1, ids_keep)
            seq_embed = self.seq_embed(seq_tokens_masked)
            x = x + seq_embed
        x = x + pos_embed_masked
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        new_mask = torch.cat((self.cls_mask.expand(x.shape[0], -1), new_mask), dim=1)
        x, _, _ = self.encoder(
            x,
            attention_mask=new_mask,
            affine=None,
            affine_mask=None,
            sequence_id=sequence_id,
        )
        return x, ids_restore, random_mask

    def forward_decoder(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        ids_restore: torch.Tensor,
        sequence_id: torch.Tensor | None,
    ) -> dict:
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x_ = x_ * mask.unsqueeze(-1)
        if mask is None:
            mask = torch.ones(
                x_.shape[0], x_.shape[1], dtype=torch.bool, device=x.device
            )
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        mask = torch.cat((self.cls_mask.expand(x.shape[0], -1), mask), dim=1)
        x, _, _ = self.decoder(
            x,
            attention_mask=mask,
            affine=None,
            affine_mask=None,
            sequence_id=sequence_id,
        )
        # remove cls token
        x = x[:, 1:, :]
        mask = mask[:, 1:]
        x = self.decoder_head(x, mask)
        return x

    def forward(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        residue_index: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
        seq_tokens: torch.Tensor | None = None,
        mask_ratio: float = 0.0,
        noise: float = 0.0,
    ):
        x, ids_restore, random_mask = self.forward_encoder(
            coords, mask, residue_index, sequence_id, seq_tokens, mask_ratio, noise
        )
        x = self.forward_decoder(x, mask, ids_restore, sequence_id)
        return x

    def forward_encoder_with_masked_ids(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        residue_index: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
        seq_tokens: torch.Tensor | None = None,
        ids_masked: torch.Tensor | None = None,
    ):
        coords = coords[..., :3, :]
        # assert ids_masked is not None
        if mask is None:
            mask = torch.ones_like(coords[..., 0, 0], dtype=torch.bool)
        if ids_masked is not None:
            coords_masked = coords.gather(
                1,
                ids_masked.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, 3, coords.shape[-1]),
            )
            new_mask = mask.gather(1, ids_masked)
            residue_index_masked = residue_index.gather(1, ids_masked)
        else:
            coords_masked = coords
            new_mask = mask
            residue_index_masked = residue_index
        pos_embed_masked = self.pos_embed(residue_index_masked)
        affine, affine_mask = build_affine3d_from_coordinates(coords=coords_masked)
        x = self.input_embed.expand(affine.shape[0], affine.shape[1], -1)
        x, _, _ = self.geometric_encoder(x, new_mask, affine, affine_mask, sequence_id)
        if (self.seq_embed is not None) and (seq_tokens is not None):
            seq_tokens_masked = seq_tokens.gather(1, ids_masked)
            seq_embed = self.seq_embed(seq_tokens_masked)
            x = x + seq_embed
        x = x + pos_embed_masked
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        new_mask = torch.cat((self.cls_mask.expand(x.shape[0], -1), new_mask), dim=1)
        x, _, _ = self.encoder(
            x,
            attention_mask=new_mask,
            affine=None,
            affine_mask=None,
            sequence_id=sequence_id,
        )
        return x

    def forward_with_masked_ids(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        residue_index: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
        seq_tokens: torch.Tensor | None = None,
        ids_masked: torch.Tensor | None = None,
        ids_restore: torch.Tensor | None = None,
    ):
        x = self.forward_encoder_with_masked_ids(
            coords, mask, residue_index, sequence_id, seq_tokens, ids_masked
        )
        x = self.forward_decoder(x, mask, ids_restore, sequence_id)
        return x


def mae_small_dec512d2b(**kwargs) -> MiAECore:
    return MiAECore(
        embed_dim=512,
        depth=6,
        num_heads=8,
        geometric_depth=2,
        v_heads=64,
        decoder_embed_dim=512,
        decoder_depth=2,
        decoder_num_heads=16,
        **kwargs,
    )


def mae_base_dec512d8b(**kwargs) -> MiAECore:
    return MiAECore(
        embed_dim=768,
        depth=12,
        num_heads=12,
        geometric_depth=2,
        v_heads=96,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        **kwargs,
    )


def mae_large_dec512d2b(**kwargs) -> MiAECore:
    return MiAECore(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        geometric_depth=2,
        v_heads=128,
        decoder_embed_dim=512,
        decoder_depth=2,
        decoder_num_heads=16,
        **kwargs,
    )


def mae_huge_dec512d2b(**kwargs) -> MiAECore:
    return MiAECore(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        geometric_depth=2,
        v_heads=128,
        decoder_embed_dim=512,
        decoder_depth=2,
        decoder_num_heads=16,
        **kwargs,
    )


def mae_base_dec512d2b(**kwargs) -> MiAECore:
    return MiAECore(
        embed_dim=768,
        depth=12,
        num_heads=12,
        geometric_depth=2,
        v_heads=96,
        decoder_embed_dim=512,
        decoder_depth=2,
        decoder_num_heads=16,
        **kwargs,
    )


def mae_base_dec512d1b(**kwargs) -> MiAECore:
    return MiAECore(
        embed_dim=768,
        depth=12,
        num_heads=12,
        geometric_depth=2,
        v_heads=96,
        decoder_embed_dim=512,
        decoder_depth=1,
        decoder_num_heads=16,
        **kwargs,
    )


def mae_base_dec512d4b(**kwargs) -> MiAECore:
    return MiAECore(
        embed_dim=768,
        depth=12,
        num_heads=12,
        geometric_depth=2,
        v_heads=96,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=16,
        **kwargs,
    )


def mae_base_dec512d6b(**kwargs) -> MiAECore:
    return MiAECore(
        embed_dim=768,
        depth=12,
        num_heads=12,
        geometric_depth=2,
        v_heads=96,
        decoder_embed_dim=512,
        decoder_depth=6,
        decoder_num_heads=16,
        **kwargs,
    )


def mae_base_dec256d2b(**kwargs) -> MiAECore:
    return MiAECore(
        embed_dim=768,
        depth=12,
        num_heads=12,
        geometric_depth=2,
        v_heads=96,
        decoder_embed_dim=256,
        decoder_depth=2,
        decoder_num_heads=16,
        **kwargs,
    )


def mae_base_dec768d2b(**kwargs) -> MiAECore:
    return MiAECore(
        embed_dim=768,
        depth=12,
        num_heads=12,
        geometric_depth=2,
        v_heads=96,
        decoder_embed_dim=768,
        decoder_depth=2,
        decoder_num_heads=16,
        **kwargs,
    )


def miae_model(name: str = "miae_b", **kwargs) -> MiAECore:
    """Instantiate a MiAE pretraining model by name.

    Args:
        name: Model variant. Primary names: ``"miae_s"`` (29M, 6 layers,
            512-dim), ``"miae_b"`` (102M, 12 layers, 768-dim, default),
            ``"miae_l"`` (339M, 24 layers, 1024-dim), ``"miae_h"`` (huge).
            Ablation variants: ``"miae_b_dec512d1b"``, ``"miae_b_dec512d2b"``,
            ``"miae_b_dec512d4b"``, ``"miae_b_dec512d6b"``,
            ``"miae_b_dec256d2b"``, ``"miae_b_dec768d2b"``.
            Legacy ``"mae_*"`` aliases are also accepted.
        **kwargs: Forwarded to :class:`MiAECore` (e.g. ``masking_strategy``,
            ``use_seq_input``, ``use_inverse_folding_loss``).

    Returns:
        A configured :class:`MiAECore` instance.
    """
    if name in ("miae_s", "mae_small"):
        return mae_small_dec512d2b(**kwargs)
    elif name in ("miae_b", "mae_base"):
        return mae_base_dec512d2b(**kwargs)
    elif name in ("miae_b_dec512d8b", "mae_base_dec512d8b"):
        return mae_base_dec512d8b(**kwargs)
    elif name in ("miae_b_dec512d4b", "mae_base_dec512d4b"):
        return mae_base_dec512d4b(**kwargs)
    elif name in ("miae_b_dec512d2b", "mae_base_dec512d2b"):
        return mae_base_dec512d2b(**kwargs)
    elif name in ("miae_b_dec512d1b", "mae_base_dec512d1b"):
        return mae_base_dec512d1b(**kwargs)
    elif name in ("miae_b_dec256d2b", "mae_base_dec256d2b"):
        return mae_base_dec256d2b(**kwargs)
    elif name in ("miae_b_dec768d2b", "mae_base_dec768d2b"):
        return mae_base_dec768d2b(**kwargs)
    elif name in ("miae_l", "mae_large"):
        return mae_large_dec512d2b(**kwargs)
    elif name in ("miae_h", "mae_huge"):
        return mae_huge_dec512d2b(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")
