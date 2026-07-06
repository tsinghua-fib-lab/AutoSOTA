import torch
from torch import nn
from minesm.utils.constants import esm3 as C
from minesm.layers.structure_proj import Dim6RotStructureHead
from minesm.models.vqvae import PairwisePredictionHead


class PredictionHead(nn.Module):
    """Decoder prediction head that produces backbone coordinates and pairwise logits.

    Given decoder hidden states it outputs:
    - ``bb_pred``: predicted backbone atom coordinates ``(B, L, 3, 3)`` for N/Cα/C.
    - ``pairwise_dist_logits``: pairwise Cβ-distance logits ``(B, L, L, 64)`` over
      64 distance bins (Appendix B.2 of the paper).
    - ``pairwise_dir_logits``: pairwise direction logits ``(B, L, L, 96)`` over
      16 bins × 6 direction vectors.
    - ``last_hidden_state``: the raw hidden states passed through unchanged for
      the inverse-folding classification head.

    Args:
        hidden_dim: Decoder hidden dimension (width of the decoder).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.affine_output_projection = Dim6RotStructureHead(
            self.hidden_dim, 10, predict_torsion_angles=False
        )

        direction_loss_bins = C.VQVAE_DIRECTION_LOSS_BINS
        self.pairwise_bins = [
            64,  # distogram
            direction_loss_bins * 6,  # direction bins
        ]
        self.pairwise_classification_head = PairwisePredictionHead(
            self.hidden_dim,
            downproject_dim=128,
            hidden_dim=128,
            n_bins=sum(self.pairwise_bins),
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        bb_pred_only: bool = False,
    ) -> torch.Tensor | dict:
        # x: [B, N, D]

        _, bb_pred = self.affine_output_projection(
            x, affine=None, affine_mask=torch.zeros_like(mask)
        )  # [B, L, 12], [B, L, 3, 3]

        if bb_pred_only:
            return bb_pred

        pairwise_logits = self.pairwise_classification_head(
            x
        )  # [B, L, L, 64 + 96 + 64]
        pairwise_dist_logits, pairwise_dir_logits = [
            (o if o.numel() > 0 else None)
            for o in pairwise_logits.split(self.pairwise_bins, dim=-1)
        ]

        return dict(
            bb_pred=bb_pred,
            pairwise_dist_logits=pairwise_dist_logits,
            pairwise_dir_logits=pairwise_dir_logits,
            last_hidden_state=x,
        )
