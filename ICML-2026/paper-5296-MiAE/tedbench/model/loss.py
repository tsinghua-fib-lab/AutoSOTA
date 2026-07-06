import torch
import torch.nn.functional as F
from torch import nn

from minesm.models.vqvae import RegressionHead
from minesm.utils.constants import esm3 as C
from minesm.utils.structure.protein_chain import ProteinChain
from minesm.utils.structure.protein_structure import infer_cbeta_from_atom37


class ESM3Loss(nn.Module):
    """Composite reconstruction loss for MiAE pretraining (Appendix B.2).

    Combines five terms following the ESM3 reconstruction objective:

    1. **Geometric distance** (``L_dist``): MSE on pairwise Cα distances, clamped at 25 Å².
    2. **Geometric direction** (``L_dir``): MSE on pairwise backbone-atom direction
       vectors, clamped at 20.
    3. **Binned distance** (``L_binned_dist``): Cross-entropy over 64 Cβ distance bins.
    4. **Binned direction** (``L_binned_dir``): Cross-entropy over 16 × 6 direction bins.
    5. **Inverse folding** (``L_invf``, optional): Cross-entropy predicting amino-acid
       identity from decoder representations.  Omitting this term hurts downstream
       performance (see Table 4d in the paper).

    Args:
        hidden_dim: Decoder hidden dimension, used to build the inverse-folding head.
        use_inverse_folding_loss: Whether to include the inverse-folding term.
    """

    def __init__(self, hidden_dim: int, use_inverse_folding_loss: bool = True) -> None:
        super().__init__()

        self.inverse_folding_head = None

        if use_inverse_folding_loss:
            self.inverse_folding_head = RegressionHead(
                embed_dim=hidden_dim, output_dim=len(C.SEQUENCE_VOCAB)
            )

    def forward(
        self, pred_dict: dict, true_dict: dict
    ) -> tuple[torch.Tensor, dict]:
        # (1) backbone geometric distance loss: pairwise L2 distance matrix for
        # the predicted and true coordinates of the 3 backbone atoms (N, Cα, C)
        geom_dist_loss, geom_dist_metrics = self.compute_geometric_distance(
            pred_dict["bb_pred"], true_dict["coords"], true_dict["mask"]
        )  # [B, L, 3, 3]
        # (2) backbone geometric direction loss
        geom_dir_loss, geom_dir_metrics = self.compute_geometric_direction(
            pred_dict["bb_pred"], true_dict["coords"], true_dict["mask"]
        )
        # (3) backbone binned distance classification
        binned_dist_loss, binned_dist_metrics = self.compute_binned_distance(
            pred_dict["pairwise_dist_logits"], true_dict["coords"], true_dict["mask"]
        )
        # (4) backbone binned direction classification
        binned_dir_loss, binned_dir_metrics = self.compute_binned_direction(
            pred_dict["pairwise_dir_logits"], true_dict["coords"], true_dict["mask"]
        )
        # (5) inverse folding
        if self.inverse_folding_head is not None:
            inverse_folding_loss, inverse_folding_metrics = (
                self.compute_inverse_folding(
                    pred_dict["last_hidden_state"],
                    true_dict["seq_tokens"],
                    true_dict["mask"],
                )
            )
        else:
            inverse_folding_loss = 0.0
            inverse_folding_metrics = {}
        loss = (
            geom_dist_loss
            + geom_dir_loss
            + binned_dist_loss
            + binned_dir_loss
            + inverse_folding_loss
        )
        metrics = {
            **geom_dist_metrics,
            **geom_dir_metrics,
            **binned_dist_metrics,
            **binned_dir_metrics,
            **inverse_folding_metrics,
        }
        return loss, metrics

    def compute_geometric_distance(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        clamp_value: float = 25,
    ) -> tuple[torch.Tensor, dict]:
        """
        x_recon: [B, L, 3, 3]
        x: [B, L, 3, 3]
        """
        assert x_recon.shape[-2] == 3 and x_recon.shape[-1] == 3

        # ignore padding regions
        x_recon[~attention_mask] = 0
        x[~attention_mask] = 0
        B, L, E = x.shape[0], x.shape[1], x.shape[-1]
        x_recon, x = (
            x_recon.reshape(B, -1, E),
            x.reshape(B, -1, E),
        )  # [B, L, 3, 3] -> [B, L * 3, 3]

        # dist_pred = torch.sum((x_recon[:, :, None, :] - x_recon[:, None, :, :]) ** 2, dim=-1) # [B, L * 3, L * 3]
        # dist_true = torch.sum((x[:, :, None, :] - x[:, None, :, :]) ** 2, dim=-1)
        # Bug fixed: https://github.com/KatarinaYuan/StructTokenBench/blob/cd4cfe5026dc0e6919a8116367ec2d129afabd29/src/vqvae_model.py#L608C9-L609C45
        # dist_pred = torch.cdist(x_recon, x_recon) # [B, L * 3, L * 3]
        # dist_true = torch.cdist(x, x)
        dist_pred = (x_recon.unsqueeze(-2) - x_recon.unsqueeze(-3)).norm(dim=-1)
        dist_true = (x.unsqueeze(-2) - x.unsqueeze(-3)).norm(dim=-1)

        dist_mask = attention_mask.repeat(1, 3)
        dist_mask = torch.logical_and(
            dist_mask.unsqueeze(-1), dist_mask.unsqueeze(1)
        )  # [B, L * 3, L * 3]
        dist_pred, dist_true = dist_pred[dist_mask], dist_true[dist_mask]
        loss = F.mse_loss(dist_pred, dist_true, reduction="none")  # flattened
        loss = torch.clamp(loss, max=clamp_value)
        metric = {
            f"geom_dist_loss": loss.mean(),
            f"geom_dist_loss_below_clamp": loss[loss != clamp_value].mean(),
            f"geom_dist_loss_clamp_ratio_{clamp_value}": (loss != clamp_value)
            .float()
            .mean(),
        }
        # metrics like spearman R is too time consuming to calculate
        return loss.mean(), metric

    def compute_direction_vectors(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: [B, L, 3, 3]
        """
        # N -> Ca
        v1 = coords[:, :, 1, :] - coords[:, :, 0, :]  # [B, 0~L, 3]
        # Ca -> C
        v2 = coords[:, :, 2, :] - coords[:, :, 1, :]  # [B, 0~L, 3]
        # C -> N_next
        v3 = coords[:, 1:, 0, :] - coords[:, :-1, 2, :]  # [B, 0~L-1, 3]
        # -(N -> Ca) x (Ca -> C)
        v4 = -torch.cross(v1, v2, dim=-1)  # [B, 0~L, 3]
        # (C_prev -> N) x (N -> Ca)
        tmp = coords[:, 1:, 0, :] - coords[:, :-1, 2, :]  # [B, 1~L, 3]
        v5 = torch.cross(tmp, v1[:, 1:], dim=-1)
        # (Ca -> C) x (C -> N_next)
        v6 = torch.cross(v2[:, :-1], v3, dim=-1)  # [B, 0~L-1, 3]

        ret = [
            v1[:, 1:-1],
            v2[:, 1:-1],
            v3[:, 1:],
            v4[:, 1:-1],
            v5[:, :-1],
            v6[:, 1:],
        ]  # [B, L-2, 3]
        ret = torch.stack(ret, dim=1)  # [B, 6, L-2, 3]
        ret = ret.reshape(ret.shape[0], -1, ret.shape[-1])  # [B, 6 * (L-2), 3]

        return ret

    def compute_geometric_direction(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        clamp_value: float = 20,
    ) -> tuple[torch.Tensor, dict]:
        """
        x_recon: [B, L, 3, 3]
        x: [B, L, 3, 3]
        attention_mask: [B, L]
        """
        vec_pred = self.compute_direction_vectors(x_recon)
        vec = self.compute_direction_vectors(x)

        # pairwise dot product
        dist_pred = torch.matmul(
            vec_pred, torch.transpose(vec_pred, 1, 2)
        )  # [B, 6(L-2), 6(L-2)]
        dist_true = torch.matmul(vec, torch.transpose(vec, 1, 2))  # [B, 6(L-2), 6(L-2)]

        dist_mask = attention_mask[:, 1:-1].repeat(1, 6)  # [B, 6(L-2)]
        dist_mask = torch.logical_and(
            dist_mask.unsqueeze(-1), dist_mask.unsqueeze(1)
        )  # [B, 6(L-2), 6(L-2)]
        dist_pred, dist_true = dist_pred[dist_mask], dist_true[dist_mask]
        loss = F.mse_loss(dist_pred, dist_true, reduction="none")  # flattened
        loss = torch.clamp(loss, max=clamp_value)
        metric = {
            f"geom_dir_loss": loss.mean(),
            f"geom_dir_loss_below_clamp": loss[loss != clamp_value].mean(),
            f"geom_dir_loss_clamp_ratio_{clamp_value}": (loss != clamp_value)
            .float()
            .mean(),
        }
        return loss.mean(), metric

    def compute_binned_direction(
        self,
        pairwise_logits: torch.Tensor,
        coords: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        pairwise_logits: [B, L, L, 96]
        coords: [B, L, 3, 3]
        attention_mask: [B, L]
        """
        # compute from ground truth
        # unit vectors
        # Ca -> C
        v1 = coords[:, :, 2, :] - coords[:, :, 1, :]  # [B, 0~L, 3]
        # Ca -> N
        v2 = coords[:, :, 0, :] - coords[:, :, 1, :]  # [B, 0~L, 3]
        # (Ca -> C) x (Ca -> N)
        v3 = torch.cross(v1, v2, dim=-1)  # [B, L, 3]
        v1 = F.normalize(v1, p=2, dim=-1)
        v2 = F.normalize(v2, p=2, dim=-1)
        v3 = F.normalize(v3, p=2, dim=-1)

        # dot products
        pairwise_prod = torch.stack(
            [
                torch.matmul(v1, torch.transpose(v2, 1, 2)),  # [B, L, L]
                torch.matmul(v1, torch.transpose(v3, 1, 2)),
                torch.matmul(v2, torch.transpose(v1, 1, 2)),
                torch.matmul(v2, torch.transpose(v3, 1, 2)),
                torch.matmul(v3, torch.transpose(v1, 1, 2)),
                torch.matmul(v3, torch.transpose(v2, 1, 2)),
            ],
            dim=-1,
        )  # [B, L, L, 6]
        NUM_BIN = 16
        bin_edges = [-1 + 0.125 * i for i in range(NUM_BIN)] + [1]
        bin_edges = torch.tensor(bin_edges, device=pairwise_logits.device)
        binned_labels = (
            torch.bucketize(pairwise_prod, bin_edges, right=True) - 1
        )  # [B, L, L, 6]
        binned_labels = torch.clamp(binned_labels, max=NUM_BIN - 1, min=0)
        pairwise_logits = pairwise_logits.reshape(
            [_ for _ in binned_labels.shape] + [-1]
        )  # [B, L, L, 6, NUM_BIN]

        mask = torch.logical_and(
            attention_mask.unsqueeze(-1), attention_mask.unsqueeze(1)
        )  # [B, L, L]
        pairwise_logits, binned_labels = (
            pairwise_logits[mask].reshape(-1, NUM_BIN),
            binned_labels[mask].reshape(-1),
        )

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(pairwise_logits, binned_labels)

        metric = {
            f"binned_dir_loss": loss.mean(),
            f"binned_dir_accuracy": (pairwise_logits.argmax(dim=-1) == binned_labels)
            .float()
            .mean(),
        }
        return loss.mean(), metric

    def compute_binned_distance(
        self,
        pairwise_logits: torch.Tensor,
        coords: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        pairwise_logits: [B, L, L, 64]
        coords: [B, L, 37, 3]
        attention_mask: [B, L]
        """

        # calculate Cbeta
        cbeta = infer_cbeta_from_atom37(coords)  # [B, L, 3]

        # pairwise Cbeta distance
        NUM_BIN = 64
        dist_true = torch.cdist(cbeta, cbeta, p=2.0)
        bin_edges = [0] + [(2.3125 + 0.3075 * i) ** 2 for i in range(NUM_BIN)]
        bin_edges = torch.tensor(bin_edges, device=pairwise_logits.device)
        binned_labels = (
            torch.bucketize(dist_true, bin_edges, right=True) - 1
        )  # [B, L, L]
        binned_labels = torch.clamp(binned_labels, max=NUM_BIN - 1, min=0)
        assert binned_labels.min() >= 0 and binned_labels.max() < NUM_BIN

        mask = torch.logical_and(
            attention_mask.unsqueeze(-1), attention_mask.unsqueeze(1)
        )  # [B, L, L]
        pairwise_logits, binned_labels = pairwise_logits[mask], binned_labels[mask]

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(pairwise_logits, binned_labels)

        metric = {
            f"binned_dist_loss": loss.mean(),
            f"binned_dist_accuracy": (pairwise_logits.argmax(dim=-1) == binned_labels)
            .float()
            .mean(),
        }
        return loss.mean(), metric

    def compute_inverse_folding(
        self,
        h: torch.Tensor,
        residue_labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        h: [B, L, d_model=1024]
        residue_labels: [B, L]
        attention_mask: [B, L]
        """
        logits = self.inverse_folding_head(h)  # [B, L, num_AAs]

        if not (
            logits.shape[0] == attention_mask.shape[0]
            and logits.shape[1] == attention_mask.shape[1]
        ):
            raise ValueError

        logits, residue_labels = logits[attention_mask], residue_labels[attention_mask]

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits, residue_labels)

        metric = {
            f"inverse_folding_loss": loss.mean(),
            f"inverse_folding_accuracy": (logits.argmax(dim=-1) == residue_labels)
            .float()
            .mean(),
        }
        return loss.mean(), metric


class RMSDLoss:
    @torch.no_grad()
    def __call__(
        self, pred_dict: dict, true_dict: dict, reduction: bool = True
    ) -> dict:
        bb_rmsd_list = []
        lddt_list = []
        device = pred_dict["bb_pred"][0].device
        for i in range(len(pred_dict["bb_pred"])):
            pdb_chain_recon = ProteinChain.from_backbone_atom_coordinates(
                pred_dict["bb_pred"][i].detach()
            )
            pdb_chain_recon = pdb_chain_recon[: len(true_dict["protein_chain"][i])]

            bb_rmsd = pdb_chain_recon.rmsd(
                true_dict["protein_chain"][i], only_compute_backbone_rmsd=True
            ).float()
            lddt = pdb_chain_recon.lddt_ca(true_dict["protein_chain"][i]).mean()
            bb_rmsd_list.append(bb_rmsd)
            lddt_list.append(lddt)
        if not reduction:
            return dict(
                rmsd=torch.stack(bb_rmsd_list).to(device),
                lddt=torch.stack(lddt_list).to(device),
            )
        return dict(
            rmsd=torch.stack(bb_rmsd_list).to(device).mean(),
            lddt=torch.stack(lddt_list).to(device).mean(),
        )
