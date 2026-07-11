import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAlignmentLoss(nn.Module):
    def __init__(self, num_samples, num_clusters, js_weight=0.1):
        super().__init__()
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.js_weight = js_weight

    def mask_correlated_samples(self, num_items):
        mask = torch.ones((num_items, num_items), dtype=torch.bool)
        mask = mask.fill_diagonal_(False)

        for i in range(num_items // 2):
            mask[i, num_items // 2 + i] = False
            mask[num_items // 2 + i, i] = False

        return mask

    def target_distribution(self, q):
        weight = (q ** 2.0) / torch.sum(q, dim=0)
        return (weight.t() / torch.sum(weight, dim=1)).t()

    def forward_prob(self, q_i, q_j):
        q_i = self.target_distribution(q_i)
        q_j = self.target_distribution(q_j)

        p_i = q_i.sum(0).view(-1)
        p_i = p_i / (p_i.sum() + 1e-9)
        ne_i = (p_i * torch.log(p_i + 1e-9)).sum()

        p_j = q_j.sum(0).view(-1)
        p_j = p_j / (p_j.sum() + 1e-9)
        ne_j = (p_j * torch.log(p_j + 1e-9)).sum()

        kl_ij = torch.sum(p_i * torch.log((p_i + 1e-9) / (p_j + 1e-9)))
        kl_ji = torch.sum(p_j * torch.log((p_j + 1e-9) / (p_i + 1e-9)))
        js_divergence = 0.5 * (kl_ij + kl_ji)

        return ne_i + ne_j + self.js_weight * js_divergence

    def forward_label(self, q_i, q_j, temperature_l, normalized=False):
        q_i = self.target_distribution(q_i).t()
        q_j = self.target_distribution(q_j).t()

        num_items = 2 * self.num_clusters
        q = torch.cat((q_i, q_j), dim=0)

        if normalized:
            sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / temperature_l
        else:
            sim = torch.matmul(q, q.t()) / temperature_l

        sim_i_j = torch.diag(sim, self.num_clusters)
        sim_j_i = torch.diag(sim, -self.num_clusters)
        positives = torch.cat((sim_i_j, sim_j_i), dim=0).view(num_items, 1)

        mask = self.mask_correlated_samples(num_items).to(q.device)
        negatives = sim[mask].view(num_items, -1)

        logits = torch.cat((positives, negatives), dim=1)
        labels = torch.zeros(num_items, dtype=torch.long, device=q.device)

        return self.criterion(logits, labels) / num_items


def _info_nce(z1, z2, tau=0.2, valid=None):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    if valid is not None:
        idx = valid.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() <= 1:
            return torch.tensor(0.0, device=z1.device)
        z1 = z1[idx]
        z2 = z2[idx]

    batch_size = z1.size(0)
    if batch_size <= 1:
        return torch.tensor(0.0, device=z1.device)

    sim = torch.mm(z1, z2.t()) / tau
    labels = torch.arange(batch_size, device=z1.device)

    return F.cross_entropy(sim, labels)


def _sym_kl(p, q, eps=1e-8):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)

    kl_pq = (p * (p.log() - q.log())).sum(dim=1).mean()
    kl_qp = (q * (q.log() - p.log())).sum(dim=1).mean()

    return kl_pq + kl_qp


class MultiPathConsensusLoss(nn.Module):
    def __init__(
        self,
        temperature=0.2,
        lambda_fuse=1.0,
        lambda_uni=1.0,
        lambda_mask=1.0,
        lambda_cons=0.05,
    ):
        super().__init__()
        self.tau = temperature
        self.l_fuse = lambda_fuse
        self.l_uni = lambda_uni
        self.l_mask = lambda_mask
        self.l_cons = lambda_cons

    def forward(self, aug):
        z_fuse_1 = aug["Z_fuse_1"]
        z_fuse_2 = aug["Z_fuse_2"]
        z_uni_1 = aug["Z_uni_1"]
        z_uni_2 = aug["Z_uni_2"]
        z_mask_1 = aug["Z_mask_1"]
        z_mask_2 = aug["Z_mask_2"]

        q_fuse_1 = aug["Q_fuse_1"]
        q_fuse_2 = aug["Q_fuse_2"]
        q_uni_1 = aug["Q_uni_1"]
        q_uni_2 = aug["Q_uni_2"]
        q_mask_1 = aug["Q_mask_1"]
        q_mask_2 = aug["Q_mask_2"]

        visible_masks = aug.get("visible_masks", aug.get("vis_masks", None))
        device = z_fuse_1.device

        loss_fuse = _info_nce(z_fuse_1, z_fuse_2, tau=self.tau)

        loss_uni = torch.tensor(0.0, device=device)
        if visible_masks is None:
            visible_masks = [
                torch.ones(z_uni_1[v].size(0), dtype=torch.bool, device=device)
                for v in range(len(z_uni_1))
            ]

        for v in range(len(z_uni_1)):
            valid = visible_masks[v]
            if valid.sum() <= 1:
                continue

            loss_uni = loss_uni + 0.5 * _info_nce(
                z_uni_1[v],
                z_fuse_2,
                tau=self.tau,
                valid=valid,
            )
            loss_uni = loss_uni + 0.5 * _info_nce(
                z_uni_2[v],
                z_fuse_1,
                tau=self.tau,
                valid=valid,
            )

        loss_mask = torch.tensor(0.0, device=device)
        if len(z_mask_1) > 0:
            for k in range(len(z_mask_1)):
                loss_mask = loss_mask + 0.5 * _info_nce(
                    z_mask_1[k],
                    z_fuse_1,
                    tau=self.tau,
                )
                loss_mask = loss_mask + 0.5 * _info_nce(
                    z_mask_2[k],
                    z_fuse_2,
                    tau=self.tau,
                )
            loss_mask = loss_mask / float(len(z_mask_1))

        loss_cons = _sym_kl(q_fuse_1, q_fuse_2)

        if q_mask_1 is not None and q_mask_2 is not None:
            loss_cons = loss_cons + _sym_kl(q_mask_1, q_mask_2)

        for v in range(len(q_uni_1)):
            loss_cons = loss_cons + _sym_kl(q_uni_1[v], q_uni_2[v])

        loss_total = (
            self.l_fuse * loss_fuse
            + self.l_uni * loss_uni
            + self.l_mask * loss_mask
            + self.l_cons * loss_cons
        )

        return {
            "loss_total": loss_total,
            "L_fuse": loss_fuse.detach(),
            "L_uni": loss_uni.detach(),
            "L_mask": loss_mask.detach(),
            "L_cons": loss_cons.detach(),
        }
