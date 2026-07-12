import torch
import torch.nn as nn
import numpy as np

from utils.util import to_np

class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        elif self.distance_type == 'l2':
            return torch.square(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        elif self.distance_type == 'l3':
            return torch.pow(torch.abs(labels[:, None, :] - labels[None, :, :]), 3).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='L2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'L2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        elif self.similarity_type == 'cosine':
            # return (features)
            return
        else:
            raise ValueError(self.similarity_type)

class ConOrdLoss(nn.Module):

    def __init__(self, label_diff='l1', feature_sim='cosine', temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ConOrdLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)
        self.similarity_type = feature_sim

    def forward(self, features, labels=None, cfg=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = cfg.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
            weight = self.label_diff_fn(labels).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        if self.similarity_type == 'cosine':
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature)
        elif self.similarity_type == 'L2':
            anchor_dot_contrast = self.feature_sim_fn(anchor_feature).div(self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # original ver
 
        numerator_weight = 1 / (weight + cfg.epsilon)
        numerator = (numerator_weight * torch.exp(logits) * logits_mask).sum(1, keepdim=True)
        denominator = (weight * exp_logits).sum(1, keepdim=True)

        log_prob = torch.log(numerator / denominator)
        log_prob_sum = log_prob.sum(1)
        loss = - (self.temperature / self.base_temperature) * log_prob_sum
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def compute_center_loss(embs, rank_labels, fdc_points, cfg, record=False):
    fdc_points = nn.functional.normalize(fdc_points, dim=-1)

    # Support both integer and float rank labels (e.g. float MOS values)
    if isinstance(rank_labels, torch.Tensor):
        ranks_np = rank_labels.detach().cpu().numpy().astype(np.float64)
    else:
        ranks_np = np.asarray(rank_labels, dtype=np.float64)

    # Fiducial point ranks span [rank_min, rank_max] in the same scale as rank_labels
    rank_min = getattr(cfg, 'rank_min', 0.0)
    rank_max = getattr(cfg, 'rank_max', float(cfg.n_ranks - 1))
    fdc_point_ranks = np.linspace(rank_min, rank_max, cfg.fiducial_point_num)

    def get_pos_neg_idxs(ranks, fdc_ranks, cfg):
        # Always find the nearest fiducial point — works for both int and float ranks
        nn_idxs, margins, emb_idxs = [], [], []
        for emb_idx, r in enumerate(ranks):
            abs_diff = np.abs(fdc_ranks - r)
            min_val = abs_diff.min()
            nn = np.argwhere(abs_diff == min_val).flatten()
            nn_idxs.append(nn)
            margin_val = min_val * cfg.margin / max(float(cfg.tau), 1.0)
            margins.extend([margin_val] * len(nn))
            emb_idxs.extend([emb_idx] * len(nn))
        return np.concatenate(nn_idxs), np.array(emb_idxs), np.array(margins)

    nn_idxs, emb_idxs, margins = get_pos_neg_idxs(ranks_np, fdc_point_ranks, cfg)

    if cfg.metric == 'L2':
        dists = torch.cdist(fdc_points, embs)
    elif cfg.metric == 'cosine':
        dists = 1 - torch.matmul(fdc_points, embs.transpose(1, 0))

    loss = dists[nn_idxs, emb_idxs]

    if record:
        return torch.sum(loss) / (torch.sum(loss > 0) + 1e-7), to_np(loss)
    return torch.sum(loss) / embs.shape[0]

