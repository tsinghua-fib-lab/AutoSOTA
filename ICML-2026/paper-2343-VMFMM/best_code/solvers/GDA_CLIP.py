import torch
import torch.nn.functional as F
import torch.nn as nn

def get_zero_shot_logits(query_features, query_labels, clip_prototypes):
    clip_logits = 100 * query_features @ clip_prototypes
    return clip_logits.squeeze()

def GDA_CLIP_solver(query_features, query_labels, clip_prototypes, alpha=5.0):
    query_labels = query_labels.cuda().float()
    clip_prototypes = clip_prototypes.cuda().float()
    query_features = query_features.cuda().float()

    # Zero-shot logits and probabilities
    zs_logits = get_zero_shot_logits(query_features, query_labels, clip_prototypes)
    y_hat = F.softmax(zs_logits, dim=1)  # [N, C]

    # Parameter Estimation (unsupervised using soft assignments y_hat)
    # mus: class means
    sum_z = y_hat.sum(dim=0, keepdim=True).clamp_min(1e-6)  # [1, C]
    mus = (y_hat.T @ query_features) / sum_z.T             # [C, D]

    # Global covariance and its pseudo-inverse (ridge for stability)
    # Use covariance over query features, following original spirit
    D = query_features.shape[-1]
    cov = query_features.T.cov()                           # [D, D]
    # KS-like stabilizer: add trace * I
    cov = cov + (cov.trace() / D) * torch.eye(D).cuda()

    cov_inv = torch.linalg.pinv(cov)                       # [D, D]

    # GDA classifier (W, b), keep notation from original code
    # W must be [D, C] to match (features @ W + b)
    W = cov_inv @ mus.T                                    # [D, C]

    # b = log ps - (mu^T Sigma^{-1} mu)/2
    # compute diagonal of mus @ cov_inv @ mus^T
    quad = (mus @ cov_inv) * mus                           # [C, D]
    quad = quad.sum(dim=1)                                 # [C]
    ps = torch.ones(mus.shape[0]).cuda() * (1.0 / mus.shape[0])  # uniform prior
    b = ps.log() - 0.5 * quad                              # [C]

    # Fuse zero-shot with GDA logits
    gda_logits = query_features @ W + b                    # [N, C]
    fused_logits = zs_logits + alpha * gda_logits          # [N, C]

    z = F.softmax(fused_logits, dim=1)

    return y_hat.cpu(), z.cpu()