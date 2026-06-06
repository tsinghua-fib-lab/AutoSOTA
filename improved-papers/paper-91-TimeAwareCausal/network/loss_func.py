import torch
import torch.nn.functional as F
from .types_ import *
import numpy as np
from torch.distributions import Normal, Independent


def kl_divergence(latent_space_a, latent_space_b):
    return torch.mean(torch.distributions.kl_divergence(latent_space_a, latent_space_b))


def temporal_smooth_loss(latent_variables: Tensor, batch_first=True):
    if batch_first:
        return F.l1_loss(latent_variables[:, 1:, :], latent_variables[:, :-1, :], reduction='mean')
    else:
        return F.l1_loss(latent_variables[1:, :, :], latent_variables[-1:, :, :], reduction='mean')

def calculate_mi_loss(dy_latent_variables, st_latent_variables):
    '''
    input features shape: [BxT, D]
    '''
    dy_latent_variables = [torch.transpose(i, 0, 1) for i in dy_latent_variables] # [B, T, D] -> [T, B, D]
    domains, batch_size = dy_latent_variables[0].shape[:2]
    st_latent_variables = [torch.transpose(i.view(batch_size, domains, -1), 0, 1)
                           for i in st_latent_variables]  # [BxT, D] -> [T, B, D]

    dynamic_mu, dynamic_sigma = dy_latent_variables
    static_mu, static_sigma = st_latent_variables

    z_dy1 = Normal(loc=dynamic_mu.unsqueeze(1), scale=dynamic_sigma.unsqueeze(1))
    z_dy2 = Normal(loc=dynamic_mu.unsqueeze(2), scale=dynamic_sigma.unsqueeze(2))
    log_q_dy = z_dy1.log_prob(z_dy2.rsample()).sum(-1)

    z_st1 = Normal(loc=static_mu.unsqueeze(1), scale=static_sigma.unsqueeze(1))
    z_st2 = Normal(loc=static_mu.unsqueeze(2), scale=static_sigma.unsqueeze(2))
    log_q_st = z_st1.log_prob(z_st2.rsample()).sum(-1)

    H_dy = log_q_dy.logsumexp(2).mean(1) - np.log(log_q_dy.shape[2])
    H_st = log_q_st.logsumexp(2).mean(1) - np.log(log_q_st.shape[2])
    H_dy_st = (log_q_dy + log_q_st).logsumexp(2).mean(1) - np.log(log_q_st.shape[2])

    mi_loss = - (H_dy.sum() + H_st.sum() - H_dy_st.sum())

    return mi_loss


def cross_domain_contrastive_loss(static_features, labels, domains, batch_size, temperature=0.07):
    '''
    static features shape: [BxT, D]
    labels shape: [BxD]
    '''
    # to [T, B, D]
    static_features = torch.transpose(static_features.view(batch_size, domains, -1), 0, 1)

    device = (torch.device('cuda')
              if static_features.is_cuda
              else torch.device('cpu'))

    labels = torch.transpose(labels.contiguous().view(-1, domains, 1), 0, 1)
    loss = 0
    for t in range(domains - 1):
        features_t, features_t1 = static_features[t], static_features[t+1]
        labels_t, labels_t1 = labels[t], labels[t+1]
        mask = torch.eq(labels_t, labels_t1.T).float().to(device)
        anchor_dot_contrast = torch.div(torch.matmul(features_t, features_t1.T), temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(features_t.shape[0]).view(-1, 1).to(device),
            0
        )
        pos_mask = mask * logits_mask
        neg_mask = torch.ones_like(mask).to(device) - mask

        exp_logits = torch.exp(logits)
        pos_exp_logits = exp_logits * pos_mask
        neg_exp_logits = exp_logits * neg_mask

        exp_logits_sum = pos_exp_logits + neg_exp_logits.sum(1, keepdim=True)
        log_prob = (logits - torch.log(exp_logits_sum + 1e-6)) * pos_mask

        mask_pos_pairs = pos_mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = log_prob.sum(1) / mask_pos_pairs

        loss += - mean_log_prob_pos.mean()

    return loss


def inter_domain_contrastive_loss(dynamic_features, static_features, labels, domains, batch_size, temperature=0.07):
    '''
    dynamic features shape: [BxT, D]
    static features shape: [BxT, D]
    labels shape: [BxD]
    '''
    # to [T, B, D]
    dynamic_features = torch.transpose(dynamic_features.view(batch_size, domains, -1), 0, 1)
    static_features = torch.transpose(static_features.view(batch_size, domains, -1), 0, 1)

    device = (torch.device('cuda')
              if dynamic_features.is_cuda
              else torch.device('cpu'))

    labels = torch.transpose(labels.contiguous().view(-1, domains, 1), 0, 1)
    if labels.shape[1] != batch_size:
        raise ValueError('Num of labels does not match num of features')
    mask = torch.eq(labels, torch.transpose(labels, 1, 2)).float().to(device) # shape: [T, B, B]

    # compute logits
    contrast_features = dynamic_features
    anchor_features = static_features
    contrast_dot_anchor = torch.div(
        torch.matmul(contrast_features, torch.transpose(anchor_features, 1, 2)),
        temperature)

    # for numerical stability
    logits_max, _ = torch.max(contrast_dot_anchor, dim=2, keepdim=True)
    logits = contrast_dot_anchor - logits_max.detach()

    logits_mask = torch.scatter(
        torch.ones_like(mask),
        2,
        torch.arange(batch_size).view(-1, 1).repeat(domains, 1, 1).to(device),
        0
    )
    pos_mask = mask * logits_mask
    neg_mask = torch.ones_like(mask).to(device) - mask

    exp_logits = torch.exp(logits)
    pos_exp_logits = exp_logits * pos_mask
    neg_exp_logits = exp_logits * neg_mask

    exp_logits_sum = pos_exp_logits + neg_exp_logits.sum(2, keepdim=True)
    log_prob = (logits - torch.log(exp_logits_sum + 1e-6)) * pos_mask

    mask_pos_pairs = pos_mask.sum(2)
    mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
    mean_log_prob_pos = log_prob.sum(2) / mask_pos_pairs  # shape: [T, B]

    loss = - mean_log_prob_pos.mean(1).sum()

    return loss