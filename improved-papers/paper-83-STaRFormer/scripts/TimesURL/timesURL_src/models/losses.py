import torch
from torch import nn
import torch.nn.functional as F


def hierarchical_contrastive_loss(z1, z2, alpha=0.8, temporal_unit=0, temp=1.0):
    loss = torch.tensor(0., device=z1.device)
    d = 0

    while z1.size(1) > 1:

        if alpha != 0:
            if d == 0:
                loss += alpha * instance_contrastive_loss_mixup(z1, z2, temp)
            else:
                loss += alpha * instance_contrastive_loss_mixup(z1, z2, temp)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                if d == 0:
                    loss += (1 - alpha) * temporal_contrastive_loss_mixup(z1, z2, temp)
                else:
                    loss += (1 - alpha) * temporal_contrastive_loss_mixup(z1, z2, temp)
        d += 1

        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss_mixup(z1, z2, temp)
            d += 1
    return loss / d


def temporal_contrastive_loss_mixup(z1, z2, temp=1.0):
    B, T = z1.size(0), z1.size(1)
    alpha = 0.2
    beta = 0.2

    if T == 1:
        return z1.new_tensor(0.)

    uni_z1 = alpha * z1 + (1 - alpha) * z1[:, torch.randperm(z1.shape[1]), :].view(z1.size())
    uni_z2 = beta * z2 + (1 - beta) * z2[:, torch.randperm(z1.shape[1]), :].view(z2.size())

    z = torch.cat([z1, z2, uni_z1, uni_z2], dim=1)

    sim = torch.matmul(z[:, : 2 * T, :], z.transpose(1, 2)) / temp  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]

    if T > 1500:
        z, sim = z.cpu(), sim.cpu()
        torch.cuda.empty_cache()

    logits = -F.log_softmax(logits, dim=-1)

    logits = logits[:, :2 * T, :(2 * T - 1)]

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


def instance_contrastive_loss_mixup(z1, z2, temp=1.0):
    B, T = z1.size(0), z1.size(1)
    alpha = 0.2
    beta = 0.2

    if B == 1:
        return z1.new_tensor(0.)

    uni_z1 = alpha * z1 + (1 - alpha) * z1[torch.randperm(z1.shape[0]), :, :].view(z1.size())
    uni_z2 = beta * z2 + (1 - beta) * z2[torch.randperm(z2.shape[0]), :, :].view(z2.size())

    z = torch.cat([z1, z2, uni_z1, uni_z2], dim=0)
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z[:, : 2 * B, :], z.transpose(1, 2)) / temp  # T x 2B x 2B

    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B  x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    logits = logits[:, :2 * B, :(2 * B - 1)]

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss
