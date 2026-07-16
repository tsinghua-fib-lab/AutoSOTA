import torch
from typing import Optional, Literal
IGNORE_INDEX = -100


def get_loss_func(loss_type: str):
    if loss_type == "ranktuner":
        return ranktuner_loss_func
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")


@torch.no_grad()
def _direct_rank_scale(logits, shift_labels, gt_prob):
    gt_logits = logits[torch.arange(logits.size(0), device=logits.device), shift_labels]

    # Compute rank more efficiently - count how many logits are >= gt_logits
    gt_rank = (gt_logits.unsqueeze(1) <= logits).sum(dim=1).clamp_min(1)

    # Compute entropy more efficiently using logsumexp trick
    max_logits, max_indices = logits.max(dim=1, keepdim=True)
    log_probs = logits - max_logits - torch.logsumexp(logits - max_logits, dim=1, keepdim=True)
    max_probs = torch.exp(log_probs.gather(1, max_indices)).squeeze(1)

    # Entropy computation without storing full probs tensor
    eps = torch.finfo(log_probs.dtype).eps
    entropy_nats = -(torch.exp(log_probs) * (log_probs + eps)).sum(dim=1)
    position_entropy = entropy_nats / torch.log(
        torch.tensor(2.0, device=entropy_nats.device, dtype=entropy_nats.dtype)
    )

    expected_rank = torch.where(
        position_entropy >= 2.0,
        0.25
        * torch.pow(
            torch.tensor(2.0, device=position_entropy.device, dtype=position_entropy.dtype),
            position_entropy,
        )
        + 1.0,
        1 + (1 - max_probs)
    )
    xi = torch.maximum(expected_rank, gt_rank.float())
    
    xi = xi.clamp_min(1.0)
    K_xi = 1.0 / (torch.log2(xi + 1.0)) ** 2
    rev_coeff = -1.0
    scale = torch.pow((gt_prob * expected_rank).clamp_min(1e-6), rev_coeff * K_xi)
    scale = scale * gt_prob

    
    return scale
    


def ranktuner_loss_func(outputs, labels, num_items_in_batch=None):
    logits = outputs.get("logits")
    if logits is None:
        return outputs.get("loss", torch.tensor(0.0))


    logits = logits.float()
    vocab_size = logits.size(-1)

    # Align labels: pad then right-shift
    labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:].contiguous()

    batch_size = shift_labels.size(0)
    seq_len = shift_labels.size(1)

    logits = logits.view(-1, vocab_size)
    shift_labels_flat = shift_labels.view(-1).to(logits.device)

    ignore_index = IGNORE_INDEX if "IGNORE_INDEX" in globals() else -100

    # Filter valid tokens
    valid_mask = shift_labels_flat != ignore_index
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    per_token_ce_flat = torch.nn.functional.cross_entropy(
        logits, shift_labels_flat, ignore_index=ignore_index, reduction="none"
    )

    logits = logits[valid_mask]
    shift_labels = shift_labels_flat[valid_mask]
    per_token_ce = per_token_ce_flat[valid_mask]
    gt_prob = torch.exp(-per_token_ce)
    with torch.no_grad():
        scale = _direct_rank_scale(
            logits,
            shift_labels,
            gt_prob,
        )

    # Apply scale to losses
    weighted_losses = per_token_ce * scale

    # Reduce to final loss
    if num_items_in_batch is not None:
        total_loss = weighted_losses.sum()
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(total_loss.device)
        loss = total_loss / num_items_in_batch
    else:
        loss = weighted_losses.mean()


    return loss
