from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _sequence_cross_entropy_bits(
    logits: torch.Tensor,  # [batch, sequence, vocabulary]
    labels: torch.Tensor,  # [batch, sequence]
    mask: torch.Tensor,  # [batch, sequence]
    *,
    align_offset: int = 0,
) -> float:
    """Total masked next-token cross-entropy, in bits."""
    aligned_logits = logits[:, align_offset:, :]
    shifted_logits = aligned_logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    shifted_mask = mask[:, 1:].contiguous()

    flat_logits = shifted_logits.view(-1, shifted_logits.size(-1))
    flat_labels = shifted_labels.view(-1)
    flat_mask = shifted_mask.view(-1).bool()

    if flat_mask.sum() == 0:
        return 0.0

    ce_sum = F.cross_entropy(
        flat_logits[flat_mask],
        flat_labels[flat_mask],
        reduction="sum",
    )
    return ce_sum.item() / math.log(2)


@torch.no_grad()
def compute_information_gain(
    *,
    model: nn.Module,
    input_ids: torch.Tensor,  # [batch, sequence] -- the continuation (loss target)
    attention_mask: torch.Tensor,  # [batch, sequence]
    token_embeddings: torch.Tensor,  # [batch, sequence, hidden]
    compression_token_embeddings: torch.Tensor,  # [batch, compression, hidden]
    compression_attention_mask: torch.Tensor,  # [batch, compression]
    prefix_token_embeddings: torch.Tensor | None = None,  # [batch, prefix, hidden]
    prefix_attention_mask: torch.Tensor | None = None,  # [batch, prefix]
) -> list[float]:
    """Per-sample Information Gain in bits (paper Eq. 9).

    Measures how many bits the compression token saves on the continuation:
    ``H_lm(continuation | prefix) - H_comp_lm(continuation | mem, prefix)``. The continuation is the
    sequence described by ``input_ids``/``token_embeddings``. When ``prefix_token_embeddings`` is
    given, the real (uncompressed) prefix is prepended to BOTH forwards so the gain is measured on
    top of the context the model already has; the prefix positions are never scored. With no prefix
    this reduces to the original ``H_lm(seq) - H_comp_lm(seq | mem)``.
    """
    batch_size = input_ids.size(0)
    num_compression_tokens = compression_token_embeddings.size(1)
    prefix_len = prefix_token_embeddings.size(1) if prefix_token_embeddings is not None else 0
    information_gains: list[float] = []

    for j in range(batch_size):
        sample_input_ids = input_ids[j : j + 1]  # [1, sequence] (continuation, used as labels)
        sample_attention_mask = attention_mask[j : j + 1]  # [1, sequence]
        sample_token_embeddings = token_embeddings[j : j + 1]  # [1, sequence, hidden]
        sample_compression_token_embeddings = (  # [1, compression, hidden]
            compression_token_embeddings[j : j + 1].to(sample_token_embeddings.device).to(sample_token_embeddings.dtype)
        )
        sample_compression_attention_mask = compression_attention_mask[j : j + 1]  # [1, compression]

        sample_prefix_embeddings = None
        sample_prefix_attention_mask = None
        if prefix_len > 0:
            sample_prefix_embeddings = (
                prefix_token_embeddings[j : j + 1].to(sample_token_embeddings.device).to(sample_token_embeddings.dtype)
            )
            sample_prefix_attention_mask = prefix_attention_mask[j : j + 1]

        # H_lm: continuation cross-entropy WITHOUT the compression token (the real prefix, if any, is
        # still in context). align_offset positions the scored window at the start of the continuation.
        if prefix_len > 0:
            base_token_embeddings = torch.cat((sample_prefix_embeddings, sample_token_embeddings), dim=1)
            base_attention_mask = torch.cat((sample_prefix_attention_mask, sample_attention_mask), dim=1)
            base_outputs = model(inputs_embeds=base_token_embeddings, attention_mask=base_attention_mask)
        else:
            base_outputs = model(input_ids=sample_input_ids, attention_mask=sample_attention_mask)
        h_lm_bits = _sequence_cross_entropy_bits(
            base_outputs.logits, sample_input_ids, sample_attention_mask, align_offset=prefix_len
        )

        # H_comp_lm: continuation cross-entropy WITH the compression token (+ optional prefix).
        embedding_blocks = [sample_compression_token_embeddings]
        mask_blocks = [sample_compression_attention_mask]
        if prefix_len > 0:
            embedding_blocks.append(sample_prefix_embeddings)
            mask_blocks.append(sample_prefix_attention_mask)
        embedding_blocks.append(sample_token_embeddings)
        mask_blocks.append(sample_attention_mask)
        united_token_embeddings = torch.cat(embedding_blocks, dim=1)  # [1, compression + prefix + sequence, hidden]
        united_attention_mask = torch.cat(mask_blocks, dim=1)

        outputs = model(inputs_embeds=united_token_embeddings, attention_mask=united_attention_mask)
        h_comp_lm_bits = _sequence_cross_entropy_bits(
            outputs.logits,
            sample_input_ids,
            sample_attention_mask,
            align_offset=num_compression_tokens + prefix_len,
        )

        information_gains.append(h_lm_bits - h_comp_lm_bits)

    return information_gains


@torch.no_grad()
def compute_prefix_surprisal_bits_per_token(
    *,
    model: nn.Module,
    prefix_input_ids: torch.Tensor,  # [batch, prefix]
    prefix_attention_mask: torch.Tensor,  # [batch, prefix]
) -> list[float]:
    """Per-sample base-model surprisal over the (uncompressed) prefix tokens, in bits per token.

    Runs the plain LM over the prefix tokens p_1..p_P (no compression token) and returns the mean
    next-token cross-entropy in bits, normalized by the number of scored prefix positions so the
    value is comparable across prefix lengths. Returns 0.0 for a sample with no scorable prefix.
    """
    batch_size = prefix_input_ids.size(0)
    bits_per_token: list[float] = []
    for j in range(batch_size):
        sample_ids = prefix_input_ids[j : j + 1]
        sample_mask = prefix_attention_mask[j : j + 1]
        scored = int(sample_mask[:, 1:].sum().item())
        if scored == 0:
            bits_per_token.append(0.0)
            continue
        outputs = model(input_ids=sample_ids, attention_mask=sample_mask)
        total_bits = _sequence_cross_entropy_bits(outputs.logits, sample_ids, sample_mask, align_offset=0)
        bits_per_token.append(total_bits / scored)
    return bits_per_token
