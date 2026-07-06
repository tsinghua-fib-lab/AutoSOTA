# core/influence_phase2.py

import time
import heapq
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm


def compute_sample_grads_qkonly(model, npy_batch, layer_idx, head_idx, seq_length, dtype, device):
    """
    Compute per-sample gradient for QK-only.
    Returns g0 (d_model x 2*d_head) and loss scalar.
    """
    input_ids = npy_batch[:, :seq_length].to(device, non_blocking=True)
    labels = npy_batch[:, 1:seq_length + 1].to(device, non_blocking=True)

    with torch.enable_grad():
        with autocast(dtype=dtype):
            logits = model(input_ids)

        Vocab = logits.shape[-1]
        loss = F.cross_entropy(
            logits.reshape(-1, Vocab).float(),
            labels.reshape(-1),
            reduction="sum"
        )

        attn = model.blocks[layer_idx].attn
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=[attn.W_Q, attn.W_K],
            retain_graph=False, create_graph=False, allow_unused=False
        )
        gQ = grads[0][head_idx].float()
        gK = grads[1][head_idx].float()
        g0 = torch.cat([gQ, gK], dim=-1)

    return g0, float(loss.item())


def compute_sample_grads_qkvo(model, npy_batch, layer_idx, head_idx, seq_length, dtype, device):
    """
    Compute per-sample gradient for QKVO.
    Returns g0, g1, g2 and loss scalar.
    """
    input_ids = npy_batch[:, :seq_length].to(device, non_blocking=True)
    labels = npy_batch[:, 1:seq_length + 1].to(device, non_blocking=True)

    with torch.enable_grad():
        with autocast(dtype=dtype):
            logits = model(input_ids)

        Vocab = logits.shape[-1]
        loss = F.cross_entropy(
            logits.reshape(-1, Vocab).float(),
            labels.reshape(-1),
            reduction="sum"
        )

        attn = model.blocks[layer_idx].attn
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=[attn.W_Q, attn.W_K, attn.W_V, attn.W_O],
            retain_graph=False, create_graph=False, allow_unused=False
        )
        gQ = grads[0][head_idx].float()
        gK = grads[1][head_idx].float()
        gV = grads[2][head_idx].float()
        gO = grads[3][head_idx].float()

        g0 = torch.cat([gQ, gK], dim=-1)
        g1 = gV
        g2 = gO

    return g0, g1, g2, float(loss.item())


def phase2_score_qkonly(
    model,
    dataloader,
    p_qk,
    layer_idx,
    head_idx,
    seq_length,
    dtype,
    device,
    top_k,
    empty_cache_every=500,
    verbose=False
):
    """
    Phase2 scoring for QK-only.
    Returns (pos_heap, neg_heap, processed_samples, elapsed_time).
    """
    local_pos_heap = []
    local_neg_heap = []
    heap_cap = top_k
    uid_counter = 0
    processed_samples = 0

    start_time = time.time()
    for _, (batch_tokens, batch_indices) in enumerate(tqdm(dataloader, desc="Scoring", disable=not verbose)):
        processed_samples += 1

        g0, loss_val = compute_sample_grads_qkonly(
            model, batch_tokens, layer_idx, head_idx, seq_length, dtype, device
        )
        s0 = torch.sum(g0 * p_qk).item()
        projection_score = -s0

        payload = (int(batch_indices.item()), float(loss_val))

        entry_pos = (projection_score, uid_counter, payload)
        if len(local_pos_heap) < heap_cap:
            heapq.heappush(local_pos_heap, entry_pos)
        else:
            if projection_score > local_pos_heap[0][0]:
                heapq.heapreplace(local_pos_heap, entry_pos)

        neg_score = -projection_score
        entry_neg = (neg_score, uid_counter, payload)
        if len(local_neg_heap) < heap_cap:
            heapq.heappush(local_neg_heap, entry_neg)
        else:
            if neg_score > local_neg_heap[0][0]:
                heapq.heapreplace(local_neg_heap, entry_neg)

        uid_counter += 1

        if device.type == "cuda" and (processed_samples % empty_cache_every == 0):
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    return local_pos_heap, local_neg_heap, processed_samples, elapsed


def phase2_score_qkvo(
    model,
    dataloader,
    p_qk,
    p_v,
    p_o,
    layer_idx,
    head_idx,
    seq_length,
    dtype,
    device,
    top_k,
    empty_cache_every=500,
    verbose=False
):
    """
    Phase2 scoring for QKVO.
    Returns (pos_heap, neg_heap, processed_samples, elapsed_time).
    """
    local_pos_heap = []
    local_neg_heap = []
    heap_cap = top_k
    uid_counter = 0
    processed_samples = 0

    start_time = time.time()
    for _, (batch_tokens, batch_indices) in enumerate(tqdm(dataloader, desc="Scoring", disable=not verbose)):
        processed_samples += 1

        g0, g1, g2, loss_val = compute_sample_grads_qkvo(
            model, batch_tokens, layer_idx, head_idx, seq_length, dtype, device
        )
        s0 = torch.sum(g0 * p_qk).item()
        s1 = torch.sum(g1 * p_v).item()
        s2 = torch.sum(g2 * p_o).item()
        projection_score = -(s0 + s1 + s2)

        payload = (int(batch_indices.item()), float(loss_val))

        entry_pos = (projection_score, uid_counter, payload)
        if len(local_pos_heap) < heap_cap:
            heapq.heappush(local_pos_heap, entry_pos)
        else:
            if projection_score > local_pos_heap[0][0]:
                heapq.heapreplace(local_pos_heap, entry_pos)

        neg_score = -projection_score
        entry_neg = (neg_score, uid_counter, payload)
        if len(local_neg_heap) < heap_cap:
            heapq.heappush(local_neg_heap, entry_neg)
        else:
            if neg_score > local_neg_heap[0][0]:
                heapq.heapreplace(local_neg_heap, entry_neg)

        uid_counter += 1

        if device.type == "cuda" and (processed_samples % empty_cache_every == 0):
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    return local_pos_heap, local_neg_heap, processed_samples, elapsed
