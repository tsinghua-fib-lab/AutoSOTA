# core/ekfac_fit.py

import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
from tqdm import tqdm

from core.ekfac_blocks import EKFAC_QK_Head, EKFAC_QKVO_Head
from model.hooks import QKActivationCache, QKVOActivationCache, setup_qk_hooks, setup_qkvo_hooks


def compute_pseudo_labels(logits: torch.Tensor) -> torch.Tensor:
    V = logits.shape[-1]
    probs_flat = torch.softmax(logits.reshape(-1, V).float(), dim=-1)
    sampled = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
    return sampled


def stage1A_accumulate_AS(
    model,
    dataloader,
    ekfac,
    layer_idx: int,
    head_idx: int,
    seq_length: int,
    d_model: int,
    d_head: int,
    dtype,
    device,
    empty_cache_every: int = 100,
    verbose: bool = False
):
    if isinstance(ekfac, EKFAC_QK_Head):
        cache = QKActivationCache()
        hooks = setup_qk_hooks(model, layer_idx, cache, verbose=verbose)
    elif isinstance(ekfac, EKFAC_QKVO_Head):
        cache = QKVOActivationCache()
        hooks = setup_qkvo_hooks(model, layer_idx, cache, verbose=verbose)
    else:
        raise TypeError("Unsupported EKFAC type")

    start_time = time.time()
    for batch_idx, (batch_tokens, _) in enumerate(tqdm(dataloader, desc="Accum A/S", disable=not verbose)):
        cache.clear()
        input_ids = batch_tokens[:, :seq_length].to(device, non_blocking=True)

        with torch.enable_grad():
            with autocast(dtype=dtype):
                logits = model(input_ids)
            sampled_labels = compute_pseudo_labels(logits)
            Vocab = logits.shape[-1]
            loss = F.cross_entropy(logits.reshape(-1, Vocab).float(), sampled_labels, reduction="sum")

            if isinstance(ekfac, EKFAC_QK_Head):
                if cache.Q is None or cache.K is None or cache.X is None:
                    raise RuntimeError("X/Q/K activations not captured.")
                grads_Q, grads_K = torch.autograd.grad(
                    outputs=loss,
                    inputs=[cache.Q, cache.K],
                    retain_graph=False, create_graph=False, allow_unused=False
                )
                dQ = grads_Q[:, :, head_idx, :].reshape(-1, d_head).float()
                dK = grads_K[:, :, head_idx, :].reshape(-1, d_head).float()
                X_flat = cache.X.reshape(-1, d_model).float()
                ekfac.accumulate_A_S(X_flat, dQ, dK)

            else:
                if (cache.Q is None or cache.K is None or cache.V is None or
                        cache.Z is None or cache.result is None or cache.X is None):
                    raise RuntimeError("X/Q/K/V/Z/result activations not captured.")
                grads_Q, grads_K, grads_V, grads_R = torch.autograd.grad(
                    outputs=loss,
                    inputs=[cache.Q, cache.K, cache.V, cache.result],
                    retain_graph=False, create_graph=False, allow_unused=False
                )
                dQ = grads_Q[:, :, head_idx, :].reshape(-1, d_head).float()
                dK = grads_K[:, :, head_idx, :].reshape(-1, d_head).float()
                dV = grads_V[:, :, head_idx, :].reshape(-1, d_head).float()
                dR = grads_R.reshape(-1, d_model).float()
                X_flat = cache.X.reshape(-1, d_model).float()
                Z_flat = cache.Z[:, :, head_idx, :].reshape(-1, d_head).float()
                ekfac.accumulate_A_S(X_flat, dQ, dK, dV, Z_flat, dR)

        cache.clear()
        if device.type == "cuda" and ((batch_idx + 1) % empty_cache_every == 0):
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    ekfac.finalize_eigendecomposition(device=device)

    try:
        model.reset_hooks(hooks)
    except Exception:
        try:
            model.remove_all_hook_fns()
        except Exception:
            pass

    return elapsed


def stage1B_fit_lambda(
    model,
    dataloader,
    ekfac,
    layer_idx: int,
    head_idx: int,
    seq_length: int,
    d_model: int,
    d_head: int,
    dtype,
    device,
    empty_cache_every: int = 100,
    verbose: bool = False
):
    if isinstance(ekfac, EKFAC_QK_Head):
        cache = QKActivationCache()
        hooks = setup_qk_hooks(model, layer_idx, cache, verbose=False)
        lambda_sum = torch.zeros(ekfac.block['d_in'], ekfac.block['d_out'], device=device, dtype=torch.float32)
        weight_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
    elif isinstance(ekfac, EKFAC_QKVO_Head):
        cache = QKVOActivationCache()
        hooks = setup_qkvo_hooks(model, layer_idx, cache, verbose=False)
        lambda_sum = [
            torch.zeros(ekfac.blocks[0]['d_in'], ekfac.blocks[0]['d_out'], device=device, dtype=torch.float32),
            torch.zeros(ekfac.blocks[1]['d_in'], ekfac.blocks[1]['d_out'], device=device, dtype=torch.float32),
            torch.zeros(ekfac.blocks[2]['d_in'], ekfac.blocks[2]['d_out'], device=device, dtype=torch.float32),
        ]
        weight_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
    else:
        raise TypeError("Unsupported EKFAC type")

    start_time = time.time()
    for batch_idx, (batch_tokens, _) in enumerate(tqdm(dataloader, desc="Fit Lambda", disable=not verbose)):
        cache.clear()
        input_ids = batch_tokens[:, :seq_length].to(device, non_blocking=True)
        B = int(input_ids.shape[0])

        with torch.enable_grad():
            with autocast(dtype=dtype):
                logits = model(input_ids)
            sampled_labels = compute_pseudo_labels(logits)
            Vocab = logits.shape[-1]
            loss = F.cross_entropy(logits.reshape(-1, Vocab).float(), sampled_labels, reduction="sum")

            if isinstance(ekfac, EKFAC_QK_Head):
                if cache.Q is None or cache.K is None or cache.X is None:
                    raise RuntimeError("X/Q/K activations not captured in Lambda pass.")
                grads_Q, grads_K = torch.autograd.grad(
                    outputs=loss,
                    inputs=[cache.Q, cache.K],
                    retain_graph=False, create_graph=False, allow_unused=False
                )
                dQ = grads_Q[:, :, head_idx, :].reshape(-1, d_head).float().detach()
                dK = grads_K[:, :, head_idx, :].reshape(-1, d_head).float().detach()
                X_flat = cache.X.reshape(-1, d_model).float().detach()

                dW0 = X_flat.t() @ torch.cat([dQ, dK], dim=-1)
                ge0 = ekfac.Q_A.t() @ dW0 @ ekfac.Q_S
                lambda_sum.add_(ge0.pow(2))
                weight_sum += float(B)

            else:
                if (cache.Q is None or cache.K is None or cache.V is None or
                        cache.Z is None or cache.result is None or cache.X is None):
                    raise RuntimeError("X/Q/K/V/Z/result activations not captured in Lambda pass.")
                grads_Q, grads_K, grads_V, grads_R = torch.autograd.grad(
                    outputs=loss,
                    inputs=[cache.Q, cache.K, cache.V, cache.result],
                    retain_graph=False, create_graph=False, allow_unused=False
                )
                dQ = grads_Q[:, :, head_idx, :].reshape(-1, d_head).float().detach()
                dK = grads_K[:, :, head_idx, :].reshape(-1, d_head).float().detach()
                dV = grads_V[:, :, head_idx, :].reshape(-1, d_head).float().detach()
                dR = grads_R.reshape(-1, d_model).float().detach()
                X_flat = cache.X.reshape(-1, d_model).float().detach()
                Z_flat = cache.Z[:, :, head_idx, :].reshape(-1, d_head).float().detach()

                dW0 = X_flat.t() @ torch.cat([dQ, dK], dim=-1)
                dW1 = X_flat.t() @ dV
                dW2 = Z_flat.t() @ dR

                ge0 = ekfac.Q_A[0].t() @ dW0 @ ekfac.Q_S[0]
                ge1 = ekfac.Q_A[1].t() @ dW1 @ ekfac.Q_S[1]
                ge2 = ekfac.Q_A[2].t() @ dW2 @ ekfac.Q_S[2]

                lambda_sum[0].add_(ge0.pow(2))
                lambda_sum[1].add_(ge1.pow(2))
                lambda_sum[2].add_(ge2.pow(2))
                weight_sum += float(B)

        cache.clear()
        if device.type == "cuda" and ((batch_idx + 1) % empty_cache_every == 0):
            torch.cuda.empty_cache()

    if dist.is_initialized():
        if isinstance(ekfac, EKFAC_QK_Head):
            dist.all_reduce(lambda_sum, op=dist.ReduceOp.SUM)
        else:
            for i in range(3):
                dist.all_reduce(lambda_sum[i], op=dist.ReduceOp.SUM)
        dist.all_reduce(weight_sum, op=dist.ReduceOp.SUM)

    w_total = max(1.0, weight_sum.item())
    if isinstance(ekfac, EKFAC_QK_Head):
        ekfac.Lambda = (lambda_sum / w_total).flatten()
    else:
        ekfac.Lambda[0] = (lambda_sum[0] / w_total).flatten()
        ekfac.Lambda[1] = (lambda_sum[1] / w_total).flatten()
        ekfac.Lambda[2] = (lambda_sum[2] / w_total).flatten()

    elapsed = time.time() - start_time

    try:
        model.reset_hooks(hooks)
    except Exception:
        try:
            model.remove_all_hook_fns()
        except Exception:
            pass

    return elapsed

