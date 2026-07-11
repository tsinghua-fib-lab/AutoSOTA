# nc_minibatch/mb_eval.py
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def _infer_on_loader(
    model,
    loader,
    *,
    device: torch.device,
    criterion,
    forward_kwargs: Optional[dict] = None,
    return_logits: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Run inference over a NeighborLoader.

    Returns:
      node_ids_cpu: LongTensor [M]   (global node ids for all seed nodes encountered)
      logits_cpu:   FloatTensor [M,C] (seed logits in the same order as node_ids_cpu) if return_logits else empty
      avg_loss:     float (average over seed nodes; 0.0 if criterion is None)

    Notes:
      - We ONLY evaluate on seed nodes (first batch.batch_size nodes in batch.n_id).
      - Neighbor nodes are present only to support message passing and are not scored directly.
    """
    model_was_training = model.training
    model.eval()

    fw = {} if forward_kwargs is None else dict(forward_kwargs)

    all_ids = []
    all_logits = []

    loss_sum = 0.0
    count = 0

    for batch in loader:
        batch = batch.to(device)

        out = model(
            batch.x,
            batch.edge_index,
            node_ids=getattr(batch, "n_id", None),
            **fw,
        )
        seed_n = int(batch.batch_size)

        # Seed node ids in global indexing
        seed_ids = batch.n_id[:seed_n].to(torch.device("cpu"), dtype=torch.long)
        all_ids.append(seed_ids)

        if return_logits:
            all_logits.append(out[:seed_n].detach().to(torch.device("cpu")))

        if criterion is not None:
            y = batch.y[:seed_n]
            if y.dim() == 2 and y.size(1) == 1:
                y = y.squeeze(1)
            y = y.view(-1).to(torch.long)

            loss = criterion(out[:seed_n], y)
            loss_sum += float(loss.item()) * seed_n
            count += seed_n

    node_ids_cpu = torch.cat(all_ids, dim=0) if len(all_ids) else torch.empty((0,), dtype=torch.long)

    if return_logits:
        logits_cpu = torch.cat(all_logits, dim=0) if len(all_logits) else torch.empty((0, 0), dtype=torch.float32)
    else:
        logits_cpu = torch.empty((0, 0), dtype=torch.float32)

    avg_loss = (loss_sum / max(count, 1)) if (criterion is not None) else 0.0

    # restore mode
    if model_was_training:
        model.train()

    return node_ids_cpu, logits_cpu, avg_loss


@torch.no_grad()
def _score_split_from_loader(
    model,
    data,
    loader,
    split_idx: torch.Tensor,
    *,
    device: torch.device,
    eval_func: Callable[[torch.Tensor, torch.Tensor], float],
    criterion,
    forward_kwargs: Optional[dict] = None,
) -> Tuple[float, float]:
    """
    Computes (metric, loss) for a split defined by split_idx using the corresponding loader.

    We gather seed logits + seed ids from the loader, then compute the metric on exactly split_idx.
    """
    # Collect predictions for all seed nodes in this loader epoch
    ids_cpu, logits_cpu, avg_loss = _infer_on_loader(
        model,
        loader,
        device=device,
        criterion=criterion,
        forward_kwargs=forward_kwargs,
        return_logits=True,
    )

    # Build an aligned logits tensor on CPU for the nodes in split_idx
    split_idx_cpu = split_idx.to(torch.device("cpu"), dtype=torch.long)

    if split_idx_cpu.numel() == 0:
        return 0.0, avg_loss

    # We only need logits for the nodes in split_idx; create a mapping via scatter.
    # Approach: allocate logits for split nodes only.
    # We will fill by matching ids_cpu -> position in split_idx via a hash map (python dict).
    # This is fast enough for these dataset sizes and avoids allocating [N,C] on CPU.
    pos = {int(nid): i for i, nid in enumerate(split_idx_cpu.tolist())}

    C = int(logits_cpu.size(1))
    # Initialize defensively: if any node is missing from the collected seed predictions,
    # its logits stay very negative and won't win argmax.
    split_logits = torch.full((split_idx_cpu.numel(), C), -1e9, dtype=logits_cpu.dtype)

    filled = 0
    for nid, logit in zip(ids_cpu.tolist(), logits_cpu):
        j = pos.get(int(nid), None)
        if j is not None:
            split_logits[j] = logit
            filled += 1


    # Safety: if something went wrong and some nodes were not filled, set them to -inf so argmax is stable.
    if filled < split_idx_cpu.numel():
        missing = split_idx_cpu.numel() - filled
        # Fill any uninitialized rows defensively
        # (rare; can happen if a loader was built with different input_nodes than split_idx)
        mask = torch.isnan(split_logits).any(dim=1) if split_logits.numel() > 0 else None
        if mask is not None and mask.any():
            split_logits[mask] = -1e9

    y = data.y
    if y.dim() == 2 and y.size(1) == 1:
        y = y.squeeze(1)
    y = y.view(-1).to(torch.long).to(torch.device("cpu"))

    try:
        metric = float(eval_func(y[split_idx_cpu], split_logits))
    except Exception:
        metric = float(eval_func(split_logits, y[split_idx_cpu]))
    return metric, float(avg_loss)


@torch.no_grad()
def evaluate_minibatch(
    model,
    data,
    loaders: Dict[str, object],
    split_idx: Dict[str, torch.Tensor],
    eval_func: Callable[[torch.Tensor, torch.Tensor], float],
    criterion,
    *,
    device: torch.device,
    forward_kwargs: Optional[dict] = None,
) -> Tuple[float, float, float, float, Optional[torch.Tensor]]:
    """
    Mini-batch analogue of full_batch evaluate(...).

    Returns:
      train_metric, valid_metric, test_metric, valid_loss, out_return

    Notes:
      - out_return is returned as None here (to avoid allocating full-graph logits),
        because evaluation is performed via loaders.
      - This is the evaluation for flickr/ogbn-arxiv
        to avoid CUDA OOM.
    """
    tr, _ = _score_split_from_loader(
        model, data, loaders["train"], split_idx["train"],
        device=device, eval_func=eval_func, criterion=criterion, forward_kwargs=forward_kwargs
    )
    va, va_loss = _score_split_from_loader(
        model, data, loaders["valid"], split_idx["valid"],
        device=device, eval_func=eval_func, criterion=criterion, forward_kwargs=forward_kwargs
    )
    te, _ = _score_split_from_loader(
        model, data, loaders["test"], split_idx["test"],
        device=device, eval_func=eval_func, criterion=criterion, forward_kwargs=forward_kwargs
    )

    return tr, va, te, va_loss, None
