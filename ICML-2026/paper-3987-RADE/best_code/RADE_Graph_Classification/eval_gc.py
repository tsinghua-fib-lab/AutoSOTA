# graph_classification/eval_gc.py
from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .data_utils_gc import eval_acc_graph, eval_ogb_rocauc, eval_ogb_ap, get_primary_metric, eval_graph_metric
except Exception:
    from data_utils_gc import eval_acc_graph, eval_ogb_rocauc, eval_ogb_ap, get_primary_metric, eval_graph_metric

try:
    from ogb.graphproppred import Evaluator
except Exception:
    Evaluator = None


def forward_model(model: nn.Module, batch, *, use_edge_attr: bool, p: float = 0.0, q: float = 0.0) -> torch.Tensor:
    """
    Preferred API for GraphMPNN:
      model(x, edge_index, batch, edge_attr=..., p=..., q=...)
    """
    edge_attr = getattr(batch, "edge_attr", None) if use_edge_attr else None
    node_ids = getattr(batch, "n_id", None)

    # Try the full signature first
    try:
        return model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr, p=float(p), q=float(q), node_ids=node_ids)
    except TypeError:
        pass

    # Fallbacks (older models)
    try:
        return model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr, node_ids=node_ids)
    except Exception:
        try:
            return model(batch.x, batch.edge_index, batch.batch, node_ids=node_ids)
        except Exception:
            return model(batch.x, batch.edge_index)


def masked_bce_with_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y = y.to(torch.float32)
    mask = torch.isfinite(y) & (y != -1)
    if not mask.any():
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits[mask], y[mask], reduction="mean")


def _eval_metric(metric: str, dataset_name: str, y_true: torch.Tensor, logits: torch.Tensor) -> float:
    dataset_name = str(dataset_name).lower().strip()
    if dataset_name == "ogbg-molhiv":
        if Evaluator is None:
            raise ImportError("OGB Evaluator is required for ogbg-molhiv evaluation.")
        y = y_true
        if y.dim() == 1:
            y = y.view(-1, 1)
        pred = logits
        if pred.dim() == 1:
            pred = pred.view(-1, 1)
        evaluator = Evaluator(name="ogbg-molhiv")
        result = evaluator.eval(
            {
                "y_true": y.detach().cpu().numpy(),
                "y_pred": pred.detach().cpu().numpy(),
            }
        )
        return float(result["rocauc"])

    if metric == "auto":
        # uses get_primary_metric() internally
        return eval_graph_metric(dataset_name, y_true, logits)

    m = str(metric).lower().strip()
    if m == "acc":
        return eval_acc_graph(y_true, logits)
    if m == "rocauc":
        return eval_ogb_rocauc(y_true, logits)
    if m == "ap":
        return eval_ogb_ap(y_true, logits)
    raise ValueError(f"Unknown metric '{metric}'")


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader,
    dataset_name: str,
    criterion: nn.Module,
    *,
    metric: str = "auto",
    use_edge_attr: bool = False,
    device: Optional[torch.device] = None,
    p: float = 0.0,
    q: float = 0.0,
) -> Tuple[float, float]:
    model.eval()
    if device is not None:
        model = model.to(device)

    total_loss_tensor: Optional[torch.Tensor] = None
    total_graphs = 0
    ys = []
    outs = []

    for batch in loader:
        if device is not None:
            batch = batch.to(device)

        out = forward_model(model, batch, use_edge_attr=use_edge_attr, p=float(p), q=float(q))
        y = batch.y

        # loss
        if isinstance(criterion, nn.BCEWithLogitsLoss) or (y.dim() == 2 and y.dtype != torch.long):
            if y.dim() == 1:
                y = y.view(-1, 1)
            if out.dim() == 1:
                out = out.view(-1, 1)
            loss = masked_bce_with_logits(out, y)
        elif isinstance(criterion, nn.CrossEntropyLoss):
            if y.dim() == 2 and y.size(1) == 1:
                y = y.squeeze(1)
            y = y.view(-1).to(torch.long)
            loss = criterion(out, y)
        else:
            raise ValueError(f"Unsupported criterion type: {type(criterion)}")

        bsz = int(batch.num_graphs) if hasattr(batch, "num_graphs") else int(y.size(0))
        loss_term = loss.detach() * float(bsz)
        total_loss_tensor = loss_term if total_loss_tensor is None else total_loss_tensor + loss_term
        total_graphs += bsz

        ys.append(y.detach().cpu())
        outs.append(out.detach().cpu())

    if total_graphs == 0:
        return float("nan"), float("nan")

    y_all = torch.cat(ys, dim=0)
    out_all = torch.cat(outs, dim=0)

    score = _eval_metric(metric, dataset_name, y_all, out_all)
    total_loss = float(total_loss_tensor.item()) if total_loss_tensor is not None else 0.0
    avg_loss = total_loss / max(total_graphs, 1)
    return float(score), float(avg_loss)


@torch.no_grad()
def evaluate_splits(
    model: nn.Module,
    loaders: Dict[str, object],
    dataset_name: str,
    criterion: nn.Module,
    *,
    metric: str = "auto",
    use_edge_attr: bool = False,
    device: Optional[torch.device] = None,
    p: float = 0.0,
    q: float = 0.0,
) -> Tuple[float, float, float, float]:
    train_metric, _ = evaluate_loader(
        model, loaders["train"], dataset_name, criterion,
        metric=metric, use_edge_attr=use_edge_attr, device=device, p=float(p), q=float(q)
    )
    valid_metric, valid_loss = evaluate_loader(
        model, loaders["valid"], dataset_name, criterion,
        metric=metric, use_edge_attr=use_edge_attr, device=device, p=float(p), q=float(q)
    )
    test_metric, _ = evaluate_loader(
        model, loaders["test"], dataset_name, criterion,
        metric=metric, use_edge_attr=use_edge_attr, device=device, p=float(p), q=float(q)
    )
    return train_metric, valid_metric, test_metric, valid_loss
