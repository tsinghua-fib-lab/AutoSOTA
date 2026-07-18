# metrics.py
#
# loss, regularization, dataloader, and evaluation helpers for circuit-compatible models.

from __future__ import annotations

from .utils import DEVICE

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .circuit import Circuit, edge_key, node_key
from .models import CircuitModel

WeightItem = tuple[node_key, str]

# --------------------------------------------------------------------------------------
# Edge-Pruning hard-concrete / L0 helpers
# --------------------------------------------------------------------------------------

EDGE_PRUNING_LIMIT_LEFT = -0.1
EDGE_PRUNING_LIMIT_RIGHT = 1.1
EDGE_PRUNING_EPS = 1e-6
EDGE_PRUNING_TEMPERATURE = 2.0 / 3.0
EDGE_PRUNING_SCORE_FACTOR = 0.8


def edge_pruning_stretched_concrete_cdf(
    x: float,
    log_alpha: torch.Tensor,
) -> torch.Tensor:
    """
    CDF of the stretched concrete gate used by Edge-Pruning.

    This mirrors the reference implementation's L0 expected-nonzero
    calculation, but keeps it model-agnostic so algorithms can operate on any
    grouped edge-logit tensors supplied by a CircuitModel.
    """
    x_01 = (x - EDGE_PRUNING_LIMIT_LEFT) / (
        EDGE_PRUNING_LIMIT_RIGHT - EDGE_PRUNING_LIMIT_LEFT
    )
    intermediate = math.log(x_01) - math.log(1.0 - x_01)
    prob = torch.sigmoid(
        EDGE_PRUNING_TEMPERATURE * intermediate - log_alpha
    )
    return torch.clamp(prob, EDGE_PRUNING_EPS, 1.0 - EDGE_PRUNING_EPS)


def edge_pruning_expected_density(
    edge_logits: nn.ParameterList | list[torch.Tensor],
    *,
    device: str = DEVICE,
) -> torch.Tensor:
    """
    Expected fraction of nonzero edge gates under the hard-concrete relaxation.
    """
    if len(edge_logits) == 0:
        return torch.zeros((), device=device)

    total = torch.zeros((), device=device)
    n_params = 0

    for logits in edge_logits:
        total = total + (
            1.0 - edge_pruning_stretched_concrete_cdf(0.0, logits)
        ).sum()
        n_params += logits.numel()

    return total / max(n_params, 1)


def edge_pruning_expected_sparsity(
    edge_logits: nn.ParameterList | list[torch.Tensor],
    *,
    device: str = DEVICE,
) -> torch.Tensor:
    return 1.0 - edge_pruning_expected_density(edge_logits, device=device)


def edge_pruning_sample_mask(
    logits: torch.Tensor,
    *,
    reverse: bool = False,
    **_: Any,
) -> torch.Tensor:
    """
    Stochastic hard-concrete sample clipped to [0, 1].
    """
    uniform = logits.new_empty(logits.shape).uniform_(
        EDGE_PRUNING_EPS,
        1.0 - EDGE_PRUNING_EPS,
    )
    concrete = torch.sigmoid(
        (torch.log(uniform) - torch.log1p(-uniform) + logits)
        / EDGE_PRUNING_TEMPERATURE
    )
    stretched = (
        EDGE_PRUNING_LIMIT_RIGHT - EDGE_PRUNING_LIMIT_LEFT
    ) * concrete + EDGE_PRUNING_LIMIT_LEFT
    mask = F.hardtanh(stretched, 0.0, 1.0)

    if reverse:
        mask = 1.0 - mask

    return mask


def edge_pruning_edge_scores(logits: torch.Tensor) -> torch.Tensor:
    """
    Continuous learned edge scores independent of any chosen sparsity budget.
    """
    return torch.sigmoid(
        logits / EDGE_PRUNING_TEMPERATURE * EDGE_PRUNING_SCORE_FACTOR
    )


def edge_pruning_lagrangian_loss(
    *,
    model_edge_sparsity: torch.Tensor,
    target_edge_sparsity: float,
    lambda_1: torch.Tensor,
    lambda_2: torch.Tensor,
) -> torch.Tensor:
    """
    Edge-Pruning Lagrangian sparsity objective.

    The lambda parameters should be optimized with `maximize=True`; edge logits
    are optimized in the normal minimizing direction.
    """
    target = model_edge_sparsity.new_tensor(target_edge_sparsity)
    diff = model_edge_sparsity - target
    return lambda_1.reshape(()) * diff + lambda_2.reshape(()) * diff.square()


def edge_pruning_linear_sparsity_target(
    step: int,
    *,
    start_edge_sparsity: float,
    target_edge_sparsity: float,
    warmup_steps: int,
) -> float:
    if warmup_steps <= 0 or step >= warmup_steps:
        return target_edge_sparsity

    frac = step / warmup_steps
    return start_edge_sparsity + (
        target_edge_sparsity - start_edge_sparsity
    ) * frac


def edge_pruning_full_vocab_kl_loss(
    batch: dict[str, Any],
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """
    KL(student || full model) over full vocabulary distributions.

    Position selection is batch-driven:
        - `loss_mask`: compare all true positions.
        - `seq_lens`: compare next-token logits at seq_len - 1.
        - otherwise: compare every sequence position.

    teacher_logits may either match logits shape or already be preselected to
    [n_selected_positions, vocab], which allows the trainer to cache only the
    full-vocab teacher distributions actually used by the KL objective.
    """
    student = edge_pruning_select_kl_positions(batch, logits)
    if student.numel() == 0:
        return logits.new_tensor(0.0)

    if teacher_logits.shape == logits.shape:
        teacher = edge_pruning_select_kl_positions(batch, teacher_logits)
    elif teacher_logits.ndim == 2 and teacher_logits.shape[0] == student.shape[0]:
        teacher = teacher_logits
    else:
        raise ValueError(
            "teacher_logits must either match student logits shape or be "
            "preselected to [n_selected_positions, vocab]. "
            f"got teacher={tuple(teacher_logits.shape)}, "
            f"logits={tuple(logits.shape)}, selected={tuple(student.shape)}."
        )

    teacher = teacher.to(device=student.device, dtype=student.dtype)

    student_log_probs = F.log_softmax(student, dim=-1)
    teacher_log_probs = F.log_softmax(teacher.detach(), dim=-1)
    return F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction="batchmean",
        log_target=True,
    )


def edge_pruning_select_kl_positions(
    batch: dict[str, Any],
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Select the positions used by full-vocab Edge-Pruning KL.
    """
    if "loss_mask" in batch:
        mask = batch["loss_mask"].to(device=logits.device, dtype=torch.bool)
        return logits[mask]

    elif "seq_lens" in batch:
        batch_size = logits.shape[0]
        rows = torch.arange(batch_size, device=logits.device)
        pos = batch["seq_lens"].to(device=logits.device) - 1
        return logits[rows, pos]

    return logits.reshape(-1, logits.shape[-1])


def edge_pruning_two_label_loss(
    batch: dict[str, Any],
    logits: torch.Tensor,
    teacher_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Good-vs-bad target loss shared with DiscoGP datasets.

    teacher_logits is accepted so callers can swap KL and two-label objectives
    through a common signature.
    """
    del teacher_logits
    return discogp_fidelity_loss(batch, logits)


# --------------------------------------------------------------------------------------
# ACDC / EAP scoring helpers
# --------------------------------------------------------------------------------------

def acdc_loss_delta(
    *,
    candidate_loss: torch.Tensor,
    current_loss: torch.Tensor,
) -> torch.Tensor:
    """
    Loss increase caused by a candidate ACDC edge removal.

    ACDC keeps a removal when this increase is below the current threshold.
    Keeping this tiny helper in metrics.py makes the decision criterion shared
    and easy to swap when we add non-CE faithfulness targets.
    """
    return candidate_loss - current_loss


def acdc_should_prune(
    *,
    candidate_loss: torch.Tensor,
    current_loss: torch.Tensor,
    threshold: float,
) -> bool:
    return bool(
        (
            acdc_loss_delta(
                candidate_loss=candidate_loss,
                current_loss=current_loss,
            )
            < candidate_loss.new_tensor(threshold)
        )
        .detach()
        .cpu()
        .item()
    )


def eap_edge_importance_from_gradient(
    grad: torch.Tensor,
    *,
    absolute: bool = True,
) -> torch.Tensor:
    """
    Zero-ablation EAP score from the gradient of the task loss wrt an edge gate.

    With an edge gate m and zero ablation, removing an edge changes m from 1 to
    0. The first-order loss change is therefore -dL/dm. The original EAP score
    uses the magnitude of grad * (ablated - clean); for zero ablation this is
    exactly the magnitude of -dL/dm.
    """
    score = -grad.detach()
    return score.abs() if absolute else score

# --------------------------------------------------------------------------------------
# DiscoGP-specific task-related loss functions
# --------------------------------------------------------------------------------------

def discogp_fidelity_loss(batch: dict[str, Any], logits: torch.Tensor) -> torch.Tensor:
    """
    default good-vs-bad cross-entropy loss.

    expected batch keys:
        input_ids
        seq_lens
        target good
        target bad
    """
    if "target bad" not in batch:
        return logits.new_tensor(0.0)

    seq_lens = batch["seq_lens"]
    batch_size = logits.shape[0]
    rows = torch.arange(batch_size, device=logits.device)

    good = logits[rows, seq_lens - 1, batch["target good"]]
    bad = logits[rows, seq_lens - 1, batch["target bad"]]
    good_bad_logits = torch.stack([good, bad], dim=-1)

    target = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    return F.cross_entropy(good_bad_logits, target)


def discogp_completeness_loss(
    batch: dict[str, Any],
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    default complement loss.

    this pushes the complement circuit toward being uninformative between
    good/bad targets by using a uniform soft target.
    """
    if "target bad" not in batch:
        return logits.new_tensor(0.0)

    seq_lens = batch["seq_lens"]
    batch_size = logits.shape[0]
    rows = torch.arange(batch_size, device=logits.device)

    good = logits[rows, seq_lens - 1, batch["target good"]]
    bad = logits[rows, seq_lens - 1, batch["target bad"]]
    good_bad_logits = torch.stack([good, bad], dim=-1)

    target = torch.full_like(good_bad_logits, 0.5)
    return F.cross_entropy(good_bad_logits, target)

# --------------------------------------------------------------------------------------
# DiscoGP-specific density-related loss functions
# --------------------------------------------------------------------------------------

def discogp_edge_density_loss(
    edge_logits: nn.ParameterList,
    *,
    device: str = DEVICE,
) -> torch.Tensor:
    if len(edge_logits) == 0:
        return torch.zeros((), device=device)

    total = torch.zeros((), device=device)
    n_params = 0

    for logits in edge_logits:
        total = total + torch.sigmoid(logits).sum()
        n_params += logits.numel()

    return total / max(n_params, 1)


def discogp_weight_density_loss(
    weight_logits: nn.ParameterList,
    *,
    device: str = DEVICE,
) -> torch.Tensor:
    if len(weight_logits) == 0:
        return torch.zeros((), device=device)

    total = torch.zeros((), device=device)
    n_params = 0

    for logits in weight_logits:
        total = total + torch.sigmoid(logits).sum()
        n_params += logits.numel()

    return total / max(n_params, 1)


def discogp_overlap_loss(
    *,
    reference: Circuit,
    edge_keys: list[edge_key],
    edge_logits: nn.ParameterList,
    weight_items: list[WeightItem],
    weight_logits: nn.ParameterList,
    edge_logit_keys: list[list[edge_key]] | None = None,
    weight_logit_items: list[list[WeightItem]] | None = None,
    edges: bool = False,
    weights: bool = False,
    penalty: bool = True,
    device: str = DEVICE,
) -> torch.Tensor:
    """
    positive value if penalty=True, negative value if penalty=False.

    penalty=True  -> discourage overlap
    penalty=False -> encourage overlap
    """
    sign = 1.0 if penalty else -1.0
    loss = torch.zeros((), device=device)
    denom = 0

    if edges:
        if edge_logit_keys is None:
            for key, logit in zip(edge_keys, edge_logits):
                ref_mask = reference.get_edge(key).edge_mask

                if ref_mask is not None and bool(ref_mask.bool().item()):
                    loss = loss + torch.sigmoid(logit)
                    denom += 1
        else:
            for keys, logits in zip(edge_logit_keys, edge_logits):
                flat_logits = logits.reshape(-1)
                for offset, key in enumerate(keys):
                    ref_mask = reference.get_edge(key).edge_mask

                    if ref_mask is not None and bool(ref_mask.bool().item()):
                        loss = loss + torch.sigmoid(flat_logits[offset])
                        denom += 1

    if weights:
        if weight_logit_items is None:
            for (n_key, w_key), logit in zip(weight_items, weight_logits):
                ref_mask = reference.nodes[n_key].weight_masks.get(w_key)

                if ref_mask is None:
                    continue

                ref = ref_mask.to(device=logit.device, dtype=torch.bool)

                if ref.shape == torch.Size([]):
                    if bool(ref.item()):
                        loss = loss + torch.sigmoid(logit).sum()
                        denom += logit.numel()
                else:
                    loss = loss + torch.sigmoid(logit[ref]).sum()
                    denom += int(ref.sum().item())
        else:
            for items, logits in zip(weight_logit_items, weight_logits):
                for offset, (n_key, w_key) in enumerate(items):
                    logit = logits if len(items) == 1 else logits[offset]
                    ref_mask = reference.nodes[n_key].weight_masks.get(w_key)

                    if ref_mask is None:
                        continue

                    ref = ref_mask.to(device=logit.device, dtype=torch.bool)

                    if ref.shape == torch.Size([]):
                        if bool(ref.item()):
                            loss = loss + torch.sigmoid(logit).sum()
                            denom += logit.numel()
                    else:
                        loss = loss + torch.sigmoid(logit[ref]).sum()
                        denom += int(ref.sum().item())

    if denom == 0:
        return torch.zeros((), device=device)

    return sign * loss / denom

# --------------------------------------------------------------------------------------
# two-label evaluation
# --------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_good_bad_accuracy(
    *,
    model: CircuitModel,
    dataloader,
    circuit: Circuit,
) -> dict[str, Any]:
    """
    evaluate whether target good logit is larger than target bad logit.

    expected batch keys:
        input_ids
        seq_lens
        target good
        target bad
    """

    n_correct = 0
    n_total = 0

    model.eval()

    for batch in dataloader:
        logits = model(batch["input_ids"], circuit=circuit)

        if "target bad" not in batch:
            continue

        seq_lens = batch["seq_lens"]
        cur_batch_size = logits.shape[0]
        rows = torch.arange(cur_batch_size, device=logits.device)

        good = logits[rows, seq_lens - 1, batch["target good"]]
        bad = logits[rows, seq_lens - 1, batch["target bad"]]

        n_correct += int((good > bad).sum().item())
        n_total += cur_batch_size

    out: dict[str, Any] = {
        "n_correct": n_correct,
        "n_total": n_total,
        "acc": n_correct / n_total if n_total > 0 else 0.0,
    }
    out.update(circuit.stats())

    return out
