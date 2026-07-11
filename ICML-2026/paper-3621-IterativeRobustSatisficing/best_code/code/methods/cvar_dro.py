from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


EvaluationFn = Callable[[nn.Module, DataLoader], Dict[str, float]]


@dataclass
class CVaRDROConfig:
    epochs: int = 100
    lr: float = 1e-3
    alpha: float = 0.5
    weight_decay: float = 0.0
    amp: bool = True


def _amp_ctx(device: torch.device, amp: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=amp)
    return nullcontext()


def cvar_loss_from_batch(loss_vec: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Finite-sample CVaR-DRO objective.

    This is Eq. (28) in Levy et al. (2020): average the worst alpha
    fraction of losses, with fractional weight on the cutoff example when
    alpha * batch_size is not an integer.
    """
    alpha = max(min(float(alpha), 1.0), 1e-6)
    batch_size = loss_vec.size(0)
    if batch_size == 0:
        return loss_vec.new_tensor(0.0)

    sorted_losses = torch.sort(loss_vec, descending=True).values
    tail_mass = alpha * batch_size
    k = int(math.floor(tail_mass + 1e-12))
    frac = tail_mass - k
    if abs(frac) < 1e-12:
        frac = 0.0

    if k == 0:
        return sorted_losses[0]
    if k >= batch_size:
        return sorted_losses.mean()

    tail_sum = sorted_losses[:k].sum() + frac * sorted_losses[k]
    return tail_sum / tail_mass


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int = 10,
    amp: bool = True,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_n = 0
    correct_per_class = np.zeros(n_classes, dtype=np.int64)
    total_per_class = np.zeros(n_classes, dtype=np.int64)

    with _amp_ctx(device, amp):
        for x, y, *_ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            total_n += y.size(0)
            preds = logits.argmax(dim=1)
            for c in range(n_classes):
                mask = y == c
                correct_per_class[c] += (preds[mask] == c).sum().item()
                total_per_class[c] += mask.sum().item()

    per_class_acc = np.where(
        total_per_class > 0,
        correct_per_class / total_per_class,
        np.nan,
    )
    return {
        "loss": total_loss / max(total_n, 1),
        "acc": correct_per_class.sum() / max(total_n, 1),
        "balanced_acc": float(np.nanmean(per_class_acc)),
        "worst_acc": float(np.nanmin(per_class_acc)),
    }


@dataclass
class CVaRDROEpochMetrics:
    epoch: int
    train_loss: float
    test_loss: float
    train_acc: float
    test_acc: float
    test_balanced_acc: float
    test_worst_acc: float


class CVaRDROTrainer:
    def __init__(
        self,
        cfg: CVaRDROConfig,
        model: nn.Module,
        train_loader: DataLoader,
        train_eval_loader: DataLoader,
        test_loader: DataLoader,
        reference_prior: Optional[torch.Tensor] = None,
        device: Optional[Union[torch.device, str]] = None,
        n_classes: int = 10,
        evaluate_fn: Optional[EvaluationFn] = None,
    ):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.train_eval_loader = train_eval_loader
        self.test_loader = test_loader
        self.evaluate: EvaluationFn = evaluate_fn or (
            lambda model, loader: evaluate_classification(
                model,
                loader,
                self.device,
                n_classes,
                cfg.amp,
            )
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(cfg.amp and self.device.type == "cuda")
        )
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def train(self) -> pd.DataFrame:
        history: List[CVaRDROEpochMetrics] = []

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            total_ce_sum = 0.0
            total_n = 0

            for x, y, *_ in self.train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                with _amp_ctx(self.device, self.cfg.amp):
                    logits = self.model(x)
                    loss_vec = self.criterion(logits, y)
                    loss = cvar_loss_from_batch(loss_vec, alpha=self.cfg.alpha)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                bs = x.size(0)
                total_ce_sum += loss_vec.mean().detach().item() * bs
                total_n += bs

            train_loss = total_ce_sum / max(1, total_n)
            train_metrics = self.evaluate(self.model, self.train_eval_loader)
            test_metrics = self.evaluate(self.model, self.test_loader)
            history.append(
                CVaRDROEpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    test_loss=test_metrics["loss"],
                    train_acc=train_metrics["acc"],
                    test_acc=test_metrics["acc"],
                    test_balanced_acc=test_metrics["balanced_acc"],
                    test_worst_acc=test_metrics["worst_acc"],
                )
            )

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  [CVaR-DRO] ep {epoch:3d} | "
                    f"test_acc={test_metrics['acc']:.3f}  "
                    f"bal_acc={test_metrics['balanced_acc']:.3f}"
                )
        return pd.DataFrame([asdict(r) for r in history])
