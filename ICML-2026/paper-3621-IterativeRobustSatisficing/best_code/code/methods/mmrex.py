from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


EvaluationFn = Callable[[nn.Module, DataLoader], Dict[str, float]]


@dataclass
class MMREXConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    rex_lambda: float = 1.5
    penalty_anneal_epochs: int = 10
    amp: bool = True


def _amp_ctx(device: torch.device, amp: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=amp)
    return nullcontext()


def get_rex_penalty(losses, mode="vrex", lam=1.0):
    if mode == "vrex":
        mean_loss = sum(losses) / len(losses)
        var_loss = sum((loss - mean_loss) ** 2 for loss in losses) / len(losses)
        return var_loss
    if mode == "mmrex":
        loss_max = max(losses)
        loss_min = min(losses)
        return lam * loss_max + (1 - lam) * loss_min
    raise ValueError(f"Unknown REx mode: {mode}")


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
class MMREXEpochMetrics:
    epoch: int
    train_loss: float
    test_loss: float
    train_acc: float
    test_acc: float
    test_balanced_acc: float
    test_worst_acc: float


class MMREXTrainer:
    def __init__(
        self,
        cfg: MMREXConfig,
        model: nn.Module,
        train_loader: DataLoader,
        train_eval_loader: DataLoader,
        test_loader: DataLoader,
        reference_prior: Optional[torch.Tensor] = None,
        device: Optional[Union[torch.device, str]] = None,
        n_classes: int = 10,
        evaluate_fn: Optional[EvaluationFn] = None,
        env_loaders: Optional[Tuple[DataLoader, DataLoader]] = None,
    ):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.train_eval_loader = train_eval_loader
        self.test_loader = test_loader
        self.env_loaders = env_loaders
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
        self.criterion = nn.CrossEntropyLoss()

    def train(self) -> pd.DataFrame:
        if self.env_loaders is None:
            raise ValueError("MMREXTrainer requires env_loaders=(loader_env1, loader_env2).")

        train_loader_env1, train_loader_env2 = self.env_loaders
        history: List[MMREXEpochMetrics] = []

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            total_ce_sum = 0.0
            total_n = 0

            if epoch <= self.cfg.penalty_anneal_epochs:
                anneal_factor = float(epoch) / float(self.cfg.penalty_anneal_epochs)
            else:
                anneal_factor = 1.0
            current_lambda = 0.5 + anneal_factor * (self.cfg.rex_lambda - 0.5)

            iter_env1 = iter(train_loader_env1)
            iter_env2 = iter(train_loader_env2)

            for _ in range(min(len(train_loader_env1), len(train_loader_env2))):
                try:
                    x1, y1, *_ = next(iter_env1)
                    x2, y2, *_ = next(iter_env2)
                except StopIteration:
                    break

                x1, y1 = x1.to(self.device, non_blocking=True), y1.to(self.device, non_blocking=True)
                x2, y2 = x2.to(self.device, non_blocking=True), y2.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                with _amp_ctx(self.device, self.cfg.amp):
                    logits1 = self.model(x1)
                    logits2 = self.model(x2)
                    loss1 = self.criterion(logits1, y1)
                    loss2 = self.criterion(logits2, y2)
                    mean_loss = (loss1 + loss2) / 2.0
                    loss = get_rex_penalty(
                        [loss1, loss2],
                        mode="mmrex",
                        lam=current_lambda,
                    )

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                bs = x1.size(0) + x2.size(0)
                total_ce_sum += mean_loss.detach().item() * bs
                total_n += bs

            train_loss = total_ce_sum / max(1, total_n)
            train_metrics = self.evaluate(self.model, self.train_eval_loader)
            test_metrics = self.evaluate(self.model, self.test_loader)
            history.append(
                MMREXEpochMetrics(
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
                    f"  [MM-REx] ep {epoch:3d} | "
                    f"test_acc={test_metrics['acc']:.3f}  "
                    f"bal_acc={test_metrics['balanced_acc']:.3f}"
                )
        return pd.DataFrame([asdict(r) for r in history])
