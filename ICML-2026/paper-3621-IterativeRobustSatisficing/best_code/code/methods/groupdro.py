from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


EvaluationFn = Callable[[nn.Module, DataLoader], Dict[str, float]]


@dataclass
class GroupDROConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    eta_q: float = 0.01
    gamma: float = 0.1
    group_source: str = "target"
    amp: bool = True


def _amp_ctx(device: torch.device, amp: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=amp)
    return nullcontext()


class GroupDatasetInfo:
    def __init__(self, group_counts: np.ndarray):
        self.group_counts = np.asarray(group_counts, dtype=np.float32)
        self.n_groups = int(len(self.group_counts))


class LossComputer:
    """
    GroupDRO loss computer using exponentiated-gradient adversarial weights.
    """

    def __init__(
        self,
        dataset: GroupDatasetInfo,
        eta_q=0.01,
        gamma=0.1,
        adj=None,
        min_var_weight=0,
        normalize_loss=False,
        btl=False,
        device="cuda",
    ):
        self.dataset = dataset
        self.eta_q = eta_q
        self.gamma = gamma
        self.device = device
        self.n_groups = dataset.n_groups

        self.adv_probs = torch.ones(self.n_groups, device=device) / self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups, device=device)
        self.exp_avg_initialized = torch.zeros(self.n_groups, device=device).bool()

        self.group_counts = torch.tensor(dataset.group_counts, dtype=torch.float32, device=device)
        self.group_frac = self.group_counts / self.group_counts.sum().clamp_min(1.0)

        self.processed_data_counts = torch.zeros(self.n_groups, device=device)
        self.update_data_counts = torch.zeros(self.n_groups, device=device)
        self.update_batch_counts = torch.zeros(self.n_groups, device=device)

        self.adj = adj
        self.normalize_loss = normalize_loss
        self.btl = btl
        self.min_var_weight = min_var_weight

    def loss(self, yhat, y, group_idx=None, is_training=True):
        per_sample_losses = F.cross_entropy(yhat, y, reduction="none")

        if group_idx is None:
            group_idx = y

        group_losses = []
        group_counts_batch = []
        for g in range(self.n_groups):
            mask = group_idx == g
            if mask.any():
                group_loss = per_sample_losses[mask].mean()
                group_losses.append(group_loss)
                group_counts_batch.append(mask.sum().item())
            else:
                group_losses.append(torch.tensor(0.0, device=self.device))
                group_counts_batch.append(0)

        group_losses = torch.stack(group_losses)
        group_counts_batch = torch.tensor(
            group_counts_batch,
            dtype=torch.float32,
            device=self.device,
        )

        if is_training:
            for g in range(self.n_groups):
                if group_counts_batch[g] > 0:
                    if not self.exp_avg_initialized[g]:
                        self.exp_avg_loss[g] = group_losses[g].detach()
                        self.exp_avg_initialized[g] = True
                    else:
                        self.exp_avg_loss[g] = (
                            self.gamma * group_losses[g].detach()
                            + (1 - self.gamma) * self.exp_avg_loss[g]
                        )
                    self.update_data_counts[g] += group_counts_batch[g]
                    self.update_batch_counts[g] += 1

            self.processed_data_counts += group_counts_batch
            self.adv_probs = self.adv_probs * torch.exp(
                self.eta_q * group_losses.detach()
            )
            self.adv_probs = self.adv_probs / self.adv_probs.sum().clamp_min(1e-12)

        group_weights = self.adv_probs
        weighted_loss = (group_weights * group_losses).sum()

        loss_dict = {
            "avg_loss": per_sample_losses.mean().item(),
            "weighted_loss": weighted_loss.item(),
        }
        for g in range(self.n_groups):
            loss_dict[f"loss_group_{g}"] = group_losses[g].item()
            loss_dict[f"weight_group_{g}"] = group_weights[g].item()

        return weighted_loss, loss_dict


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
class GroupDROEpochMetrics:
    epoch: int
    train_loss: float
    test_loss: float
    train_acc: float
    test_acc: float
    test_balanced_acc: float
    test_worst_acc: float


class GroupDROTrainer:
    def __init__(
        self,
        cfg: GroupDROConfig,
        model: nn.Module,
        train_loader: DataLoader,
        train_eval_loader: DataLoader,
        test_loader: DataLoader,
        reference_prior: Optional[torch.Tensor] = None,
        device: Optional[Union[torch.device, str]] = None,
        n_classes: int = 10,
        evaluate_fn: Optional[EvaluationFn] = None,
        group_counts: Optional[np.ndarray] = None,
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
        if group_counts is None:
            group_counts = np.ones(n_classes, dtype=np.float32)
        self.loss_computer = LossComputer(
            GroupDatasetInfo(group_counts),
            eta_q=cfg.eta_q,
            gamma=cfg.gamma,
            device=self.device,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(cfg.amp and self.device.type == "cuda")
        )

    def train(self) -> pd.DataFrame:
        history: List[GroupDROEpochMetrics] = []

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            total_ce_sum = 0.0
            total_n = 0

            for x, y, *rest in self.train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                if self.cfg.group_source == "batch" and rest:
                    group_idx = rest[0].to(self.device, non_blocking=True)
                else:
                    group_idx = y

                self.optimizer.zero_grad(set_to_none=True)
                with _amp_ctx(self.device, self.cfg.amp):
                    logits = self.model(x)
                    loss, loss_dict = self.loss_computer.loss(
                        logits,
                        y,
                        group_idx=group_idx,
                        is_training=True,
                    )
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                bs = x.size(0)
                total_ce_sum += loss_dict["avg_loss"] * bs
                total_n += bs

            train_loss = total_ce_sum / max(1, total_n)
            train_metrics = self.evaluate(self.model, self.train_eval_loader)
            test_metrics = self.evaluate(self.model, self.test_loader)
            history.append(
                GroupDROEpochMetrics(
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
                    f"  [GroupDRO] ep {epoch:3d} | "
                    f"test_acc={test_metrics['acc']:.3f}  "
                    f"bal_acc={test_metrics['balanced_acc']:.3f}"
                )
        return pd.DataFrame([asdict(r) for r in history])
