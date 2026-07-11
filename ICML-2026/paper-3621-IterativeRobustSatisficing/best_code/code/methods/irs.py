from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


EvaluationFn = Callable[[nn.Module, DataLoader], Dict[str, float]]


@dataclass
class IRSConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 4
    tau_multiplier: float = 1.01
    target_tau: float = 0.1
    h_min: float = 1e-3
    h_max: float = 50.0
    h_grid_points: int = 64
    refine_rounds: int = 3


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int = 10,
) -> Dict[str, float]:
    """
    Compute loss and accuracy metrics on a classification DataLoader.

    The loader is expected to yield at least (x, y); extra fields such as
    group labels or environments are ignored.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_n = 0
    correct_per_class = np.zeros(n_classes, dtype=np.int64)
    total_per_class = np.zeros(n_classes, dtype=np.int64)

    for x, y, *_ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        total_n += len(y)
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


class IRSObjective:
    """
    IRS objective adapted for classification.

    Identical maths to the tabular version; only the criterion changes
    (CrossEntropyLoss instead of MSELoss). Groups are class labels.
    """

    def __init__(
        self,
        reference_prior: torch.Tensor,
        cfg: IRSConfig,
        device: torch.device,
    ):
        self.reference_prior = reference_prior.to(device)
        self.cfg = cfg
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def _group_means(
        self,
        loss_vec: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_groups = len(self.reference_prior)
        counts = torch.bincount(labels, minlength=n_groups)
        present = (counts > 0).nonzero(as_tuple=False).view(-1)
        sums = torch.zeros(n_groups, device=loss_vec.device, dtype=loss_vec.dtype)
        sums.scatter_add_(0, labels, loss_vec)
        means = sums[present] / counts[present].to(loss_vec.dtype)
        return means, present

    def _kl_path_probs(
        self,
        group_losses: torch.Tensor,
        prior: torch.Tensor,
        h: float,
    ) -> torch.Tensor:
        logits = torch.log(prior.clamp_min(1e-30)) + h * group_losses
        return torch.softmax(logits - logits.max(), dim=0)

    def _dkl(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return (p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log())).sum()

    def _maximize_along_curve(
        self,
        group_losses: torch.Tensor,
        prior: torch.Tensor,
        tau: float,
    ) -> Tuple[torch.Tensor, float, float]:
        """1-D scalar search for optimal h (IRS core, same as tabular)."""

        def eval_h(h: float) -> Tuple[float, float, torch.Tensor]:
            p = self._kl_path_probs(group_losses, prior, h)
            dkl = float(self._dkl(p, prior).item())
            num = float(torch.dot(p, group_losses).item()) - tau
            if dkl <= 1e-12:
                return -float("inf"), dkl, p
            return num / dkl, dkl, p

        log_min, log_max = math.log(self.cfg.h_min), math.log(self.cfg.h_max)
        best_val, best_h, best_dkl, best_p = (
            -float("inf"),
            self.cfg.h_min,
            float("nan"),
            prior.clone(),
        )

        for log_h in torch.linspace(log_min, log_max, self.cfg.h_grid_points).tolist():
            h = math.exp(log_h)
            v, dkl, p = eval_h(h)
            if v > best_val:
                best_val, best_h, best_dkl, best_p = v, h, dkl, p

        step = (log_max - log_min) / max(self.cfg.h_grid_points - 1, 1)
        left = max(log_min, math.log(best_h) - step)
        right = min(log_max, math.log(best_h) + step)

        for _ in range(self.cfg.refine_rounds):
            for log_h in torch.linspace(left, right, 25).tolist():
                h = math.exp(log_h)
                v, dkl, p = eval_h(h)
                if v > best_val:
                    best_val, best_h, best_dkl, best_p = v, h, dkl, p
            w = max((right - left) / 4.0, 1e-6)
            c = math.log(best_h)
            left = max(log_min, c - w)
            right = min(log_max, c + w)

        return best_p, best_val, best_dkl

    def _active_tau(self, source_loss: float) -> float:
        adaptive = self.cfg.tau_multiplier * source_loss
        return max(adaptive, self.cfg.target_tau)

    def training_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """IRS training loss. logits: (N, C), targets/class_labels: (N,) long."""
        loss_vec = self.criterion(logits, targets)
        group_losses_det, present = self._group_means(loss_vec.detach(), class_labels)

        if len(present) < 2:
            mean_loss = loss_vec.mean()
            return mean_loss, {
                "batch_loss": float(mean_loss.item()),
                "batch_kappa": float("nan"),
            }

        prior = self.reference_prior[present]
        prior = prior / prior.sum().clamp_min(1e-12)
        source_loss = float(torch.dot(prior, group_losses_det).item())
        active_tau = self._active_tau(source_loss)

        tilted_prior, kappa, dkl = self._maximize_along_curve(
            group_losses_det,
            prior,
            active_tau,
        )

        # Differentiable pass with the detached tilted prior (Danskin's theorem).
        group_losses_grad, _ = self._group_means(loss_vec, class_labels)
        irs_loss = torch.dot(tilted_prior.detach(), group_losses_grad) / max(dkl, 1e-12)

        return irs_loss, {
            "batch_loss": float(loss_vec.mean().item()),
            "batch_kappa": kappa,
            "active_tau": active_tau,
        }

    @torch.no_grad()
    def compute_kappa(
        self,
        model: nn.Module,
        loader: DataLoader,
        tau: float,
    ) -> float:
        """Compute fragility kappa_tau over the full loader."""
        model.eval()
        n_groups = len(self.reference_prior)
        loss_sum = torch.zeros(n_groups, device=self.device)
        count = torch.zeros(n_groups, device=self.device)
        criterion = nn.CrossEntropyLoss(reduction="none")

        for x, y, g, _ in loader:
            x, y, g = x.to(self.device), y.to(self.device), g.to(self.device)
            loss_vec = criterion(model(x), y)
            for c in range(n_groups):
                mask = g == c
                if mask.any():
                    loss_sum[c] += loss_vec[mask].sum()
                    count[c] += mask.sum()

        present = (count > 0).nonzero(as_tuple=False).view(-1)
        group_losses = loss_sum[present] / count[present].clamp_min(1)
        prior = self.reference_prior[present]
        prior = prior / prior.sum().clamp_min(1e-12)

        _, kappa, _ = self._maximize_along_curve(group_losses, prior, tau)
        return kappa


@dataclass
class IRSEpochMetrics:
    epoch: int
    train_loss: float
    test_loss: float
    train_acc: float
    test_acc: float
    test_balanced_acc: float
    test_worst_acc: float
    kappa: float


class IRSTrainer:
    def __init__(
        self,
        cfg: IRSConfig,
        model: nn.Module,
        train_loader: DataLoader,
        train_eval_loader: DataLoader,
        test_loader: DataLoader,
        reference_prior: torch.Tensor,
        device: Optional[Union[torch.device, str]] = None,
        n_classes: int = 10,
        evaluate_fn: Optional[EvaluationFn] = None,
    ):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.epochs,
        )

        self.irs = IRSObjective(reference_prior, cfg, self.device)

        self.train_loader = train_loader
        self.train_eval_loader = train_eval_loader
        self.test_loader = test_loader
        self.evaluate: EvaluationFn = evaluate_fn or (
            lambda model, loader: evaluate_classification(
                model,
                loader,
                self.device,
                n_classes,
            )
        )

    def train(self) -> pd.DataFrame:
        history: List[IRSEpochMetrics] = []

        for epoch in range(1, self.cfg.epochs + 1):
            warming_up = epoch <= self.cfg.warmup_epochs
            self.model.train()
            for x, y, g, _ in self.train_loader:
                x, y, g = x.to(self.device), y.to(self.device), g.to(self.device)
                logits = self.model(x)
                if warming_up:
                    loss = F.cross_entropy(logits, y)
                else:
                    loss, _ = self.irs.training_loss(logits, y, g)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            self.scheduler.step()

            train_metrics = self.evaluate(self.model, self.train_eval_loader)
            test_metrics = self.evaluate(self.model, self.test_loader)

            if warming_up:
                kappa = float("nan")
            else:
                active_tau = self.irs._active_tau(train_metrics["loss"])
                kappa = self.irs.compute_kappa(
                    self.model,
                    self.train_eval_loader,
                    active_tau,
                )

            history.append(
                IRSEpochMetrics(
                    epoch=epoch,
                    train_loss=train_metrics["loss"],
                    test_loss=test_metrics["loss"],
                    train_acc=train_metrics["acc"],
                    test_acc=test_metrics["acc"],
                    test_balanced_acc=test_metrics["balanced_acc"],
                    test_worst_acc=test_metrics["worst_acc"],
                    kappa=kappa,
                )
            )

            if epoch % 5 == 0 or epoch == 1:
                phase = "warmup" if warming_up else "IRS  "
                print(
                    f"  [{phase}] ep {epoch:3d} | "
                    f"test_acc={test_metrics['acc']:.3f}  "
                    f"bal_acc={test_metrics['balanced_acc']:.3f}  "
                    f"kappa={kappa:.4f}"
                )

        return pd.DataFrame([asdict(r) for r in history])
