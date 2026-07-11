from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


EvaluationFn = Callable[[nn.Module, DataLoader], Dict[str, float]]


@dataclass
class KLRSConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 4
    target_tau: float = 0.1
    inner_epochs: int = 20
    max_bisect_steps: int = 5
    lambda_init: float = 1.0
    lambda_bisect_tol: float = 0.0
    total_epochs: Optional[int] = None


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


class KLRSObjective:
    """
    KL-RS objective adapted for classification.

    Same maths as the tabular KLRSObjective; criterion is CrossEntropyLoss.
    """

    def __init__(
        self,
        reference_prior: torch.Tensor,
        cfg: KLRSConfig,
        device: torch.device,
    ):
        self.reference_prior = reference_prior.to(device)
        self.cfg = cfg
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def _group_losses(
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

    def r_hat(
        self,
        group_losses: torch.Tensor,
        prior: torch.Tensor,
        lam: float,
    ) -> float:
        if lam <= 0:
            return float("inf")
        log_q = torch.log(prior.clamp_min(1e-30))
        exponents = log_q + group_losses / lam
        return float((lam * torch.logsumexp(exponents, dim=0)).item())

    def feasibility_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_labels: torch.Tensor,
        lam: float,
        tau: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """KL-RS feasibility objective for fixed lambda."""
        loss_vec = self.criterion(logits, targets)
        group_losses, present = self._group_losses(loss_vec, class_labels)

        if len(present) < 2:
            return loss_vec.mean(), {"batch_loss": float(loss_vec.mean().item())}

        prior = self.reference_prior[present]
        prior = prior / prior.sum().clamp_min(1e-12)
        exponents = ((group_losses - tau) / max(lam, 1e-8)).clamp(max=50.0)
        feas_loss = torch.dot(prior, torch.exp(exponents))

        return feas_loss, {
            "batch_loss": float(loss_vec.mean().item()),
            "source_loss": float(torch.dot(prior, group_losses.detach()).item()),
        }

    @torch.no_grad()
    def _full_group_losses(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Accumulate group-mean losses over the full loader."""
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
        return group_losses, prior

    def check_feasibility(
        self,
        model: nn.Module,
        loader: DataLoader,
        lam: float,
        tau: float,
    ) -> Tuple[bool, float]:
        """Return (is_feasible, feas_value) for the current model and lambda."""
        group_losses, prior = self._full_group_losses(model, loader)
        exponents = ((group_losses - tau) / max(lam, 1e-8)).clamp(max=50.0)
        feas_val = float(torch.dot(prior, torch.exp(exponents)).item())
        return feas_val <= 1.0, feas_val


@dataclass
class KLRSEpochMetrics:
    epoch: int
    bisect_step: int
    lam: float
    train_loss: float
    test_loss: float
    train_acc: float
    test_acc: float
    test_balanced_acc: float
    test_worst_acc: float


class KLRSTrainer:
    def __init__(
        self,
        cfg: KLRSConfig,
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

        self.klrs = KLRSObjective(reference_prior, cfg, self.device)
        self.tau = cfg.target_tau

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

    def _new_scheduler(self, n_epochs: int):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(n_epochs, 1),
        )

    def _train_erm_epoch(self, scheduler) -> None:
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        for x, y, *_ in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            loss = criterion(self.model(x), y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        scheduler.step()

    def _train_klrs_epoch(self, lam: float, scheduler) -> None:
        self.model.train()
        for x, y, g, _ in self.train_loader:
            x, y, g = x.to(self.device), y.to(self.device), g.to(self.device)
            feas_loss, _ = self.klrs.feasibility_loss(self.model(x), y, g, lam, self.tau)
            self.optimizer.zero_grad()
            feas_loss.backward()
            self.optimizer.step()
        scheduler.step()

    def _log(
        self,
        history: List[KLRSEpochMetrics],
        global_epoch: int,
        bisect_step: int,
        lam: float,
    ) -> None:
        train_m = self.evaluate(self.model, self.train_eval_loader)
        test_m = self.evaluate(self.model, self.test_loader)
        history.append(
            KLRSEpochMetrics(
                epoch=global_epoch,
                bisect_step=bisect_step,
                lam=lam,
                train_loss=train_m["loss"],
                test_loss=test_m["loss"],
                train_acc=train_m["acc"],
                test_acc=test_m["acc"],
                test_balanced_acc=test_m["balanced_acc"],
                test_worst_acc=test_m["worst_acc"],
            )
        )

    def _feasibility_check(
        self,
        lam: float,
        bisect_step: int,
        global_epoch: int,
        history: List[KLRSEpochMetrics],
        n_epochs: Optional[int] = None,
    ) -> Tuple[bool, int]:
        """Algorithm 1: train at fixed lambda, then check feasibility."""
        n_epochs = self.cfg.inner_epochs if n_epochs is None else n_epochs
        scheduler = self._new_scheduler(n_epochs)
        for _ in range(n_epochs):
            self._train_klrs_epoch(lam, scheduler)
            global_epoch += 1
            self._log(history, global_epoch, bisect_step, lam)

        is_feasible, feas_val = self.klrs.check_feasibility(
            self.model,
            self.train_eval_loader,
            lam,
            self.tau,
        )
        print(
            f"    bisect_step={bisect_step}  lambda={lam:.4f}  "
            f"feas_val={feas_val:.4f}  feasible={is_feasible}"
        )
        return is_feasible, global_epoch

    def _epochs_for_bisect_step(self, global_epoch: int) -> int:
        if self.cfg.total_epochs is None:
            return self.cfg.inner_epochs

        remaining_epochs = self.cfg.total_epochs - global_epoch
        if remaining_epochs <= 0:
            return 0

        return min(self.cfg.inner_epochs, remaining_epochs)

    def train(self) -> pd.DataFrame:
        history: List[KLRSEpochMetrics] = []
        global_epoch = 0

        # Phase 1: ERM warmup.
        warmup_epochs = self.cfg.warmup_epochs
        if self.cfg.total_epochs is not None:
            warmup_epochs = min(warmup_epochs, self.cfg.total_epochs)
        scheduler = self._new_scheduler(warmup_epochs)
        for _ in range(warmup_epochs):
            self._train_erm_epoch(scheduler)
            global_epoch += 1
            self._log(history, global_epoch, bisect_step=0, lam=float("nan"))

        # Phase 2: Bisection (Algorithm 2).
        lam_lower = 0.0
        lam_upper = self.cfg.lambda_init
        bisect_step = 0
        tol = self.cfg.lambda_bisect_tol

        print(f"    Doubling phase: initial lambda={lam_upper:.4f}")
        for _ in range(self.cfg.max_bisect_steps):
            n_epochs = self._epochs_for_bisect_step(global_epoch)
            if n_epochs <= 0:
                break
            bisect_step += 1
            is_feasible, global_epoch = self._feasibility_check(
                lam_upper,
                bisect_step,
                global_epoch,
                history,
                n_epochs,
            )
            if is_feasible:
                break
            lam_lower = lam_upper
            lam_upper *= 2.0
        else:
            print("    Warning: upper bound not found within max_bisect_steps.")

        print(f"    Bisection phase: [{lam_lower:.4f}, {lam_upper:.4f}]")
        while (
            lam_upper - lam_lower >= tol
            and bisect_step < self.cfg.max_bisect_steps
        ):
            n_epochs = self._epochs_for_bisect_step(global_epoch)
            if n_epochs <= 0:
                break
            lam_mid = (lam_upper + lam_lower) / 2.0
            bisect_step += 1
            is_feasible, global_epoch = self._feasibility_check(
                lam_mid,
                bisect_step,
                global_epoch,
                history,
                n_epochs,
            )
            if is_feasible:
                lam_upper = lam_mid
            else:
                lam_lower = lam_mid

        lam_star = (lam_upper + lam_lower) / 2.0
        print(
            f"    Bisection complete: lambda* approx {lam_star:.4f}  "
            f"(bisect_steps={bisect_step})"
        )

        return pd.DataFrame([asdict(r) for r in history])
