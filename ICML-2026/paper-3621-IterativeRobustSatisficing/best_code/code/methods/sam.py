from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


EvaluationFn = Callable[[nn.Module, DataLoader], Dict[str, float]]


@dataclass
class SAMConfig:
    epochs: int = 100
    lr: float = 1e-3
    rho: float = 0.05
    weight_decay: float = 0.0
    momentum: float = 0.9
    use_cosine: bool = False
    label_smoothing: float = 0.0
    grad_clip: Optional[float] = 1.0
    max_scale: Optional[float] = None
    amp: bool = True


def _amp_ctx(device: torch.device, amp: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=amp)
    return nullcontext()


class SAMSGD(torch.optim.SGD):
    def __init__(self, params: Iterable[torch.Tensor], lr=1e-3, rho=0.05, **kwargs):
        if rho <= 0:
            raise ValueError(f"rho must be positive, got {rho}")
        self.rho = float(rho)
        super().__init__(params, lr=lr, **kwargs)

    @torch.no_grad()
    def _grad_norm(self):
        device_ = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.detach().norm(2))
        if not norms:
            return torch.tensor(0.0, device=device_)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def _epsilon(self, scale, max_scale=None):
        """Apply the SAM ascent perturbation."""
        if max_scale is not None:
            scale = min(scale, max_scale) if isinstance(scale, (int, float)) else scale.clamp(max=max_scale)
        epsilons = []
        for group in self.param_groups:
            eps_group = []
            for p in group["params"]:
                if p.grad is None:
                    eps_group.append(None)
                    continue
                e = p.grad * scale
                if not torch.isfinite(e).all():
                    eps_group.append(torch.zeros_like(p))
                    continue
                p.add_(e)
                eps_group.append(e)
            epsilons.append(eps_group)
        return epsilons

    @torch.no_grad()
    def _restore(self, epsilons):
        for group, eps_group in zip(self.param_groups, epsilons):
            for p, e in zip(group["params"], eps_group):
                if e is not None:
                    p.sub_(e)


def disable_running_stats(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m._sam_backup_momentum = m.momentum
            m.momentum = 0.0


def enable_running_stats(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and hasattr(m, "_sam_backup_momentum"):
            m.momentum = m._sam_backup_momentum
            delattr(m, "_sam_backup_momentum")


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
class SAMEpochMetrics:
    epoch: int
    train_loss: float
    test_loss: float
    train_acc: float
    test_acc: float
    test_balanced_acc: float
    test_worst_acc: float


class SAMTrainer:
    def __init__(
        self,
        cfg: SAMConfig,
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
        self.optimizer = SAMSGD(
            self.model.parameters(),
            lr=cfg.lr,
            rho=cfg.rho,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        self.criterion = (
            nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
            if cfg.label_smoothing > 0
            else nn.CrossEntropyLoss()
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(cfg.amp and self.device.type == "cuda")
        )
        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.epochs,
                eta_min=1e-6,
            )
            if cfg.use_cosine
            else None
        )

    def _unscale_sam_ascent_grads(self) -> None:
        if not self.scaler.is_enabled():
            return
        inv_scale = 1.0 / float(self.scaler.get_scale())
        for p in self.model.parameters():
            if p.grad is not None:
                if p.grad.is_sparse:
                    p.grad = p.grad.coalesce()
                    p.grad._values().mul_(inv_scale)
                else:
                    p.grad.detach().mul_(inv_scale)

    def train(self) -> pd.DataFrame:
        history: List[SAMEpochMetrics] = []

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            total_ce_sum = 0.0
            total_n = 0

            for x, y, *_ in self.train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                enable_running_stats(self.model)
                self.optimizer.zero_grad(set_to_none=True)
                with _amp_ctx(self.device, self.cfg.amp):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)

                if not torch.isfinite(loss):
                    continue

                self.scaler.scale(loss).backward()
                self._unscale_sam_ascent_grads()

                if self.cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.cfg.grad_clip,
                    )

                gnorm = self.optimizer._grad_norm()

                if not torch.isfinite(gnorm):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                if gnorm < 1e-8:
                    self.optimizer.zero_grad(set_to_none=True)
                    with _amp_ctx(self.device, self.cfg.amp):
                        logits = self.model(x)
                        loss = self.criterion(logits, y)
                    if not torch.isfinite(loss):
                        continue
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if self.cfg.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.cfg.grad_clip,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    bs = x.size(0)
                    total_ce_sum += loss.detach().item() * bs
                    total_n += bs
                    continue

                scale = self.cfg.rho / (gnorm + 1e-12)
                epsilons = self.optimizer._epsilon(scale, max_scale=self.cfg.max_scale)

                disable_running_stats(self.model)
                self.optimizer.zero_grad(set_to_none=True)
                with _amp_ctx(self.device, self.cfg.amp):
                    logits_pert = self.model(x)
                    loss_pert = self.criterion(logits_pert, y)

                if not torch.isfinite(loss_pert):
                    self.optimizer._restore(epsilons)
                    enable_running_stats(self.model)
                    self.optimizer.zero_grad(set_to_none=True)
                    with _amp_ctx(self.device, self.cfg.amp):
                        logits = self.model(x)
                        loss = self.criterion(logits, y)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if self.cfg.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.cfg.grad_clip,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    bs = x.size(0)
                    total_ce_sum += loss.detach().item() * bs
                    total_n += bs
                    continue

                self.scaler.scale(loss_pert).backward()
                self.optimizer._restore(epsilons)
                enable_running_stats(self.model)
                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                bs = x.size(0)
                total_ce_sum += loss.detach().item() * bs
                total_n += bs

            if self.scheduler is not None:
                self.scheduler.step()

            train_loss = total_ce_sum / max(1, total_n)
            train_metrics = self.evaluate(self.model, self.train_eval_loader)
            test_metrics = self.evaluate(self.model, self.test_loader)

            if not np.isfinite(test_metrics["loss"]):
                print(f"  [SAM] WARNING: test loss became NaN at epoch {epoch}.")
                break

            history.append(
                SAMEpochMetrics(
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
                    f"  [SAM] ep {epoch:3d} | "
                    f"test_acc={test_metrics['acc']:.3f}  "
                    f"bal_acc={test_metrics['balanced_acc']:.3f}"
                )
        return pd.DataFrame([asdict(r) for r in history])
