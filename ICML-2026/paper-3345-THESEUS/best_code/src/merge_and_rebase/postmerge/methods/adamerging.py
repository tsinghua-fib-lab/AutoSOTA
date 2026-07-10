from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..base import PostMergeContext, PostMergeResult
from ..registry import register
from ..task_delta_bank import TaskDeltaBank
from ._vision_training import (
    VisionPostmergeTrainer,
    visual_params_from_bank,
)
from ._vision_training import (
    prediction_entropy as prediction_entropy,
)


def clamped_alpha_from_raw(raw: torch.Tensor, *, alpha_min: float, alpha_max: float) -> torch.Tensor:
    lo = float(alpha_min)
    hi = float(alpha_max)
    if hi < lo:
        raise ValueError("postmerge.alpha_max must be >= postmerge.alpha_min")
    return torch.clamp(raw, min=lo, max=hi)


def init_clamped_raw_alpha(
    shape: tuple[int, ...],
    *,
    init_alpha: float,
    alpha_min: float,
    alpha_max: float,
    device: str | torch.device,
) -> torch.Tensor:
    lo = float(alpha_min)
    hi = float(alpha_max)
    if hi < lo:
        raise ValueError("postmerge.alpha_max must be >= postmerge.alpha_min")
    init = min(max(float(init_alpha), lo), hi)
    return torch.full(shape, fill_value=float(init), dtype=torch.float32, device=device)


def _build_vision_backward_entropy_loss(context: PostMergeContext, cfg: dict[str, Any]):
    trainer_cfg = dict(cfg)
    trainer_cfg["loss"] = "entropy"
    trainer = VisionPostmergeTrainer(context, trainer_cfg)

    def _backward_entropy_loss(bank: TaskDeltaBank, alpha_values: torch.Tensor, alpha_mode: str) -> torch.Tensor:
        detached_losses: list[torch.Tensor] = []
        alpha_grad = torch.zeros_like(alpha_values)
        for item, images, labels in trainer.iter_batches():
            visual_params = visual_params_from_bank(
                bank=bank,
                model=trainer.clf.model,
                alpha_values=alpha_values,
                alpha_mode=alpha_mode,
                device=trainer.device,
            )
            logits = trainer.visual_logits_from_params(item, visual_params, images)
            raw_loss = trainer.loss_from_logits(logits, labels)
            if not torch.isfinite(raw_loss):
                raise RuntimeError(
                    f"AdaMerging vision loss became non-finite for task '{item['task']}': "
                    f"{float(raw_loss.detach().cpu())}"
                )
            (task_alpha_grad,) = torch.autograd.grad(raw_loss, alpha_values)
            alpha_grad = alpha_grad + task_alpha_grad
            detached_losses.append(raw_loss.detach())
            del visual_params, logits, raw_loss
        if not detached_losses:
            raise RuntimeError("AdaMerging vision loss did not receive any training batches.")
        alpha_values.backward(alpha_grad)
        loss_stack = torch.stack(detached_losses)
        return loss_stack.sum()

    return _backward_entropy_loss


@dataclass(frozen=True)
class AdaMergingPostMerge:
    name: str = "adamerging"

    def run(self, context: PostMergeContext) -> PostMergeResult:
        if str(context.peft_subspace) != "full":
            raise ValueError("AdaMerging v1 supports only peft_subspace='full'.")

        cfg = dict(context.config)
        entropy_loss_fn = context.entropy_loss_fn
        backward_entropy_loss_fn = context.backward_entropy_loss_fn
        if entropy_loss_fn is None and backward_entropy_loss_fn is None and str(context.kind) == "vision":
            backward_entropy_loss_fn = _build_vision_backward_entropy_loss(context, cfg)
        if entropy_loss_fn is None and backward_entropy_loss_fn is None:
            raise ValueError("AdaMerging requires an entropy loss function in PostMergeContext.")

        alpha_mode = str(cfg.get("alpha_mode", "task")).strip().lower()
        if alpha_mode not in {"task", "layer"}:
            raise ValueError("postmerge.alpha_mode must be one of: task, layer")

        device = str(cfg.get("device", next(context.model.parameters()).device))
        steps = int(cfg.get("steps", 500))
        if steps < 0:
            raise ValueError("postmerge.steps must be >= 0.")
        lr = float(cfg.get("lr", 1e-3))
        if lr <= 0:
            raise ValueError("postmerge.lr must be > 0.")
        alpha_min = float(cfg.get("alpha_min", 0.0))
        alpha_max = float(cfg.get("alpha_max", 1.0))
        init_alpha = float(cfg.get("init_alpha", 0.3))
        beta1 = float(cfg.get("beta1", 0.9))
        beta2 = float(cfg.get("beta2", 0.999))
        weight_decay = float(cfg.get("weight_decay", 0.0))
        log_every = int(cfg.get("log_every", 25))

        bank = TaskDeltaBank.build(
            base=context.base,
            tuned=context.tuned,
            tasks=context.tasks,
            weights=context.weights,
            kind=context.kind,
        )
        raw_alpha = torch.nn.Parameter(
            init_clamped_raw_alpha(
                bank.alpha_shape(alpha_mode),
                init_alpha=init_alpha,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                device=device,
            )
        )
        optimizer = torch.optim.Adam(
            [raw_alpha],
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )

        history: list[dict[str, Any]] = []
        context.model.train(False)
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            alpha_values = clamped_alpha_from_raw(raw_alpha, alpha_min=alpha_min, alpha_max=alpha_max)
            if backward_entropy_loss_fn is not None:
                loss = backward_entropy_loss_fn(bank, alpha_values, alpha_mode)
            else:
                assert entropy_loss_fn is not None
                loss = entropy_loss_fn(bank, alpha_values, alpha_mode)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"AdaMerging entropy loss became non-finite at step {step}: {float(loss)}")
                loss.backward()
            if not torch.isfinite(loss):
                raise RuntimeError(f"AdaMerging entropy loss became non-finite at step {step}: {float(loss)}")
            optimizer.step()

            if log_every > 0 and ((step + 1) % log_every == 0 or step == 0 or step + 1 == steps):
                alpha_detached = clamped_alpha_from_raw(raw_alpha, alpha_min=alpha_min, alpha_max=alpha_max).detach()
                row = {
                    "step": int(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "alpha_mean": float(alpha_detached.mean().cpu()),
                    "alpha_min": float(alpha_detached.min().cpu()),
                    "alpha_max": float(alpha_detached.max().cpu()),
                }
                history.append(row)
                print(
                    "[adamerging] "
                    f"step={row['step']} loss={row['loss']:.6f} "
                    f"alpha_mean={row['alpha_mean']:.6f} "
                    f"alpha_range=[{row['alpha_min']:.6f}, {row['alpha_max']:.6f}]"
                )

        final_alpha = clamped_alpha_from_raw(raw_alpha, alpha_min=alpha_min, alpha_max=alpha_max).detach()
        merged_state = bank.materialize(final_alpha, mode=alpha_mode)
        metadata = {
            "method": self.name,
            "alpha_mode": alpha_mode,
            "steps": steps,
            "lr": lr,
            "betas": [beta1, beta2],
            "alpha_bounds": [alpha_min, alpha_max],
            "alpha_parameterization": "clamp",
            "init_alpha": init_alpha,
            "weight_decay": weight_decay,
            "alpha_shape": list(final_alpha.shape),
            "alpha_values": final_alpha.cpu().tolist(),
            "history": history,
            "delta_bank": bank.metadata(),
        }
        return PostMergeResult(merged_state=merged_state, metadata=metadata)


register(AdaMergingPostMerge())
