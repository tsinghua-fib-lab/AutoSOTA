from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..base import PostMergeContext, PostMergeResult
from ..registry import register
from ..task_delta_bank import TaskDeltaBank
from ._vision_training import (
    VisionPostmergeTrainer,
    visual_params_from_trainable_deltas,
    visual_params_from_trainable_merged_delta,
)


def _optimizer_cfg(context: PostMergeContext, cfg: dict[str, Any]) -> tuple[str, int, float, tuple[float, float], float, int]:
    device = str(cfg.get("device", next(context.model.parameters()).device))
    steps = int(cfg.get("steps", 500))
    if steps < 0:
        raise ValueError("postmerge.steps must be >= 0.")
    lr = float(cfg.get("lr", 1e-3))
    if lr <= 0:
        raise ValueError("postmerge.lr must be > 0.")
    beta1 = float(cfg.get("beta1", 0.9))
    beta2 = float(cfg.get("beta2", 0.999))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    log_every = int(cfg.get("log_every", 25))
    return device, steps, lr, (beta1, beta2), weight_decay, log_every


def _history_row(step: int, loss: torch.Tensor) -> dict[str, Any]:
    return {
        "step": int(step),
        "loss": float(loss.detach().cpu()),
    }


@dataclass(frozen=True)
class TaskVectorFinetunePostMerge:
    name: str = "task_vector_finetune"

    def run(self, context: PostMergeContext) -> PostMergeResult:
        if str(context.peft_subspace) != "full":
            raise ValueError("task_vector_finetune supports only peft_subspace='full'.")
        if str(context.kind) != "vision":
            raise ValueError("task_vector_finetune currently supports only vision postmerge.")

        cfg = dict(context.config)
        device, steps, lr, betas, weight_decay, log_every = _optimizer_cfg(context, cfg)
        trainer = VisionPostmergeTrainer(context, cfg)
        bank = TaskDeltaBank.build(
            base=context.base,
            tuned=context.tuned,
            tasks=context.tasks,
            weights=context.weights,
            kind=context.kind,
        )
        trainable_deltas = bank.trainable_delta_parameters(device=device)
        params = [param for task_deltas in trainable_deltas for param in task_deltas.values()]
        optimizer = torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)

        history: list[dict[str, Any]] = []
        context.model.train(False)
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)

            def _logits(item: dict[str, Any], images: torch.Tensor) -> torch.Tensor:
                visual_params = visual_params_from_trainable_deltas(
                    bank=bank,
                    model=trainer.clf.model,
                    trainable_deltas=trainable_deltas,
                    device=device,
                )
                return trainer.visual_logits_from_params(item, visual_params, images)

            loss = trainer.backward_summed_loss(_logits)
            if not torch.isfinite(loss):
                raise RuntimeError(f"task_vector_finetune loss became non-finite at step {step}: {float(loss)}")
            optimizer.step()
            if log_every > 0 and ((step + 1) % log_every == 0 or step == 0 or step + 1 == steps):
                row = _history_row(step + 1, loss)
                history.append(row)
                print(f"[task_vector_finetune] step={row['step']} loss={row['loss']:.6f}")

        merged_state = bank.materialize_trainable_deltas(trainable_deltas)
        metadata = {
            "method": self.name,
            "steps": steps,
            "lr": lr,
            "betas": list(betas),
            "loss": trainer.loss_kind,
            "weight_decay": weight_decay,
            "num_trainable_tensors": len(params),
            "history": history,
            "delta_bank": bank.metadata(),
        }
        return PostMergeResult(merged_state=merged_state, metadata=metadata)


@dataclass(frozen=True)
class MergedDeltaFinetunePostMerge:
    name: str = "merged_delta_finetune"

    def run(self, context: PostMergeContext) -> PostMergeResult:
        if str(context.peft_subspace) != "full":
            raise ValueError("merged_delta_finetune supports only peft_subspace='full'.")
        if str(context.kind) != "vision":
            raise ValueError("merged_delta_finetune currently supports only vision postmerge.")

        cfg = dict(context.config)
        device, steps, lr, betas, weight_decay, log_every = _optimizer_cfg(context, cfg)
        trainer = VisionPostmergeTrainer(context, cfg)
        bank = TaskDeltaBank.build(
            base=context.base,
            tuned=context.tuned,
            tasks=context.tasks,
            weights=context.weights,
            kind=context.kind,
        )
        trainable_keys = [
            f"visual.{name}"
            for name, _param in trainer.clf.model.visual.named_parameters()
            if f"visual.{name}" in bank.layer_for_key
        ]
        if not trainable_keys:
            raise ValueError("merged_delta_finetune found no visual tensors to train.")
        trainable_merged_delta = bank.trainable_merged_delta_parameters(device=device, trainable_keys=trainable_keys)
        params = [value for value in trainable_merged_delta.values() if isinstance(value, torch.nn.Parameter)]
        optimizer = torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)

        history: list[dict[str, Any]] = []
        context.model.train(False)
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)

            def _logits(item: dict[str, Any], images: torch.Tensor) -> torch.Tensor:
                visual_params = visual_params_from_trainable_merged_delta(
                    bank=bank,
                    model=trainer.clf.model,
                    trainable_merged_delta=trainable_merged_delta,
                    device=device,
                )
                return trainer.visual_logits_from_params(item, visual_params, images)

            loss = trainer.backward_summed_loss(_logits)
            if not torch.isfinite(loss):
                raise RuntimeError(f"merged_delta_finetune loss became non-finite at step {step}: {float(loss)}")
            optimizer.step()
            if log_every > 0 and ((step + 1) % log_every == 0 or step == 0 or step + 1 == steps):
                row = _history_row(step + 1, loss)
                history.append(row)
                print(f"[merged_delta_finetune] step={row['step']} loss={row['loss']:.6f}")

        merged_state = bank.materialize_trainable_merged_delta(trainable_merged_delta)
        metadata = {
            "method": self.name,
            "steps": steps,
            "lr": lr,
            "betas": list(betas),
            "loss": trainer.loss_kind,
            "weight_decay": weight_decay,
            "trainable_tensor_keys": trainable_keys,
            "num_trainable_tensors": len(params),
            "history": history,
            "delta_bank": bank.metadata(),
        }
        return PostMergeResult(merged_state=merged_state, metadata=metadata)


@dataclass(frozen=True)
class VisionHeadProbePostMerge:
    name: str = "vision_head_probe"

    def run(self, context: PostMergeContext) -> PostMergeResult:
        if str(context.peft_subspace) != "full":
            raise ValueError("vision_head_probe supports only peft_subspace='full'.")
        if str(context.kind) != "vision":
            raise ValueError("vision_head_probe currently supports only vision postmerge.")

        cfg = dict(context.config)
        device, steps, lr, betas, weight_decay, log_every = _optimizer_cfg(context, cfg)
        init_alpha = float(cfg.get("init_alpha", 0.3))
        trainable_prefixes = ("visual.proj", "visual.ln_post")
        bank = TaskDeltaBank.build(
            base=context.base,
            tuned=context.tuned,
            tasks=context.tasks,
            weights=context.weights,
            kind=context.kind,
        )
        trainer = VisionPostmergeTrainer(context, cfg)
        selected_keys = [
            key
            for key in bank.tensor_keys
            if any(key == prefix or key.startswith(f"{prefix}.") for prefix in trainable_prefixes)
        ]
        if not selected_keys:
            raise ValueError(
                "vision_head_probe found no final vision-head task-vector tensors "
                f"(expected prefixes: {list(trainable_prefixes)})."
            )

        delta_params: list[dict[str, torch.Tensor]] = []
        trainable: list[torch.nn.Parameter] = []
        for deltas in bank.deltas_by_task:
            task_delta: dict[str, torch.Tensor] = {}
            for key in bank.tensor_keys:
                value = init_alpha * deltas[key].detach().clone().to(device=device)
                if key in selected_keys:
                    param = torch.nn.Parameter(value)
                    task_delta[key] = param
                    trainable.append(param)
                else:
                    task_delta[key] = value
            delta_params.append(task_delta)
        optimizer = torch.optim.Adam(trainable, lr=lr, betas=betas, weight_decay=weight_decay)

        history: list[dict[str, Any]] = []
        context.model.train(False)
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)

            def _logits(item: dict[str, Any], images: torch.Tensor) -> torch.Tensor:
                visual_params = visual_params_from_trainable_deltas(
                    bank=bank,
                    model=trainer.clf.model,
                    trainable_deltas=delta_params,
                    device=device,
                )
                return trainer.visual_logits_from_params(item, visual_params, images)

            loss = trainer.backward_summed_loss(_logits)
            if not torch.isfinite(loss):
                raise RuntimeError(f"vision_head_probe loss became non-finite at step {step}: {float(loss)}")
            optimizer.step()
            if log_every > 0 and ((step + 1) % log_every == 0 or step == 0 or step + 1 == steps):
                row = _history_row(step + 1, loss)
                history.append(row)
                print(f"[vision_head_probe] step={row['step']} loss={row['loss']:.6f}")
        final_state = bank.materialize_trainable_deltas(delta_params)

        metadata = {
            "method": self.name,
            "steps": steps,
            "lr": lr,
            "betas": list(betas),
            "loss": trainer.loss_kind,
            "init_alpha": init_alpha,
            "weight_decay": weight_decay,
            "trainable_prefixes": list(trainable_prefixes),
            "trainable_tensor_keys": selected_keys,
            "num_trainable_tensors": len(trainable),
            "history": history,
            "delta_bank": bank.metadata(),
        }
        return PostMergeResult(merged_state=final_state, metadata=metadata)


register(TaskVectorFinetunePostMerge())
register(MergedDeltaFinetunePostMerge())
register(VisionHeadProbePostMerge())
