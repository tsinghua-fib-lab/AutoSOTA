# algorithms/edge_pruning.py
#
# Architecture-agnostic Edge-Pruning-style optimizer for circuit.py-compatible
# models. Architecture-specific packing stays behind CircuitModel hooks.

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
from tqdm import tqdm

from ..circuit import Circuit, edge_key
from ..metrics import (
    edge_pruning_edge_scores,
    edge_pruning_expected_sparsity,
    edge_pruning_full_vocab_kl_loss,
    edge_pruning_lagrangian_loss,
    edge_pruning_linear_sparsity_target,
    edge_pruning_sample_mask,
    edge_pruning_two_label_loss,
)
from ..models import CircuitModel, load_circuit_model
from ..utils import DEVICE

ObjectiveName = Literal["kl_full_vocab", "two_label", "kl_plus_two_label"]
AblationName = Literal["zero"]

EdgePruningLossFn = Callable[
    [dict[str, Any], torch.Tensor, torch.Tensor | None],
    torch.Tensor,
]


@dataclass
class EdgePruningConfig:
    model_name: str = "gpt2-small"

    # training
    n_epochs: int = 40
    max_steps: int | None = None
    batch_size: int = 32
    edge_learning_rate: float = 1e-2
    reg_edge_learning_rate: float = 1e-2
    lr_warmup_steps: int = 0

    # hard-concrete edge log-alpha initialization
    edge_logit_init_mean: float = 10.0
    edge_logit_init_std: float = 0.01

    # Edge-Pruning Lagrangian sparsity target. This is optional because the
    # trained float scores can also be thresholded after training.
    start_edge_sparsity: float = 0.0
    target_edge_sparsity: float | None = 0.97
    n_sparsity_warmup_steps: int = 0

    # task objective
    objective: ObjectiveName = "kl_full_vocab"
    kl_weight: float = 1.0
    two_label_weight: float = 1.0

    # currently supported ablation path. In this codebase, runtime masks
    # implement zero ablation by multiplying absent edge contributions by zero.
    ablation: AblationName = "zero"

    # Cache teacher logits for KL objectives. CPU float16 is the default because
    # full-vocab teacher vectors are large and do not need to occupy VRAM across
    # epochs. They are moved back and cast to student dtype for the current loss.
    cache_teacher_logits: bool = True
    teacher_cache_dtype: torch.dtype = torch.float16

    seed: int = 42
    tqdm_disabled: bool = False


@dataclass
class EdgePruningState:
    """
    Float-valued learned edge state independent of any final sparsity budget.
    """

    base_circuit: Circuit
    edge_logit_keys: list[list[edge_key]]
    edge_scores: list[torch.Tensor]

    def score_dict(self) -> dict[edge_key, torch.Tensor]:
        out: dict[edge_key, torch.Tensor] = {}
        for keys, scores in zip(self.edge_logit_keys, self.edge_scores):
            flat_scores = scores.reshape(-1)
            for offset, key in enumerate(keys):
                out[key] = flat_scores[offset]
        return out

    def float_circuit(self) -> Circuit:
        """
        Circuit with float edge masks. This is an intermediate state for
        inspection/scoring, not a finalized boolean circuit.
        """
        out = self.base_circuit.full_like()
        scores = self.score_dict()
        for edge in out.all_edges():
            edge.edge_mask = scores[edge.key].detach().clone()
        return out

    def circuit_for_sparsity_budget(
        self,
        edge_sparsity_budget: float,
    ) -> Circuit:
        """
        Convert scores to a boolean circuit by keeping the top
        (1 - edge_sparsity_budget) fraction of edges.
        """
        if not 0.0 <= edge_sparsity_budget <= 1.0:
            raise ValueError(
                "edge_sparsity_budget must be in [0.0, 1.0], "
                f"got {edge_sparsity_budget}."
            )

        score_items = list(self.score_dict().items())
        n_edges = len(score_items)
        n_keep = round((1.0 - edge_sparsity_budget) * n_edges)

        kept: set[edge_key]
        if n_keep <= 0:
            kept = set()
        elif n_keep >= n_edges:
            kept = {key for key, _ in score_items}
        else:
            flat_scores = torch.stack(
                [score.detach().reshape(()) for _, score in score_items]
            )
            top_indices = torch.topk(flat_scores, k=n_keep, largest=True).indices
            kept = {score_items[int(index)][0] for index in top_indices}

        out = self.base_circuit.full_like()
        device = self.edge_scores[0].device if self.edge_scores else DEVICE
        for edge in out.all_edges():
            edge.edge_mask = torch.tensor(edge.key in kept, device=device)
        return out


@dataclass
class EdgePruningResult:
    state: EdgePruningState
    history: list[dict[str, float]] = field(default_factory=list)

    def float_circuit(self) -> Circuit:
        return self.state.float_circuit()

    def circuit_for_sparsity_budget(
        self,
        edge_sparsity_budget: float,
        *,
        model: CircuitModel | None = None,
        finalize: bool = True,
    ) -> Circuit:
        circuit = self.state.circuit_for_sparsity_budget(edge_sparsity_budget)
        if finalize:
            if model is None:
                raise ValueError(
                    "model is required when finalize=True because finalization "
                    "is architecture-specific."
                )
            return model.finalize_circuit(circuit)
        return circuit


class EdgePruningMasks(nn.Module):
    """
    Trainable hard-concrete edge logits grouped by model-provided edge specs.
    """

    def __init__(
        self,
        model: CircuitModel,
        circuit: Circuit,
        *,
        init_mean: float,
        init_std: float,
        device: str,
    ) -> None:
        super().__init__()
        self.circuit = circuit
        self.edge_group_specs = list(model.edge_logit_group_specs(circuit))
        self.edge_logit_keys: list[list[edge_key]] = [
            list(spec.keys)
            for spec in self.edge_group_specs
        ]
        self.edge_logits = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(tuple(spec.shape), device=device).normal_(
                        init_mean,
                        init_std,
                    )
                )
                for spec in self.edge_group_specs
            ]
        )

    def parameters_for_optimizer(self) -> list[nn.Parameter]:
        return list(self.edge_logits.parameters())

    def edge_scores(self) -> list[torch.Tensor]:
        return [
            edge_pruning_edge_scores(logits).detach().cpu()
            for logits in self.edge_logits
        ]

    def state(self) -> EdgePruningState:
        return EdgePruningState(
            base_circuit=self.circuit,
            edge_logit_keys=self.edge_logit_keys,
            edge_scores=self.edge_scores(),
        )


class EdgePruningTeacherCache:
    """
    CPU cache of full-vocab teacher logits at KL-selected positions.

    The cache key is content-based, so it works even when the dataloader
    shuffles examples into different batches across epochs.
    """

    def __init__(
        self,
        *,
        model: CircuitModel,
        dtype: torch.dtype = torch.float16,
        storage_device: str = "cpu",
    ) -> None:
        self.model = model
        self.dtype = dtype
        self.storage_device = torch.device(storage_device)
        self.cache: dict[bytes, torch.Tensor] = {}
        self.hits = 0
        self.misses = 0

    def _selected_positions(
        self,
        batch: dict[str, Any],
        *,
        n_pos: int,
    ) -> list[torch.Tensor]:
        input_ids = batch["input_ids"]
        batch_size = input_ids.shape[0]

        if "loss_mask" in batch:
            mask = batch["loss_mask"].detach().to(device="cpu", dtype=torch.bool)
            return [
                mask[row].nonzero(as_tuple=False).flatten()
                for row in range(batch_size)
            ]

        if "seq_lens" in batch:
            seq_lens = batch["seq_lens"].detach().to(device="cpu")
            return [
                torch.tensor([int(seq_lens[row].item()) - 1], dtype=torch.long)
                for row in range(batch_size)
            ]

        all_positions = torch.arange(n_pos, dtype=torch.long)
        return [all_positions for _ in range(batch_size)]

    def _key(self, input_ids: torch.Tensor, positions: torch.Tensor) -> bytes:
        input_cpu = input_ids.detach().to(device="cpu", dtype=torch.long).contiguous()
        pos_cpu = positions.detach().to(device="cpu", dtype=torch.long).contiguous()
        digest = hashlib.blake2b(digest_size=16)
        digest.update(input_cpu.numpy().tobytes())
        digest.update(pos_cpu.numpy().tobytes())
        return digest.digest()

    def logits_for_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        input_ids = batch["input_ids"]
        positions_by_row = self._selected_positions(
            batch,
            n_pos=input_ids.shape[1],
        )
        keys = [
            self._key(input_ids[row], positions)
            for row, positions in enumerate(positions_by_row)
        ]

        missing_rows = [
            row
            for row, key in enumerate(keys)
            if key not in self.cache
        ]
        self.hits += len(keys) - len(missing_rows)
        self.misses += len(missing_rows)

        if missing_rows:
            missing_index = torch.tensor(
                missing_rows,
                device=input_ids.device,
                dtype=torch.long,
            )
            with torch.no_grad():
                teacher_logits = self.model(input_ids.index_select(0, missing_index))

            for local_row, row in enumerate(missing_rows):
                positions = positions_by_row[row].to(
                    device=teacher_logits.device,
                    dtype=torch.long,
                )
                selected = teacher_logits[local_row].index_select(0, positions)
                self.cache[keys[row]] = selected.detach().to(
                    device=self.storage_device,
                    dtype=self.dtype,
                    copy=True,
                )

        pieces = [self.cache[key] for key in keys]
        if not pieces:
            return torch.empty(
                (0, 0),
                device=self.storage_device,
                dtype=self.dtype,
            )
        return torch.cat(pieces, dim=0)

    def stats(self) -> dict[str, float]:
        return {
            "teacher_cache_entries": float(len(self.cache)),
            "teacher_cache_hits": float(self.hits),
            "teacher_cache_misses": float(self.misses),
        }


class EdgePruning:
    """
    Edge-Pruning-style optimizer.

    The implementation is deliberately architecture-agnostic: it owns only edge
    log-alpha tensors and asks the model to pack them into runtime masks.
    """

    def __init__(
        self,
        *,
        model: CircuitModel | None = None,
        config: EdgePruningConfig | None = None,
        device: str = DEVICE,
    ) -> None:
        self.config = config if config is not None else EdgePruningConfig()
        self.device_name = device

        if self.config.ablation != "zero":
            raise ValueError("only zero ablation is currently supported.")

        self.model = (
            load_circuit_model(self.config.model_name, device=device)
            if model is None
            else model
        )
        self.model.eval()
        for _, parameter in self.model.named_parameters():
            parameter.requires_grad_(False)

        self.base_circuit = self.model.full_circuit
        self.masks = EdgePruningMasks(
            self.model,
            self.base_circuit,
            init_mean=self.config.edge_logit_init_mean,
            init_std=self.config.edge_logit_init_std,
            device=device,
        ).to(device)
        self.sparsity_lambda_1 = nn.Parameter(torch.zeros((), device=device))
        self.sparsity_lambda_2 = nn.Parameter(torch.zeros((), device=device))
        self.history: list[dict[str, float]] = []
        self.teacher_cache = (
            EdgePruningTeacherCache(
                model=self.model,
                dtype=self.config.teacher_cache_dtype,
                storage_device="cpu",
            )
            if self.config.cache_teacher_logits
            else None
        )

    def _optimizer(self) -> torch.optim.Optimizer:
        groups: list[dict[str, Any]] = [
            {
                "params": self.masks.parameters_for_optimizer(),
                "lr": self.config.edge_learning_rate,
            }
        ]

        if self.config.target_edge_sparsity is not None:
            groups.append(
                {
                    "params": [self.sparsity_lambda_1, self.sparsity_lambda_2],
                    "lr": self.config.reg_edge_learning_rate,
                    "maximize": True,
                }
            )

        return torch.optim.AdamW(
            groups,
            lr=self.config.edge_learning_rate,
            fused=torch.cuda.is_available(),
        )

    def _scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        max_steps: int | None,
    ) -> torch.optim.lr_scheduler.LambdaLR | None:
        if max_steps is None:
            return None

        warmup_steps = self.config.lr_warmup_steps

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            return max(
                0.0,
                float(max_steps - step) / max(1, max_steps - warmup_steps),
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _runtime_masks(self) -> Any:
        return self.model.sample_runtime_masks(
            edge_logits=self.masks.edge_logits,
            edge_group_specs=self.masks.edge_group_specs,
            weight_logits=None,
            weight_group_specs=None,
            frozen_weight_runtime=None,
            sample_mask_fn=edge_pruning_sample_mask,
            boolean_mask_fn=None,
            mode="edge",
            reverse_edges=False,
            random_mode=None,
            gs_temp_edge=1.0,
        )

    def _teacher_logits(
        self,
        batch: dict[str, Any],
        *,
        use_cache: bool,
    ) -> torch.Tensor | None:
        if self.config.objective == "two_label":
            return None

        if use_cache and self.teacher_cache is not None:
            return self.teacher_cache.logits_for_batch(batch)

        with torch.no_grad():
            return self.model(batch["input_ids"])

    def _task_loss(
        self,
        batch: dict[str, Any],
        logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        loss_fn: EdgePruningLossFn | None,
    ) -> torch.Tensor:
        if loss_fn is not None:
            return loss_fn(batch, logits, teacher_logits)

        if self.config.objective == "kl_full_vocab":
            if teacher_logits is None:
                raise RuntimeError("teacher_logits are required for KL loss.")
            return edge_pruning_full_vocab_kl_loss(batch, logits, teacher_logits)

        if self.config.objective == "two_label":
            return edge_pruning_two_label_loss(batch, logits, teacher_logits)

        if teacher_logits is None:
            raise RuntimeError("teacher_logits are required for KL loss.")
        return (
            self.config.kl_weight
            * edge_pruning_full_vocab_kl_loss(batch, logits, teacher_logits)
            + self.config.two_label_weight
            * edge_pruning_two_label_loss(batch, logits, teacher_logits)
        )

    def _target_sparsity(self, global_step: int) -> float | None:
        if self.config.target_edge_sparsity is None:
            return None

        return edge_pruning_linear_sparsity_target(
            global_step,
            start_edge_sparsity=self.config.start_edge_sparsity,
            target_edge_sparsity=self.config.target_edge_sparsity,
            warmup_steps=self.config.n_sparsity_warmup_steps,
        )

    def fit(
        self,
        train_dataloader,
        *,
        loss_fn: EdgePruningLossFn | None = None,
    ) -> EdgePruningResult:
        torch.manual_seed(self.config.seed)
        optimizer = self._optimizer()
        max_steps = self.config.max_steps
        scheduler = self._scheduler(optimizer, max_steps=max_steps)
        global_step = 0

        epoch_iter = tqdm(
            range(self.config.n_epochs),
            desc="Edge-Pruning",
            dynamic_ncols=True,
            disable=self.config.tqdm_disabled,
        )

        for epoch in epoch_iter:
            last_log: dict[str, float] = {}

            for batch in train_dataloader:
                if max_steps is not None and global_step >= max_steps:
                    break

                optimizer.zero_grad(set_to_none=True)

                teacher_logits = self._teacher_logits(
                    batch,
                    use_cache=loss_fn is None,
                )
                logits = self.model(
                    batch["input_ids"],
                    runtime_masks=self._runtime_masks(),
                )
                task_loss = self._task_loss(
                    batch,
                    logits,
                    teacher_logits,
                    loss_fn,
                )

                expected_sparsity = edge_pruning_expected_sparsity(
                    self.masks.edge_logits,
                    device=self.device_name,
                )
                target_sparsity = self._target_sparsity(global_step)

                if target_sparsity is None:
                    reg_loss = logits.new_tensor(0.0)
                    loss = task_loss
                else:
                    reg_loss = edge_pruning_lagrangian_loss(
                        model_edge_sparsity=expected_sparsity,
                        target_edge_sparsity=target_sparsity,
                        lambda_1=self.sparsity_lambda_1,
                        lambda_2=self.sparsity_lambda_2,
                    )
                    loss = task_loss + reg_loss

                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                last_log = {
                    "epoch": float(epoch),
                    "step": float(global_step),
                    "loss": float(loss.detach().cpu().item()),
                    "task_loss": float(task_loss.detach().cpu().item()),
                    "reg_loss": float(reg_loss.detach().cpu().item()),
                    "expected_edge_sparsity": float(
                        expected_sparsity.detach().cpu().item()
                    ),
                    "target_edge_sparsity": (
                        float("nan")
                        if target_sparsity is None
                        else float(target_sparsity)
                    ),
                }
                if self.teacher_cache is not None:
                    last_log.update(self.teacher_cache.stats())
                global_step += 1

            self.history.append(last_log)
            if max_steps is not None and global_step >= max_steps:
                break

        return EdgePruningResult(
            state=self.masks.state(),
            history=self.history,
        )

    def discover_circuit(
        self,
        train_dataloader,
        *,
        edge_sparsity_budget: float,
        loss_fn: EdgePruningLossFn | None = None,
        finalize: bool = True,
    ) -> Circuit:
        """
        Convenience wrapper for callers that already know the desired final
        sparsity budget. Use fit() when you want the float-valued intermediate
        state and will choose the budget later.
        """
        result = self.fit(train_dataloader, loss_fn=loss_fn)
        return result.circuit_for_sparsity_budget(
            edge_sparsity_budget,
            model=self.model,
            finalize=finalize,
        )
