# algorithms/eap.py
#
# Architecture-agnostic Edge Attribution Patching over circuit.py-compatible
# models. GPT-specific edge packing and finalization stay behind CircuitModel.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
from tqdm import tqdm

from ..circuit import Circuit, edge_key
from ..metrics import discogp_fidelity_loss, eap_edge_importance_from_gradient
from ..models import CircuitModel, load_circuit_model
from ..utils import DEVICE

EAPLossFn = Callable[[dict[str, Any], torch.Tensor], torch.Tensor]


def _identity_mask(
    logits: torch.Tensor,
    *,
    reverse: bool = False,
    **_: Any,
) -> torch.Tensor:
    return 1.0 - logits if reverse else logits


@dataclass
class EAPConfig:
    model_name: str = "gpt2-small"
    absolute_scores: bool = True
    seed: int = 42
    tqdm_disabled: bool = False


@dataclass
class EAPState:
    base_circuit: Circuit
    edge_score_keys: list[list[edge_key]]
    edge_scores: list[torch.Tensor]

    def score_dict(self) -> dict[edge_key, torch.Tensor]:
        out: dict[edge_key, torch.Tensor] = {}
        for keys, scores in zip(self.edge_score_keys, self.edge_scores):
            flat_scores = scores.reshape(-1)
            for offset, key in enumerate(keys):
                out[key] = flat_scores[offset]
        return out

    def float_circuit(self) -> Circuit:
        out = self.base_circuit.full_like()
        scores = self.score_dict()
        for edge in out.all_edges():
            edge.edge_mask = scores[edge.key].detach().clone()
        return out

    def circuit_for_edge_budget(self, edge_density: float) -> Circuit:
        if not 0.0 <= edge_density <= 1.0:
            raise ValueError(f"edge_density must be in [0, 1], got {edge_density}.")

        score_items = list(self.score_dict().items())
        n_edges = len(score_items)
        n_keep = round(edge_density * n_edges)
        return self.circuit_for_top_k(n_keep)

    def circuit_for_top_k(self, k: int) -> Circuit:
        score_items = list(self.score_dict().items())
        n_edges = len(score_items)
        n_keep = int(k)

        if n_keep < 0:
            raise ValueError(f"k must be non-negative, got {k}.")

        if n_keep <= 0:
            kept: set[edge_key] = set()
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
class EAPResult:
    state: EAPState
    history: list[dict[str, float]] = field(default_factory=list)

    def float_circuit(self) -> Circuit:
        return self.state.float_circuit()

    def circuit_for_edge_budget(
        self,
        edge_density: float,
        *,
        model: CircuitModel | None = None,
        finalize: bool = True,
    ) -> Circuit:
        circuit = self.state.circuit_for_edge_budget(edge_density)
        if finalize:
            if model is None:
                raise ValueError(
                    "model is required when finalize=True because finalization "
                    "is architecture-specific."
                )
            return model.finalize_circuit(circuit)
        return circuit

    def circuit_for_top_k(
        self,
        k: int,
        *,
        model: CircuitModel | None = None,
        finalize: bool = True,
    ) -> Circuit:
        circuit = self.state.circuit_for_top_k(k)
        if finalize:
            if model is None:
                raise ValueError(
                    "model is required when finalize=True because finalization "
                    "is architecture-specific."
                )
            return model.finalize_circuit(circuit)
        return circuit


class EAPMasks(nn.Module):
    """
    Dense edge gates aligned with model-provided edge groups.

    The gates are initialized to one so gradients are measured around the full
    model. They are not optimized; EAP uses their gradients as first-order edge
    removal scores.
    """

    def __init__(
        self,
        model: CircuitModel,
        circuit: Circuit,
        *,
        device: str,
    ) -> None:
        super().__init__()
        self.edge_group_specs = list(model.edge_logit_group_specs(circuit))
        self.edge_score_keys: list[list[edge_key]] = [
            list(spec.keys)
            for spec in self.edge_group_specs
        ]
        self.edge_masks = nn.ParameterList(
            [
                nn.Parameter(torch.ones(tuple(spec.shape), device=device))
                for spec in self.edge_group_specs
            ]
        )

    def zero_grad(self) -> None:
        for mask in self.edge_masks:
            mask.grad = None


class EAP:
    """
    Edge Attribution Patching scorer for zero-ablation circuits.

    The algorithm is architecture-agnostic: it asks the model for grouped edge
    masks, differentiates a caller-provided loss wrt those masks, and maps the
    resulting scores back onto circuit.py edge keys.
    """

    def __init__(
        self,
        *,
        model: CircuitModel | None = None,
        config: EAPConfig | None = None,
        device: str = DEVICE,
    ) -> None:
        self.config = config if config is not None else EAPConfig()
        self.device_name = device
        self.model = (
            load_circuit_model(self.config.model_name, device=device)
            if model is None
            else model
        )
        self.model.eval()
        for _, parameter in self.model.named_parameters():
            parameter.requires_grad_(False)

        self.base_circuit = self.model.full_circuit
        self.masks = EAPMasks(
            self.model,
            self.base_circuit,
            device=device,
        ).to(device)
        self.history: list[dict[str, float]] = []

    def _runtime_masks(self) -> Any:
        return self.model.sample_runtime_masks(
            edge_logits=self.masks.edge_masks,
            edge_group_specs=self.masks.edge_group_specs,
            weight_logits=None,
            weight_group_specs=None,
            frozen_weight_runtime=None,
            sample_mask_fn=_identity_mask,
            boolean_mask_fn=None,
            mode="edge",
            reverse_edges=False,
            random_mode=None,
            gs_temp_edge=1.0,
        )

    def fit(
        self,
        dataloader,
        *,
        loss_fn: EAPLossFn = discogp_fidelity_loss,
    ) -> EAPResult:
        torch.manual_seed(self.config.seed)
        score_sums = [
            torch.zeros(tuple(spec.shape), device=self.device_name)
            for spec in self.masks.edge_group_specs
        ]
        n_batches = 0

        for batch in tqdm(
            dataloader,
            desc="EAP",
            dynamic_ncols=True,
            disable=self.config.tqdm_disabled,
        ):
            self.masks.zero_grad()
            logits = self.model(
                batch["input_ids"],
                runtime_masks=self._runtime_masks(),
            )
            loss = loss_fn(batch, logits)
            loss.backward()

            for index, mask in enumerate(self.masks.edge_masks):
                if mask.grad is None:
                    continue
                score_sums[index] = score_sums[index] + (
                    eap_edge_importance_from_gradient(
                        mask.grad,
                        absolute=self.config.absolute_scores,
                    )
                )

            n_batches += 1
            self.history.append(
                {
                    "batch": float(n_batches - 1),
                    "loss": float(loss.detach().cpu().item()),
                }
            )

        denom = max(n_batches, 1)
        scores = [
            (score_sum / denom).detach().cpu()
            for score_sum in score_sums
        ]

        return EAPResult(
            state=EAPState(
                base_circuit=self.base_circuit,
                edge_score_keys=self.masks.edge_score_keys,
                edge_scores=scores,
            ),
            history=self.history,
        )

    def discover_circuit(
        self,
        dataloader,
        *,
        edge_density: float,
        loss_fn: EAPLossFn = discogp_fidelity_loss,
        finalize: bool = True,
    ) -> Circuit:
        result = self.fit(dataloader, loss_fn=loss_fn)
        return result.circuit_for_edge_budget(
            edge_density,
            model=self.model,
            finalize=finalize,
        )
