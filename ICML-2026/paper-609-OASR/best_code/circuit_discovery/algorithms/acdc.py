# algorithms/acdc.py
#
# Architecture-agnostic ACDC-style greedy zero-ablation pruning for
# circuit.py-compatible models. Model-specific circuit finalization stays behind
# CircuitModel.finalize_circuit.

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Iterable
import random

import torch
from tqdm import tqdm

from ..circuit import Circuit, edge_key, node_key
from ..metrics import acdc_loss_delta, discogp_fidelity_loss
from ..models import CircuitModel, load_circuit_model
from ..utils import DEVICE

ACDCLossFn = Callable[[dict[str, Any], torch.Tensor], torch.Tensor]
ACDCThresholdCallback = Callable[[float, "ACDCResult"], None]


@dataclass(init=False)
class ACDCConfig:
    model_name: str = "gpt2-small"

    # Reference ACDC sweeps several tau values and assigns every pruned edge the
    # smallest tau at which it can be removed. These defaults match the same
    # grid shape as auto-circuit while leaving callers free to pass explicit
    # thresholds.
    tao_exps: tuple[int, ...] = (-5, -4, -3, -2)
    tao_bases: tuple[int, ...] = (1, 3, 5, 7, 9)
    thresholds: tuple[float, ...] | None = None

    # Original ACDC uses one batch. Larger values are supported but expensive
    # because every edge-removal candidate re-evaluates the loss.
    max_batches: int = 1
    optimized_for_acdc: bool = False
    edge_ordering: str = "fixed"

    seed: int = 42
    tqdm_disabled: bool = False

    def __init__(
        self,
        model_name: str = "gpt2-small",
        tao_exps: tuple[int, ...] = (-5, -4, -3, -2),
        tao_bases: tuple[int, ...] = (1, 3, 5, 7, 9),
        thresholds: tuple[float, ...] | None = None,
        max_batches: int = 1,
        optimized_for_acdc: bool = False,
        edge_ordering: str = "fixed",
        seed: int = 42,
        tqdm_disabled: bool = False,
        gpt_speedup: bool | None = None,
    ) -> None:
        if gpt_speedup is not None:
            optimized_for_acdc = gpt_speedup

        self.model_name = model_name
        self.tao_exps = tao_exps
        self.tao_bases = tao_bases
        self.thresholds = thresholds
        self.max_batches = max_batches
        self.optimized_for_acdc = optimized_for_acdc
        self.edge_ordering = edge_ordering
        self.seed = seed
        self.tqdm_disabled = tqdm_disabled

    def threshold_values(self) -> list[float]:
        if self.thresholds is not None:
            return sorted(float(value) for value in self.thresholds)
        return sorted(
            float(base * 10**exp)
            for base, exp in product(self.tao_bases, self.tao_exps)
        )


@dataclass
class ACDCState:
    base_circuit: Circuit
    edge_scores: dict[edge_key, float]

    def score_dict(self) -> dict[edge_key, torch.Tensor]:
        return {
            key: torch.tensor(score, dtype=torch.float32)
            for key, score in self.edge_scores.items()
        }

    def float_circuit(self) -> Circuit:
        out = self.base_circuit.full_like()
        for edge in out.all_edges():
            edge.edge_mask = torch.tensor(
                self.edge_scores[edge.key],
                dtype=torch.float32,
            )
        return out

    def circuit_for_threshold(self, threshold: float) -> Circuit:
        """
        Keep edges not removed at or below this ACDC threshold.
        """
        out = self.base_circuit.full_like()
        for edge in out.all_edges():
            keep = self.edge_scores[edge.key] > threshold
            edge.edge_mask = torch.tensor(keep, dtype=torch.bool, device=DEVICE)
        return out

    def circuit_for_edge_budget(self, edge_density: float) -> Circuit:
        if not 0.0 <= edge_density <= 1.0:
            raise ValueError(f"edge_density must be in [0, 1], got {edge_density}.")

        score_items = list(self.edge_scores.items())
        n_edges = len(score_items)
        n_keep = round(edge_density * n_edges)

        if n_keep <= 0:
            kept: set[edge_key] = set()
        elif n_keep >= n_edges:
            kept = {key for key, _ in score_items}
        else:
            flat_scores = torch.tensor(
                [score for _, score in score_items],
                dtype=torch.float32,
            )
            top_indices = torch.topk(flat_scores, k=n_keep, largest=True).indices
            kept = {score_items[int(index)][0] for index in top_indices}

        out = self.base_circuit.full_like()
        for edge in out.all_edges():
            edge.edge_mask = torch.tensor(edge.key in kept, dtype=torch.bool)
        return out


@dataclass
class ACDCResult:
    state: ACDCState
    history: list[dict[str, float]] = field(default_factory=list)

    def float_circuit(self) -> Circuit:
        return self.state.float_circuit()

    def circuit_for_threshold(
        self,
        threshold: float,
        *,
        model: CircuitModel | None = None,
        finalize: bool = True,
    ) -> Circuit:
        circuit = self.state.circuit_for_threshold(threshold)
        if finalize:
            if model is None:
                raise ValueError(
                    "model is required when finalize=True because finalization "
                    "is architecture-specific."
                )
            return model.finalize_circuit(circuit)
        return circuit

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


class ACDC:
    """
    Greedy ACDC-style zero-ablation circuit discovery.

    The algorithm starts from the full circuit for each tau value, tries
    removing edges in reverse layer order, and leaves a removal in place if the
    task-loss increase is below tau. Edges are scored by the smallest tau at
    which they could be removed; edges never removed receive +inf.
    """

    def __init__(
        self,
        *,
        model: CircuitModel | None = None,
        config: ACDCConfig | None = None,
        device: str = DEVICE,
    ) -> None:
        self.config = config if config is not None else ACDCConfig()
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
        self.history: list[dict[str, float]] = []

    def _edge_order(self, circuit: Circuit) -> list[edge_key]:
        ordered = sorted(
            circuit.edge_keys(),
            key=self._edge_sort_key,
            reverse=True,
        )
        if self.config.edge_ordering == "fixed":
            return ordered
        if self.config.edge_ordering == "random_per_layer":
            return self._shuffle_edge_order_within_parallel_nodes(ordered)
        raise ValueError(
            "ACDCConfig.edge_ordering must be one of "
            f"{('fixed', 'random_per_layer')}, got {self.config.edge_ordering!r}."
        )

    @staticmethod
    def _receiver_stage_rank(kind: str) -> int:
        return {
            "output": 4,
            "mlp": 3,
            "attn_q": 2,
            "attn_k": 2,
            "attn_v": 2,
        }.get(kind, 0)

    @staticmethod
    def _sender_stage_rank(kind: str) -> int:
        return {
            "mlp": 2,
            "attn_o": 1,
            "emb": 0,
        }.get(kind, 0)

    def _edge_sort_key(self, key: edge_key) -> tuple[int, int, int, str, int, int, int, str]:
        dst, src = key
        return (
            dst[0],
            self._receiver_stage_rank(dst[2]),
            dst[1],
            dst[2],
            src[0],
            self._sender_stage_rank(src[2]),
            src[1],
            src[2],
        )

    def _shuffle_edge_order_within_parallel_nodes(
        self,
        ordered_edges: list[edge_key],
    ) -> list[edge_key]:
        """
        Preserve reverse-topological receiver and sender stages while
        randomizing only where nodes are parallel.

        In GPT-style blocks this keeps output before MLP and MLP before
        attention-input receivers inside a layer. For each receiver node, parent
        senders are still ordered from later-layer MLPs/heads to earlier-layer
        MLPs/heads, with MLP senders before head senders in the same layer.
        Randomness only reorders nodes inside the same topological stage, such
        as same-layer heads.
        """
        rng = random.Random(self.config.seed)
        grouped: dict[tuple[int, int], dict[node_key, list[edge_key]]] = {}
        stage_order: list[tuple[int, int]] = []
        node_order_by_stage: dict[tuple[int, int], list[node_key]] = {}

        for key in ordered_edges:
            dst = key[0]
            dst_layer = dst[0]
            dst_kind = dst[2]
            if dst_kind == "output":
                stage_rank = 4
            elif dst_kind == "mlp":
                stage_rank = 3
            elif dst_kind in {"attn_q", "attn_k", "attn_v"}:
                stage_rank = 2
            else:
                stage_rank = 0

            stage = (dst_layer, stage_rank)
            if stage not in grouped:
                grouped[stage] = {}
                stage_order.append(stage)
                node_order_by_stage[stage] = []
            if dst not in grouped[stage]:
                grouped[stage][dst] = []
                node_order_by_stage[stage].append(dst)
            grouped[stage][dst].append(key)

        shuffled: list[edge_key] = []
        for stage in stage_order:
            stage_nodes = list(node_order_by_stage[stage])
            rng.shuffle(stage_nodes)
            for dst in stage_nodes:
                shuffled.extend(
                    self._shuffle_parent_edges_within_parallel_sender_stages(
                        grouped[stage][dst],
                        rng,
                    )
                )
        return shuffled

    def _shuffle_parent_edges_within_parallel_sender_stages(
        self,
        ordered_edges: list[edge_key],
        rng: random.Random,
    ) -> list[edge_key]:
        grouped: dict[tuple[int, int], list[edge_key]] = {}
        stage_order: list[tuple[int, int]] = []

        for key in ordered_edges:
            src = key[1]
            stage = (src[0], self._sender_stage_rank(src[2]))
            if stage not in grouped:
                grouped[stage] = []
                stage_order.append(stage)
            grouped[stage].append(key)

        shuffled: list[edge_key] = []
        for stage in stage_order:
            stage_edges = list(grouped[stage])
            rng.shuffle(stage_edges)
            shuffled.extend(stage_edges)
        return shuffled

    def _batches(self, dataloader) -> list[dict[str, Any]]:
        batches = []
        for index, batch in enumerate(dataloader):
            if index >= self.config.max_batches:
                break
            batches.append(batch)
        if not batches:
            raise ValueError("ACDC requires at least one batch.")
        return batches

    @torch.inference_mode()
    def _average_loss(
        self,
        circuit: Circuit,
        batches: Iterable[dict[str, Any]],
        loss_fn: ACDCLossFn,
    ) -> torch.Tensor:
        total: torch.Tensor | None = None
        n_batches = 0

        for batch in batches:
            logits = self.model(batch["input_ids"], circuit=circuit)
            loss = loss_fn(batch, logits).detach()
            total = loss if total is None else total + loss
            n_batches += 1

        if total is None:
            return torch.zeros((), device=self.device_name)

        return total / max(n_batches, 1)

    def _per_layer_cache_methods(self):
        if not self.config.optimized_for_acdc:
            return None

        per_layer_cache = getattr(self.model, "per_layer_cache", None)
        forward_from_cache = getattr(self.model, "forward_from_per_layer_cache", None)
        if callable(per_layer_cache) and callable(forward_from_cache):
            return per_layer_cache, forward_from_cache

        raise AttributeError(
            "ACDCConfig.optimized_for_acdc=True, but this model does not expose "
            "per_layer_cache / forward_from_per_layer_cache.",
        )

    @torch.inference_mode()
    def _build_per_layer_caches(
        self,
        batches: Iterable[dict[str, Any]],
        per_layer_cache: Callable[..., Any],
    ) -> list[Any]:
        return [
            per_layer_cache(batch["input_ids"])
            for batch in batches
        ]

    @torch.inference_mode()
    def _average_loss_from_per_layer_cache(
        self,
        *,
        circuit: Circuit,
        candidate_edge: edge_key,
        batches: Iterable[dict[str, Any]],
        per_layer_caches: Iterable[Any],
        forward_from_cache: Callable[..., torch.Tensor],
        loss_fn: ACDCLossFn,
    ) -> torch.Tensor:
        total: torch.Tensor | None = None
        n_batches = 0

        for batch, cache in zip(batches, per_layer_caches):
            logits = forward_from_cache(
                cache,
                circuit=circuit,
                edge_key=candidate_edge,
            )
            loss = loss_fn(batch, logits).detach()
            total = loss if total is None else total + loss
            n_batches += 1

        if total is None:
            return torch.zeros((), device=self.device_name)

        return total / max(n_batches, 1)

    def _remove_edge(
        self,
        edge,
        false_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        previous = edge.edge_mask
        edge.edge_mask = false_mask
        return previous

    def _restore_edge(self, edge, previous: torch.Tensor | None) -> None:
        edge.edge_mask = previous

    def fit(
        self,
        dataloader,
        *,
        loss_fn: ACDCLossFn = discogp_fidelity_loss,
        threshold_callback: ACDCThresholdCallback | None = None,
    ) -> ACDCResult:
        torch.manual_seed(self.config.seed)
        batches = self._batches(dataloader)
        edge_order = self._edge_order(self.base_circuit)
        edge_scores = {key: float("inf") for key in edge_order}
        false_mask = torch.tensor(False, dtype=torch.bool, device=DEVICE)
        cache_methods = self._per_layer_cache_methods()
        per_layer_caches = None
        if cache_methods is not None:
            per_layer_cache, forward_from_cache = cache_methods
            per_layer_caches = self._build_per_layer_caches(
                batches,
                per_layer_cache,
            )
        else:
            forward_from_cache = None

        thresholds = self.config.threshold_values()
        for threshold in tqdm(
            thresholds,
            desc="ACDC thresholds",
            dynamic_ncols=True,
            disable=self.config.tqdm_disabled,
        ):
            current = self.base_circuit.full_like()
            current_edges = current.edge_dict()
            current_loss = self._average_loss(current, batches, loss_fn)
            current_loss_value = float(current_loss.detach().cpu().item())
            n_removed = 0

            for edge_index, key in enumerate(edge_order):
                edge = current_edges[key]
                previous_mask = self._remove_edge(edge, false_mask)
                if per_layer_caches is None or forward_from_cache is None:
                    candidate_loss = self._average_loss(current, batches, loss_fn)
                else:
                    candidate_loss = self._average_loss_from_per_layer_cache(
                        circuit=current,
                        candidate_edge=key,
                        batches=batches,
                        per_layer_caches=per_layer_caches,
                        forward_from_cache=forward_from_cache,
                        loss_fn=loss_fn,
                    )
                delta = acdc_loss_delta(
                    candidate_loss=candidate_loss,
                    current_loss=current_loss,
                )
                delta_value = float(delta.detach().cpu().item())
                candidate_loss_value = current_loss_value + delta_value

                if delta_value < threshold:
                    current_loss = candidate_loss
                    current_loss_value = candidate_loss_value
                    edge_scores[key] = min(edge_scores[key], threshold)
                    n_removed += 1
                else:
                    self._restore_edge(edge, previous_mask)

                self.history.append(
                    {
                        "threshold": float(threshold),
                        "edge_index": float(edge_index),
                        "current_loss": current_loss_value,
                        "candidate_loss": candidate_loss_value,
                        "loss_delta": delta_value,
                        "n_removed": float(n_removed),
                    }
                )

            if threshold_callback is not None:
                threshold_callback(
                    float(threshold),
                    ACDCResult(
                        state=ACDCState(
                            base_circuit=self.base_circuit,
                            edge_scores=dict(edge_scores),
                        ),
                        history=list(self.history),
                    ),
                )

        return ACDCResult(
            state=ACDCState(
                base_circuit=self.base_circuit,
                edge_scores=edge_scores,
            ),
            history=self.history,
        )

    def discover_circuit(
        self,
        dataloader,
        *,
        threshold: float | None = None,
        edge_density: float | None = None,
        loss_fn: ACDCLossFn = discogp_fidelity_loss,
        finalize: bool = True,
    ) -> Circuit:
        if (threshold is None) == (edge_density is None):
            raise ValueError("pass exactly one of threshold or edge_density.")

        result = self.fit(dataloader, loss_fn=loss_fn)
        if threshold is not None:
            return result.circuit_for_threshold(
                threshold,
                model=self.model,
                finalize=finalize,
            )
        assert edge_density is not None
        return result.circuit_for_edge_budget(
            edge_density,
            model=self.model,
            finalize=finalize,
        )
