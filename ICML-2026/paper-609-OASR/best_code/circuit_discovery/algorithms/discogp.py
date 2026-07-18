# algorithms/discogp.py
#
# differentiable circuit discovery for circuit.py + circuit-compatible models.
#
# this module keeps float-valued mask logits outside Circuit. Circuit remains the
# structural/boolean representation; models own any architecture-specific packed
# runtime representation used during optimization.
#
# supported modes:
#   - weight pruning
#   - edge pruning
#
# random_mode=None gives deterministic straight-through thresholding.
# random_mode="gumbel_sigmoid" gives stochastic straight-through perturbation.
#
# task losses, sparsity losses, and overlap losses live in metrics.py.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal

import torch
import torch.nn as nn
from tqdm import tqdm

from ..circuit import Circuit, edge_key, node_key
from ..metrics import (
    discogp_completeness_loss,
    discogp_fidelity_loss,
    discogp_edge_density_loss,
    discogp_overlap_loss,
    discogp_weight_density_loss,
)
from ..models import CircuitModel, load_circuit_model
from ..utils import DEVICE

RandomMode = Literal[None, "gumbel_sigmoid"]
PruningMode = Literal["weight", "edge"]

WeightItem = tuple[node_key, str]

# --------------------------------------------------------------------------------------
# mask samplers
# --------------------------------------------------------------------------------------

def gumbel_sigmoid(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    straight-through gumbel-sigmoid binary relaxation.

    binary Gumbel-Max uses the difference of two iid Gumbels. This implementation
    keeps the two-uniform construction close to that derivation.
    """
    uniform = logits.new_empty((2, *logits.shape)).uniform_(0, 1)
    noise = -((uniform[0] + eps).log() / (uniform[1] + eps).log() + eps).log()

    soft = torch.sigmoid((logits + noise) / temperature)
    hard = (soft > 0.5).to(soft.dtype)

    return (hard - soft).detach() + soft


def sample_mask(
    logits: torch.Tensor,
    *,
    random_mode: RandomMode,
    reverse: bool = False,
    gs_temp: float = 1.0,
) -> torch.Tensor:
    if random_mode == "gumbel_sigmoid":
        mask = gumbel_sigmoid(logits, temperature=gs_temp)
    else:
        # deterministic straight-through threshold:
        # forward value is hard, gradient flows through sigmoid(logits).
        soft = torch.sigmoid(logits)
        hard = (logits > 0.0).to(soft.dtype)
        mask = (hard - soft).detach() + soft

    if reverse:
        mask = 1.0 - mask

    return mask


def boolean_mask(logits: torch.Tensor, *, reverse: bool = False) -> torch.Tensor:
    mask = logits > 0.0

    if reverse:
        mask = torch.logical_not(mask)

    return mask

# --------------------------------------------------------------------------------------
# learning rate scheduling
# --------------------------------------------------------------------------------------

def schedule_epoch_lambda(
    epoch: int,
    *,
    lambda_0: float,
    min_times: float = 1.0,
    max_times: float = 1.0,
    n_epoch_warmup: int = 0,
    n_epoch_cooldown: int = 0,
) -> float:
    """
    DiscoGP learning rate schedule.

    phase 1:
        linearly increase lambda_0 -> lambda_0 * max_times

    phase 2:
        linearly decrease lambda_0 * max_times -> lambda_0 * min_times

    phase 3:
        hold lambda_0 * min_times
    """
    if n_epoch_warmup > 0 and epoch < n_epoch_warmup:
        return (
            lambda_0
            + lambda_0
            * (max_times - 1.0)
            * epoch
            / n_epoch_warmup
        )

    if (
        n_epoch_cooldown > 0
        and epoch < n_epoch_warmup + n_epoch_cooldown
    ):
        cooldown_epoch = epoch - n_epoch_warmup
        return (
            lambda_0 * max_times
            - lambda_0
            * (max_times - min_times)
            * cooldown_epoch
            / n_epoch_cooldown
        )

    return lambda_0 * min_times

# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------

@dataclass
class DiscoGPConfig:
    model_name: str = "gpt2-small"

    # pruning switches
    prune_edges: bool = True
    prune_weights: bool = False

    # training
    n_epochs_e: int = 40
    n_epochs_w: int = 120
    batch_size: int = 32

    lr_e: float = 0.07
    lr_w: float = 0.1

    # mask initialization
    edge_logit_init_mean: float = 0.1
    edge_logit_init_std: float = 0.01
    weight_logit_init_mean: float = 0.01
    weight_logit_init_std: float = 0.01

    # mask sampling
    random_mode: RandomMode = "gumbel_sigmoid"
    gs_temp_edge: float = 1.0
    gs_temp_weight: float = 0.01

    # scheduled sparsity regularization
    lambda_sparse_e: float = 1.0
    lambda_sparse_w: float = 1.0

    min_times_lambda_sparse_e: float = 0.01
    max_times_lambda_sparse_e: float = 20.0
    min_times_lambda_sparse_w: float = 1.0
    max_times_lambda_sparse_w: float = 1000.0

    n_epoch_warmup_lambda_sparse_e: int = int(0.8 * n_epochs_e)
    n_epoch_cooldown_lambda_sparse_e: int = int(0.2 * n_epochs_e)
    n_epoch_warmup_lambda_sparse_w: int = int(0.8 * n_epochs_w)
    n_epoch_cooldown_lambda_sparse_w: int = int(0.1 * n_epochs_w)

    # completeness regularization
    lambda_complete_e: float = 0.01
    lambda_complete_w: float = 1.0
    completeness_start_frac: float = 0.8

    # overlap / alternative-circuit regularization
    lambda_overlap_e: float = 0.66
    lambda_overlap_w: float = 0.0

    min_times_lambda_overlap_e: float = 0.01
    max_times_lambda_overlap_e: float = 20.0
    min_times_lambda_overlap_w: float = 1.0
    max_times_lambda_overlap_w: float = 1000.0

    n_epoch_warmup_lambda_overlap_e: int = int(0.8 * n_epochs_e)
    n_epoch_cooldown_lambda_overlap_e: int = int(0.2 * n_epochs_e)
    n_epoch_warmup_lambda_overlap_w: int = int(0.8 * n_epochs_w)
    n_epoch_cooldown_lambda_overlap_w: int = int(0.2 * n_epochs_w)

    overlap_penalty: bool = False
    tqdm_disabled: bool = False

# --------------------------------------------------------------------------------------
# float mask store
# --------------------------------------------------------------------------------------

class DiscoGPMasks(nn.Module):
    """
    float-valued trainable mask logits aligned with a structural Circuit.

    edge and weight logits are grouped according to model-provided specs.
    The grouping is opaque to DiscoGP; model code owns runtime packing.
    """

    def __init__(
        self,
        model: CircuitModel,
        circuit: Circuit,
        *,
        init_weights: bool = True,
        edge_init_mean: float = 1.0,
        edge_init_std: float = 0.01,
        weight_init_mean: float = 1.0,
        weight_init_std: float = 0.01,
        device: str = DEVICE,
    ) -> None:
        super().__init__()

        self.model = model
        self.circuit = circuit
        self.device_name = device

        self.edge_keys: list[edge_key] = circuit.edge_keys()
        self.edge_group_specs = list(model.edge_logit_group_specs(circuit))
        self.edge_logit_keys: list[list[edge_key]] = [
            list(spec.keys)
            for spec in self.edge_group_specs
        ]
        self.edge_logit_shapes = [
            tuple(spec.shape)
            for spec in self.edge_group_specs
        ]
        self.edge_logits = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(shape, device=device).normal_(
                        edge_init_mean,
                        edge_init_std,
                    )
                )
                for shape in self.edge_logit_shapes
            ]
        )
        self.edge_index: dict[edge_key, tuple[int, int]] = {
            key: (group_index, offset)
            for group_index, keys in enumerate(self.edge_logit_keys)
            for offset, key in enumerate(keys)
        }

        self.weight_group_specs = (
            list(model.weight_logit_group_specs(circuit))
            if init_weights
            else []
        )
        self.weight_logit_items: list[list[WeightItem]] = []
        weight_params: list[nn.Parameter] = []

        if init_weights:
            self.weight_logit_items = [
                list(spec.items)
                for spec in self.weight_group_specs
            ]

            for spec in self.weight_group_specs:
                weight_params.append(
                    nn.Parameter(
                        torch.empty(tuple(spec.shape), device=device).normal_(
                            weight_init_mean,
                            weight_init_std,
                        )
                    )
                )

        self.weight_items: list[WeightItem] = [
            (n_key, w_key)
            for n_key, node in circuit.nodes.items()
            for w_key in node.weight_masks.keys()
        ]

        self.weight_logits = nn.ParameterList(weight_params)
        self.weight_index: dict[WeightItem, tuple[int, int | None]] = {
            item: (group_index, offset if len(items) > 1 else None)
            for group_index, items in enumerate(self.weight_logit_items)
            for offset, item in enumerate(items)
        }

    def edge_parameters(self) -> Iterable[nn.Parameter]:
        return self.edge_logits.parameters()

    def weight_parameters(self) -> Iterable[nn.Parameter]:
        return self.weight_logits.parameters()

    def sampled_circuit(
        self,
        *,
        use_edges: bool = True,
        use_weights: bool = True,
        reverse_edges: bool = False,
        reverse_weights: bool = False,
        random_mode: RandomMode = "gumbel_sigmoid",
        gs_temp_edge: float = 1.0,
        gs_temp_weight: float = 1.0,
        frozen_weight_masks: list[torch.Tensor] | None = None,
    ) -> Circuit:
        """
        build a temporary differentiable circuit.

        Circuit.__init__ coerces masks to bool, so we first clone the structural
        circuit and then overwrite edge_mask / weight_masks with float tensors.
        """
        out = self.circuit.full_like()

        if use_edges:
            for keys, logits in zip(self.edge_logit_keys, self.edge_logits):
                masks = sample_mask(
                    logits,
                    random_mode=random_mode,
                    reverse=reverse_edges,
                    gs_temp=gs_temp_edge,
                )
                flat_masks = masks.reshape(-1)
                for offset, key in enumerate(keys):
                    out.get_edge(key).edge_mask = flat_masks[offset]

        if frozen_weight_masks is not None:
            for items, masks in zip(self.weight_logit_items, frozen_weight_masks):
                if len(items) == 1:
                    n_key, w_key = items[0]
                    out.nodes[n_key].weight_masks[w_key] = masks
                else:
                    for offset, (n_key, w_key) in enumerate(items):
                        out.nodes[n_key].weight_masks[w_key] = masks[offset]

        elif use_weights:
            for items, logits in zip(self.weight_logit_items, self.weight_logits):
                masks = sample_mask(
                    logits,
                    random_mode=random_mode,
                    reverse=reverse_weights,
                    gs_temp=gs_temp_weight,
                )
                if len(items) == 1:
                    n_key, w_key = items[0]
                    out.nodes[n_key].weight_masks[w_key] = masks
                else:
                    for offset, (n_key, w_key) in enumerate(items):
                        out.nodes[n_key].weight_masks[w_key] = masks[offset]

        return out

    @torch.no_grad()
    def edge_reference_masks(self, reference: Circuit) -> list[torch.Tensor]:
        edge_dict = reference.edge_dict()
        masks: list[torch.Tensor] = []

        for keys, logits in zip(self.edge_logit_keys, self.edge_logits):
            flat_values = []
            for key in keys:
                ref_mask = edge_dict[key].edge_mask
                flat_values.append(
                    ref_mask is not None and bool(ref_mask.bool().item())
                )

            masks.append(
                torch.tensor(
                    flat_values,
                    dtype=torch.bool,
                    device=logits.device,
                ).reshape(logits.shape)
            )

        return masks

    @torch.no_grad()
    def weight_reference_masks(self, reference: Circuit) -> list[torch.Tensor]:
        masks: list[torch.Tensor] = []

        for items, logits in zip(self.weight_logit_items, self.weight_logits):
            if len(items) == 1:
                n_key, w_key = items[0]
                ref_mask = reference.nodes[n_key].weight_masks.get(w_key)

                if ref_mask is None:
                    masks.append(torch.zeros_like(logits, dtype=torch.bool))
                else:
                    ref = ref_mask.to(device=logits.device, dtype=torch.bool)
                    if ref.shape == torch.Size([]):
                        ref = torch.full_like(
                            logits,
                            bool(ref.item()),
                            dtype=torch.bool,
                        )
                    masks.append(ref)
                continue

            pieces: list[torch.Tensor] = []
            for offset, (n_key, w_key) in enumerate(items):
                ref_mask = reference.nodes[n_key].weight_masks.get(w_key)
                logit = logits[offset]

                if ref_mask is None:
                    pieces.append(torch.zeros_like(logit, dtype=torch.bool))
                    continue

                ref = ref_mask.to(device=logit.device, dtype=torch.bool)
                if ref.shape == torch.Size([]):
                    ref = torch.full_like(
                        logit,
                        bool(ref.item()),
                        dtype=torch.bool,
                    )
                pieces.append(ref)

            masks.append(torch.stack(pieces, dim=0))

        return masks

    @torch.no_grad()
    def boolean_weight_masks(self) -> list[torch.Tensor]:
        return [
            boolean_mask(logits).detach()
            for logits in self.weight_logits
        ]

    @torch.no_grad()
    def boolean_circuit(
        self,
        *,
        use_edges: bool = True,
        use_weights: bool = True,
        reverse_edges: bool = False,
        reverse_weights: bool = False,
    ) -> Circuit:
        out = self.circuit.full_like()

        if use_edges:
            for keys, logits in zip(self.edge_logit_keys, self.edge_logits):
                masks = boolean_mask(logits, reverse=reverse_edges)
                flat_masks = masks.reshape(-1)
                for offset, key in enumerate(keys):
                    out.get_edge(key).edge_mask = flat_masks[offset]

        if use_weights:
            for items, logits in zip(self.weight_logit_items, self.weight_logits):
                masks = boolean_mask(logits, reverse=reverse_weights)
                if len(items) == 1:
                    n_key, w_key = items[0]
                    out.nodes[n_key].weight_masks[w_key] = masks
                else:
                    for offset, (n_key, w_key) in enumerate(items):
                        out.nodes[n_key].weight_masks[w_key] = masks[offset]

        return out

# --------------------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------------------

class DiscoGP:
    def __init__(
        self,
        *,
        model: CircuitModel | None = None,
        config: DiscoGPConfig | None = None,
        device: str = DEVICE,
    ) -> None:
        self.config = config if config is not None else DiscoGPConfig()
        self.device_name = device

        if model is None:
            self.model = load_circuit_model(
                self.config.model_name,
                device=device,
            )
        else:
            self.model = model

        self.model.eval()

        self.base_circuit = self.model.full_circuit
        self.masks = DiscoGPMasks(
            self.model,
            self.base_circuit,
            init_weights=self.config.prune_weights,
            edge_init_mean=self.config.edge_logit_init_mean,
            edge_init_std=self.config.edge_logit_init_std,
            weight_init_mean=self.config.weight_logit_init_mean,
            weight_init_std=self.config.weight_logit_init_std,
            device=device,
        ).to(device)

        self.reference_circuit: Circuit | None = None
        self.reference_edge_masks: list[torch.Tensor] | None = None
        self.reference_weight_masks: list[torch.Tensor] | None = None
        self.reference_edge_denom = 0
        self.reference_weight_denom = 0

    def load_reference_circuit(self, circuit: Circuit | None) -> None:
        self.reference_circuit = circuit
        if circuit is None:
            self.reference_edge_masks = None
            self.reference_weight_masks = None
            self.reference_edge_denom = 0
            self.reference_weight_denom = 0
            return

        self.reference_edge_masks = self.masks.edge_reference_masks(circuit)
        self.reference_weight_masks = self.masks.weight_reference_masks(circuit)
        self.reference_edge_denom = sum(
            int(mask.sum().item())
            for mask in self.reference_edge_masks
        )
        self.reference_weight_denom = sum(
            int(mask.sum().item())
            for mask in self.reference_weight_masks
        )

    def _optimizer(self, mode: PruningMode) -> torch.optim.Optimizer:
        fused = torch.cuda.is_available()

        if mode == "edge":
            return torch.optim.AdamW(
                list(self.masks.edge_parameters()),
                lr=self.config.lr_e,
                fused=fused,
            )

        return torch.optim.AdamW(
            list(self.masks.weight_parameters()),
            lr=self.config.lr_w,
            fused=fused,
        )

    def _sampled_circuit_for_mode(
        self,
        mode: PruningMode,
        *,
        reverse: bool = False,
        frozen_weight_masks: list[torch.Tensor] | None = None,
    ) -> Circuit:
        return self.masks.sampled_circuit(
            use_edges=mode == "edge",
            use_weights=mode == "weight",
            reverse_edges=reverse if mode == "edge" else False,
            reverse_weights=reverse if mode == "weight" else False,
            random_mode=self.config.random_mode,
            gs_temp_edge=self.config.gs_temp_edge,
            gs_temp_weight=self.config.gs_temp_weight,
            frozen_weight_masks=frozen_weight_masks,
        )

    def _sampled_runtime_masks_for_mode(
        self,
        mode: PruningMode,
        *,
        reverse: bool = False,
        frozen_weight_runtime: Any | None = None,
    ) -> Any:
        return self.model.sample_runtime_masks(
            edge_logits=self.masks.edge_logits if mode == "edge" else None,
            edge_group_specs=self.masks.edge_group_specs,
            weight_logits=self.masks.weight_logits if mode == "weight" else None,
            weight_group_specs=self.masks.weight_group_specs,
            frozen_weight_runtime=frozen_weight_runtime,
            sample_mask_fn=sample_mask,
            boolean_mask_fn=boolean_mask,
            mode=mode,
            reverse_edges=reverse if mode == "edge" else False,
            reverse_weights=reverse if mode == "weight" else False,
            random_mode=self.config.random_mode,
            gs_temp_weight=self.config.gs_temp_weight,
            gs_temp_edge=self.config.gs_temp_edge,
        )

    def _frozen_weight_runtime_masks(self) -> Any | None:
        if not self.config.prune_weights:
            return None

        return self.model.boolean_runtime_weight_masks(
            weight_logits=self.masks.weight_logits,
            weight_group_specs=self.masks.weight_group_specs,
            boolean_mask_fn=boolean_mask,
        )

    def _assert_no_weight_grads(self) -> None:
        for parameter in self.masks.weight_parameters():
            if parameter.grad is not None:
                raise RuntimeError(
                    "weight mask gradients were produced during edge pruning; "
                    "edge phase must use frozen deterministic weight masks."
                )

    def _sparsity_loss(self, mode: PruningMode) -> torch.Tensor:
        if mode == "edge":
            return discogp_edge_density_loss(
                self.masks.edge_logits,
                device=self.device_name,
            )

        return discogp_weight_density_loss(
            self.masks.weight_logits,
            device=self.device_name,
        )

    def _overlap_loss(self, mode: PruningMode) -> torch.Tensor:
        if self.reference_circuit is None:
            return torch.zeros((), device=self.device_name)

        if mode == "edge" and self.reference_edge_masks is not None:
            return self._cached_overlap_loss(
                self.masks.edge_logits,
                self.reference_edge_masks,
                self.reference_edge_denom,
            )

        if mode == "weight" and self.reference_weight_masks is not None:
            return self._cached_overlap_loss(
                self.masks.weight_logits,
                self.reference_weight_masks,
                self.reference_weight_denom,
            )

        return discogp_overlap_loss(
            reference=self.reference_circuit,
            edge_keys=self.masks.edge_keys,
            edge_logits=self.masks.edge_logits,
            edge_logit_keys=self.masks.edge_logit_keys,
            weight_items=self.masks.weight_items,
            weight_logits=self.masks.weight_logits,
            weight_logit_items=self.masks.weight_logit_items,
            edges=mode == "edge",
            weights=mode == "weight",
            penalty=self.config.overlap_penalty,
            device=self.device_name,
        )

    def _cached_overlap_loss(
        self,
        logits: nn.ParameterList,
        reference_masks: list[torch.Tensor],
        denom: int,
    ) -> torch.Tensor:
        if denom == 0:
            return torch.zeros((), device=self.device_name)

        sign = 1.0 if self.config.overlap_penalty else -1.0
        loss = torch.zeros((), device=self.device_name)

        for logit, ref_mask in zip(logits, reference_masks):
            loss = loss + torch.sigmoid(logit[ref_mask]).sum()

        return sign * loss / denom

    def _n_epochs(self, mode: PruningMode) -> int:
        if mode == "edge":
            return self.config.n_epochs_e

        return self.config.n_epochs_w

    def _scheduled_lambda_sparse(
        self,
        *,
        mode: PruningMode,
        epoch: int,
    ) -> float:
        if mode == "edge":
            return schedule_epoch_lambda(
                epoch,
                lambda_0=self.config.lambda_sparse_e,
                min_times=self.config.min_times_lambda_sparse_e,
                max_times=self.config.max_times_lambda_sparse_e,
                n_epoch_warmup=self.config.n_epoch_warmup_lambda_sparse_e,
                n_epoch_cooldown=self.config.n_epoch_cooldown_lambda_sparse_e,
            )

        return schedule_epoch_lambda(
            epoch,
            lambda_0=self.config.lambda_sparse_w,
            min_times=self.config.min_times_lambda_sparse_w,
            max_times=self.config.max_times_lambda_sparse_w,
            n_epoch_warmup=self.config.n_epoch_warmup_lambda_sparse_w,
            n_epoch_cooldown=self.config.n_epoch_cooldown_lambda_sparse_w,
        )

    def _scheduled_lambda_overlap(
        self,
        *,
        mode: PruningMode,
        epoch: int,
    ) -> float:
        if mode == "edge":
            return schedule_epoch_lambda(
                epoch,
                lambda_0=self.config.lambda_overlap_e,
                min_times=self.config.min_times_lambda_overlap_e,
                max_times=self.config.max_times_lambda_overlap_e,
                n_epoch_warmup=self.config.n_epoch_warmup_lambda_overlap_e,
                n_epoch_cooldown=self.config.n_epoch_cooldown_lambda_overlap_e,
            )

        return schedule_epoch_lambda(
            epoch,
            lambda_0=self.config.lambda_overlap_w,
            min_times=self.config.min_times_lambda_overlap_w,
            max_times=self.config.max_times_lambda_overlap_w,
            n_epoch_warmup=self.config.n_epoch_warmup_lambda_overlap_w,
            n_epoch_cooldown=self.config.n_epoch_cooldown_lambda_overlap_w,
        )

    def _lambda_complete(self, mode: PruningMode) -> float:
        if mode == "edge":
            return self.config.lambda_complete_e

        return self.config.lambda_complete_w

    def _train_mode(
        self,
        train_dataloader,
        *,
        mode: PruningMode,
        frozen_weight_runtime: Any | None = None,
        fidelity_loss_fn: Callable[[dict[str, Any], torch.Tensor], torch.Tensor],
        completeness_loss_fn: Callable[[dict[str, Any], torch.Tensor], torch.Tensor],
    ) -> None:
        optimizer = self._optimizer(mode)

        n_epochs = self._n_epochs(mode)
        lambda_complete = self._lambda_complete(mode)
        complete_start = int(self.config.completeness_start_frac * n_epochs)

        epoch_iter = tqdm(
            range(n_epochs),
            desc=f"DiscoGP {mode} pruning",
            dynamic_ncols=True,
            disable=self.config.tqdm_disabled,
        )

        for epoch in epoch_iter:
            lambda_sparse = self._scheduled_lambda_sparse(
                mode=mode,
                epoch=epoch,
            )
            lambda_overlap = self._scheduled_lambda_overlap(
                mode=mode,
                epoch=epoch,
            )

            for batch in train_dataloader:

                sparsity = self._sparsity_loss(mode)
                overlap = self._overlap_loss(mode)

                runtime_masks = self._sampled_runtime_masks_for_mode(
                    mode,
                    frozen_weight_runtime=frozen_weight_runtime,
                )
                logits = self.model(
                    batch["input_ids"],
                    runtime_masks=runtime_masks,
                )

                fidelity = fidelity_loss_fn(batch, logits)
                loss = fidelity + lambda_sparse * sparsity + lambda_overlap * overlap

                loss.backward()
                if mode == "edge":
                    self._assert_no_weight_grads()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if epoch >= complete_start and lambda_complete > 0.0:
                    reverse_runtime_masks = self._sampled_runtime_masks_for_mode(
                        mode,
                        reverse=True,
                        frozen_weight_runtime=frozen_weight_runtime,
                    )
                    reverse_logits = self.model(
                        batch["input_ids"],
                        runtime_masks=reverse_runtime_masks,
                    )
                    completeness = (
                        lambda_complete
                        * completeness_loss_fn(batch, reverse_logits)
                    )

                    completeness.backward()
                    if mode == "edge":
                        self._assert_no_weight_grads()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    
    def discover_circuit(
        self,
        train_dataloader,
        *,
        fidelity_loss_fn: Callable[
            [dict[str, Any], torch.Tensor],
            torch.Tensor,
        ] = discogp_fidelity_loss,
        completeness_loss_fn: Callable[
            [dict[str, Any], torch.Tensor],
            torch.Tensor,
        ] = discogp_completeness_loss,
        finalize: bool = True,
    ) -> Circuit:
        """
        train float masks and return a boolean Circuit.

        train_dataloader should yield batches compatible with the chosen loss
        functions. Dataset loading is intentionally handled outside this file.
        """
        frozen_weight_runtime = None

        if self.config.prune_weights:
            self._train_mode(
                train_dataloader,
                mode="weight",
                frozen_weight_runtime=None,
                fidelity_loss_fn=fidelity_loss_fn,
                completeness_loss_fn=completeness_loss_fn,
            )
            frozen_weight_runtime = self._frozen_weight_runtime_masks()

        if self.config.prune_edges:
            self._train_mode(
                train_dataloader,
                mode="edge",
                frozen_weight_runtime=frozen_weight_runtime,
                fidelity_loss_fn=fidelity_loss_fn,
                completeness_loss_fn=completeness_loss_fn,
            )

        circuit = self.masks.boolean_circuit(
            use_edges=self.config.prune_edges,
            use_weights=self.config.prune_weights,
        )

        if finalize:
            circuit = self.model.finalize_circuit(circuit)

        return circuit
