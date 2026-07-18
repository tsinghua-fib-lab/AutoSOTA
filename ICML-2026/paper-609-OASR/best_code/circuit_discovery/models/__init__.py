# models/__init__.py
# registry and protocol definitions for circuit-oriented modeling.

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Callable, Protocol, TypeAlias

import torch

from ..circuit import Circuit, Edge, node_key

from .modeling_gpt import CircuitGPT
from .modeling_pythia import CircuitPythia

# --------------------------------------------------------------------------------------
# intervention typing
# --------------------------------------------------------------------------------------

EdgeIntervention: TypeAlias = Callable[
    [Edge, torch.Tensor, node_key, node_key],
    torch.Tensor | None,
]

# --------------------------------------------------------------------------------------
# circuit model protocol
# --------------------------------------------------------------------------------------

class CircuitModel(Protocol):
    """
    minimal interface required by algorithms/discogp.py.
    """

    full_circuit: Circuit

    def forward(
        self,
        tokens: torch.Tensor,
        circuit: Circuit | None = None,
        *,
        runtime_masks: Any | None = None,
        edge_intervention: EdgeIntervention | None = None,
        return_residual: bool = False,
    ) -> torch.Tensor: ...

    def __call__(
        self,
        tokens: torch.Tensor,
        circuit: Circuit | None = None,
        *,
        runtime_masks: Any | None = None,
        edge_intervention: EdgeIntervention | None = None,
        return_residual: bool = False,
    ) -> torch.Tensor: ...

    def eval(self) -> CircuitModel: ...

    def to(self, *args, **kwargs) -> CircuitModel: ...

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, torch.nn.Parameter]]: ...

    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]: ...

    def lookup_weight(
        self,
        n_key: node_key,
        w_key: str,
    ) -> torch.Tensor: ...

    def edge_logit_group_specs(self, circuit: Circuit) -> list[Any]: ...

    def weight_logit_group_specs(self, circuit: Circuit) -> list[Any]: ...

    def sample_runtime_masks(self, **kwargs: Any) -> Any: ...

    def boolean_runtime_weight_masks(self, **kwargs: Any) -> Any: ...

    def finalize_circuit(self, circuit: Circuit) -> Circuit: ...

# --------------------------------------------------------------------------------------
# model registry
# --------------------------------------------------------------------------------------

CircuitModelClass: TypeAlias = type[CircuitGPT] | type[CircuitPythia]

MODEL_REGISTRY: dict[str, CircuitModelClass] = {
    "gpt2-small": CircuitGPT,
    "gpt2-medium": CircuitGPT,
    "pythia-160m": CircuitPythia,
}

# --------------------------------------------------------------------------------------
# loader
# --------------------------------------------------------------------------------------

def load_circuit_model(
    model_name: str,
    *,
    device: str | None = None,
) -> CircuitModel:
    """
    load a circuit-compatible model by name.

    model classes are expected to expose:
        - full_circuit: Circuit
        - forward(tokens, circuit=...)
        - lookup_weight(node_key, weight_key)
        - load_model(model_name=..., device=...)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown circuit model {model_name!r}. "
            f"available models: {sorted(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[model_name]

    return model_cls.load_model(
        model_name=model_name,
        device=device,
    ) # type: ignore
