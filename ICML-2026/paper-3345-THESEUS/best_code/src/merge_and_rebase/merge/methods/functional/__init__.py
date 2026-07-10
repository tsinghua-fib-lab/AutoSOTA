from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from . import actmerge as _actmerge  # noqa: F401
from . import cart_merge as _cart_merge  # noqa: F401
from . import dare_merge as _dare_merge  # noqa: F401
from . import dc_merge as _dc_merge  # noqa: F401
from . import isoc_merge as _isoc_merge  # noqa: F401
from . import isocts_merge as _isocts_merge  # noqa: F401
from . import pcb as _pcb  # noqa: F401
from . import task_arithmetic as _task_arithmetic  # noqa: F401
from . import ties_merge as _ties_merge  # noqa: F401
from . import tsv_merge as _tsv_merge  # noqa: F401
from . import weighted_average as _weighted_average  # noqa: F401
from . import wudi as _wudi  # noqa: F401
from ._registry import list_functional_methods, merge_functional, merge_raw_matrices

_MERGE_EXPORTS: dict[str, str] = {
    "merge_actmerge": "actmerge",
    "merge_task_arithmetic": "task_arithmetic",
    "merge_weighted_average": "weighted_average",
    "merge_wudi": "wudi",
    "merge_tsv": "tsv_merge",
    "merge_isoc": "isoc_merge",
    "merge_isocts": "isocts_merge",
    "merge_dc": "dc_merge",
    "merge_dare": "dare_merge",
    "merge_ties": "ties_merge",
    "merge_pcb": "pcb",
    "merge_cart": "cart_merge",
}


def _merge_named(
    method_name: str,
    *,
    matrices: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    alpha: float = 1.0,
    method_params: Mapping[str, Any] | None = None,
    **technical_params: Any,
) -> torch.Tensor:
    return merge_functional(
        method_name,
        matrices=matrices,
        weights=weights,
        alpha=alpha,
        method_params=method_params,
        **technical_params,
    )


def _make_merge_export(export_name: str, method_name: str):
    def _merge_export(
        *,
        matrices: Sequence[torch.Tensor],
        weights: Sequence[float] | None = None,
        alpha: float = 1.0,
        method_params: Mapping[str, Any] | None = None,
        **technical_params: Any,
    ) -> torch.Tensor:
        return _merge_named(
            method_name,
            matrices=matrices,
            weights=weights,
            alpha=alpha,
            method_params=method_params,
            **technical_params,
        )

    _merge_export.__name__ = export_name
    _merge_export.__qualname__ = export_name
    _merge_export.__doc__ = f"Convenience wrapper for functional merge method `{method_name}`."
    return _merge_export


globals().update(
    {
        export_name: _make_merge_export(export_name, method_name)
        for export_name, method_name in _MERGE_EXPORTS.items()
    }
)


__all__ = [
    "list_functional_methods",
    "merge_actmerge",
    "merge_functional",
    "merge_raw_matrices",
    *_MERGE_EXPORTS.keys(),
]
