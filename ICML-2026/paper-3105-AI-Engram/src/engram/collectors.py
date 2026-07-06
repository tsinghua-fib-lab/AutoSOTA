"""Covariance collector.

Registers forward pre-hooks that accumulate the **mean** input covariance
``mean(x^T x)`` plus a sample count for each supported layer. Forward-only (no
backward pass), in-place incremental-mean update, with matrices held on
``config.storage_device``. The mean keeps the magnitude bounded regardless of
corpus size; the count is what lets the editor recover the paper's ``n / N``
weighting (see :class:`engram.stats.Statistics`).

A per-batch token mask (``mask_fn`` in ``EngramEditor.collect_statistics``)
restricts the covariance to selected tokens for **every** layer type. MoE experts
receive only their routed tokens, so the routed rows are matched back to the most
recent full-token reference (the router/gate input) to recover their global mask;
an expert's second projection (whose input has no full-token counterpart) reuses
the routing mask just computed. This is automatic and a no-op for dense models.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from .config import EditorConfig
from .handlers import LayerHandler, handler_for

# Container names whose ".<name>.<idx>." segment carries the decoder-layer index,
# scanned when layers_to_transform is set but layers_pattern is not (Llama="layers",
# GPT-2="h", etc.). Mirrors PEFT/LoRA's layer-index selection.
_COMMON_LAYER_PATTERNS = ("layers", "h", "blocks", "block", "layer")

# Routed-token alignment (masked MoE on the per-expert nn.Linear layout): an expert's
# input rows are an exact gather of the full-token reference, recovered by a small
# random-projection fingerprint. A few dims make a collision (two distinct rows agreeing
# in every dim) effectively impossible — a 1-D fingerprint could alias and mis-map a token.
_FP_DIM = 4


def _module_name_matches(name: str, target_modules: Optional[Union[str, List[str]]]) -> bool:
    """LoRA/PEFT-style module match: None=all, str=regex (fullmatch), list=exact-or-dotted-suffix."""
    if target_modules is None:
        return True
    if isinstance(target_modules, str):
        return re.fullmatch(target_modules, name) is not None
    return name in target_modules or any(name.endswith(f".{t}") for t in target_modules)


def _layer_index_matches(
    name: str,
    layers_to_transform: Optional[Union[int, List[int]]],
    layers_pattern: Optional[Union[str, List[str]]],
) -> bool:
    """PEFT-style index filter: True if the module's decoder-layer index is selected.

    No-op when layers_to_transform is None. A module with no parseable layer index
    (e.g. ``lm_head``) is excluded once an index filter is requested.
    """
    if layers_to_transform is None:
        return True
    want = {layers_to_transform} if isinstance(layers_to_transform, int) else set(layers_to_transform)
    if isinstance(layers_pattern, str):
        patterns = [layers_pattern]
    elif layers_pattern:
        patterns = list(layers_pattern)
    else:
        patterns = list(_COMMON_LAYER_PATTERNS)
    for pat in patterns:
        m = re.search(rf"(?:^|\.){pat}\.(\d+)(?:\.|$)", name)
        if m is not None:
            return int(m.group(1)) in want
    return False


class CovarianceCollector:
    """Context manager accumulating per-layer **mean** input covariance + sample count."""

    def __init__(
        self,
        model: nn.Module,
        config: EditorConfig,
        registry: Dict[Type[nn.Module], LayerHandler],
        target_modules: Optional[Union[str, List[str]]] = None,
        layers_to_transform: Optional[Union[int, List[int]]] = None,
        layers_pattern: Optional[Union[str, List[str]]] = None,
        adapters: Optional[List[Any]] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.registry = registry
        self.target_modules = target_modules
        self.layers_to_transform = layers_to_transform
        self.layers_pattern = layers_pattern
        self.adapters = list(adapters or [])  # opt-in extensions (e.g. fused MoE experts)
        self.covariance_matrices: Dict[str, torch.Tensor] = {}  # per-layer MEAN of x^T x
        self.sample_counts: Dict[str, int] = {}                 # per-layer rows that entered the mean
        self.current_mask: Optional[torch.Tensor] = None
        # per-batch routing-alignment state (used only when a layer sees a subset)
        self._proj: Dict[int, torch.Tensor] = {}                       # H -> random projection [H, _FP_DIM]
        self._ref: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}   # H -> (fingerprint[N,_FP_DIM], mask[N])
        self._routed_mask: Optional[torch.Tensor] = None
        self._storage_device: Optional[torch.device] = None  # resolved in __enter__
        self._hook_handles: List[Any] = []

    def set_mask(self, mask: Optional[torch.Tensor]) -> None:
        """Set the per-batch token mask and reset routing-alignment state."""
        self.current_mask = mask
        self._ref.clear()
        self._routed_mask = None

    def _fingerprint(self, x: torch.Tensor) -> torch.Tensor:
        """Per-row signature (random projection to ``_FP_DIM`` dims) to match routed rows back."""
        h = x.shape[1]
        proj = self._proj.get(h)
        if proj is None:
            g = torch.Generator(device=x.device).manual_seed(1234567 + h)  # local RNG, deterministic
            proj = torch.randn(h, _FP_DIM, generator=g, device=x.device, dtype=x.dtype)
            self._proj[h] = proj
        return x @ proj  # [N, _FP_DIM]

    def _select(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the boolean row-mask selecting wanted tokens for this layer input."""
        m = self.current_mask
        if m is None:
            return None
        m = m.reshape(-1).to(x.device)
        n_full, n, h = m.shape[0], x.shape[0], x.shape[1]

        if n == n_full:  # full-token layer: also stash a reference for routed layers
            self._ref[h] = (self._fingerprint(x), m)
            return m

        # n != n_full: a routed subset (e.g. an MoE expert seeing only its tokens)
        ref = self._ref.get(h)
        if ref is not None:  # match rows back to the full-token reference (exact gather)
            fp_ref, mask_full = ref
            dist = torch.cdist(self._fingerprint(x), fp_ref)  # [n, n_full] over _FP_DIM dims
            idx = dist.argmin(1)
            if (dist.gather(1, idx[:, None]).squeeze(1) > 1e-3 * fp_ref.std().clamp(min=1e-9)).any():
                raise ValueError(
                    f"can't align mask: a layer received {n}/{n_full} rows that don't match the "
                    f"full-token reference (not a token gather)."
                )
            self._routed_mask = mask_full[idx]
            return self._routed_mask
        if self._routed_mask is not None and self._routed_mask.shape[0] == n:
            return self._routed_mask  # 2nd projection of the same expert (no own reference)
        raise ValueError(f"can't align mask: a layer received {n}/{n_full} rows and no reference.")

    def __enter__(self) -> "CovarianceCollector":
        # storage_device=None (default) follows the model's device: fastest, since
        # each batch's D×D is accumulated in place with no GPU->CPU transfer. Set
        # "cpu" when the covariances don't fit in VRAM (large/wide models).
        storage = self.config.storage_device
        if storage is None:
            try:
                storage = next(self.model.parameters()).device
            except StopIteration:
                storage = torch.device("cpu")
        self._storage_device = storage

        for name, module in self.model.named_modules():
            if not (
                _module_name_matches(name, self.target_modules)
                and _layer_index_matches(name, self.layers_to_transform, self.layers_pattern)
            ):
                continue

            handler = handler_for(self.registry, module)
            if handler is None:
                continue

            absorb = self.config.absorb_bias and getattr(module, "bias", None) is not None
            dim = handler.get_input_dim(module, absorb_bias=absorb)
            # float32 throughout: float64's finer pinv rcond keeps the near-null
            # directions of an ill-conditioned Sigma and 1/sigma-amplifies them into
            # a catastrophic edit (TOFU Overall ~0); float32's coarser cut happens to
            # discard them, which is the regularization that makes the edit work.
            self.covariance_matrices[name] = torch.zeros(
                (dim, dim), device=self._storage_device, dtype=torch.float32
            )
            self.sample_counts[name] = 0

            def make_hook(layer_name: str, layer_handler: LayerHandler, absorb_bias: bool):
                def hook(mod: nn.Module, inputs: Any) -> None:
                    x = layer_handler.reshape_input(mod, inputs, absorb_bias=absorb_bias).to(
                        torch.float32
                    )
                    sel = self._select(x)
                    if sel is not None:
                        x = x[sel]
                    k = x.shape[0]
                    if k == 0:
                        return
                    # incremental mean: C := (N*C + sum(x^T x)) / (N + k), magnitude-bounded
                    batch = (x.mT @ x).to(self._storage_device)
                    n = self.sample_counts[layer_name]
                    cov = self.covariance_matrices[layer_name]
                    cov.add_((batch - k * cov) / (n + k))
                    self.sample_counts[layer_name] = n + k

                return hook

            self._hook_handles.append(
                module.register_forward_pre_hook(make_hook(name, handler, absorb))
            )

        # opt-in adapters register their own hooks (e.g. fused MoE experts, which
        # have no per-expert module for the loop above to find). No-op if none.
        for adapter in self.adapters:
            adapter.attach(self)

        # Nothing matched -> the covariance would be silently empty (and the edit a
        # no-op). Almost always a target_modules / layers_to_transform typo, or a
        # selection that only hit non-hookable types (e.g. LayerNorm). Warn early,
        # before iterating the whole dataloader for nothing.
        if not self.covariance_matrices:
            warnings.warn(
                "CovarianceCollector: no supported layers matched the selection, so the "
                "covariance is empty and any resulting edit is a no-op. Check target_modules / "
                "layers_to_transform (a typo?), or whether the matched modules are hookable "
                "(nn.Linear / Conv1D / fused-MoE experts via engram.moe).",
                stacklevel=3,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for adapter in self.adapters:
            adapter.detach()
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
