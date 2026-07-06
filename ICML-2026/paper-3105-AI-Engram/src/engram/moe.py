"""Optional, detachable support for transformers >=5 fused-expert MoE layers.

Why this is a separate module: the core (collectors / editor) hooks nn.Linear-style
modules — one module, one weight, one input. transformers >=5 fuses each MoE layer's
experts into 3D ``nn.Parameter``s (``gate_up_proj`` / ``down_proj``) computed by a
single batched op, so there is **no per-expert module to hook**. Recovering each
expert's weight slice and its routed input is intrinsically messy. All of that mess
is quarantined here, behind a tiny generic interface the core understands
(``owns`` / ``weight_for``). **Import nothing from this module and the core stays
exactly as MoE-unaware as before.**

Opt-in::

    from engram import EngramEditor
    from engram.moe import FusedExpertAdapter
    editor = EngramEditor(model, adapters=[FusedExpertAdapter()])
    target = editor.collect_statistics(forget_loader, batch_fn=bf, mask_fn=mf)  # Statistics
    total  = editor.collect_statistics(all_loader,    batch_fn=bf, mask_fn=mf)
    edited = editor.edit(target, total, alpha=0.6)   # per-expert 3D-Parameter slices edited here
    # per-expert engrams are keyed "<experts>.gate_up_proj.<e>" / ".down_proj.<e>".
    # Low-level alternative for pre-computed deltas: engram.moe.apply_engram_weights(model, deltas, alpha).

Scope — the dominant "standard fused" pattern (~35 arches: Mixtral, Qwen2/3/3.5-MoE,
DeepSeek-V2/V3, GLM4-MoE, Ernie4.5-MoE, MiniMax, Mistral4, Jamba, OLMoE, Phi-MoE, …):
experts module exposes 3D ``gate_up_proj`` ``(E, 2*inter, hidden)`` + ``down_proj``
``(E, hidden, inter)`` + ``act_fn``, and ``forward(hidden_states, top_k_index,
top_k_weights)``. Other fused variants (gpt_oss bias, llama4 bmm, granite parallel,
aria grouped-gemm) are detected and skipped with a one-time warning.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

GATE_UP, DOWN = "gate_up_proj", "down_proj"


def _is_standard_fused(module: nn.Module) -> bool:
    """Duck-typed detection of the standard fused-experts module (no class names)."""
    gu = getattr(module, GATE_UP, None)
    dn = getattr(module, DOWN, None)
    if not (torch.is_tensor(gu) and torch.is_tensor(dn) and gu.ndim == 3 and dn.ndim == 3):
        return False
    if not hasattr(module, "act_fn") or hasattr(module, "gate_up_proj_bias"):
        return False  # act_fn needed to recompute down's input; bias = gpt_oss variant
    E, two_inter, hidden = gu.shape          # gate_up_proj: [E, 2*inter, hidden]
    Ed, hidden2, inter = dn.shape            # down_proj:    [E, hidden, inter]
    return E == Ed and hidden == hidden2 and two_inter == 2 * inter


def _looks_like_experts(module: nn.Module) -> bool:
    return type(module).__name__.endswith("Experts") or any(
        p.ndim == 3 for p in module.parameters(recurse=False)
    )


def _split_key(key: str) -> Optional[Tuple[str, str, int]]:
    """"<experts-name>.gate_up_proj.<e>" -> (name, "gate_up_proj", e)."""
    for proj in (GATE_UP, DOWN):
        marker = f".{proj}."
        if marker in key:
            name, e = key.rsplit(marker, 1)
            if e.isdigit():
                return name, proj, int(e)
    return None


def _accumulate_mean(
    cov: Dict[str, torch.Tensor], counts: Dict[str, int], key: str,
    batch_sum: torch.Tensor, k: int, store: Any,
) -> None:
    """Incremental running mean for one key (mirrors CovarianceCollector):
    ``C := (N*C + batch_sum) / (N + k)`` — magnitude-bounded, exact running mean."""
    n = counts[key]
    c = cov[key]
    c.add_((batch_sum.to(store) - k * c) / (n + k))
    counts[key] = n + k


class FusedExpertAdapter:
    """Collects per-expert input covariance for standard fused MoE experts (opt-in).

    Plugs into the core via three calls: ``attach``/``detach`` (the collector adds
    its hooks during a covariance pass) and ``owns``/``weight_for`` (the editor asks
    it to resolve covariance keys it doesn't recognize as modules).
    """

    def __init__(self) -> None:
        self._handles: List[Any] = []
        self._modules: Dict[str, nn.Module] = {}     # experts-module name -> module
        self._warned: set = set()

    # ---- collection: called by CovarianceCollector.__enter__/__exit__ ----
    def attach(self, collector: Any) -> None:
        store = collector._storage_device
        for name, module in collector.model.named_modules():
            if not _is_standard_fused(module):
                if _looks_like_experts(module) and type(module).__name__ not in self._warned:
                    self._warned.add(type(module).__name__)
                    warnings.warn(
                        f"engram.moe: {type(module).__name__} is a fused-expert module but not "
                        f"the standard gate_up_proj/down_proj pattern — skipping (not yet supported)."
                    )
                continue
            self._modules[name] = module
            E, two_inter, hidden = module.gate_up_proj.shape
            inter = two_inter // 2
            for e in range(E):  # pre-allocate per-expert MEAN covariances + counts (gate_up over hidden, down over inter)
                for key, d in ((f"{name}.{GATE_UP}.{e}", hidden), (f"{name}.{DOWN}.{e}", inter)):
                    collector.covariance_matrices[key] = torch.zeros((d, d), device=store, dtype=torch.float32)
                    collector.sample_counts[key] = 0
            self._handles.append(module.register_forward_pre_hook(self._make_hook(collector, name, module)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, collector: Any, name: str, module: nn.Module):
        cov = collector.covariance_matrices
        counts = collector.sample_counts
        store = collector._storage_device

        def hook(mod: nn.Module, args: Tuple[Any, ...]) -> None:
            # standard fused forward: (hidden_states, top_k_index, top_k_weights)
            if len(args) < 2 or not torch.is_tensor(args[1]) or torch.is_floating_point(args[1]):
                return  # unexpected signature -> skip safely
            hidden_states, top_k_index = args[0], args[1]
            x = hidden_states.reshape(-1, hidden_states.shape[-1]).to(torch.float32)
            idx = top_k_index.reshape(x.shape[0], -1)               # [N, k] token -> expert ids
            m = collector.current_mask
            if m is not None:
                m = m.reshape(-1).to(x.device)
            gate_up = module.gate_up_proj.to(torch.float32)
            for e in range(gate_up.shape[0]):
                sel = (idx == e).any(dim=-1)                        # tokens routed to expert e
                if m is not None:
                    sel = sel & m                                  # ... that are also answer tokens
                if not bool(sel.any()):
                    continue
                x_e = x[sel]                                        # gate_up's input
                k = x_e.shape[0]
                _accumulate_mean(cov, counts, f"{name}.{GATE_UP}.{e}", x_e.mT @ x_e, k, store)
                gate, up = F.linear(x_e, gate_up[e]).chunk(2, dim=-1)
                inter_e = module.act_fn(gate) * up                 # down's input (recomputed, exact)
                _accumulate_mean(cov, counts, f"{name}.{DOWN}.{e}", inter_e.mT @ inter_e, k, store)

        return hook

    # ---- compute / apply: called by EngramEditor for keys it doesn't recognise as modules ----
    def owns(self, key: str, model: nn.Module) -> bool:
        """True if ``key`` names a fused-expert slice present on ``model``.

        Stateless — resolved from ``model``, so it works whether or not a covariance
        pass was run this session (e.g. applying engrams loaded from disk).
        """
        split = _split_key(key)
        if split is None:
            return False
        name, proj, e = split
        mod = dict(model.named_modules()).get(name)
        return mod is not None and _is_standard_fused(mod) and 0 <= e < getattr(mod, proj).shape[0]

    def weight_for(self, key: str, model: nn.Module) -> torch.Tensor:
        name, proj, e = _split_key(key)  # type: ignore[misc]
        return getattr(dict(model.named_modules())[name], proj)[e]  # already [out, in]

    def apply_delta(self, model: nn.Module, key: str, delta: torch.Tensor) -> None:
        """Subtract ``delta`` from the expert's 3D-Parameter slice, in place."""
        name, proj, e = _split_key(key)  # type: ignore[misc]
        param = getattr(dict(model.named_modules())[name], proj)
        param.data[e] -= delta.to(param.dtype)


def apply_engram_weights(model: nn.Module, weight_engrams: Dict[str, torch.Tensor], alpha: float = 1.0) -> None:
    """Low-level: subtract ``alpha * weight_engrams[key]`` from each weight, handling fused keys.

    ``weight_engrams`` is a plain ``{key: delta}`` of **final** per-key tensors (already
    scaled by any per-layer factor). Real-module keys edit ``module.weight``; fused-expert
    keys (``"<experts>.gate_up_proj.<e>"``) edit ``param.data[e]`` of the 3D Parameter. The
    primary path is :meth:`EngramEditor.apply`, which dispatches fused keys here via the
    adapter and applies the pluggable scaling for you; use this only when you hold raw deltas.
    """
    modules = dict(model.named_modules())
    with torch.no_grad():
        for key, w in weight_engrams.items():
            if key in modules:
                modules[key].weight.data -= (alpha * w).to(modules[key].weight.dtype)
                continue
            split = _split_key(key)
            if split is None:
                continue
            name, proj, e = split
            param = getattr(modules[name], proj)
            param.data[e] -= (alpha * w).to(param.dtype)
