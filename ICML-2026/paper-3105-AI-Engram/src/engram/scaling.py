"""Pluggable per-layer scaling functions for engram edits.

The edit subtracts, per layer ``l``, ``alpha * f_l * P_l`` where
``P_l = W_l . C_target . pinv(C_total)`` is the pure projection (no sample-count
factor) and ``f_l`` is a scalar from a **scaling function**. ``alpha`` is the paper's
global edit strength; ``f_l`` weights each layer's edit relative to the others.

A :data:`ScaleFn` maps the whole ``{name: LayerScaleInfo}`` dict to ``{name: float}``
— it sees every layer at once, so it can normalize globally (e.g. divide by the max).

Built-ins:

- :func:`count_ratio` ``f_l = (n_l / N_l) ** power``  — **default**; ``power=1`` is the paper.
- :func:`weight_norm`  ``f_l = (rel_l / max rel) ** power``, ``rel_l = ||P_l|| / ||W_l||``.
- :func:`effective_rank` ``f_l = (er(C_target) / er(C_total)) ** power`` per layer.
- :func:`uniform`      ``f_l = 1``.
- :func:`compose`      multiply several together, e.g. ``compose(count_ratio(1.0), weight_norm(1.0))``.

The default is ``count_ratio(1.0)``: the paper's engram is exactly ``(n / N) . P`` (see
:mod:`engram.stats`), so subtracting ``alpha * (n / N) * P`` at ``alpha=1`` is the paper edit.

**Dense vs MoE.** In a dense model every token passes through every layer, so ``n / N`` is
the *same constant* for all layers — :func:`count_ratio` is then a global rescale (folds
into ``alpha``), and per-layer structure comes from :func:`weight_norm` /
:func:`effective_rank`. For fused-MoE experts the routed counts ``n_e / N_e`` differ per
expert, so :func:`count_ratio` carries genuine per-expert weighting there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import torch

_EPS = 1e-8


def _erank(cov: torch.Tensor) -> float:
    """Effective rank: ``exp(H)`` of the normalized eigenvalue spectrum (Roy & Vetterli, 2007).

    Scale-invariant (so the mean covariance gives the same value as the summed one);
    ranges in ``[1, D]``. Returns 0 for an all-zero covariance.
    """
    ev = torch.linalg.eigvalsh(cov.to(torch.float32)).clamp(min=0)
    s = ev.sum()
    if s <= 0:
        return 0.0
    p = ev / s
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


@dataclass
class LayerScaleInfo:
    """Per-layer inputs handed to a scaling function."""

    name: str
    weight_fro: float          # ||W_l||_F (scalar); the full weight tensor is not retained — see weight_norm
    projection: torch.Tensor   # P_l = W . C_target . pinv(C_total), in module.weight shape
    n: int                     # target sample count
    N: int                     # total sample count
    target_erank: Optional[float] = None  # effective rank of C_target — only set for effective_rank
    total_erank: Optional[float] = None   # effective rank of C_total


@dataclass
class EngramResult:
    """Output of :meth:`EngramEditor.compute_engram_weights`: projections + scale inputs.

    ``layers[name].projection`` is the pure engram projection (no ``n / N``);
    ``bias[name]`` is the matching bias projection for bias-absorbed layers. Pass the
    whole result to :meth:`EngramEditor.apply`.
    """

    layers: Dict[str, LayerScaleInfo] = field(default_factory=dict)
    bias: Dict[str, torch.Tensor] = field(default_factory=dict)


ScaleFn = Callable[[Dict[str, "LayerScaleInfo"]], Dict[str, float]]


def count_ratio(power: float = 1.0) -> ScaleFn:
    """``f_l = (n_l / N_l) ** power``. Default (``power=1``) reproduces the paper exactly.

    ``n_l / N_l`` is the target/total sample-count ratio implicit in the paper's summed
    covariances. ``n == 0`` or ``N == 0`` -> 0 (a layer no target token reached is not edited).
    """

    def fn(infos: Dict[str, LayerScaleInfo]) -> Dict[str, float]:
        return {
            k: (i.n / i.N) ** power if (i.n > 0 and i.N > 0) else 0.0
            for k, i in infos.items()
        }

    return fn


def weight_norm(power: float = 1.0) -> ScaleFn:
    """``f_l = (rel_l / max rel) ** power`` with ``rel_l = ||P_l|| / ||W_l||``.

    Edits each layer in proportion to how strongly the engram occupies it (the relative
    Frobenius norm). This is the per-layer "adaptive" weighting; compose with
    :func:`count_ratio` to also keep the paper's ``n / N`` factor.
    """

    def fn(infos: Dict[str, LayerScaleInfo]) -> Dict[str, float]:
        rel = {
            k: i.projection.norm().item() / (i.weight_fro + _EPS)
            for k, i in infos.items()
        }
        mx = max(rel.values(), default=1.0) or 1.0
        return {k: (r / mx) ** power for k, r in rel.items()}

    return fn


def effective_rank(power: float = 1.0) -> ScaleFn:
    """``f_l = (er(C_target_l) / er(C_total_l)) ** power`` per layer.

    The ratio of the **effective rank** (see :func:`_erank`) of the target covariance to
    that of the total — how dimensionally rich the forget inputs are relative to the whole
    at each layer. Needs the per-layer effective ranks, i.e.
    ``compute_engram_weights(..., compute_erank=True)``.
    """

    def fn(infos: Dict[str, LayerScaleInfo]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, i in infos.items():
            if i.target_erank is None or i.total_erank is None:
                raise ValueError(
                    "effective_rank needs per-layer effective ranks; recompute with "
                    "compute_engram_weights(..., compute_erank=True)."
                )
            ratio = (i.target_erank / i.total_erank) if i.total_erank > 0 else 0.0
            out[k] = ratio ** power
        return out

    return fn


def uniform() -> ScaleFn:
    """``f_l = 1`` for every layer (subtract the bare projection, ignoring counts/norms)."""

    def fn(infos: Dict[str, LayerScaleInfo]) -> Dict[str, float]:
        return {k: 1.0 for k in infos}

    return fn


def compose(*fns: ScaleFn) -> ScaleFn:
    """Multiply scaling functions per layer: ``f_l = prod_i fns[i](infos)[l]``.

    e.g. ``compose(count_ratio(1.0), weight_norm(1.0))`` is the paper's ``n / N``
    weighting further modulated by relative weight-norm.
    """

    def fn(infos: Dict[str, LayerScaleInfo]) -> Dict[str, float]:
        out: Dict[str, float] = {k: 1.0 for k in infos}
        for f in fns:
            d = f(infos)
            for k in out:
                out[k] *= d.get(k, 0.0)
        return out

    return fn
