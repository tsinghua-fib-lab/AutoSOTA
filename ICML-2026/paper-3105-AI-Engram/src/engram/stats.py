"""Statistics container: per-layer **mean** input covariance + sample counts.

A :class:`Statistics` holds, per layer (and per fused-MoE expert), the *mean*
input covariance ``C = mean_k(x_k^T x_k)`` over the rows (tokens) that entered it,
plus the integer count ``N`` of those rows. Storing the mean — not the raw sum
``sum x^T x`` the earlier versions used — keeps the magnitude bounded (~``E[x^2]``)
regardless of corpus size, and makes :meth:`Statistics.merge` a count-weighted
average rather than a plain sum.

The closed-form engram is recovered **exactly** from means + counts: the paper's
``W . Sigma_target . pinv(Sigma_total)`` equals
``(n / N) . W . C_target . pinv(C_total)`` because ``pinv`` is scale-invariant (its
rcond cut is relative to the largest singular value). The ``n / N`` factor is applied
at edit time through the default scaling function — see :func:`engram.scaling.count_ratio`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Union

import torch

_FORMAT = 2  # on-disk tag; the legacy unreleased "sum-only dict" (no tag) is rejected on load


@dataclass
class Statistics:
    """Per-layer mean input covariance (``cov``) and sample count (``count``).

    ``cov[name]`` is the mean of ``x^T x`` over the ``count[name]`` rows that entered
    layer ``name`` during collection. Keys are module names, or
    ``"<experts>.gate_up_proj.<e>"`` / ``"....down_proj.<e>"`` for fused-MoE experts.
    Behaves like a read-only mapping over ``cov`` (``stats[name]``, ``name in stats``,
    iteration) with the parallel ``count`` dict alongside.
    """

    cov: Dict[str, torch.Tensor] = field(default_factory=dict)
    count: Dict[str, int] = field(default_factory=dict)

    # ---- mapping-like surface over cov (count is the parallel dict) ----
    def __getitem__(self, key: str) -> torch.Tensor:
        return self.cov[key]

    def __contains__(self, key: str) -> bool:
        return key in self.cov

    def __iter__(self) -> Iterator[str]:
        return iter(self.cov)

    def __len__(self) -> int:
        return len(self.cov)

    def keys(self):
        return self.cov.keys()

    def items(self):
        return self.cov.items()

    def to(self, device: Union[str, torch.device]) -> "Statistics":
        """Move every covariance to ``device`` (counts are plain ints, copied as-is)."""
        return Statistics({k: v.to(device) for k, v in self.cov.items()}, dict(self.count))

    @staticmethod
    def merge(*stats: "Statistics") -> "Statistics":
        """Count-weighted merge: ``C = sum(n_i C_i) / sum(n_i)``, ``N = sum(n_i)``.

        Equivalent to having collected over the concatenated token streams. Keys are
        unioned; a key present in only some inputs contributes only its own
        ``(count, mean)``. Combined incrementally (``C += n_i/(N+n_i) (C_i - C)``) so
        no large ``n_i * C_i`` intermediate is formed.
        """
        cov: Dict[str, torch.Tensor] = {}
        count: Dict[str, int] = {}
        for s in stats:
            for k, c in s.cov.items():
                n = int(s.count.get(k, 0))
                if k not in cov:
                    cov[k] = c.to(torch.float32).clone()
                    count[k] = n
                    continue
                prev = count[k]
                total = prev + n
                if total > 0:
                    c = c.to(cov[k].device, torch.float32)
                    cov[k] = cov[k] + (n / total) * (c - cov[k])
                count[k] = total
        return Statistics(cov, count)

    def save(self, path: Union[str, Path]) -> None:
        """Save with ``torch.save`` (tagged ``format=2``)."""
        torch.save({"format": _FORMAT, "cov": self.cov, "count": self.count}, path)

    @staticmethod
    def load(
        path: Union[str, Path], map_location: Union[str, torch.device, None] = None
    ) -> "Statistics":
        """Load a :class:`Statistics`. Rejects the legacy raw-covariance dict format."""
        obj = torch.load(path, map_location=map_location, weights_only=True)
        if not (isinstance(obj, dict) and obj.get("format") == _FORMAT and "cov" in obj):
            raise ValueError(
                f"{path!r} is not a Statistics file (format={_FORMAT}). It looks like a "
                "legacy raw-covariance dict; re-collect with EngramEditor.collect_statistics() "
                "— the mean+count format cannot be reconstructed from summed covariances."
            )
        return Statistics(obj["cov"], obj["count"])
