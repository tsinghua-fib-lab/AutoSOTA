from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from latentis.transform import Estimator
from cycloreps.utils.dim_matcher import ZeroPaddingN


class MultiSpaceBase(Estimator):
    """
    Base class for N-space translators.
    Handles: stats, (optional) zero-padding, universe helpers, caching, metadata, and API.
    Subclasses must implement `_fit_impl(spaces, dims)` and `_to_universe_impl/_from_universe_impl/_pairwise_map_impl`.
    """

    def __init__(
        self,
        *,
        name: str,
        max_iter: int = 50,
        tol: float = 1e-6,
        device: str = "cpu",
    ) -> None:
        super().__init__(name=name, invertible=False)
        self.max_iter, self.tol, self.device = max_iter, tol, device

        # dim matching (only used by subclasses that need it)
        self.dim_matcher = ZeroPaddingN()

        # learned params (subclasses will populate the relevant ones)
        self.R_out: Dict[str, torch.Tensor] = {}
        self.T_out: Dict[str, torch.Tensor] = {}
        self.means: Dict[str, torch.Tensor] = {}
        self.stds: Dict[str, torch.Tensor] = {}

        self._pairwise_cache: Dict[Tuple[str, str], torch.Tensor] = {}
        self._fitted: bool = False

        self._uses_padding: bool = False  # whether to apply ZeroPaddingN

    # ---------- core fit ----------
    @torch.no_grad()
    def fit(self, spaces: Dict[str, torch.Tensor]) -> "MultiSpaceBase":
        """
        Fit alignment on all encoders in `spaces`.
        """
        dims = {name: t.shape[1] for name, t in spaces.items()}

        # (optional) match to same dimension
        if self._uses_padding:
            self.dim_matcher.fit(spaces)
            spaces_padded: Dict[str, torch.Tensor] = self.dim_matcher.transform(spaces)
        else:
            spaces_padded = spaces

        # stats for z-score
        self.means = {
            name: v.mean(0, keepdim=True) for name, v in spaces_padded.items()
        }
        self.stds = {
            name: v.std(0, unbiased=False, keepdim=True)
            for name, v in spaces_padded.items()
        }

        # apply z-score
        spaces_std = {
            name: self._zscore(v, name=name) for name, v in spaces_padded.items()
        }

        self._fit_impl(spaces_std, dims=dims)

        self._pairwise_cache.clear()
        self._fitted = True
        return self

    # to be implemented by subclasses
    def _fit_impl(
        self, spaces_std: Dict[str, torch.Tensor], dims: Dict[str, int]
    ) -> None:
        raise NotImplementedError

    # ---------- universe helpers ----------
    def _pad(self, *, name: str, x: torch.Tensor) -> torch.Tensor:
        if not self._uses_padding:
            return x
        return self.dim_matcher.transform({name: x})[name]

    def _unpad(self, *, name: str, x: torch.Tensor) -> torch.Tensor:
        if not self._uses_padding:
            return x
        return self.dim_matcher.inverse_transform({name: x})[name]

    def to_universe(self, x: torch.Tensor, *, src: str) -> torch.Tensor:
        x = self._pad(name=src, x=x)
        z = self._zscore(x, name=src)
        return self._to_universe_impl(z, src=src)

    def from_universe(self, u: torch.Tensor, *, tgt: str) -> torch.Tensor:
        z = self._from_universe_impl(u, tgt=tgt)
        x_pad = self._un_zscore(z, name=tgt)
        return self._unpad(name=tgt, x=x_pad)

    # to be implemented by subclasses
    def _to_universe_impl(self, z: torch.Tensor, *, src: str) -> torch.Tensor:
        raise NotImplementedError

    def _from_universe_impl(self, u: torch.Tensor, *, tgt: str) -> torch.Tensor:
        raise NotImplementedError

    # ---------- pairwise map (centred) ----------
    def pairwise_map(self, src: str, tgt: str) -> torch.Tensor:
        """
        Linear (centred) map src → tgt in their *standardized widths*.
        Padding/means are handled outside.
        """
        key = (src, tgt)
        if key not in self._pairwise_cache:
            self._pairwise_cache[key] = self._pairwise_map_impl(src, tgt)
        return self._pairwise_cache[key]

    # to be implemented by subclasses
    def _pairwise_map_impl(self, src: str, tgt: str) -> torch.Tensor:
        raise NotImplementedError

    # ---------- public API ----------
    def transform(self, x: torch.Tensor, *, src: str, tgt: str) -> torch.Tensor:
        """Convert a batch from *src* encoder into *tgt* encoder space."""
        assert self._fitted, "Call .fit(...) first."
        u = self.to_universe(x, src=src)
        return self.from_universe(u, tgt=tgt)

    def inverse_transform(self, y: torch.Tensor, *, src: str, tgt: str) -> torch.Tensor:
        """Approximate inverse (tgt → src) via the same two-step path."""
        return self.transform(y, src=tgt, tgt=src)

    # ---------- z-score ----------
    def _zscore(self, x: torch.Tensor, *, name: str) -> torch.Tensor:
        mean = self.means[name].to(device=x.device, dtype=x.dtype)
        std = self.stds[name].to(device=x.device, dtype=x.dtype)
        return (x - mean) / (std + 1e-8)

    def _un_zscore(self, z: torch.Tensor, *, name: str) -> torch.Tensor:
        mean = self.means[name].to(device=z.device, dtype=z.dtype)
        std = self.stds[name].to(device=z.device, dtype=z.dtype)
        return z * std + mean
