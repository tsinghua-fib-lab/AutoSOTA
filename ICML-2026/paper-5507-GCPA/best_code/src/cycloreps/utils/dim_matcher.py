# latentis/transform/dim_matcher/zero_padding_n.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from latentis.transform.dim_matcher import DimMatcher


def pad_to_dimension(X: np.ndarray, dim: int) -> np.ndarray:
    """
    Pad a 2D array X of shape (N, D) with zeros so that it becomes (N, dim).

    Args:
        X (np.ndarray): Input array of shape (N, D).
        dim (int): Desired second dimension (must be >= D).

    Returns:
        np.ndarray: Zero-padded array of shape (N, dim).
    """
    N, D = X.shape

    if D > dim:
        raise ValueError(f"Target dim {dim} must be >= current dim {D}.")

    return np.pad(X, ((0, 0), (0, dim - D)), mode="constant")


class ZeroPaddingN(DimMatcher):
    """
    Zero-pad each (N, d_k) matrix on the right so all spaces share width D = max d_k at fit-time.

    Dict-first API:
        zp = ZeroPaddingN().fit(spaces)            # spaces: dict[str, np.ndarray | torch.Tensor] with shape (N, d)
        spaces_pad = zp.transform(spaces)
        spaces_unpad = zp.inverse_transform(spaces_pad)

    2-way convenience:
        zp.fit_pair(x, y);  x_pad, y_pad = zp.transform_pair(x, y);  x, y = zp.inverse_transform_pair(x_pad, y_pad)

    Notes:
      • Returns match the input type per key (NumPy in → NumPy out; torch in → torch out).
      • Only feature width may differ; batch size N must match across keys in a call.
    """

    def __init__(self):
        super().__init__(name="zero_padding_n", invertible=True)
        self._pad_by_key: Dict[str, int] = {}
        self._max_dim: int = -1

    # ------------------------------- FIT -------------------------------- #
    def fit(self, spaces: Dict[str, np.ndarray | torch.Tensor]) -> "ZeroPaddingN":
        self._validate_spaces(spaces)
        # Compute max width using shapes as provided (no conversion needed)
        max_dim = max(self._feat_dim(v) for v in spaces.values())
        self._pad_by_key = {k: (max_dim - self._feat_dim(v)) for k, v in spaces.items()}
        self._max_dim = max_dim

        # Optional: register buffers for serialization (if DimMatcher supports it)
        state = {"max_dim": torch.tensor(max_dim, dtype=torch.long)}
        state.update(
            {
                f"pad_{k}": torch.tensor(p, dtype=torch.long)
                for k, p in self._pad_by_key.items()
            }
        )
        self._register_state(state)
        return self

    # ---------------------------- TRANSFORM ----------------------------- #
    def transform(
        self, spaces: Dict[str, np.ndarray | torch.Tensor]
    ) -> Dict[str, np.ndarray | torch.Tensor]:
        self._ensure_fitted()
        self._validate_spaces(spaces, allow_new_keys=True)

        out: Dict[str, np.ndarray | torch.Tensor] = {}
        for k, arr in spaces.items():
            ten, typ, meta = self._to_tensor(arr)
            pad = self._pad_by_key.get(k, self._max_dim - ten.shape[1])
            ten_out = F.pad(ten, (0, int(pad))) if pad > 0 else ten
            out[k] = self._to_original_type(ten_out, typ, meta)
        return out

    def inverse_transform(
        self, spaces: Dict[str, np.ndarray | torch.Tensor]
    ) -> Dict[str, np.ndarray | torch.Tensor]:
        self._ensure_fitted()
        self._validate_spaces(spaces, allow_new_keys=True)
        out: Dict[str, np.ndarray | torch.Tensor] = {}
        for k, arr in spaces.items():
            ten, typ, meta = self._to_tensor(arr)
            pad = self._pad_by_key.get(k, max(ten.shape[1] - self._max_dim, 0))
            ten_out = ten[..., : ten.shape[1] - int(pad)] if pad > 0 else ten
            out[k] = self._to_original_type(ten_out, typ, meta)
        return out

    # -------------------------- PAIR ADAPTERS --------------------------- #
    def fit_pair(
        self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> "ZeroPaddingN":
        return self.fit({"x": x, "y": y})

    def transform_pair(
        self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        out = self.transform({"x": x, "y": y})
        return out["x"], out["y"]

    def inverse_transform_pair(
        self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        out = self.inverse_transform({"x": x, "y": y})
        return out["x"], out["y"]

    # ----------------------------- UTILS -------------------------------- #
    def _ensure_fitted(self):
        if self._max_dim < 0:
            raise RuntimeError("ZeroPaddingN must be fitted before transform().")

    @staticmethod
    def _feat_dim(v: np.ndarray | torch.Tensor) -> int:
        return int(v.shape[1])

    @staticmethod
    def _batch_dim(v: np.ndarray | torch.Tensor) -> int:
        return int(v.shape[0])

    @staticmethod
    def _to_tensor(x: np.ndarray | torch.Tensor) -> Tuple[torch.Tensor, str, dict]:
        """
        Convert input to a torch.Tensor for internal ops.
        Returns (tensor, type_tag, meta) where:
          - type_tag in {"torch", "numpy"}
          - meta carries dtype/device to restore on output
        """
        if torch.is_tensor(x):
            return (
                x,
                "torch",
                {
                    "device": x.device,
                    "dtype": x.dtype,
                    "requires_grad": x.requires_grad,
                },
            )
        if isinstance(x, np.ndarray):
            ten = torch.from_numpy(x)  # shares memory when possible
            return ten, "numpy", {"dtype": x.dtype}  # numpy has no device
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}.")

    @staticmethod
    def _to_original_type(
        t: torch.Tensor, typ: str, meta: dict
    ) -> np.ndarray | torch.Tensor:
        if typ == "torch":
            # Keep original dtype/device; padding ops keep dtype/device already
            # Preserve requires_grad flag semantics
            if meta.get("requires_grad", False) and not t.requires_grad:
                t = t.requires_grad_()
            return t
        # typ == "numpy": ensure on CPU and detach
        return t.detach().cpu().numpy()

    @staticmethod
    def _validate_spaces(
        spaces: Dict[str, np.ndarray | torch.Tensor],
        allow_new_keys: bool = False,  # kept for parity; not used here but could be for stricter behavior
    ) -> None:
        if not isinstance(spaces, dict) or not spaces:
            raise ValueError(
                "`spaces` must be a non-empty dict[str, np.ndarray | torch.Tensor]."
            )

        N: Optional[int] = None
        for k, v in spaces.items():
            if not (torch.is_tensor(v) or isinstance(v, np.ndarray)):
                raise TypeError(
                    f"spaces['{k}'] must be np.ndarray or torch.Tensor, got {type(v)}."
                )
            if v.ndim != 2:
                raise ValueError(
                    f"spaces['{k}'] must be 2D (N, d). Got shape {tuple(v.shape)}."
                )
            n = v.shape[0]
            if N is None:
                N = n
            elif n != N:
                raise ValueError(
                    f"All spaces must share the same batch size N; got N={N} and N={n} for key '{k}'."
                )
