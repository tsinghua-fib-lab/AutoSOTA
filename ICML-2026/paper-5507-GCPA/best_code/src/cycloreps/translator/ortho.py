from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Tuple
import torch
import torch.nn.functional as F
import itertools

from latentis.data import DATA_DIR
from latentis.space._base import Space

from latentis.transform.base import StandardScaling
from latentis.transform.dim_matcher import ZeroPadding
from latentis.transform.translate.aligner import MatrixAligner, Translator
from latentis.transform.translate.functional import (
    svd_align_state,
    lstsq_align_state,
    lstsq_ortho_align_state,
)

from typing import Dict, Tuple
from tqdm import tqdm


from typing import Dict
import torch
from tqdm.auto import tqdm
import torch
from latentis.transform import Estimator

from cycloreps.translator.translator import MultiSpaceBase
from cycloreps.utils.dim_matcher import ZeroPaddingN


class OrthogonalMultiSpaceTranslator(MultiSpaceBase):
    """
    N-space translator with *orthogonal* maps (incl. Generalized Procrustes).
    Uses ZeroPaddingN to equalize widths.
    """

    def __init__(
        self,
        *,
        max_iter: int = 50,
        tol: float = 1e-6,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            name="orthogonal_multi_space_translator",
            max_iter=max_iter,
            tol=tol,
            device=device,
        )
        self._uses_padding = True

    def _fit_impl(
        self, spaces_std: Dict[str, torch.Tensor], dims: Dict[str, int]
    ) -> None:
        self.R_out = self.align(
            dims=dims,
            spaces=spaces_std,
            max_iter=self.max_iter,
            tol=self.tol,
            device=self.device,
        )
        self.T_out.clear()  # not used here

    def _to_universe_impl(self, z: torch.Tensor, *, src: str) -> torch.Tensor:
        return z @ self.R_out[src]

    def _from_universe_impl(self, u: torch.Tensor, *, tgt: str) -> torch.Tensor:
        return u @ self.R_out[tgt].T

    def _pairwise_map_impl(self, src: str, tgt: str) -> torch.Tensor:
        # centred linear map in standardized, padded widths
        return self.R_out[src] @ self.R_out[tgt].T

    @torch.no_grad()
    def align(
        self,
        spaces: Dict[str, torch.Tensor],
        dims,
        *,
        max_iter: int = 50,
        tol: float = 1e-6,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        Orthogonal rotation-synchronisation (cycle-consistent alignment).

        Parameters
        ----------
        spaces : dict[str, Tensor]   each tensor (N, d) **same shape across models**
        max_iter, tol, device : optimisation controls

        Returns
        -------
        R_out  : dict[str, Tensor]   orthogonal matrices (d, d) for every model
        """

        # --- basic checks ---------------------------------------------------- #
        names = list(spaces)
        N, d = spaces[names[0]].shape
        if any(t.shape != (N, d) for t in spaces.values()):
            raise ValueError(
                "All spaces must share shape (N, d) — pad first if needed."
            )

        # --- move data to device -------------------------------------------- #
        X = {n: t.to(device, torch.float32) for n, t in spaces.items()}

        # --- initialise rotations ------------------------------------------- #
        # R = {n: torch.eye(d, device=device) for n in names}
        R = {n: torch.randn(d, d, device=device) for n in names}

        def obj() -> torch.Tensor:
            """∑_{m≠n} tr(R_mᵀ X_mᵀ X_n R_n)"""
            return sum(
                torch.trace(R[m].T @ (X[m].T @ X[n]) @ R[n])
                for m in names
                for n in names
                if m != n
            )

        prev = obj()

        # --- main loop ------------------------------------------------------- #
        for _ in tqdm(range(max_iter)):
            for p in names:
                Gp = sum(
                    X[p].T @ X[n] @ R[n] for n in names if n != p
                )  # full d_max×d_max

                d_p = dims[p]  # native width of encoder p
                G_block = Gp[:d_p, :d_p]  # isolate real feature sub-matrix

                U, _, Vt = torch.linalg.svd(G_block, full_matrices=False)
                R_block = U @ Vt  # (d_p × d_p) orthogonal

                # overwrite R[p] as block-diagonal  diag(R_block , I_pad)
                R[p].zero_()
                R[p][:d_p, :d_p] = R_block
                if d_p < d:
                    R[p][d_p:, d_p:] = torch.eye(d - d_p, device=device)

            cur = obj()
            if abs((cur - prev) / (prev + 1e-12)) < tol:
                break
            prev = cur

        return {n: R[n].cpu() for n in names}
