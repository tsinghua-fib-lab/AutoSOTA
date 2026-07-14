from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from latentis.transform import Estimator

from cycloreps.translator.translator import MultiSpaceBase
from cycloreps.utils.dim_matcher import ZeroPaddingN
from tqdm import tqdm
import torch.nn.functional as F


class LinearMultiSpaceTranslator(MultiSpaceBase):
    """
    N-space translator with *reconstruction* maps (general linear maps to a shared latent).
    No zero-padding; chooses an explicit `out_dim`.
    """

    def __init__(
        self,
        *,
        out_dim: int,
        reg: float = 0.0,  # Frobenius norm regularization on T
        enforce_norm: Optional[float] = None,  # enforce ||T[p]||_F
        lr: float = 0.01,  # learning rate
        ortho_reg: float = 0.0,  # encourage orthogonality within T (if desired)
        max_iter: int = 50,
        tol: float = 1e-6,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            name="reconstruction_multi_space_translator",
            max_iter=max_iter,
            tol=tol,
            device=device,
        )
        self._alignment_kind = "reconstruction"
        self._uses_padding = False

        self.out_dim = out_dim
        self.reg = reg
        self.enforce_norm = enforce_norm
        self.lr = lr
        self.ortho_reg = ortho_reg

    @property
    def metadata(self) -> Mapping[str, Any]:
        md = super().metadata
        md.update(
            dict(
                out_dim=self.out_dim,
                reg=self.reg,
                enforce_norm=self.enforce_norm,
                lr=self.lr,
                ortho_reg=self.ortho_reg,
            )
        )
        return md

    def _fit_impl(
        self, spaces_std: Dict[str, torch.Tensor], dims: Dict[str, int]
    ) -> None:
        # dims unused in reconstruction, but kept for signature parity
        _ = dims
        if self.out_dim is None:
            raise ValueError(
                "`out_dim` must be specified for reconstruction alignment."
            )

        self.T_out = self.align(
            spaces=spaces_std,
            out_dim=self.out_dim,
            max_iter=self.max_iter,
            tol=self.tol,
            device=self.device,
            reg=self.reg,
            enforce_norm=self.enforce_norm,
            lr=self.lr,
            ortho_reg=self.ortho_reg,
        )
        self.R_out.clear()  # not used here

    def _to_universe_impl(self, z: torch.Tensor, *, src: str) -> torch.Tensor:
        # map to shared latent via T[src]^T (consistent with your original)
        return z @ self.T_out[src].to(z.device, z.dtype).T

    def _from_universe_impl(self, u: torch.Tensor, *, tgt: str) -> torch.Tensor:
        # (pseudo-)inverse to return to tgt centred space
        T_inv = torch.linalg.pinv(self.T_out[tgt].to(u.device, u.dtype))
        return u @ T_inv.T

    def _pairwise_map_impl(self, src: str, tgt: str) -> torch.Tensor:
        # centred linear map (standardized widths): (T_src^T) @ pinv(T_tgt)
        T_src = self.T_out[src]
        T_tgt = self.T_out[tgt]
        T_tgt_inv = torch.linalg.pinv(T_tgt)
        return T_src.T @ T_tgt_inv.T

    def align(
        self,
        spaces: Dict[str, torch.Tensor],
        *,
        out_dim: int,
        max_iter: int = 50,
        tol: float = 1e-6,
        device: str = "cpu",
        reg: float = 0.0,  # Frobenius norm regularization strength
        enforce_norm: float = None,  # If set, enforce all T[p] to have this Frobenius norm
        lr: float = 0.01,  # learning rate for gradient descent
        ortho_reg: float = 0.0,  # Orthogonality regularization strength
        det_reg: float = 0.0,  # Determinant regularization to encourage invertibility
    ) -> Dict[str, torch.Tensor]:
        """
        Cycle-consistent alignment using reconstruction losses.

        This method minimizes reconstruction errors when transforming embeddings
        between different spaces, ensuring that transformations are consistent
        across all pairs of spaces.

        Parameters
        ----------
        spaces   : dict model_name -> (N, d) tensor (same N across models, d can differ)
        out_dim  : output embedding dimension after alignment
        max_iter : maximum number of iterations
        tol      : tolerance for convergence
        device   : device to run on
        reg      : regularization strength for Frobenius norm of T[p]
        enforce_norm : if set, enforce all T[p] to have this Frobenius norm
        lr       : learning rate for gradient descent
        ortho_reg : regularization strength to enforce T[p] to be almost orthogonal
                    (penalizes ||T[p]^T T[p] - I||^2)

        Returns
        -------
        T_out : dict model_name -> (out_dim, d) transformation matrix
        """
        # Move to device and get dimensions
        names = list(spaces.keys())
        Z = {name: spaces[name].to(device) for name in names}
        dims = {name: Z[name].shape[1] for name in names}

        # Initialize transformation matrices
        T = {}
        for name in names:
            # Initialize with random orthogonal matrix scaled by sqrt(out_dim / d)
            d = dims[name]
            T[name] = torch.randn(out_dim, d, device=device)
            # Orthogonalize and scale
            U, _, Vt = torch.linalg.svd(T[name], full_matrices=False)
            T[name] = U @ Vt * (out_dim / d) ** 0.5

        def objective() -> torch.Tensor:
            """Compute reconstruction loss objective."""
            total_loss = 0.0

            # Pairwise reconstruction losses
            for m in names:
                for n in names:
                    if m != n:
                        # Transform m -> n -> m and compute reconstruction loss
                        Z_m = Z[m]  # (N, d_m)
                        Z_n = Z[n]  # (N, d_n)

                        # Forward transform: m -> n
                        T_m = T[m]  # (out_dim, d_m)
                        T_n = T[n]  # (out_dim, d_n)

                        # Transform m to common space, then to n space
                        Z_m_common = Z_m @ torch.linalg.pinv(T_m)  # (N, out_dim)
                        Z_m_to_n = Z_m_common @ T_n  # (N, d_n)

                        # Reconstruction loss: ||Z_m_to_n - Z_n||^2
                        # recon_loss = torch.norm(Z_m_to_n - Z_n, p='fro')**2 / (Z_m.shape[0] * Z_m.shape[1])
                        # total_loss += recon_loss

                        # Cosine similarity loss: minimize angle between corresponding embeddings
                        # Normalize embeddings to unit vectors
                        Z_m_to_n_norm = F.normalize(Z_m_to_n, p=2, dim=1)  # (N, d_n)
                        Z_n_norm = F.normalize(Z_n, p=2, dim=1)  # (N, d_n)

                        # Compute cosine similarity for each pair of corresponding embeddings
                        cosine_sim = torch.sum(Z_m_to_n_norm * Z_n_norm, dim=1)  # (N,)

                        # Loss = 1 - cosine_similarity (minimize angle)
                        angle_loss = torch.mean(1 - cosine_sim)
                        # if n == 'DINOv2':
                        #     angle_loss = 5.0*angle_loss
                        total_loss += angle_loss

            # Add regularization
            if reg > 0:
                reg_loss = reg * sum(torch.norm(T[p], p="fro") ** 2 for p in names)
                total_loss += reg_loss

            # Add orthogonality regularization: penalize ||T[p]^T T[p] - I||^2
            if ortho_reg > 0:
                for p in names:
                    # T[p] has shape (out_dim, d), so T[p]^T @ T[p] has shape (d, d)
                    gram_matrix = T[p].T @ T[p]  # (d, d)
                    identity = torch.eye(dims[p], device=device)
                    total_loss += (
                        ortho_reg * torch.norm(gram_matrix - identity, p="fro") ** 2
                    )

            return total_loss

        # Main optimization loop
        prev_f = objective()
        for it in tqdm(range(max_iter)):
            # Enable gradients for all T[p]
            for p in names:
                T[p].requires_grad_(True)

            # Compute full objective
            obj = objective()

            # Compute gradients for all T[p]
            with torch.enable_grad():
                obj = objective()
                assert obj.requires_grad, "loss has no grad; check no_grad/inference_mode"
                obj.backward()

            # Update all T[p] simultaneously
            with torch.no_grad():
                for p in names:
                    T[p] = T[p] - lr * T[p].grad
                    # T[p].grad.zero_()
                    T[p].requires_grad_(False)

                    # Check for NaNs
                    if torch.isnan(T[p]).any():
                        raise RuntimeError(
                            f"NaNs detected in T['{p}'] at iteration {it}. Check your data and regularization."
                        )

                    # Enforce Frobenius norm if requested
                    if enforce_norm is not None:
                        norm = torch.norm(T[p], p="fro")
                        if norm > 0:
                            T[p] = T[p] * (enforce_norm / norm)

            # Check convergence
            curr_f = objective()
            if it % 10 == 0:
                print(f"Iteration {it}: {curr_f:.2f}")
            if abs(curr_f - prev_f) < tol:
                print(f"Converged at iteration {it}")
                break
            prev_f = curr_f

        T_out = {m: T[m].detach().cpu() for m in names}
        return T_out
