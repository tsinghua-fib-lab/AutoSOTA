# RACING-GCCA translator
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
from cycloreps.translator.translator import MultiSpaceBase


class GeneralizedCCATranslator(MultiSpaceBase):
    def __init__(
        self,
        device: str = "cpu",
        shared_rank: Optional[int] = None,
    ) -> None:
        super().__init__(name="racing_gcca_multi_space_translator", device=device)
        self._uses_padding = True
        self.R = shared_rank
        self.Q_out: Dict[str, torch.Tensor] = {}
        self.M: Optional[torch.Tensor] = None
        self._dims: Dict[str, int] = {}
        self._N = 0
        self._d = 0

    def _fit_impl(self, spaces_std: Dict[str, torch.Tensor], dims: Dict[str, int]) -> None:
        Q_out, M = self.align(spaces_std, dims, device=self.device, shared_rank=self.R)
        self.Q_out, self.M = Q_out, M
        self._dims = dims
        any_name = next(iter(spaces_std))
        self._N, self._d = spaces_std[any_name].shape
        self.T_out.clear()
    
    def _to_universe_impl(self, z: torch.Tensor, *, src: str) -> torch.Tensor:
        return z @ self.Q_out[src].to(z.device, z.dtype)

    def _from_universe_impl(self, u: torch.Tensor, *, tgt: str) -> torch.Tensor:
        return u @ self.Q_out[tgt].to(u.device, u.dtype).T

    def _pairwise_map_impl(self, src: str, tgt: str) -> torch.Tensor:
        return self.Q_out[src] @ self.Q_out[tgt].T


    @torch.no_grad()
    def align(
        self,
        spaces: Dict[str, torch.Tensor], # standardized & padded
        dims: Dict[str, int], # view name -> active width (<= d)
        *,
        device: str = "cpu",
        shared_rank: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        RACING-GCCA.

        Inputs
        ------
        spaces:
            These are our padded and standardised spaces.

        dims:
            Dict mapping space/view_name -> d_view 
        
        device:
            CPU or GPU
        
        shared_rank:
            Desired dimensionality R of the shared representation.
            If None, R is chosen as min(min(d_view), N), min(d_view) is the min view dimension, N is the number of samples
        
        Outputs
        -------
        (Q_out, M)
        Q_out:
            Dict view_name -> Q_full of shape (d, R).
            Q_full contains per-view loadings in the *original* feature coordinates, padded back to the global width d.
            Only the first dims[view_name] rows contain nonzero information; trailing rows are zero (padding).
        M:
            Tensor of shape (N, R).
            Columns of M form an orthonormal basis for the shared sample subspace across all views.
        
        Steps
        ----------------------------------
        1) For each view, take a thin SVD of the active submatrix X_view[:, :d_view] and keep up to r_view columns,
            where r_view = min(R + private_allowance, N, d_view). The private allowance lets each view keep some
            extra variance beyond the shared rank before we build the small problem.

        2) Build the small square matrix S of shape (sum_r, sum_r), where sum_r = sum(r_view).
            Its diagonal blocks are scaled identities and its off-diagonal blocks are -U_i.T @ U_j,
            where U_i are the left singular vectors from step (1).

        3) Compute the eigen-decomposition of S and keep the eigenvectors for the smallest R eigenvalues.
            Stack them into Phi of shape (sum_r, R) and slice Phi per view to get Phi_view of shape (r_view, R).

        4) Build G by horizontally concatenating U_view @ Phi_view for all views (each piece is (N, R); G is (N, M*R)).
            Take the top-R left singular vectors of G; this gives M (N, R), the shared sample basis.

        5) For each view, map back to original feature coordinates via Q_view = V_view @ Phi_view (d_view, R),
            then pad Q_view with zeros to (d, R) so all Qs share the same global width.
        """

        names = list(spaces.keys())

        # Basic shape check: all (N,d) equal
        N, d = spaces[names[0]].shape
        if any(t.shape != (N, d) for t in spaces.values()):
            raise ValueError("All spaces must share shape (N, d) — pad first if needed.")
        X = {n: spaces[n].to(device, dtype=torch.float32) for n in names}

        # Choose R (shared universe dim) safely
        if shared_rank is None:
            R = min(min(int(dims[n]) for n in names), N)
        else:
            R = min(int(shared_rank), min(int(dims[n]) for n in names), N)

        self.R = R

        # Per-view extra allowance beyond R for SVD truncation.
        L = {n: max(0, int(dims[n]) - R) for n in names}

        # per-view thin SVDs on active columns
        U_list: Dict[str, torch.Tensor] = {}
        V_list: Dict[str, torch.Tensor] = {}
        r_list: Dict[str, int] = {}
        for n in names:
            d_n = int(dims[n]) # active width for view n
            Xn = X[n][:, :d_n] # (N,d_n)
            r_n = min(R + L[n], N, d_n)
            Un, Sn, Vhn = torch.linalg.svd(Xn, full_matrices=False)
            U_list[n] = Un[:, :r_n] # (N,r_n) orthonormal columns
            V_list[n] = Vhn.transpose(-2, -1)[:, :r_n]  # (d_n,r_n) orthonormal columns
            r_list[n] = r_n

        # Offsets for block indexing in the concatenated small space
        offsets: Dict[str, int] = {}
        total_r = 0
        for n in names:
            offsets[n] = total_r
            total_r += r_list[n]

        Mviews = len(names)

        # Step 2: Build S, the small square system of size (total_r, total_r).
        # Layout: blocks correspond to per-view truncated U spaces.
        S = torch.zeros(total_r, total_r, device=device, dtype=torch.float32)

        # Diagonal blocks
        for n in names:
            r_n = r_list[n]
            c0 = offsets[n]
            S[c0:c0+r_n, c0:c0+r_n].diagonal().add_(Mviews - 1)

        # Off-diagonal blocks: set to -U_i.T @ U_j (and mirror to keep S symmetric).
        # Each block captures how similar the truncated U spaces are across views.
        for i in range(Mviews - 1):
            ni = names[i]
            Ui = U_list[ni] # (N,r_i)
            r_i, c_i = r_list[ni], offsets[ni]
            for j in range(i + 1, Mviews):
                nj = names[j]
                Uj = U_list[nj] # (N,r_j)
                r_j, c_j = r_list[nj], offsets[nj]
                Cij = Ui.T @ Uj # (r_i,r_j)
                S[c_i:c_i+r_i, c_j:c_j+r_j].add_(-Cij)
                S[c_j:c_j+r_j, c_i:c_i+r_i].add_(-Cij.T)

        # Step 3: Find the "right-nullspace-like" directions of S by eigen-decomposition.
        # We sort eigenvalues ascending and keep the eigenvectors for the smallest R eigenvalues.
        # Collect them in Phi of shape (total_r, R). Later we slice Phi into per-view pieces using offsets/ranks.
        evals, evecs = torch.linalg.eigh(S) # ascending eigenvalues
        Phi = evecs[:, :R]

        # Step 4: Build G by concatenating U_view @ Phi_view for all views and extract the shared sample basis M.
        # For each view:
        #   - Slice Phi rows for this view's block -> Phi_view (r_n, R)
        #   - Multiply U_view (N, r_n) by Phi_view (r_n, R) -> contribution (N, R)
        # Concatenate contributions across views along feature axis to form G (N, Mviews*R).
        G_parts = []
        Phi_slices: Dict[str, torch.Tensor] = {}
        for n in names:
            c0 = offsets[n]; r_n = r_list[n]
            Phi_n = Phi[c0:c0+r_n, :] # (r_n,R)
            Phi_slices[n] = Phi_n
            G_parts.append(U_list[n] @ Phi_n) # (N,R)
        G = torch.cat(G_parts, dim=1) # (N, Mviews * R)

        # Get the top-R left singular vectors of G.
        # These span the shared subspace across samples, giving us M with orthonormal columns.
        U_m, _, _ = torch.linalg.svd(G, full_matrices=False)
        M = U_m[:, :R] # (N,R) orthonormal shared basis

        # Step 5: Map each view back to its original feature coordinates.
        # For a view:
        #   - Take V_view (d_n, r_n) from the SVD of X_view
        #   - Multiply by Phi_view (r_n, R) -> Q_view (d_n, R)
        #   - Zero-pad Q_view to (d, R) so all views share the same global height.
        Q_out: Dict[str, torch.Tensor] = {}
        for n in names:
            d_n = int(dims[n])
            Vn = V_list[n] # (d_n,r_n)
            Phi_n = Phi_slices[n] # (r_n,R)
            Qn = Vn @ Phi_n # (d_n,R) rectangular loading
            Q_full = torch.zeros(d, R, device=device, dtype=torch.float32)
            Q_full[:d_n, :] = Qn # pad to global width d
            Q_out[n] = Q_full

        # Return per-view loadings (padded) and the shared sample basis
        return Q_out, M
