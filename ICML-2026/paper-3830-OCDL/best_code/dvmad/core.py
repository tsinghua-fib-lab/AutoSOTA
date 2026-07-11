"""
Core DVM-AD algorithm (NumPy-only, framework-agnostic).

This is the reference implementation from the ICML 2026 paper:
    "DVM-AD: Discriminant Vector Machine for Anomaly Detection".

The PyOD-compatible wrapper lives in ``dvmad.dvmad``. Users that want a
plain NumPy interface (no scikit-learn / PyOD dependency) can use this class
directly.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

try:  # FAISS is optional — accelerates exact NN search
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    FAISS_AVAILABLE = False


class DVMADCore:
    """Closed-form spectral one-class anomaly detector.

    Parameters
    ----------
    mode : {"min", "max", "both"}, default="both"
        Two-tailed eigenvector selection. ``"both"`` keeps the smallest and
        largest discriminant directions; ``"min"`` / ``"max"`` keep only one
        tail.
    eps : float, default=0.1
        Tail threshold in (0, 0.5). Eigenvalues with ``< eps`` (low tail) or
        ``> 1 - eps`` (high tail) are kept.
    artificial_mode : {"max", "min", "farthest"}, default="max"
        How to construct the single artificial reference point used to make
        the within-class scatter non-degenerate in the one-class regime.
    use_faiss : bool, default=True
        Use FAISS for exact NN distance lookup at predict time, when
        available. Falls back to a chunked NumPy implementation otherwise.
    faiss_float32 : bool, default=True
        Cast projected features to float32 (required by FAISS).
    chunk_size : int, default=4096
        Row chunk size for the NumPy NN fallback.
    """

    def __init__(
        self,
        mode: str = "both",
        eps: float = 0.1,
        artificial_mode: str = "max",
        use_faiss: bool = True,
        faiss_float32: bool = True,
        chunk_size: int = 4096,
    ):
        if mode not in {"min", "max", "both"}:
            raise ValueError("mode must be one of {'min', 'max', 'both'}")
        if artificial_mode not in {"max", "min", "farthest"}:
            raise ValueError(
                "artificial_mode must be one of {'max', 'min', 'farthest'}"
            )
        if not (0.0 < eps < 0.5):
            raise ValueError("eps must lie in (0, 0.5) for two-tailed selection")

        self.mode = mode
        self.eps = float(eps)
        self.artificial_mode = artificial_mode
        self.use_faiss = bool(use_faiss) and FAISS_AVAILABLE
        self.faiss_float32 = bool(faiss_float32)
        self.chunk_size = int(chunk_size)

        # learned attributes (populated by fit)
        self.npd_ = None              # discriminant projection (d, m)
        self.basepoint_X_ = None      # projected augmented train data (n+1, m)
        self.filtered_eigvals_ = None # selected eigenvalues (m,)
        self.faiss_index_ = None

    # ------------------------------------------------------------------
    # Reference point construction
    # ------------------------------------------------------------------
    def _construct_reference_point(self, X: np.ndarray) -> np.ndarray:
        if self.artificial_mode == "max":
            return np.max(X, axis=0)
        if self.artificial_mode == "min":
            return np.min(X, axis=0)
        # "farthest"
        mu = np.mean(X, axis=0)
        idx = int(np.argmax(np.linalg.norm(X - mu, axis=1)))
        return X[idx]

    # ------------------------------------------------------------------
    # Algorithm 1: discriminant computation
    # ------------------------------------------------------------------
    def compute_discriminants(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # NumPy 2.x emits spurious BLAS warnings on strided matmul views.
        # We pass through ascontiguousarray and silence the warning class --
        # all intermediate values are verified finite under the tolerance below.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return self._compute_discriminants_unchecked(X, y)

    def _compute_discriminants_unchecked(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        classes = np.unique(y)
        mean_total = X.mean(axis=0)

        # Within-class scatter: P_W stacks (Xc - mean_c)^T per class.
        P_W_blocks = []
        for c in classes:
            Xc = X[y == c]
            mc = Xc.mean(axis=0)
            P_W_blocks.append((Xc - mc).T)
        P_W = np.ascontiguousarray(np.hstack(P_W_blocks))  # (d, sum n_c)

        # Total / structural scatter (Ledoit-Wolf shrunk for stability)
        lw = LedoitWolf().fit(X)
        S_S = lw.covariance_ * (X.shape[0] - 1)  # covariance -> scatter

        # Eigen-basis of S_S (ascending eigenvalues)
        eigvals_S, Q_S = np.linalg.eigh(S_S)
        # Moore-Penrose: 1/lambda for lambda > tol, else 0 (paper Thm 4.1).
        tol = (
            np.finfo(eigvals_S.dtype).eps
            * max(S_S.shape)
            * (eigvals_S.max() if eigvals_S.size else 0.0)
        )
        inv_eigvals_S = np.zeros_like(eigvals_S)
        nz = eigvals_S > tol
        inv_eigvals_S[nz] = 1.0 / eigvals_S[nz]
        D_pinv = np.diag(inv_eigvals_S)

        M = P_W @ P_W.T
        T = D_pinv @ (Q_S.T @ M @ Q_S)

        eigvals, eigvecs = np.linalg.eigh(T)

        eps = self.eps
        if self.mode == "max":
            mask = eigvals > (1.0 - eps)
        elif self.mode == "min":
            mask = eigvals < eps
        else:  # both
            mask = (eigvals < eps) | (eigvals > (1.0 - eps))

        if not np.any(mask):  # degenerate fallback: keep all
            mask = np.ones_like(eigvals, dtype=bool)

        self.filtered_eigvals_ = eigvals[mask]
        return Q_S @ eigvecs[:, mask]  # (d, m)

    # ------------------------------------------------------------------
    # Fit / transform / predict
    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray | None = None):
        X_train = np.asarray(X_train)
        n, _ = X_train.shape

        if y_train is None:
            y_train = np.zeros(n, dtype=int)
        else:
            y_train = np.asarray(y_train)

        x_ref = self._construct_reference_point(X_train)
        new_label = (int(np.max(y_train)) + 1) if y_train.size else 1

        X_aug = np.vstack([X_train, x_ref])
        y_aug = np.hstack([y_train, new_label])

        self.npd_ = self.compute_discriminants(X_aug, y_aug)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            Z_aug = X_aug @ self.npd_
        if self.faiss_float32:
            Z_aug = Z_aug.astype(np.float32)
        self.basepoint_X_ = Z_aug

        if self.use_faiss:
            m = self.basepoint_X_.shape[1]
            index = faiss.IndexFlatL2(m)
            index.add(self.basepoint_X_.astype(np.float32))
            self.faiss_index_ = index

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.npd_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return np.ascontiguousarray(X) @ self.npd_

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return raw anomaly scores (NN distance in projected space)."""
        if self.basepoint_X_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        Z = self.transform(X_test)
        if self.faiss_float32:
            Z = Z.astype(np.float32)

        if self.use_faiss and self.faiss_index_ is not None:
            D, _ = self.faiss_index_.search(Z.astype(np.float32), k=1)
            return np.sqrt(D[:, 0])

        return self._min_l2_to_set_chunked(
            Z, self.basepoint_X_, chunk_size=self.chunk_size
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _min_l2_to_set_chunked(
        A: np.ndarray, B: np.ndarray, chunk_size: int = 4096
    ) -> np.ndarray:
        A = np.ascontiguousarray(A)
        B = np.ascontiguousarray(B)
        B_T = np.ascontiguousarray(B.T)
        nA = A.shape[0]
        norm_B = np.sum(B ** 2, axis=1)[None, :]
        out = np.empty(nA, dtype=A.dtype)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for s in range(0, nA, chunk_size):
                e = min(s + chunk_size, nA)
                Ac = A[s:e]
                norm_A = np.sum(Ac ** 2, axis=1)[:, None]
                dot = Ac @ B_T
                dist2 = norm_A + norm_B - 2.0 * dot
                out[s:e] = np.sqrt(np.maximum(dist2.min(axis=1), 0.0))
        return out

    def convergence_stats(self, X: np.ndarray) -> dict:
        if self.basepoint_X_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        centroid = self.basepoint_X_.mean(axis=0)
        Z = self.transform(X)
        if self.faiss_float32:
            Z = Z.astype(np.float32)
        dists = np.linalg.norm(Z - centroid, axis=1)
        sq_train = np.mean(np.sum((self.basepoint_X_ - centroid) ** 2, axis=1))
        sq_proj = np.mean(np.sum((Z - centroid) ** 2, axis=1))
        return {
            "mean": float(dists.mean()),
            "max": float(dists.max()),
            "min": float(dists.min()),
            "std": float(dists.std()),
            "convergence": (
                float("nan") if sq_train == 0 else float(sq_proj / sq_train)
            ),
            "n_selected": (
                None
                if self.filtered_eigvals_ is None
                else int(self.filtered_eigvals_.size)
            ),
        }
