"""Spectral Graph Wavelet Transform (SGWT)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

KernelPair = Tuple[
    Callable[[float, np.ndarray], np.ndarray],  # g(s, lambda)
    Callable[[np.ndarray], np.ndarray],  # h(lambda)
]


@dataclass(frozen=True)
class WaveletKernelInfo:
    name: str
    tight_frame: bool
    description: str


TIGHT_FRAME_WAVELETS: tuple[str, ...] = ("meyer", "tight_shannon")
NON_TIGHT_FRAME_WAVELETS: tuple[str, ...] = ("mexican_hat", "diffusion", "gaussian_dog")


def list_wavelet_kernels() -> dict[str, WaveletKernelInfo]:
    """Return available kernel types and metadata."""
    return {
        "meyer": WaveletKernelInfo(
            name="meyer",
            tight_frame=True,
            description="Smooth Meyer-like tight-frame construction (approx.).",
        ),
        "tight_shannon": WaveletKernelInfo(
            name="tight_shannon",
            tight_frame=True,
            description="Ideal (non-smooth) Shannon tiling; exact partition-of-unity in bands.",
        ),
        "mexican_hat": WaveletKernelInfo(
            name="mexican_hat",
            tight_frame=False,
            description="Non-tight Mexican-hat-like bandpass kernels.",
        ),
        "diffusion": WaveletKernelInfo(
            name="diffusion",
            tight_frame=False,
            description="Diffusion wavelets via differences of heat kernels.",
        ),
        "gaussian_dog": WaveletKernelInfo(
            name="gaussian_dog",
            tight_frame=False,
            description="Difference-of-Gaussians bandpass kernels.",
        ),
    }


class GraphWaveletKernelFactory:
    """Factory for SGWT kernels g(s, lambda) and scaling function h(lambda)."""

    def __init__(self, lmax: float):
        self.lmax = float(lmax)

    def get_kernels(
        self, kernel_type: str = "meyer", M: int = 4
    ) -> Tuple[
        Callable[[float, np.ndarray], np.ndarray],
        Callable[[np.ndarray], np.ndarray],
        Sequence[float],
    ]:
        kernel_type = str(kernel_type).lower()
        if kernel_type in {"meyer", "tight_meyer"}:
            return self._create_meyer(int(M))
        if kernel_type == "mexican_hat":
            return self._create_mexican_hat(int(M))
        if kernel_type in {"diffusion", "heat_diffusion", "heat"}:
            return self._create_diffusion(int(M))
        if kernel_type in {"gaussian_dog", "dog", "gaussian"}:
            return self._create_gaussian_dog(int(M))
        if kernel_type in {"tight_shannon", "shannon"}:
            return self._create_tight_shannon(int(M))
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    def _create_mexican_hat(
        self, M: int
    ) -> Tuple[Callable[[float, np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], Sequence[float]]:
        def g(scale: float, x: np.ndarray) -> np.ndarray:
            return (scale * x) * np.exp(-scale * x)

        def h(x: np.ndarray) -> np.ndarray:
            return np.exp(-((x / (0.4 * self.lmax)) ** 4))

        scales = np.logspace(0, np.log10(10), int(M))
        return g, h, scales

    def _create_diffusion(
        self, M: int
    ) -> Tuple[Callable[[float, np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], Sequence[float]]:
        """Diffusion wavelets: g_j(λ)=exp(-t_j λ)-exp(-t_{j+1} λ), h(λ)=exp(-t_M λ)."""
        M = int(M)
        if M <= 0:
            raise ValueError("M must be >= 1")

        denom = float(self.lmax) if self.lmax > 1e-12 else 1.0
        base = np.logspace(-1, 0.5, M + 1).astype(float)  # ~[0.1, 3.16]
        times = base / denom  # scale to graph spectrum
        times_list = [float(t) for t in times.tolist()]
        t_last = float(times_list[-1])

        def h(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            return np.exp(-t_last * x)

        def g(scale: float, x: np.ndarray) -> np.ndarray:
            j = int(round(float(scale)))
            j = max(0, min(M - 1, j))
            t0 = float(times_list[j])
            t1 = float(times_list[j + 1])
            x = np.asarray(x, dtype=float)
            return np.exp(-t0 * x) - np.exp(-t1 * x)

        scales = list(range(M))  # 0: high-ish -> M-1: low-ish (since times increase with j)
        return g, h, scales

    def _create_gaussian_dog(
        self, M: int
    ) -> Tuple[Callable[[float, np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], Sequence[float]]:
        """Difference-of-Gaussians: g_j(λ)=exp(-(t_j λ)^2)-exp(-(t_{j+1} λ)^2), h(λ)=exp(-(t_M λ)^2)."""
        M = int(M)
        if M <= 0:
            raise ValueError("M must be >= 1")

        denom = float(self.lmax) if self.lmax > 1e-12 else 1.0
        base = np.logspace(-1, 0.5, M + 1).astype(float)  # ~[0.1, 3.16]
        times = base / denom
        times_list = [float(t) for t in times.tolist()]
        t_last = float(times_list[-1])

        def h(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            z = t_last * x
            return np.exp(-(z * z))

        def g(scale: float, x: np.ndarray) -> np.ndarray:
            j = int(round(float(scale)))
            j = max(0, min(M - 1, j))
            t0 = float(times_list[j])
            t1 = float(times_list[j + 1])
            x = np.asarray(x, dtype=float)
            z0 = t0 * x
            z1 = t1 * x
            return np.exp(-(z0 * z0)) - np.exp(-(z1 * z1))

        scales = list(range(M))
        return g, h, scales

    def _create_tight_shannon(
        self, M: int
    ) -> Tuple[Callable[[float, np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], Sequence[float]]:
        """Ideal Shannon tiling (non-smooth) with disjoint spectral bands.

        Scaling covers the lowest band; wavelets cover remaining M bands from high to low.
        """
        M = int(M)
        if M <= 0:
            raise ValueError("M must be >= 1")
        if self.lmax <= 0:
            edges = np.linspace(0.0, 1.0, M + 2).astype(float)
        else:
            edges = np.linspace(0.0, float(self.lmax), M + 2).astype(float)

        # scaling band: [0, edges[1]]
        low_edge = float(edges[1])

        # wavelet bands: M bands (excluding scaling), ordered high->low for compatibility.
        bands: list[tuple[float, float]] = []
        # edges indices: 1..M+1 gives M bands; reverse to get high->low.
        for k in range(M, 0, -1):
            lo = float(edges[k])
            hi = float(edges[k + 1])
            bands.append((lo, hi))

        def h(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            return (x <= low_edge).astype(float)

        def g(scale: float, x: np.ndarray) -> np.ndarray:
            j = int(round(float(scale)))
            j = max(0, min(M - 1, j))
            lo, hi = bands[j]
            x = np.asarray(x, dtype=float)
            return ((x > lo) & (x <= hi)).astype(float)

        scales = list(range(M))  # 0: highest band ... M-1: lowest wavelet band
        return g, h, scales

    def _create_meyer(
        self, M: int
    ) -> Tuple[Callable[[float, np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], Sequence[float]]:
        a = self.lmax / 3.0
        print(f"Creating Meyer wavelet kernels with a={a:.4f}")
        print(f"lmax={self.lmax:.4f}")

        scales = [2**i for i in range(int(M))]
        s_max = float(scales[-1])
        s_min = float(scales[0])

        def nu(x: np.ndarray) -> np.ndarray:
            x = np.clip(x, 0, 1)
            return (x**4) * (35 - 84 * x + 70 * (x**2) - 20 * (x**3))

        def theta(x: np.ndarray) -> np.ndarray:
            return (np.pi / 2) * nu(np.clip(x - 1, 0, 1))

        def h(x: np.ndarray) -> np.ndarray:
            x_scaled = s_max * x
            y = np.ones_like(x_scaled)
            mask_decay = (x_scaled > a) & (x_scaled <= 2 * a)
            mask_zero = x_scaled > 2 * a
            y[mask_decay] = np.cos(theta(x_scaled[mask_decay] / a))
            y[mask_zero] = 0.0
            return y

        def g_wrapper(scale: float, x: np.ndarray) -> np.ndarray:
            x_scaled = float(scale) * x
            y = np.zeros_like(x_scaled)

            mask_rise = (x_scaled > a) & (x_scaled <= 2 * a)
            y[mask_rise] = np.sin(theta(x_scaled[mask_rise] / a))

            if float(scale) == s_min:
                mask_high = x_scaled > 2 * a
                y[mask_high] = 1.0
            else:
                mask_decay = (x_scaled > 2 * a) & (x_scaled <= 4 * a)
                y[mask_decay] = np.cos(theta(x_scaled[mask_decay] / (2 * a)))

            return y

        return g_wrapper, h, scales


class SpectralGraphWaveletTransform:
    """SGWT on a graph defined by its adjacency matrix."""

    def __init__(self, adj_matrix: np.ndarray):
        adj_matrix = np.asarray(adj_matrix, dtype=float)
        self.N = int(adj_matrix.shape[0])

        d = np.sum(adj_matrix, axis=1)
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt = np.diag(d_inv_sqrt)
        self.L = np.eye(self.N) - d_inv_sqrt @ adj_matrix @ d_inv_sqrt

        evals, evecs = np.linalg.eigh(self.L)
        self.evals = np.maximum(0.0, evals)
        self.evecs = evecs
        self.lmax = float(self.evals[-1])

    def transform(
        self, signal: np.ndarray, kernel_type: str = "meyer", M: int = 4
    ) -> Tuple[Dict[str, object], Sequence[float], KernelPair]:
        factory = GraphWaveletKernelFactory(self.lmax)
        g_func, h_func, scales = factory.get_kernels(kernel_type, int(M))

        x_hat = self.evecs.T @ signal
        coeffs: Dict[str, object] = {"scaling": None, "wavelet": []}

        coeffs["scaling"] = self.evecs @ (h_func(self.evals) * x_hat)
        wavelets: List[np.ndarray] = []
        for scale in scales:
            wavelets.append(self.evecs @ (g_func(float(scale), self.evals) * x_hat))
        coeffs["wavelet"] = wavelets

        return coeffs, scales, (g_func, h_func)

    def adjoint(self, coeffs: Dict[str, object], scales: Sequence[float], kernels: KernelPair) -> np.ndarray:
        g_func, h_func = kernels

        scaling = coeffs["scaling"]
        if scaling is None:
            raise ValueError("coeffs['scaling'] must not be None")

        recon_hat = h_func(self.evals) * (self.evecs.T @ scaling)
        wavelets = coeffs.get("wavelet", [])
        for i, scale in enumerate(scales):
            wavelet = wavelets[i]
            recon_hat += g_func(float(scale), self.evals) * (self.evecs.T @ wavelet)

        return self.evecs @ recon_hat
