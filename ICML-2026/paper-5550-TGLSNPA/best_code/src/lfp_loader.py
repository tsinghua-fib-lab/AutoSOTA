"""
Load CWT angles on the fly.

"""
__date__ = "July - October 2025"

from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
try:
    from scipy.signal import morlet2
except ImportError:
    from .morlet import morlet2
from typing import Optional


def _sliding_window(x: jnp.ndarray, w: int) -> jnp.ndarray:
    # Returns shape (N - w + 1, w, *), jit-friendly
    starts = jnp.arange(x.shape[0] - w + 1)
    return jax.vmap(lambda s: lax.dynamic_slice_in_dim(x, s, w))(starts)

sliding_window = jax.jit(_sliding_window, static_argnames=("w",))


# ---- JIT kernels used by __next__ ----

@partial(jax.jit, static_argnames=("N", "B"))
def _contig_batch_kernel(
    lfps: jnp.ndarray,          # (T, C), real
    wavelets_conj: jnp.ndarray, # (F, N), complex
    t0: int,                    # start time for the first window
    N: int,
    B: int,
) -> jnp.ndarray:
    # Slice a compact segment (N+B-1, C) and slide to get (B, N, C)
    C = lfps.shape[1]
    seg = lax.dynamic_slice(lfps, (t0, 0), (N + B - 1, C))   # (N+B-1, C)
    wins = sliding_window(seg, N)                            # (B, N, C)
    # (F,N) @ (B,N,C) -> (B,C,F)
    coeffs = jnp.einsum('fn,bnc->bcf', wavelets_conj, wins)
    return jnp.angle(coeffs)                                 # (B, C, F)


@partial(jax.jit, static_argnames=("N",))
def _arbitrary_batch_kernel(
    lfps: jnp.ndarray,          # (T, C)
    wavelets_conj: jnp.ndarray, # (F, N)
    starts: jnp.ndarray,        # (B,), int32
    N: int,
) -> jnp.ndarray:
    C = lfps.shape[1]
    def grab(s):
        return lax.dynamic_slice(lfps, (s, 0), (N, C))       # (N, C)
    wins = jax.vmap(grab)(starts)                            # (B, N, C)
    coeffs = jnp.einsum('fn,bnc->bcf', wavelets_conj, wins)  # (B, C, F)
    return jnp.angle(coeffs)                                 # (B, C, F)


class SinglePhaseLoader:
    def __init__(
        self,
        lfps: jnp.ndarray,                    # (T, C), real
        fs: float,                            # Hz
        freqs: jnp.ndarray,                   # (F,), Hz
        window_length_s: float,               # seconds per window
        batch_size: int,
        causal_filter: bool = False,
        wavelet_w: int = 5,
        labels = None,
        label_value: Optional[int] = None,
        complex_dtype=jnp.complex64,
        shuffle: bool = True,
        key: jax.random.PRNGKey = None,
    ):
        """Streams instantaneous phase via sliding Morlet correlations, in batches of fixed size."""
        self.lfps = lfps
        self.fs = fs
        self.freqs = freqs
        self.batch_size = int(batch_size)
        self.w = int(wavelet_w)
        self.causal = bool(causal_filter)
        if key is None:
            key = jax.random.PRNGKey(np.random.randint(2**30))
        self._key = key

        T, C = lfps.shape
        self.T, self.C = int(T), int(C)
        self.N = int(round(window_length_s * fs))
        if self.N < 1:
            raise ValueError("window_length_s * fs must be >= 1 sample.")
        if self.N > T:
            raise ValueError(f"Window too long ({self.N} > {T})")
        self.F = int(freqs.shape[0])

        # Optional label gating of start indices
        if labels is not None:
            if label_value is None:
                raise ValueError("If you pass labels, also pass label_value.")
            if labels.shape[0] < T:
                print(f"Truncating LFPs from {T} to {labels.shape[0]} samples.")
                lfps = lfps[:labels.shape[0]]
                T = len(lfps)
                if self.N > T:
                    raise ValueError(f"Window too long ({self.N} > {T})")
            good = jnp.where(labels[: T - self.N + 1] == label_value, size=T - self.N + 1, fill_value=-1)[0]
            good = good[good >= 0]
            self.starts = good.astype(jnp.int32)  # arbitrary starts
            self._contiguous = False
        else:
            self.starts = jnp.arange(0, T - self.N + 1, dtype=jnp.int32)
            self._contiguous = True

        # Optionally shuffle.
        if shuffle:
            self._contiguous = False
            self._key, sub = jax.random.split(self._key)
            self.starts = jax.random.permutation(sub, self.starts)

        self.num_valid = int(self.starts.shape[0])
        self._idx = 0  # pointer into self.starts

        # ---- Precompute Morlet wavelets ----
        # Scale selection: s = w * fs / (2π f0)
        scales = self.w * self.fs / (2.0 * jnp.pi * self.freqs)
        Wf_list = []
        temp_N = 2 * self.N if self.causal else self.N
        for s in np.asarray(scales, dtype=np.float64):
            raw = morlet2(int(temp_N), float(s), w=float(self.w)).astype(np.complex128)
            w_t = jnp.asarray(raw[: self.N], dtype=complex_dtype)
            if self.causal:
                w_t = w_t * jnp.sqrt(jnp.array(2.0, dtype=w_t.real.dtype))
            Wf_list.append(w_t)
        wavelets = jnp.stack(Wf_list, axis=0)       # (F, N), complex
        self.wavelets_conj = jnp.conj(wavelets)

        # JIT kernels bound to this instance via small lambdas (keeps call sites clean)
        self._contig_kernel = lambda t0: _contig_batch_kernel(
            self.lfps, self.wavelets_conj, t0, self.N, self.batch_size
        )
        self._arb_kernel = lambda starts: _arbitrary_batch_kernel(
            self.lfps, self.wavelets_conj, starts, self.N
        )

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self) -> jnp.ndarray:
        if self._idx >= self.num_valid:
            raise StopIteration

        remaining = self.num_valid - self._idx
        if remaining < self.batch_size:
            # Fixed output shapes are important for JIT; skip truncated tail.
            raise StopIteration

        if self._contiguous:
            # starts is [0, 1, 2, ...], so a contiguous batch is defined by its first start t0
            t0 = int(self.starts[self._idx])
            out = self._contig_kernel(t0)                # (B, C, F)
            self._idx += self.batch_size
            return out
        else:
            # Gather a fixed-length vector of arbitrary starts
            starts_batch = self.starts[self._idx : self._idx + self.batch_size]
            out = self._arb_kernel(starts_batch)         # (B, C, F)
            self._idx += self.batch_size
            return out


@partial(jax.jit, static_argnames=("N", "L"))
def _random_block_kernel(
    lfps: jnp.ndarray,          # (T, C), real
    wavelets_conj: jnp.ndarray, # (F, N), complex
    starts: jnp.ndarray,        # (B,), int32
    N: int,
    L: int,
) -> jnp.ndarray:
    """
    For each start s, gather segment (N+L, C), slide to (L+1, N, C),
    correlate with wavelets to get (L+1, C, F), then stack over batch.
    Output: (B, L+1, C, F)
    """
    C = lfps.shape[1]

    def seg(s):
        return lax.dynamic_slice(lfps, (s, 0), (N + L, C))  # (N+L, C)

    segs = jax.vmap(seg)(starts)                            # (B, N+L, C)
    wins = jax.vmap(lambda x: sliding_window(x, N))(segs)   # (B, L+1, N, C)

    # (F,N) @ (B, L+1, N, C) -> (B, L+1, C, F)
    coeffs = jnp.einsum('fn,blnc->blcf', wavelets_conj, wins)
    return jnp.angle(coeffs)                                # (B, L+1, C, F)


class BlockPhaseLoader:
    def __init__(
        self,
        lfps: jnp.ndarray,                    # (T, C), real
        fs: float,                            # Hz
        freqs: jnp.ndarray,                   # (F,), Hz
        window_length_s: float,               # seconds per window
        batch_size: int,
        L: int,                               # lag length; output length is L+1
        causal_filter: bool = False,
        wavelet_w: int = 5,
        labels = None,
        label_value: Optional[int] = None,
        complex_dtype=jnp.complex64,
        replace_within_batch: Optional[bool] = None,  # None => auto (no-replacement if possible)
        num_batches: Optional[int] = None,
        shuffle: bool = True,
        key: jax.random.PRNGKey = None,
    ):
        """
        Each iteration returns phases of shape (B, L+1, C, F).
        Within a batch, windows are sampled uniformly from all valid start indices.
        """
        self.lfps = lfps
        self.fs = fs
        self.freqs = freqs
        self.batch_size = int(batch_size)
        self.L = int(L)
        self.w = int(wavelet_w)
        self.causal = bool(causal_filter)
        self.complex_dtype = complex_dtype
        self.num_batches = num_batches
        if key is None:
            key = jax.random.PRNGKey(np.random.randint(2**30))
        self._key = key

        T, C = lfps.shape
        self.T, self.C = int(T), int(C)
        self.N = int(round(window_length_s * fs))
        if self.N < 1:
            raise ValueError("window_length_s * fs must be >= 1 sample.")
        if self.N > T:
            raise ValueError(f"Window too long ({self.N} > {T})")
        self.F = int(freqs.shape[0])

        # Valid start indices must allow L extra steps and an N-length window
        max_start = T - (self.N + self.L)
        if max_start < 0:
            raise ValueError(f"L ({self.L}) too large for T={T}, N={self.N}.")
        valid_len = max_start + 1

        # Optional label gating of start indices
        if labels is not None:
            if label_value is None:
                raise ValueError("If you pass labels, also pass label_value.")
            if labels.shape[0] < T:
                print(f"Truncating LFPs from {T} to {labels.shape[0]} samples.")
                lfps = lfps[:labels.shape[0]]
                T = len(lfps)
                if self.N > T:
                    raise ValueError(f"Window too long ({self.N} > {T})")
            good = jnp.where(labels[: valid_len] == label_value, size=valid_len, fill_value=-1)[0]
            good = good[good >= 0].astype(jnp.int32)
            self.starts = good.astype(jnp.int32)  # arbitrary starts
        else:
            self.starts = jnp.arange(0, valid_len, dtype=jnp.int32)

        # Optionally shuffle.
        if shuffle:
            self._contiguous = False
            self._key, sub = jax.random.split(self._key)
            self.starts = jax.random.permutation(sub, self.starts)

        self.num_valid = int(self.starts.shape[0])
        if self.num_valid == 0:
            raise ValueError("No valid start indices for windows (check L, N, labels).")

        # Replacement policy within a batch
        if replace_within_batch is None:
            self.replace_within_batch = not (self.num_valid >= self.batch_size)
        else:
            if not replace_within_batch and self.num_valid < self.batch_size:
                raise ValueError("replace_within_batch=False requires num_valid >= batch_size.")
            self.replace_within_batch = bool(replace_within_batch)

        # ---- Precompute wavelets ----
        # s = w * fs / (2π f0)
        scales = self.w * self.fs / (2.0 * jnp.pi * self.freqs)
        Wf_list = []
        temp_N = 2 * self.N if self.causal else self.N
        for s in np.asarray(scales, dtype=np.float64):
            raw = morlet2(int(temp_N), float(s), w=float(self.w)).astype(np.complex128)
            w_t = jnp.asarray(raw[: self.N], dtype=complex_dtype)
            if self.causal:
                w_t = w_t * jnp.sqrt(jnp.array(2.0, dtype=w_t.real.dtype))
            Wf_list.append(w_t)
        wavelets = jnp.stack(Wf_list, axis=0)       # (F, N), complex
        self.wavelets_conj = jnp.conj(wavelets)

        # Bind JIT kernel
        self._kernel = lambda starts: _random_block_kernel(
            self.lfps, self.wavelets_conj, starts, self.N, self.L
        )

        self._batches_emitted = 0

    def set_key(self, key: jax.Array):
        self._key = key

    def __iter__(self):
        self._batches_emitted = 0
        return self

    def __next__(self) -> jnp.ndarray:
        if self.num_batches is not None and self._batches_emitted >= self.num_batches:
            raise StopIteration

        key, sub = jax.random.split(self._key)
        self._key = key

        B = self.batch_size
        n = self.num_valid
        if not self.replace_within_batch and n >= B:
            idxs = jax.random.choice(sub, n, shape=(B,), replace=False)
        else:
            idxs = jax.random.randint(sub, (B,), minval=0, maxval=n)

        starts_batch = self.starts[idxs]            # (B,)
        out = self._kernel(starts_batch)            # (B, L+1, C, F)
        self._batches_emitted += 1
        return out

