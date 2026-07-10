from __future__ import annotations

import numba as nb
import numpy as np
import numpy.linalg as LA
from softmatcha.typing import NDArrayF32
import time

NUMPY_AVX512F_SUPPORTED: bool = "AVX512F" in np.show_config(mode="dicts").get(
	"SIMD Extensions", {}
).get("found", [])


def normalize(x: NDArrayF32, axis: int = -1, eps: float = 1e-12) -> NDArrayF32:
	"""Normalize the input vectors by its norm.

	Args:
		x (NDArrayF32): Input vectors.
		axis (int): The axis of `x` along which to compute the vector norms.
		eps (float): Epsilon for the numerical stability.
	"""
	return (x / (LA.norm(x, axis=axis, keepdims=True) + eps)).astype(np.float32)


@nb.njit(
	nb.float32[:, :](nb.float32[:, :], nb.float32[:, :], nb.float32),
	parallel=True,
	fastmath=True,
	cache=True,
)
def __matmul_numba_impl(a: NDArrayF32, b: NDArrayF32, minimum: float) -> NDArrayF32:
	"""Compute matrix multiplication.

	This implementaion accelerates by leveraging SIMD.

	Args:
		a (NDArrayF32): The input matrix of shape `(a_length, embed_dim)`.
		b (NDArrayF32): The other matrix of shape `(b_length, embed_dim)`.

	Returns:
		NDArrayF32: Multiplied matrix of shape `(a_length, b_length)`.
	"""
	len_a, embed_dim = a.shape
	len_b = b.shape[0]

	a = np.ascontiguousarray(a)
	b = np.ascontiguousarray(b)
	mm = np.zeros((len_a, len_b), dtype=np.float32)
	for j in nb.prange(len_b):
		for i in range(len_a):
			acc = 0.0
			for d in range(embed_dim):
				acc += a[i, d] * b[j, d]
			mm[i, j] = max(acc, minimum) 
	return mm


def matmul(a: NDArrayF32, b: NDArrayF32, minimum: float = 0.0) -> NDArrayF32:
	"""Compute matrix multiplication.

	Args:
		a (NDArrayF32): The input matrix of shape `(a_length, embed_dim)`.
		b (NDArrayF32): The other matrix of shape `(b_length, embed_dim)`.

	Returns:
		NDArrayF32: Multiplied matrix of shape `(a_length, b_length)`.
	"""
	# 例: a.shape = (len_a, embed_dim), b.shape = (len_b, embed_dim)
	if NUMPY_AVX512F_SUPPORTED:
		return np.maximum(a @ b.T, minimum)
	else:
		return __matmul_numba_impl(a, b, minimum)
