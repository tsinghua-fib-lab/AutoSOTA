"""Tests for FlashSinkhorn vs OTT-JAX kernel parity.

Compares FlashSinkhorn alternating Sinkhorn solver with OTT-JAX's
Sinkhorn implementation to verify numerical equivalence.

NOTE: These tests require JAX with GPU support and OTT-JAX library.
They will be skipped if JAX GPU backend fails to initialize (common
due to cuDNN version mismatches between PyTorch and JAX).
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest
import torch

from flash_sinkhorn.kernels.sinkhorn_flashstyle_sqeuclid import (
    sinkhorn_flashstyle_alternating,
)
from flash_sinkhorn.kernels.sinkhorn_triton_ott_sqeuclid import (
    apply_lse_kernel_sqeuclid,
    apply_transport_from_potentials_sqeuclid,
)


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("ott")
from ott.geometry import pointcloud  # noqa: E402

# Check if JAX GPU backend works (may fail due to cuDNN conflicts with PyTorch)
def _jax_gpu_works():
    try:
        if jax.default_backend() != "gpu":
            return False
        # Try a simple GPU operation
        x = jnp.ones(10)
        _ = (x + x).block_until_ready()
        return True
    except Exception:
        return False

if not _jax_gpu_works():
    pytestmark = pytest.mark.skip(
        reason="JAX GPU backend not working (likely cuDNN conflict with PyTorch). "
               "These tests require JAX GPU to compare with OTT-JAX."
    )


def _to_jax(x):
    dev = jax.devices("gpu")[0]
    return jax.device_put(jnp.asarray(x), device=dev)


def _rand_np(shape, seed, dtype):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_ott_backend_is_gpu():
    assert jax.default_backend() == "gpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("vec_mode", ["none", "pos", "neg", "mixed"])
def test_apply_lse_kernel_matches_ott(axis, vec_mode):
    if jax.default_backend() != "gpu":
        pytest.skip("JAX GPU backend required for OTT parity tests.")

    n, m, d = 64, 48, 32
    eps = 0.5

    x_np = _rand_np((n, d), seed=0, dtype=np.float32)
    y_np = _rand_np((m, d), seed=1, dtype=np.float32)
    f_np = _rand_np((n,), seed=2, dtype=np.float32)
    g_np = _rand_np((m,), seed=3, dtype=np.float32)

    if vec_mode == "none":
        vec_np = None
    else:
        vec_len = m if axis == 1 else n
        if vec_mode == "pos":
            vec_np = np.ones((vec_len,), dtype=np.float32)
        elif vec_mode == "neg":
            vec_np = -np.ones((vec_len,), dtype=np.float32)
        else:
            vec_np = np.linspace(-1.0, 1.0, vec_len, dtype=np.float32)

    device = torch.device("cuda")
    x_t = torch.from_numpy(x_np).to(device)
    y_t = torch.from_numpy(y_np).to(device)
    f_t = torch.from_numpy(f_np).to(device)
    g_t = torch.from_numpy(g_np).to(device)
    x2_t = (x_t.float() * x_t.float()).sum(dim=1)
    y2_t = (y_t.float() * y_t.float()).sum(dim=1)
    vec_t = None if vec_np is None else torch.from_numpy(vec_np).to(device)

    out_triton, sgn_triton = apply_lse_kernel_sqeuclid(
        x_t,
        y_t,
        f_t,
        g_t,
        eps,
        axis,
        vec=vec_t,
        x2=x2_t,
        y2=y2_t,
    )

    x_j = _to_jax(x_np)
    y_j = _to_jax(y_np)
    f_j = _to_jax(f_np)
    g_j = _to_jax(g_np)
    vec_j = None if vec_np is None else _to_jax(vec_np)

    geom = pointcloud.PointCloud(x_j, y_j, batch_size=256, epsilon=eps)
    out_ott, sgn_ott = geom.apply_lse_kernel(f_j, g_j, eps, vec=vec_j, axis=axis)

    out_ott_t = torch.from_numpy(np.array(out_ott, dtype=np.float32, copy=True))
    sgn_ott_t = torch.from_numpy(np.array(sgn_ott, dtype=np.float32, copy=True))

    torch.testing.assert_close(out_triton.cpu(), out_ott_t, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        sgn_triton.cpu(), sgn_ott_t, rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("vec_mode", ["pos", "neg", "mixed"])
def test_apply_transport_from_potentials_matches_ott(axis, vec_mode):
    if jax.default_backend() != "gpu":
        pytest.skip("JAX GPU backend required for OTT parity tests.")

    n, m, d = 64, 48, 32
    eps = 0.7

    x_np = _rand_np((n, d), seed=10, dtype=np.float32)
    y_np = _rand_np((m, d), seed=11, dtype=np.float32)
    f_np = _rand_np((n,), seed=12, dtype=np.float32)
    g_np = _rand_np((m,), seed=13, dtype=np.float32)

    vec_len = m if axis == 1 else n
    if vec_mode == "pos":
        vec_np = np.ones((vec_len,), dtype=np.float32)
    elif vec_mode == "neg":
        vec_np = -np.ones((vec_len,), dtype=np.float32)
    else:
        vec_np = np.linspace(-1.0, 1.0, vec_len, dtype=np.float32)

    device = torch.device("cuda")
    x_t = torch.from_numpy(x_np).to(device)
    y_t = torch.from_numpy(y_np).to(device)
    f_t = torch.from_numpy(f_np).to(device)
    g_t = torch.from_numpy(g_np).to(device)
    vec_t = torch.from_numpy(vec_np).to(device)

    out_triton = apply_transport_from_potentials_sqeuclid(
        x_t,
        y_t,
        f_t,
        g_t,
        vec_t,
        eps,
        axis,
    )

    x_j = _to_jax(x_np)
    y_j = _to_jax(y_np)
    f_j = _to_jax(f_np)
    g_j = _to_jax(g_np)
    vec_j = _to_jax(vec_np)
    geom = pointcloud.PointCloud(x_j, y_j, batch_size=256, epsilon=eps)

    out_ott = geom.apply_transport_from_potentials(f_j, g_j, vec_j, axis=axis)
    out_ott_t = torch.from_numpy(np.array(out_ott, dtype=np.float32, copy=True))

    torch.testing.assert_close(
        out_triton.cpu(), out_ott_t, rtol=2e-3, atol=2e-3
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton.")
def test_sinkhorn_potentials_match_ott():
    """Test FlashSinkhorn alternating solver matches OTT-JAX potentials."""
    if jax.default_backend() != "gpu":
        pytest.skip("JAX GPU backend required for OTT parity tests.")

    n, m, d = 64, 48, 32
    eps = 1.0
    n_iters = 5

    x_np = _rand_np((n, d), seed=20, dtype=np.float32)
    y_np = _rand_np((m, d), seed=21, dtype=np.float32)
    # FlashSinkhorn takes probability weights, not log weights
    a_np = np.full((n,), 1.0 / n, dtype=np.float32)
    b_np = np.full((m,), 1.0 / m, dtype=np.float32)
    loga_np = np.log(a_np)
    logb_np = np.log(b_np)

    device = torch.device("cuda")
    x_t = torch.from_numpy(x_np).to(device)
    y_t = torch.from_numpy(y_np).to(device)
    a_t = torch.from_numpy(a_np).to(device)
    b_t = torch.from_numpy(b_np).to(device)
    # Use FlashSinkhorn alternating solver with OTT convention for potential comparison
    f_t, g_t = sinkhorn_flashstyle_alternating(
        x_t,
        y_t,
        a_t,
        b_t,
        eps=eps,
        n_iters=n_iters,
        ott_convention=True,  # Return potentials in OTT format (f_hat = f + eps*log(a))
    )

    x_j = _to_jax(x_np)
    y_j = _to_jax(y_np)
    loga_j = _to_jax(loga_np)
    logb_j = _to_jax(logb_np)
    geom = pointcloud.PointCloud(x_j, y_j, batch_size=256, epsilon=eps)

    f_j = jnp.zeros((n,), dtype=jnp.float32)
    g_j = jnp.zeros((m,), dtype=jnp.float32)
    for _ in range(n_iters):
        g_j = geom.update_potential(f_j, g_j, logb_j, axis=0)
        f_j = geom.update_potential(f_j, g_j, loga_j, axis=1)

    f_ott_t = torch.from_numpy(np.array(f_j, dtype=np.float32, copy=True))
    g_ott_t = torch.from_numpy(np.array(g_j, dtype=np.float32, copy=True))
    torch.testing.assert_close(f_t.cpu(), f_ott_t, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(g_t.cpu(), g_ott_t, rtol=1e-2, atol=1e-2)
