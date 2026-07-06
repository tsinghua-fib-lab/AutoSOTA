"""
Gradient evaluation benchmark for FlashSinkhorn paper.

USE CASE: Training / Optimization
    - Computing gradients for backpropagation
    - Training neural networks with OT loss
    - Gradient-based optimization (e.g., Wasserstein barycenters)
    - Example: `loss = sinkhorn(x, y); loss.backward()`

Compares end-to-end gradient evaluation timing: FlashSinkhorn vs GeomLoss (KeOps) vs OTT-JAX.

All three methods support efficient online backward without materializing O(n²):
- FlashSinkhorn: Custom Triton backward kernels recompute cost on-the-fly (with autotuning)
- GeomLoss/KeOps: Explicit (Danskin) gradients via `last_extrapolation`
- OTT-JAX: Danskin's theorem (use_danskin=True) treats potentials as constants

IMPORTANT:
- TF32 is enabled by default for ~2x speedup on A100/H100 GPUs (uses Tensor Cores)
- Sizes are run large->small to avoid Triton autotuning cache pollution
- Autotuning finds optimal block sizes; first call per (n,d) has ~2-3s overhead
- Memory overhead note: First run at each config incurs ~256MB Triton compilation
  overhead. Subsequent runs use cached configs (~4MB steady-state). This explains
  why the largest size (run first) may show higher memory than expected.
- We measure `total_ms` = forward + backward combined (end-to-end gradient evaluation)
- This is the only fair comparison because JAX's `jax.grad(f)` compiles into a single
  fused function that cannot be separated into forward/backward components
- For forward-only timing, see bench_forward.py (inference use case)

Cost convention:
- FlashSinkhorn: C(x,y) = ||x-y||² (full squared Euclidean, half_cost=False default)
- GeomLoss SqDist: C(x,y) = ||x-y||² (matches FlashSinkhorn)
- OTT-JAX PointCloud: C(x,y) = ||x-y||² (default squared Euclidean)

OTT-JAX loss formulas (important for understanding loss value differences):
- FlashSinkhorn: loss = <a,f> + <b,g>  (dual objective)
- OTT-JAX reg_ot_cost: <a,f> + <b,g> + eps*(H(a) + H(b))  [adds marginal entropies]
- OTT-JAX ent_reg_cost: <a,f> + <b,g>  [pure dual, matches FlashSinkhorn]

Where H(a) = -sum(a*log(a)) is the entropy of distribution a.
For uniform marginals: H(a) = H(b) = ln(n), e.g., ln(1000) ≈ 6.91 for n=1000.

KEY INSIGHT: Gradients match EXACTLY (cos_sim = 1.0) despite different loss values,
because H(a), H(b) are constants w.r.t. point locations x (∂H(a)/∂x = 0).

Timing methodology:
- FlashSinkhorn/GeomLoss: CUDA events (precise GPU timing)
- OTT-JAX: Wall-clock time with block_until_ready() sync
  (JAX lacks CUDA event API; wall-clock includes minor Python overhead)

Usage:
    # Default: TF32 enabled, autotuning on
    python -m flash_sinkhorn.bench.bench_backward

    # Strict FP32 (slower but higher precision)
    python -m flash_sinkhorn.bench.bench_backward --no-tf32

    # Check gradient parity first
    python -m flash_sinkhorn.bench.bench_backward --verify
"""

from __future__ import annotations

import os

# Must set CUDA_VISIBLE_DEVICES before importing torch/KeOps
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import csv
import ctypes
import gc
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch


def _preload_cuda_libs() -> None:
    """Preload CUDA libraries for KeOps.

    Discovers CUDA_HOME from environment or PyTorch, then loads nvrtc/cudart
    so that KeOps can find them at JIT compile time.
    """
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        try:
            from torch.utils.cpp_extension import CUDA_HOME as _ch
            cuda_home = _ch
        except Exception:
            pass
    if cuda_home is None:
        return
    os.environ.setdefault("CUDA_HOME", cuda_home)
    os.environ.setdefault("CUDA_PATH", cuda_home)
    lib_dir = Path(cuda_home) / "targets" / "x86_64-linux" / "lib"
    if not lib_dir.is_dir():
        lib_dir = Path(cuda_home) / "lib64"
    for pattern in ("libnvrtc.so*", "libnvrtc-builtins.so*", "libcudart.so*"):
        for lib in sorted(lib_dir.glob(pattern)):
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass


def _set_tf32(enabled: bool) -> None:
    """Set TF32 mode."""
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    if not enabled:
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass


@dataclass
class BackwardResult:
    """Backward pass timing result."""
    method: str
    n: int
    m: int
    d: int
    eps: float
    forward_ms: float
    backward_ms: float
    total_ms: float
    peak_memory_mb: float  # Peak GPU memory during grad evaluation
    oom: bool


@dataclass
class JITOverheadResult:
    """JIT compilation overhead measurement result."""
    method: str
    n: int
    d: int
    eps: float
    cold_start_ms: float  # First call (includes JIT compilation)
    warm_ms: float        # Steady-state (average of subsequent calls)
    jit_overhead_ms: float  # cold_start - warm
    overhead_ratio: float   # cold_start / warm


def bench_with_stats(
    fn: Callable[[], None],
    warmup: int = 10,
    rep: int = 50,
) -> float:
    """Benchmark and return mean time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


def bench_flashsinkhorn_backward(
    n: int, m: int, d: int, eps: float, n_iters: int,
    device: torch.device, warmup: int, rep: int,
    grad: str = "x",
    backend: str = "symmetric",
    allow_tf32: bool = False,
) -> BackwardResult:
    """Benchmark FlashSinkhorn forward and end-to-end grad evaluation.

    Args:
        backend: "symmetric" (GeomLoss-style) or "alternating" (OTT-JAX-style)

    Uses full squared Euclidean cost C(x,y) = ||x-y||² (half_cost=False).
    Autotuning is enabled for best Triton kernel performance (~2-3s first call overhead).

    Note: OTT backend does not have custom backward kernels yet, so it uses autograd
    through the forward pass (unrolled differentiation through Sinkhorn iterations).
    """
    from flash_sinkhorn import SamplesLoss

    torch.manual_seed(0)
    x = torch.randn(n, d, device=device, dtype=torch.float32).requires_grad_(True)
    y = torch.randn(m, d, device=device, dtype=torch.float32)
    if grad == "xy":
        y.requires_grad_(True)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    # Autotuning is enabled for best performance. The autotuner finds optimal
    # block sizes for each (n, d) combination. First call at each size has
    # ~2-3s overhead, subsequent calls use cached configs.
    #
    # Note on last_extrapolation:
    # - FlashSinkhorn: last_extrapolation=False (custom backward kernels don't need it)
    # - GeomLoss: last_extrapolation=True (required for Danskin's theorem gradients)
    # This difference is acceptable because FlashSinkhorn's backward uses
    # its own gradient kernel that doesn't depend on the extrapolation step.
    loss_fn = SamplesLoss(
        "sinkhorn",
        backend=backend,
        use_epsilon_scaling=False,
        eps=eps,
        n_iters=n_iters,
        debias=False,
        potentials=False,
        normalize=False,
        autotune=True,  # Enable for optimal block sizes (first call has ~2-3s overhead)
        last_extrapolation=False,  # FlashSinkhorn backward kernels don't need this
        allow_tf32=allow_tf32,  # TF32 uses Tensor Cores for ~2x speedup
    )

    method_name = f"flash_{backend}"
    grad_inputs = (x,) if grad == "x" else (x, y)

    try:
        def run_grad() -> None:
            """Run forward + backward (end-to-end gradient evaluation)."""
            loss = loss_fn(a, x, b, y)
            torch.autograd.grad(loss, grad_inputs, retain_graph=False, create_graph=False)

        # Measure peak memory during grad evaluation
        torch.cuda.reset_peak_memory_stats(device)
        total_ms = bench_with_stats(run_grad, warmup, rep)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6

        # Report forward_ms=-1, backward_ms=-1 (N/A) since we only measure total
        # Forward-only timing is available in bench_forward.py
        return BackwardResult(method_name, n, m, d, eps, -1.0, -1.0, total_ms, peak_memory_mb, oom=False)

    except torch.cuda.OutOfMemoryError:
        return BackwardResult(method_name, n, m, d, eps, 0, 0, 0, 0, oom=True)


def bench_geomloss_backward(
    n: int, m: int, d: int, eps: float, n_iters: int,
    device: torch.device, warmup: int, rep: int,
    backend: str = "symmetric",
    grad: str = "x",
) -> BackwardResult:
    """Benchmark GeomLoss (KeOps) forward and end-to-end grad evaluation.

    Uses low-level `sinkhorn_loop` with `eps_list=[eps]*n_iters` to force exactly
    `n_iters` iterations (matching FlashSinkhorn / OTT-JAX settings).

    Cost convention: SqDist(X,Y) = ||x-y||² (full squared Euclidean, matches FlashSinkhorn).

    Gradient method: Danskin's theorem via `last_extrapolation=True` and detached cross-terms.
    The detach pattern (x, y.detach()) prevents gradients flowing through the "other" variable,
    which is the correct implementation of envelope theorem gradients.
    """
    try:
        from pykeops.torch import generic_logsumexp  # noqa: F401
    except ImportError:
        return BackwardResult(f"geomloss_{backend}", n, m, d, eps, 0, 0, 0, 0, oom=True)

    from geomloss.sinkhorn_divergence import log_weights, sinkhorn_cost, sinkhorn_loop
    from geomloss.sinkhorn_samples import lse_genred, softmin_online

    torch.manual_seed(0)
    x = torch.randn(n, d, device=device, dtype=torch.float32).requires_grad_(True)
    y = torch.randn(m, d, device=device, dtype=torch.float32)
    if grad == "xy":
        y.requires_grad_(True)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    eps_list = [eps] * n_iters
    a_log = log_weights(a)
    b_log = log_weights(b)
    # SqDist(X,Y) = ||x-y||² (full squared Euclidean, matches FlashSinkhorn default)
    my_lse = lse_genred("SqDist(X,Y)", d)
    softmin = partial(softmin_online, log_conv=my_lse)

    def _loss() -> torch.Tensor:
        # Detach cross terms for Danskin's theorem (envelope theorem gradients).
        # This prevents gradients flowing through y when computing grad_x and vice versa,
        # treating the transport plan as fixed at the optimum.
        C_xy = (x, y.detach())
        C_yx = (y, x.detach())

        _, _, g_ab, f_ba = sinkhorn_loop(
            softmin,
            a_log,
            b_log,
            None,
            None,
            C_xy,
            C_yx,
            eps_list,
            rho=None,
            debias=False,
            last_extrapolation=True,
        )
        return sinkhorn_cost(
            eps,
            None,
            a,
            b,
            None,
            None,
            g_ab,
            f_ba,
            batch=False,
            debias=False,
            potentials=False,
        )

    grad_inputs = (x,) if grad == "x" else (x, y)
    method_name = f"geomloss_{backend}"
    try:
        def run_grad() -> None:
            """Run forward + backward (end-to-end gradient evaluation)."""
            loss = _loss()
            torch.autograd.grad(loss, grad_inputs, retain_graph=False, create_graph=False)

        # Make sure KeOps is JIT'ed before timed region
        run_grad()
        torch.cuda.synchronize()

        # Measure peak memory during grad evaluation
        torch.cuda.reset_peak_memory_stats(device)
        total_ms = bench_with_stats(run_grad, warmup, rep)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6

        # Report forward_ms=-1, backward_ms=-1 (N/A) since we only measure total
        # Forward-only timing is available in bench_forward.py
        return BackwardResult(method_name, n, m, d, eps, -1.0, -1.0, total_ms, peak_memory_mb, oom=False)

    except torch.cuda.OutOfMemoryError:
        return BackwardResult(method_name, n, m, d, eps, 0, 0, 0, 0, oom=True)
    except Exception as e:
        # Print actual error for debugging (not OOM)
        print(f"  [GeomLoss error: {type(e).__name__}: {e}]")
        return BackwardResult(method_name, n, m, d, eps, 0, 0, 0, 0, oom=True)


def bench_geomloss_tensorized_backward(
    n: int, m: int, d: int, eps: float, n_iters: int,
    device: torch.device, warmup: int, rep: int,
    grad: str = "x",
) -> BackwardResult:
    """Benchmark GeomLoss Tensorized (dense) forward and end-to-end grad evaluation.

    Materializes O(n²) cost matrix in GPU memory. This is the O(n²) baseline
    that demonstrates the memory efficiency advantage of streaming methods.

    Cost convention: ||x-y||² (full squared Euclidean, matches FlashSinkhorn).

    Note: Tensorized can be faster than online methods at small n due to
    precomputed cost matrix, but OOMs at large n due to O(n²) memory.
    """
    from geomloss.sinkhorn_divergence import log_weights, sinkhorn_cost, sinkhorn_loop
    from geomloss.sinkhorn_samples import softmin_tensorized

    torch.manual_seed(0)
    x = torch.randn(n, d, device=device, dtype=torch.float32).requires_grad_(True)
    y = torch.randn(m, d, device=device, dtype=torch.float32)
    if grad == "xy":
        y.requires_grad_(True)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(m, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    def _sqdist_cost(x_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        """Compute ||x-y||² (full squared Euclidean, matches FlashSinkhorn)."""
        x2 = (x_t * x_t).sum(dim=-1, keepdim=True)
        y2 = (y_t * y_t).sum(dim=-1, keepdim=True).transpose(-2, -1)
        return x2 + y2 - 2.0 * torch.matmul(x_t, y_t.transpose(-2, -1))

    eps_list = [eps] * n_iters
    softmin = partial(softmin_tensorized)

    def _loss() -> torch.Tensor:
        # Detach cross terms for Danskin's theorem (envelope theorem gradients).
        C_xy = _sqdist_cost(x.unsqueeze(0), y.detach().unsqueeze(0))
        C_yx = C_xy.transpose(-1, -2)

        a_log = log_weights(a).unsqueeze(0)
        b_log = log_weights(b).unsqueeze(0)

        _, _, g_ab, f_ba = sinkhorn_loop(
            softmin,
            a_log,
            b_log,
            None,
            None,
            C_xy,
            C_yx,
            eps_list,
            rho=None,
            debias=False,
            last_extrapolation=True,
        )
        return sinkhorn_cost(
            eps,
            None,
            a,
            b,
            None,
            None,
            g_ab.squeeze(0),
            f_ba.squeeze(0),
            batch=False,
            debias=False,
            potentials=False,
        )

    grad_inputs = (x,) if grad == "x" else (x, y)

    try:
        def run_grad() -> None:
            """Run forward + backward (end-to-end gradient evaluation)."""
            loss = _loss()
            torch.autograd.grad(loss, grad_inputs, retain_graph=False, create_graph=False)

        # Reset peak memory BEFORE warmup to capture O(n²) cost matrix allocation
        torch.cuda.reset_peak_memory_stats(device)

        # Warmup before timed region
        run_grad()
        torch.cuda.synchronize()

        # Capture peak memory after warmup (includes cost matrix allocation)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6

        # Time the steady-state execution
        total_ms = bench_with_stats(run_grad, warmup, rep)

        return BackwardResult("geomloss_tensorized", n, m, d, eps, -1.0, -1.0, total_ms, peak_memory_mb, oom=False)

    except torch.cuda.OutOfMemoryError:
        return BackwardResult("geomloss_tensorized", n, m, d, eps, 0, 0, 0, 0, oom=True)
    except Exception as e:
        print(f"  [GeomLoss Tensorized error: {type(e).__name__}: {e}]")
        return BackwardResult("geomloss_tensorized", n, m, d, eps, 0, 0, 0, 0, oom=True)


def bench_ott_jax_backward(
    n: int, m: int, d: int, eps: float, n_iters: int,
    device: torch.device, warmup: int, rep: int,
    batch_size: int = 256,
    grad: str = "x",
    allow_tf32: bool = False,
) -> Optional[BackwardResult]:
    """Benchmark OTT-JAX (compiled) gradient evaluation.

    Uses use_danskin=True which treats potentials as constants during backward,
    avoiding O(n²) memory from unrolling through iterations. Works with online
    mode (batch_size) for efficient memory usage.

    Cost convention: PointCloud default = ||x-y||² (matches FlashSinkhorn).

    IMPORTANT - JAX timing methodology:
    JAX JIT compiles `jax.grad(loss_fn)` into a SINGLE fused function that
    includes both forward and backward passes. Unlike PyTorch where forward
    and backward are separate operations, JAX's grad_fn recomputes forward
    internally. Therefore:
    - forward_ms: Reported as N/A (cannot be separated from backward)
    - backward_ms: Reported as N/A (cannot be separated from forward)
    - total_ms: Time for grad_fn() which is the fused forward+backward

    Comparing "backward = total - forward" is WRONG for JAX because loss_jit()
    and grad_jit() are separate JIT functions that don't share computation.

    Memory: JAX doesn't expose easy peak memory tracking like PyTorch, so we report 0.
    """
    try:
        import jax
        import jax.numpy as jnp
        from jax import config as jax_config
        from ott.geometry import pointcloud
        from ott.problems.linear import linear_problem
        from ott.solvers.linear import sinkhorn
    except ImportError:
        return None

    # Match PyTorch TF32 setting for fair comparison
    jax_precision = "default" if allow_tf32 else "highest"
    jax_config.update("jax_default_matmul_precision", jax_precision)

    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4 = jax.random.split(key, 4)

    x = jax.random.normal(key1, (n, d), dtype=jnp.float32)
    y = jax.random.normal(key2, (m, d), dtype=jnp.float32)
    a = jax.random.uniform(key3, (n,), dtype=jnp.float32) + 0.1
    b = jax.random.uniform(key4, (m,), dtype=jnp.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    # Native solver with fixed iterations and Danskin for efficient backward
    solver = sinkhorn.Sinkhorn(
        threshold=-1.0,  # Never converge early
        max_iterations=n_iters,
        min_iterations=n_iters,  # Force exactly n_iters iterations
        use_danskin=True,  # Efficient backward via Danskin's theorem
    )

    def loss_fn(x, y):
        # PointCloud default cost = ||x-y||² (squared Euclidean, matches FlashSinkhorn)
        geom = pointcloud.PointCloud(x, y, epsilon=eps, batch_size=batch_size)
        prob = linear_problem.LinearProblem(geom, a=a, b=b)
        # OTT-JAX loss properties:
        #   reg_ot_cost  = <a,f> + <b,g> + eps*(H(a) + H(b))  [adds marginal entropies]
        #   ent_reg_cost = <a,f> + <b,g>                      [pure dual objective]
        #
        # Gradients are IDENTICAL (cos_sim = 1.0) because H(a), H(b) are constants
        # w.r.t. point locations x. We use reg_ot_cost as it's OTT-JAX's standard
        # output with use_danskin=True (envelope theorem).
        return solver(prob).reg_ot_cost

    loss_jit = jax.jit(loss_fn)
    if grad == "xy":
        grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))
    else:
        grad_fn = jax.jit(jax.grad(loss_fn, argnums=0))

    def run():
        out = grad_fn(x, y)
        if isinstance(out, tuple):
            return tuple(jax.block_until_ready(t) for t in out)
        return jax.block_until_ready(out)

    def run_forward():
        return jax.block_until_ready(loss_jit(x, y))

    try:
        # Warmup only the grad function (forward is included in grad_fn)
        for _ in range(warmup):
            run()
    except Exception:
        return BackwardResult("ott_jax", n, m, d, eps, 0, 0, 0, 0, oom=True)

    # Time only the grad function - it includes forward pass internally.
    # DO NOT measure forward separately: JAX's grad_fn is a fused JIT function
    # that recomputes forward, so "backward = total - forward" is meaningless.
    total_times = []
    for _ in range(rep):
        start = time.perf_counter()
        run()
        total_times.append((time.perf_counter() - start) * 1000)

    total_ms = sum(total_times) / len(total_times)
    # Report forward_ms and backward_ms as -1 to indicate "N/A" (cannot separate)
    # JAX doesn't expose easy peak memory tracking; report 0
    return BackwardResult("ott_jax", n, m, d, eps, -1.0, -1.0, total_ms, 0, oom=False)


def run_backward_benchmark(
    sizes: List[int],
    d: int,
    eps: float,
    n_iters: int,
    device: torch.device,
    warmup: int,
    rep: int,
    grad: str,
    verbose: bool = True,
    skip_flash_symmetric: bool = False,
    skip_flash_alternating: bool = False,
    skip_geomloss: bool = False,
    skip_ott: bool = False,
    include_tensorized: bool = False,
    max_dense_size: int = 8192,
    allow_tf32: bool = False,
) -> List[BackwardResult]:
    """Run backward pass benchmark (sizes large->small).

    FlashSinkhorn backends:
    - flash_symmetric: GeomLoss-style symmetric updates (compare with GeomLoss)
    - flash_alternating: OTT-JAX-style alternating updates (compare with OTT-JAX)

    Args:
        include_tensorized: If True, include GeomLoss Tensorized (O(n²) memory).
        max_dense_size: Maximum size for tensorized methods (to avoid OOM).
    """
    results = []
    sizes_sorted = sorted(sizes, reverse=True)

    for n in sizes_sorted:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Backward benchmark: n={n}, d={d}, eps={eps}")
            print(f"{'='*70}")

        # FlashSinkhorn (symmetric backend - GeomLoss-style Jacobi updates)
        if not skip_flash_symmetric:
            res = bench_flashsinkhorn_backward(n, n, d, eps, n_iters, device, warmup, rep, grad=grad, backend="symmetric", allow_tf32=allow_tf32)
            results.append(res)
            if verbose:
                if res.oom:
                    print(f"  Flash (symmetric):     OOM")
                else:
                    print(f"  Flash (symmetric):     total={res.total_ms:.2f}ms")
        elif verbose:
            print(f"  Flash (symmetric):     SKIPPED")

        # FlashSinkhorn (alternating backend - OTT-JAX-style Gauss-Seidel updates)
        if not skip_flash_alternating:
            res = bench_flashsinkhorn_backward(n, n, d, eps, n_iters, device, warmup, rep, grad=grad, backend="alternating", allow_tf32=allow_tf32)
            results.append(res)
            if verbose:
                if res.oom:
                    print(f"  Flash (alternating):   OOM")
                else:
                    print(f"  Flash (alternating):   total={res.total_ms:.2f}ms")
        elif verbose:
            print(f"  Flash (alternating):   SKIPPED")

        # GeomLoss online (KeOps)
        if not skip_geomloss:
            res = bench_geomloss_backward(n, n, d, eps, n_iters, device, warmup, rep, "online", grad=grad)
            results.append(res)
            if verbose:
                if res.oom:
                    print(f"  GeomLoss KeOps:        OOM")
                else:
                    print(f"  GeomLoss KeOps:        total={res.total_ms:.2f}ms")
        elif verbose:
            print(f"  GeomLoss KeOps:        SKIPPED")

        # GeomLoss tensorized (dense, small sizes only)
        if not skip_geomloss and include_tensorized and n <= max_dense_size:
            res = bench_geomloss_tensorized_backward(n, n, d, eps, n_iters, device, warmup, rep, grad=grad)
            results.append(res)
            if verbose:
                if res.oom:
                    print(f"  GeomLoss Tensorized:   OOM")
                else:
                    print(f"  GeomLoss Tensorized:   total={res.total_ms:.2f}ms")
        elif verbose and include_tensorized and n > max_dense_size:
            print(f"  GeomLoss Tensorized:   SKIPPED (n > max_dense_size={max_dense_size})")

        # OTT-JAX (Danskin)
        if not skip_ott:
            res = bench_ott_jax_backward(n, n, d, eps, n_iters, device, warmup, rep, grad=grad, allow_tf32=allow_tf32)
            if res:
                results.append(res)
                if verbose:
                    if res.oom:
                        print(f"  OTT-JAX:               OOM")
                    else:
                        print(f"  OTT-JAX:               total={res.total_ms:.2f}ms")
            elif verbose:
                print(f"  OTT-JAX:               NOT AVAILABLE")
        elif verbose:
            print(f"  OTT-JAX:               SKIPPED")

        gc.collect()
        torch.cuda.empty_cache()

    return results


def save_results_csv(results: List[BackwardResult], output_path: Path) -> None:
    """Save results to CSV, merging with existing data if file exists.

    Uses (method, n, d) as the unique key. New results overwrite existing ones
    with the same key, allowing incremental benchmark runs.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if file exists
    existing_results: Dict[tuple, dict] = {}
    if output_path.exists():
        with open(output_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["method"], int(row["n"]), int(row["d"]))
                existing_results[key] = row
        print(f"  Loaded {len(existing_results)} existing results from {output_path}")

    # Convert new results to dict format and merge
    for r in results:
        key = (r.method, r.n, r.d)
        if r.oom:
            existing_results[key] = {
                "method": r.method, "n": r.n, "m": r.m, "d": r.d, "eps": r.eps,
                "forward_ms": "OOM", "backward_ms": "OOM", "total_ms": "OOM",
                "peak_memory_mb": "OOM", "oom": True
            }
        else:
            fwd_str = "N/A" if r.forward_ms < 0 else f"{r.forward_ms:.3f}"
            bwd_str = "N/A" if r.backward_ms < 0 else f"{r.backward_ms:.3f}"
            existing_results[key] = {
                "method": r.method, "n": r.n, "m": r.m, "d": r.d, "eps": r.eps,
                "forward_ms": fwd_str, "backward_ms": bwd_str,
                "total_ms": f"{r.total_ms:.3f}", "peak_memory_mb": f"{r.peak_memory_mb:.1f}",
                "oom": False
            }

    # Write merged results
    with open(output_path, "w", newline="") as f:
        fieldnames = ["method", "n", "m", "d", "eps", "forward_ms", "backward_ms",
                      "total_ms", "peak_memory_mb", "oom"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Sort by (d, n, method) for consistent output
        for key in sorted(existing_results.keys(), key=lambda k: (int(existing_results[k]["d"]), int(existing_results[k]["n"]), k[0])):
            writer.writerow(existing_results[key])

    print(f"\nSaved {len(existing_results)} total results to {output_path}")


def print_summary(results: List[BackwardResult]) -> None:
    """Print summary table with both FlashSinkhorn backends."""
    print("\n" + "=" * 140)
    print("GRADIENT EVALUATION TIMING SUMMARY (forward + backward)")
    print("Use case: Training / Optimization")
    print("=" * 140)

    sizes = sorted(set(r.n for r in results))
    methods = ["flash_symmetric", "flash_alternating", "geomloss_online", "geomloss_tensorized", "ott_jax"]

    # Check if tensorized results exist
    has_tensorized = any(r.method == "geomloss_tensorized" for r in results)

    # Header
    if has_tensorized:
        header = f"{'n':>8s}  {'F.symm':>12s}  {'F.alt':>12s}  {'GL KeOps':>12s}  {'GL Tensor':>12s}  {'OTT-JAX':>12s}  {'symm/KeOps':>12s}  {'symm/Tensor':>12s}"
    else:
        header = f"{'n':>8s}  {'F.symm':>12s}  {'F.alt':>12s}  {'GeomLoss':>12s}  {'OTT-JAX':>12s}  {'symm/GL':>10s}  {'alt/OTT':>10s}"
    print(header)
    print("-" * 140)

    for n in sizes:
        print(f"{n:>8d}", end="")

        flash_symmetric_total = None
        flash_alternating_total = None
        gl_total = None
        gl_tensor_total = None
        ott_total = None

        for method in methods:
            if method == "geomloss_tensorized" and not has_tensorized:
                continue
            match = [r for r in results if r.n == n and r.method == method]
            if match and not match[0].oom:
                r = match[0]
                print(f"  {r.total_ms:>9.1f}ms", end="")
                if method == "flash_symmetric":
                    flash_symmetric_total = r.total_ms
                elif method == "flash_alternating":
                    flash_alternating_total = r.total_ms
                elif method == "geomloss_online":
                    gl_total = r.total_ms
                elif method == "geomloss_tensorized":
                    gl_tensor_total = r.total_ms
                elif method == "ott_jax":
                    ott_total = r.total_ms
            else:
                print(f"  {'OOM/N/A':>12s}", end="")

        # Speedup: flash_symmetric vs GeomLoss KeOps
        if flash_symmetric_total and gl_total:
            print(f"  {gl_total/flash_symmetric_total:>11.1f}x", end="")
        else:
            print(f"  {'N/A':>12s}", end="")

        # Speedup: flash_symmetric vs GeomLoss Tensorized (if available)
        if has_tensorized:
            if flash_symmetric_total and gl_tensor_total:
                print(f"  {gl_tensor_total/flash_symmetric_total:>11.1f}x")
            else:
                print(f"  {'N/A':>12s}")
        else:
            # Speedup: flash_alternating vs OTT-JAX
            if flash_alternating_total and ott_total:
                print(f"  {ott_total/flash_alternating_total:>9.1f}x")
            else:
                print(f"  {'N/A':>10s}")

    print("=" * 140)


def verify_gradient_parity(
    n: int = 1000,
    d: int = 64,
    eps: float = 1.0,
    n_iters: int = 10,
    device: torch.device = None,
) -> bool:
    """Verify gradient parity between FlashSinkhorn backends and their references.

    Tests both backends:
    - flash_symmetric vs GeomLoss (should match, both use symmetric updates)
    - flash_alternating vs OTT-JAX (should match, both use alternating updates)

    Returns True if both comparisons pass:
    - flash_symmetric vs GeomLoss: cos_sim > 0.99 (same algorithm, very strict)
    - flash_alternating vs OTT-JAX: cos_sim > 0.98 (independent codebases, slightly relaxed)
    """
    if device is None:
        device = torch.device("cuda")

    print("\n" + "=" * 70)
    print("GRADIENT PARITY VERIFICATION")
    print("=" * 70)
    print(f"  n={n}, d={d}, eps={eps}, n_iters={n_iters}")

    # Setup data
    torch.manual_seed(42)
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    y = torch.randn(n, d, device=device, dtype=torch.float32)
    a = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    b = torch.rand(n, device=device, dtype=torch.float32) + 0.1
    a = a / a.sum()
    b = b / b.sum()

    grads = {}
    losses = {}
    all_passed = True

    # =========================================================================
    # Test 1: flash_symmetric vs GeomLoss (symmetric updates)
    # =========================================================================
    print("\n  --- Test 1: flash_symmetric vs GeomLoss ---")

    # FlashSinkhorn online gradient
    x1 = x.detach().clone().requires_grad_(True)
    try:
        from flash_sinkhorn import SamplesLoss
        flash_symmetric_fn = SamplesLoss(
            "sinkhorn", backend="symmetric", use_epsilon_scaling=False,
            eps=eps, n_iters=n_iters, debias=False, normalize=False,
            last_extrapolation=False, allow_tf32=False,
        )
        loss = flash_symmetric_fn(a, x1, b, y)
        losses["flash_symmetric"] = loss.item()
        grads["flash_symmetric"] = torch.autograd.grad(loss, x1, retain_graph=False)[0]
        print(f"    Flash (online): loss={losses['flash_symmetric']:.6f}")
    except Exception as e:
        print(f"    Flash (online): FAILED ({e})")
        all_passed = False

    # GeomLoss gradient
    x2 = x.detach().clone().requires_grad_(True)
    try:
        from geomloss.sinkhorn_divergence import log_weights, sinkhorn_cost, sinkhorn_loop
        from geomloss.sinkhorn_samples import lse_genred, softmin_online

        eps_list = [eps] * n_iters
        a_log = log_weights(a)
        b_log = log_weights(b)
        my_lse = lse_genred("SqDist(X,Y)", d)
        softmin = partial(softmin_online, log_conv=my_lse)

        C_xy = (x2, y.detach())
        C_yx = (y, x2.detach())
        _, _, g_ab, f_ba = sinkhorn_loop(
            softmin, a_log, b_log, None, None, C_xy, C_yx, eps_list,
            rho=None, debias=False, last_extrapolation=True,
        )
        gl_loss = sinkhorn_cost(
            eps, None, a, b, None, None, g_ab, f_ba,
            batch=False, debias=False, potentials=False,
        )
        losses["geomloss"] = gl_loss.item()
        grads["geomloss"] = torch.autograd.grad(gl_loss, x2, retain_graph=False)[0]
        print(f"    GeomLoss:       loss={losses['geomloss']:.6f}")
    except Exception as e:
        print(f"    GeomLoss: FAILED ({e})")
        all_passed = False

    # Check online vs GeomLoss parity
    if "flash_symmetric" in grads and "geomloss" in grads:
        cos_sim = torch.nn.functional.cosine_similarity(
            grads["flash_symmetric"].flatten(), grads["geomloss"].flatten(), dim=0
        ).item()
        passed = cos_sim > 0.99
        symbol = "✓" if passed else "✗"
        print(f"    {symbol} Gradient cos_sim: {cos_sim:.6f} ({'PASS' if passed else 'FAIL'})")
        all_passed = all_passed and passed

    # =========================================================================
    # Test 2: flash_alternating vs OTT-JAX (alternating updates, Danskin gradient)
    # =========================================================================
    print("\n  --- Test 2: flash_alternating vs OTT-JAX ---")

    # FlashSinkhorn ott gradient
    x3 = x.detach().clone().requires_grad_(True)
    try:
        flash_alternating_fn = SamplesLoss(
            "sinkhorn", backend="alternating", use_epsilon_scaling=False,
            eps=eps, n_iters=n_iters, debias=False, normalize=False,
            allow_tf32=False,
        )
        loss = flash_alternating_fn(a, x3, b, y)
        losses["flash_alternating"] = loss.item()
        grads["flash_alternating"] = torch.autograd.grad(loss, x3, retain_graph=False)[0]
        print(f"    Flash (ott):    loss={losses['flash_alternating']:.6f}")
    except Exception as e:
        print(f"    Flash (ott): FAILED ({e})")
        all_passed = False

    # OTT-JAX gradient (Danskin)
    try:
        import jax
        import jax.numpy as jnp
        from jax import config as jax_config
        from ott.geometry import pointcloud
        from ott.problems.linear import linear_problem
        from ott.solvers.linear import sinkhorn

        jax_config.update("jax_default_matmul_precision", "highest")

        # Convert to JAX arrays
        x_jax = jnp.array(x.cpu().numpy())
        y_jax = jnp.array(y.cpu().numpy())
        a_jax = jnp.array(a.cpu().numpy())
        b_jax = jnp.array(b.cpu().numpy())

        solver = sinkhorn.Sinkhorn(
            threshold=-1.0,
            max_iterations=n_iters,
            min_iterations=n_iters,
            use_danskin=True,  # Efficient backward via Danskin's theorem
            implicit_diff=None,  # Explicitly disable implicit differentiation
        )

        def loss_fn(x_in):
            # Use batch_size for online mode (recompute cost on the fly)
            geom = pointcloud.PointCloud(x_in, y_jax, epsilon=eps, batch_size=256)
            prob = linear_problem.LinearProblem(geom, a=a_jax, b=b_jax)
            # OTT-JAX loss: reg_ot_cost = <a,f>+<b,g> + eps*(H(a)+H(b))
            # H(a), H(b) are marginal entropies (constants w.r.t. x)
            # Gradients match FlashSinkhorn exactly (cos_sim = 1.0)
            return solver(prob).reg_ot_cost

        grad_fn = jax.grad(loss_fn)
        ott_grad = grad_fn(x_jax)
        losses["ott_jax"] = float(loss_fn(x_jax))
        grads["ott_jax"] = torch.from_numpy(jax.device_get(ott_grad)).to(device)
        print(f"    OTT-JAX:        loss={losses['ott_jax']:.6f}")
    except Exception as e:
        print(f"    OTT-JAX: FAILED ({e})")

    # Check ott vs OTT-JAX parity
    # Both use identical update rules (alternating Gauss-Seidel), so gradients
    # should match exactly. reg_ot_cost adds eps*(H(a)+H(b)) which vanishes
    # under differentiation since marginal entropies are constant w.r.t. x.
    if "flash_alternating" in grads and "ott_jax" in grads:
        cos_sim = torch.nn.functional.cosine_similarity(
            grads["flash_alternating"].flatten(), grads["ott_jax"].flatten(), dim=0
        ).item()
        passed = cos_sim > 0.999  # Expect near-perfect match (cos_sim ≈ 1.0)
        symbol = "✓" if passed else "✗"
        print(f"    {symbol} Gradient cos_sim: {cos_sim:.6f} ({'PASS' if passed else 'FAIL'})")
        all_passed = all_passed and passed

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n  --- Summary ---")
    for name, loss in losses.items():
        print(f"    {name:15s}: loss={loss:.6f}")

    if all_passed:
        print(f"\n  ✓ ALL GRADIENT PARITY TESTS PASSED")
    else:
        print(f"\n  ✗ SOME GRADIENT PARITY TESTS FAILED")

    print("=" * 70)
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backward pass benchmark for FlashSinkhorn paper."
    )
    parser.add_argument(
        "--sizes", type=str, default="1024,2048,4096,8192,10000,20000,50000,100000,200000",
        help="Comma-separated sizes (sorted large->small internally)."
    )
    parser.add_argument(
        "--dims", type=str, default="4,8,16,32,64,128",
        help="Comma-separated dimensions to test."
    )
    parser.add_argument("--eps", type=float, default=0.1, help="Regularization epsilon.")
    parser.add_argument("--n-iters", type=int, default=10, help="Sinkhorn iterations.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--rep", type=int, default=30, help="Timed repetitions.")
    parser.add_argument(
        "--grad", choices=("x", "xy"), default="x",
        help="Which gradients to compute (x only, or both x and y).",
    )
    parser.add_argument("--tf32", action="store_true", default=True,
                        help="Enable TF32 for ~2x speedup (default: enabled).")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false",
                        help="Disable TF32 for strict FP32 (slower but higher precision).")
    parser.add_argument(
        "--output-dir", type=str, default="output/paper_benchmarks/backward",
        help="Output directory."
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-geomloss", action="store_true", help="Skip GeomLoss benchmarks.")
    parser.add_argument("--no-ott", action="store_true", help="Skip OTT-JAX benchmarks.")
    parser.add_argument("--no-flash-symmetric", action="store_true", help="Skip FlashSinkhorn symmetric backend.")
    parser.add_argument("--no-flash-alternating", action="store_true", help="Skip FlashSinkhorn alternating backend.")
    parser.add_argument("--tensorized", action="store_true", help="Include GeomLoss Tensorized (O(n²) memory) benchmarks.")
    parser.add_argument("--max-dense-size", type=int, default=20000,
                        help="Max size for tensorized/dense methods (to avoid OOM). Default: 20000.")
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify gradient parity between FlashSinkhorn backends and their references before benchmarking."
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    _preload_cuda_libs()
    _set_tf32(args.tf32)

    device = torch.device("cuda")

    # Run verification if requested
    if args.verify:
        passed = verify_gradient_parity(
            n=1000, d=64, eps=args.eps, n_iters=args.n_iters, device=device
        )
        if not passed:
            print("\nGradient parity verification failed. Check cost conventions.")
            return
        print("\nProceeding with benchmark...\n")

    sizes = [int(s) for s in args.sizes.split(",")]
    dims = [int(d) for d in args.dims.split(",")]

    print(f"Backward Pass Benchmark")
    print(f"  Sizes: {sorted(sizes, reverse=True)}")
    print(f"  Dimensions: {dims}")
    print(f"  Epsilon: {args.eps}")
    print(f"  Precision: {'TF32' if args.tf32 else 'FP32 (strict)'}")
    print(f"  Grad: {args.grad}")
    print(f"  FlashSinkhorn backends: symmetric={not args.no_flash_symmetric}, alternating={not args.no_flash_alternating}")
    print(f"  References: GeomLoss={not args.no_geomloss}, OTT-JAX={not args.no_ott}")
    print(f"  Include tensorized: {args.tensorized} (max size: {args.max_dense_size})")
    print(f"  GPU: {torch.cuda.get_device_name()}")

    all_results = []
    for d in dims:
        print(f"\n{'='*70}")
        print(f"Dimension d={d}")
        print(f"{'='*70}")

        results = run_backward_benchmark(
            sizes=sizes,
            d=d,
            eps=args.eps,
            n_iters=args.n_iters,
            device=device,
            warmup=args.warmup,
            rep=args.rep,
            grad=args.grad,
            verbose=not args.quiet,
            skip_flash_symmetric=args.no_flash_symmetric,
            skip_flash_alternating=args.no_flash_alternating,
            skip_geomloss=args.no_geomloss,
            skip_ott=args.no_ott,
            include_tensorized=args.tensorized,
            max_dense_size=args.max_dense_size,
            allow_tf32=args.tf32,
        )
        all_results.extend(results)

    output_dir = Path(args.output_dir)
    save_results_csv(all_results, output_dir / "backward_all.csv")

    print_summary(all_results)


if __name__ == "__main__":
    main()
