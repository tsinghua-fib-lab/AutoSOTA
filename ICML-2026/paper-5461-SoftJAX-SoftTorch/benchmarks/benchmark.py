import argparse
import gc
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import equinox as eqx
import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import pandas as pd
import softjax


# ---------------------------------------------------------------------------
# Function registry
# ---------------------------------------------------------------------------


@dataclass
class FunctionSpec:
    name: str
    fn: Callable
    kwargs: dict = field(default_factory=dict)
    methods: list[str] | None = None
    two_d: bool = False


AXISWISE_SPECS = [
    FunctionSpec(
        "argmax",
        softjax.argmax,
        {"axis": 1},
        ["ot", "softsort", "neuralsort", "sorting_network"],
        two_d=True,
    ),
    FunctionSpec(
        "argmin",
        softjax.argmin,
        {"axis": 1},
        ["ot", "softsort", "neuralsort", "sorting_network"],
        two_d=True,
    ),
    FunctionSpec(
        "argsort",
        softjax.argsort,
        {"axis": 1},
        ["ot", "softsort", "neuralsort", "sorting_network"],
        two_d=True,
    ),
    FunctionSpec(
        "max",
        softjax.max,
        {"axis": 1},
        ["ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"],
        two_d=True,
    ),
    FunctionSpec(
        "min",
        softjax.min,
        {"axis": 1},
        ["ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"],
        two_d=True,
    ),
    FunctionSpec(
        "sort",
        softjax.sort,
        {"axis": 1},
        ["ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"],
        two_d=True,
    ),
    FunctionSpec(
        "rank",
        softjax.rank,
        {"axis": 1},
        ["ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"],
        two_d=True,
    ),
    FunctionSpec(
        "top_k",
        softjax.top_k,
        {"axis": 1, "k": 5},
        ["ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"],
        two_d=True,
    ),
    FunctionSpec(
        "argquantile",
        softjax.argquantile,
        {"axis": 1, "q": 0.25},
        ["ot", "softsort", "neuralsort", "sorting_network"],
        two_d=True,
    ),
    FunctionSpec(
        "quantile",
        softjax.quantile,
        {"axis": 1, "q": 0.25},
        ["ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"],
        two_d=True,
    ),
    FunctionSpec(
        "argmedian",
        softjax.argmedian,
        {"axis": 1},
        ["ot", "softsort", "neuralsort", "sorting_network"],
        two_d=True,
    ),
    FunctionSpec(
        "median",
        softjax.median,
        {"axis": 1},
        ["ot", "softsort", "neuralsort", "fast_soft_sort", "smooth_sort", "sorting_network"],
        two_d=True,
    ),
]

ELEMENTWISE_SPECS = [
    FunctionSpec("heaviside", softjax.heaviside),
    FunctionSpec("round", softjax.round),
    FunctionSpec("sign", softjax.sign),
    FunctionSpec("abs", softjax.abs),
    FunctionSpec("relu", softjax.relu),
    FunctionSpec("clip", softjax.clip, {"a": -1.0, "b": 1.0}),
    FunctionSpec("greater", softjax.greater, {"y": 0.0}),
    FunctionSpec("greater_equal", softjax.greater_equal, {"y": 0.0}),
    FunctionSpec("less", softjax.less, {"y": 0.0}),
    FunctionSpec("less_equal", softjax.less_equal, {"y": 0.0}),
    FunctionSpec("equal", softjax.equal, {"y": 0.0}),
    FunctionSpec("not_equal", softjax.not_equal, {"y": 0.0}),
    FunctionSpec("isclose", softjax.isclose, {"y": 0.0}),
]

ALL_SPECS = AXISWISE_SPECS + ELEMENTWISE_SPECS

AXISWISE_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
ELEMENTWISE_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
OT_MAX_SIZE = None  # No cap; OOM is caught by try/except

MODES = ["hard", "smooth", "c0", "c1", "c2"]


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------


def _input_array(
    size: int, dtype: np.dtype, two_d: bool, batch_size: int = 1, seed: int = 42
) -> jax.Array:
    shape = (batch_size, size) if two_d else (batch_size * size,)
    arr = jax.random.normal(jax.random.key(seed), shape, dtype=dtype)
    return jax.device_put(arr).block_until_ready()


def _reduce_output(res):
    if isinstance(res, tuple):
        return res[0].sum()
    return res.sum()


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


def measure_jit_time(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    num_trials: int = 1,
) -> tuple[float, float, float]:
    times = []
    for _ in range(num_trials):

        def wrapper(*a, **kw):
            return fn(*a, **kw)

        t0 = time.perf_counter_ns()
        eqx.filter_jit(wrapper).lower(*args, **kwargs).compile()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)

    arr = np.array(times)
    return arr.mean().item(), arr.std().item(), arr.min().item()


def _microtime(fn: Callable, num_times: int, warmup: int) -> float:
    for i in range(num_times + warmup):
        if i == warmup:
            start = time.perf_counter_ns()
        fn()
    end = time.perf_counter_ns()
    return (end - start) / (num_times * 1e6)


def measure_runtime(
    compiled_fn: Callable,
    args: tuple,
    kwargs: dict,
    num_runs: int = 1,
    warmup: int = 3,
) -> tuple[float, float, float]:
    gc.disable()

    def run():
        res = compiled_fn(*args, **kwargs)
        if isinstance(res, tuple):
            res[0].block_until_ready()
        elif isinstance(res, jax.Array):
            res.block_until_ready()

    # Adaptive micro-batching
    first = _microtime(run, 1, warmup=0)
    num_micro = max(1, int(400 / first))

    times = []
    for i in range(num_runs + warmup):
        dur = _microtime(run, num_micro, warmup=warmup)
        if i >= warmup:
            times.append(dur)

    gc.enable()
    gc.collect(1)

    arr = np.array(times)
    return arr.mean().item(), arr.std().item(), arr.min().item()


# ---------------------------------------------------------------------------
# Memory measurement (GPU only, in-process)
# ---------------------------------------------------------------------------


def measure_peak_memory(
    fn: Callable,
    args: tuple,
    kwargs: dict,
) -> int:
    from functools import partial

    bound_fn = partial(fn, **kwargs)

    @jax.grad
    def grad_fn(*a):
        return _reduce_output(bound_fn(*a))

    compiled = jax.jit(grad_fn).lower(*args).compile()
    mem = compiled.memory_analysis()
    if mem is None:
        return -1
    return (
        mem.temp_size_in_bytes + mem.argument_size_in_bytes + mem.output_size_in_bytes
    )


# ---------------------------------------------------------------------------
# Core benchmark driver
# ---------------------------------------------------------------------------


def benchmark_single(
    spec: FunctionSpec,
    mode: str,
    method: str | None,
    size: int,
    dtype: np.dtype,
    is_gpu: bool,
    benchmark_jit: bool,
    num_trials: int = 1,
    jit_trials: int = 1,
    softness: float = 1.0,
    batch_size: int = 1,
) -> dict:
    call_kwargs = dict(spec.kwargs)
    call_kwargs["mode"] = mode
    if method is not None:
        call_kwargs["method"] = method
    call_kwargs["softness"] = jnp.array(softness)

    fn = spec.fn

    # Use first input for compilation (shape-dependent, not data-dependent)
    inp0 = _input_array(size, dtype, spec.two_d, batch_size=batch_size, seed=0)
    args0 = (inp0,)

    # --- Forward JIT (shape-dependent) ---
    if benchmark_jit:
        jit_fwd_mean, jit_fwd_std, jit_fwd_min = measure_jit_time(
            fn, args0, call_kwargs, num_trials=jit_trials
        )
    else:
        jit_fwd_mean = jit_fwd_std = jit_fwd_min = 0.0

    # --- Compile forward and grad ---
    fwd_compiled = eqx.filter_jit(fn).lower(*args0, **call_kwargs).compile()

    @jax.grad
    def grad_fn(*a, **kw):
        return _reduce_output(fn(*a, **kw))

    # --- Grad JIT (shape-dependent) ---
    if benchmark_jit:
        jit_grad_mean, jit_grad_std, jit_grad_min = measure_jit_time(
            grad_fn, args0, call_kwargs, num_trials=jit_trials
        )
    else:
        jit_grad_mean = jit_grad_std = jit_grad_min = 0.0

    grad_compiled = eqx.filter_jit(grad_fn).lower(*args0, **call_kwargs).compile()

    # --- Runtime: fresh random input per trial (data-dependent convergence) ---
    # Seed is deterministic so all modes/methods see identical inputs per trial.
    fwd_times = []
    grad_times = []
    for trial in range(num_trials):
        inp = _input_array(size, dtype, spec.two_d, batch_size=batch_size, seed=trial)
        args = (inp,)
        fwd_mean, _, _ = measure_runtime(fwd_compiled, args, call_kwargs, num_runs=1)
        fwd_times.append(fwd_mean)
        grad_mean, _, _ = measure_runtime(grad_compiled, args, call_kwargs, num_runs=1)
        grad_times.append(grad_mean)

    fwd_arr = np.array(fwd_times)
    grad_arr = np.array(grad_times)

    # --- Peak memory (shape-dependent) ---
    peak_mem = measure_peak_memory(fn, args0, call_kwargs)

    category = "axiswise" if spec.methods is not None else "elementwise"
    return {
        "function": spec.name,
        "category": category,
        "mode": mode,
        "method": method if method is not None else "",
        "softness": softness,
        "batch_size": batch_size,
        "dtype": str(dtype),
        "device": "gpu" if is_gpu else "cpu",
        "problem_size": size,
        "jit_fwd_ms_mean": jit_fwd_mean,
        "jit_fwd_ms_std": jit_fwd_std,
        "jit_fwd_ms_min": jit_fwd_min,
        "fwd_ms_mean": fwd_arr.mean().item(),
        "fwd_ms_std": fwd_arr.std().item(),
        "fwd_ms_min": fwd_arr.min().item(),
        "jit_grad_ms_mean": jit_grad_mean,
        "jit_grad_ms_std": jit_grad_std,
        "jit_grad_ms_min": jit_grad_min,
        "grad_ms_mean": grad_arr.mean().item(),
        "grad_ms_std": grad_arr.std().item(),
        "grad_ms_min": grad_arr.min().item(),
        "peak_memory_bytes": peak_mem,
    }


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


def run_benchmark(
    specs: list[FunctionSpec],
    modes: list[str],
    sizes: list[int] | None,
    dtype: np.dtype,
    is_gpu: bool,
    benchmark_jit: bool,
    out_path: Path | None,
    plot_callback: Callable | None = None,
    num_trials: int = 1,
    jit_trials: int = 1,
    softness_values: list[float] | None = None,
    batch_sizes: list[int] | None = None,
) -> pd.DataFrame:
    if softness_values is None:
        softness_values = [0.1]
    if batch_sizes is None:
        batch_sizes = [1]

    rows: list[dict] = []

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    def _record(row: dict):
        rows.append(row)
        if out_path is not None:
            pd.DataFrame(rows).to_csv(out_path, index=False)

    # Collect all (spec, sizes) pairs
    spec_sizes_map = {}
    for spec in specs:
        is_axiswise = spec.methods is not None
        default_sizes = AXISWISE_SIZES if is_axiswise else ELEMENTWISE_SIZES
        spec_sizes_map[spec.name] = sizes if sizes is not None else default_sizes

    # All unique sizes, sorted ascending (outer loop)
    all_sizes = sorted({s for ss in spec_sizes_map.values() for s in ss})

    def _run_one(spec, mode, method, size, softness, batch_size):
        parts = [f"{spec.name} | mode={mode}"]
        if method is not None:
            parts.append(f"method={method}")
        parts.append(f"size={size}")
        if len(softness_values) > 1:
            parts.append(f"softness={softness}")
        if len(batch_sizes) > 1:
            parts.append(f"batch={batch_size}")
        label = " | ".join(parts)
        print(f"  {label}", flush=True)
        try:
            row = benchmark_single(
                spec,
                mode,
                method,
                size,
                dtype,
                is_gpu,
                benchmark_jit,
                num_trials,
                jit_trials=jit_trials,
                softness=softness,
                batch_size=batch_size,
            )
            _record(row)
        except Exception as e:
            print(f"    SKIP: {e}", flush=True)

    for size in all_sizes:
        for spec in specs:
            if size not in spec_sizes_map[spec.name]:
                continue
            is_axiswise = spec.methods is not None

            for softness in softness_values:
                for batch_size in batch_sizes:
                    for mode in modes:
                        if mode == "hard":
                            _run_one(spec, mode, None, size, softness, batch_size)
                        elif is_axiswise:
                            for method in spec.methods:
                                if (
                                    OT_MAX_SIZE is not None
                                    and method == "ot"
                                    and size > OT_MAX_SIZE
                                ):
                                    continue
                                _run_one(spec, mode, method, size, softness, batch_size)
                        else:
                            _run_one(spec, mode, None, size, softness, batch_size)

        # Plot after each size is complete
        if plot_callback is not None and rows:
            plot_callback(pd.DataFrame(rows))

    df = pd.DataFrame(rows)

    if out_path is not None:
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SoftJAX operators.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cpu", action="store_true", help="Run on CPU (default: GPU)")
    parser.add_argument(
        "--no-jit", action="store_true", help="Skip JIT compilation time measurement"
    )
    parser.add_argument(
        "--functions", type=str, default=None, help="Comma-separated function names"
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help="Comma-separated modes (default: hard,smooth,c0,c1,c2)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated methods (e.g. fast_soft_sort,softsort)",
    )
    parser.add_argument(
        "--sizes", type=str, default=None, help="Comma-separated problem sizes"
    )
    parser.add_argument(
        "--dtype", type=str, default="float64", help="Dtype (default: float64)"
    )
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["axiswise", "elementwise", "all"],
        help="Category to benchmark (default: all)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of runtime trials with fresh random inputs (default: 3)",
    )
    parser.add_argument(
        "--jit-trials",
        type=int,
        default=3,
        help="Number of JIT compilation trials (default: 3)",
    )
    parser.add_argument(
        "--softness",
        type=str,
        default="0.1",
        help="Comma-separated softness values (default: 0.1)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1",
        help="Comma-separated batch sizes (default: 1)",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate plots after each problem size"
    )
    parser.add_argument(
        "--plot-format", type=str, default="png", choices=["png", "pdf", "svg"]
    )
    args = parser.parse_args()

    # --- Device setup ---
    is_gpu = not args.cpu
    if is_gpu:
        if "cuda" not in jax.extend.backend.backends():
            parser.error("GPU requested but CUDA-enabled jaxlib is not installed.")
        elif not jax.devices("gpu"):
            parser.error("GPU requested but no GPU devices are available.")
        else:
            jax.config.update("jax_platform_name", "gpu")
    else:
        jax.config.update("jax_platform_name", "cpu")

    jax.config.update("jax_enable_x64", "64" in args.dtype)

    # --- Select specs ---
    if args.category == "axiswise":
        specs = list(AXISWISE_SPECS)
    elif args.category == "elementwise":
        specs = list(ELEMENTWISE_SPECS)
    else:
        specs = list(ALL_SPECS)

    if args.functions is not None:
        names = {n.strip() for n in args.functions.split(",")}
        specs = [s for s in specs if s.name in names]
        if not specs:
            parser.error(f"No matching functions found for: {args.functions}")

    # --- Modes ---
    modes = [m.strip() for m in args.modes.split(",")] if args.modes else list(MODES)

    # --- Methods filter ---
    methods_filter = None
    if args.methods is not None:
        methods_filter = {m.strip() for m in args.methods.split(",")}
        for spec in specs:
            if spec.methods is not None:
                spec.methods = [m for m in spec.methods if m in methods_filter]

    # --- Sizes ---
    sizes = [int(s.strip()) for s in args.sizes.split(",")] if args.sizes else None

    # --- Softness ---
    softness_values = [float(s.strip()) for s in args.softness.split(",")]

    # --- Batch sizes ---
    batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",")]

    # --- Dtype ---
    dtype = np.dtype(args.dtype)

    # --- Output path ---
    if args.output:
        out_path = Path(args.output)
    else:
        platform = "gpu" if is_gpu else "cpu"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(f"benchmarks/results/{timestamp}")
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / f"softjax_benchmark_{platform}_{args.dtype}.csv"

    # --- Plot callback ---
    plot_callback = None
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")

        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from plot_results import plot_scaling

        plot_dir = out_path.parent
        fmt = args.plot_format

        def plot_callback(df):
            import matplotlib.pyplot as plt

            plt.style.use("seaborn-v0_8-whitegrid")
            print("  [plotting...]", flush=True)
            plot_scaling(df, plot_dir, fmt)

    # --- Run ---
    platform_label = "GPU" if is_gpu else "CPU"
    benchmark_jit = not args.no_jit
    print(f"SoftJAX Benchmark — {platform_label}, dtype={dtype}, jit={benchmark_jit}")
    print(f"Functions: {[s.name for s in specs]}")
    print(f"Modes: {modes}")
    if len(softness_values) > 1:
        print(f"Softness: {softness_values}")
    if len(batch_sizes) > 1:
        print(f"Batch sizes: {batch_sizes}")
    print(f"Output: {out_path}\n")

    run_benchmark(
        specs,
        modes,
        sizes,
        dtype,
        is_gpu,
        benchmark_jit,
        out_path,
        plot_callback,
        num_trials=args.trials,
        jit_trials=args.jit_trials,
        softness_values=softness_values,
        batch_sizes=batch_sizes,
    )


if __name__ == "__main__":
    main()
