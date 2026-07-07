import time
from dataclasses import dataclass
from logging import getLogger
from typing import Any

import numpy as np
import torch
from accelerate import cpu_offload
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from ..utils import clear_cache
from .evaluator import ModelEvaluator
from .utils import eval_mode

logger = getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Prefill settings
    eval_prefill: bool = True
    prefill_batch_size: int = 16
    prefill_sequence_length: int = 4096
    prefill_num_warmup_steps: int = 5
    prefill_num_steps: int = 20

    # Decode settings
    eval_decode: bool = True
    decode_batch_size: int = 16
    decode_context_length: int = 512  # Prefill context before measuring decode
    decode_num_tokens_to_generate: int = 128
    decode_num_warmup_steps: int = 5
    decode_num_steps: int = 20

    # Measurement settings
    use_cpu_offload: bool = False
    use_cuda_events: bool = True  # More accurate GPU timing
    use_torch_compile: bool = False
    compile_mode: str = "max-autotune"
    nvtx_range: bool = False
    report_memory: bool = True
    report_percentiles: bool = True
    seed: int = 42


def get_device_info() -> dict[str, Any]:
    """Collect GPU device information for reproducibility."""
    if not torch.cuda.is_available():
        return {"device": "cpu"}

    n_devices = torch.cuda.device_count()
    devices_info = []
    for i in range(n_devices):
        props = torch.cuda.get_device_properties(i)
        devices_info.append(
            {
                "device_name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )

    return {
        "num_devices": n_devices,
        "devices": devices_info,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }


class CUDATimer:
    """
    Context manager that captures both CUDA events and wall-clock time.

    CUDA events measure actual GPU execution time, avoiding Python
    interpreter overhead. Wall-clock time (perf_counter) includes
    CPU overhead and can be useful for end-to-end comparisons.

    Synchronizes all visible CUDA devices for accurate multi-GPU timing.
    """

    def __init__(self, use_cuda_events=True):
        self.use_cuda_events = use_cuda_events and torch.cuda.is_available()

    def __enter__(self):
        if self.use_cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.use_cuda_events:
            self.end_event.record()
            torch.cuda.synchronize()  # Sync all devices for multi-GPU
        self._wall_elapsed = time.perf_counter() - self._start_time

    def elapsed_cuda(self) -> float | None:
        """GPU time in seconds (None if CUDA events disabled)."""
        if self.use_cuda_events:
            return self.start_event.elapsed_time(self.end_event) / 1000.0
        return None

    def elapsed_wall(self) -> float:
        """Wall-clock time in seconds."""
        return self._wall_elapsed


def compute_statistics(latencies: list[float]) -> dict[str, float]:
    """Compute comprehensive statistics from latency measurements."""
    return {
        "mean": float(np.mean(latencies)),
        "median": float(np.median(latencies)),
        "std": float(np.std(latencies)),
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies)),
        "p90": float(np.percentile(latencies, 90)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
    }


def _get_oom_results() -> dict[str, Any]:
    """Return minimal dummy results when OOM occurs."""
    return {"oom_error": True}


def get_memory_stats() -> dict[str, float]:
    """Get current GPU memory statistics in GB, aggregated across all visible devices."""
    if not torch.cuda.is_available():
        return {}

    torch.cuda.synchronize()  # Sync all devices

    n_devices = torch.cuda.device_count()
    total_allocated = 0.0
    total_reserved = 0.0
    total_peak = 0.0
    for i in range(n_devices):
        total_allocated += torch.cuda.memory_allocated(i)
        total_reserved += torch.cuda.memory_reserved(i)
        total_peak += torch.cuda.max_memory_allocated(i)

    return {
        "memory_allocated_gb": round(total_allocated / (1024**3), 3),
        "memory_reserved_gb": round(total_reserved / (1024**3), 3),
        "memory_peak_gb": round(total_peak / (1024**3), 3),
        "num_devices": n_devices,
    }


class EfficiencyModelEvaluator(ModelEvaluator):
    """
    Evaluates inference efficiency with separate Prefill and Decode stages.

    Improvements over basic implementation:
    - CUDA Events for accurate GPU timing
    - Statistical analysis with percentiles
    - Memory usage tracking
    - Optional torch.compile support
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        config: BenchmarkConfig = None,
        prefix: str | None = "eff",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config or BenchmarkConfig()
        self.prefix = prefix

    def __call__(
        self,
        model: nn.Module | PreTrainedModel,
        prefix: str | None = "eff",
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert isinstance(model, PreTrainedModel), (
            f"Expected PreTrainedModel, got {type(model).__name__}. If using EntQuantModel, pass entquant_model.model."
        )
        prefix = self.prefix if self.prefix is not None else prefix
        prefix = prefix + "/" if prefix else ""
        results = {"device_info": get_device_info()}

        # Optionally compile the model
        if self.config.use_torch_compile:
            logger.info(f"Compiling model with mode={self.config.compile_mode}")
            model = torch.compile(model, mode=self.config.compile_mode)

        if self.config.use_cpu_offload:
            logger.info(f"Using CPU offload, execution_device={model.device}")
            model = cpu_offload(model, execution_device=model.device)

        with torch.no_grad(), eval_mode(model):
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

            if self.config.eval_prefill:
                try:
                    prefill_result = evaluate_prefill(
                        model=model,
                        tokenizer=self.tokenizer,
                        config=self.config,
                    )
                except torch.cuda.OutOfMemoryError as e:
                    logger.warning(f"Prefill OOM: {e}. Returning dummy results.")
                    torch.cuda.empty_cache()
                    prefill_result = _get_oom_results()
                logger.info(f"Prefill Results:\n{prefill_result}")
                for k, v in prefill_result.items():
                    results[f"{prefix}prefill_{k}"] = v

                clear_cache()

            if self.config.eval_decode:
                try:
                    decode_result = evaluate_decode(
                        model=model,
                        tokenizer=self.tokenizer,
                        config=self.config,
                    )
                except torch.cuda.OutOfMemoryError as e:
                    logger.warning(f"Decode OOM: {e}. Returning dummy results.")
                    torch.cuda.empty_cache()
                    decode_result = _get_oom_results()
                logger.info(f"Decode Results:\n{decode_result}")
                for k, v in decode_result.items():
                    results[f"{prefix}decode_{k}"] = v

                clear_cache()

        return results


def evaluate_prefill(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """
    Evaluates the 'Prefill' phase (compute-bound).
    Forward pass on a large sequence without generation.
    """
    bs = config.prefill_batch_size
    seq_len = config.prefill_sequence_length
    device = model.device if model.device.type != "meta" else "cpu"

    generator = torch.Generator(device=device).manual_seed(config.seed)
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (bs, seq_len), dtype=torch.long, device=device, generator=generator
    )
    attention_mask = torch.ones((bs, seq_len), dtype=torch.long, device=device)

    use_nvtx = config.nvtx_range and torch.cuda.is_available()

    # Warmup
    for it in tqdm(range(config.prefill_num_warmup_steps), desc="Prefill Warmup"):
        if use_nvtx:
            torch.cuda.nvtx.range_push(f"Prefill {it} (Warmup)")
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Sync all devices for multi-GPU
        if use_nvtx:
            torch.cuda.nvtx.range_pop()

    # Measurement
    latencies_cuda = []
    latencies_wall = []
    logger.debug(f"Prefill benchmark: BS={bs}, SeqLen={seq_len}")

    for it in tqdm(range(config.prefill_num_steps), desc="Prefill Eval"):
        if use_nvtx:
            torch.cuda.nvtx.range_push(f"Prefill {it}")

        with CUDATimer(use_cuda_events=config.use_cuda_events) as timer:
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

        if use_nvtx:
            torch.cuda.nvtx.range_pop()

        if timer.elapsed_cuda() is not None:
            latencies_cuda.append(timer.elapsed_cuda())
        latencies_wall.append(timer.elapsed_wall())

    # Results
    stats_wall = compute_statistics(latencies_wall)
    total_tokens = bs * seq_len

    results = {
        "batch_size": bs,
        "sequence_length": seq_len,
        "latency_wall_mean_s": stats_wall["mean"],
        "latency_wall_median_s": stats_wall["median"],
        "latency_wall_std_s": stats_wall["std"],
        "throughput_wall_tokens_per_s": total_tokens / stats_wall["mean"],
    }

    if latencies_cuda:
        stats_cuda = compute_statistics(latencies_cuda)
        results["latency_cuda_mean_s"] = stats_cuda["mean"]
        results["latency_cuda_median_s"] = stats_cuda["median"]
        results["latency_cuda_std_s"] = stats_cuda["std"]
        results["throughput_cuda_tokens_per_s"] = total_tokens / stats_cuda["mean"]

    if config.report_percentiles:
        results["latency_wall_p90_s"] = stats_wall["p90"]
        results["latency_wall_p95_s"] = stats_wall["p95"]
        results["latency_wall_p99_s"] = stats_wall["p99"]
        if latencies_cuda:
            results["latency_cuda_p90_s"] = stats_cuda["p90"]
            results["latency_cuda_p95_s"] = stats_cuda["p95"]
            results["latency_cuda_p99_s"] = stats_cuda["p99"]

    if config.report_memory:
        results.update(get_memory_stats())

    return results


def evaluate_decode(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """
    Evaluates the 'Decode' phase (memory-bound).
    Token generation starting from a context of configurable length to simulate realistic KV-cache.
    """
    bs = config.decode_batch_size
    ctx_len = config.decode_context_length
    num_new = config.decode_num_tokens_to_generate
    device = model.device if model.device.type != "meta" else "cpu"

    # Create context to build realistic KV-cache before measuring decode
    generator = torch.Generator(device=device).manual_seed(config.seed)
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (bs, ctx_len), dtype=torch.long, device=device, generator=generator
    )

    gen_kwargs = {
        "max_new_tokens": num_new,
        "min_new_tokens": num_new,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    use_nvtx = config.nvtx_range and torch.cuda.is_available()

    # Warmup
    for it in tqdm(range(config.decode_num_warmup_steps), desc="Decode Warmup"):
        if use_nvtx:
            torch.cuda.nvtx.range_push(f"Decode {it} (Warmup)")
        _ = model.generate(input_ids, **gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Sync all devices for multi-GPU
        if use_nvtx:
            torch.cuda.nvtx.range_pop()

    # Measurement
    latencies_cuda = []
    latencies_wall = []
    logger.debug(f"Decode benchmark: BS={bs}, NewTokens={num_new}")

    for it in tqdm(range(config.decode_num_steps), desc="Decode Eval"):
        if use_nvtx:
            torch.cuda.nvtx.range_push(f"Decode {it}")

        with CUDATimer(use_cuda_events=config.use_cuda_events) as timer:
            _ = model.generate(input_ids, **gen_kwargs)

        if use_nvtx:
            torch.cuda.nvtx.range_pop()

        if timer.elapsed_cuda() is not None:
            latencies_cuda.append(timer.elapsed_cuda())
        latencies_wall.append(timer.elapsed_wall())

    # Results
    stats_wall = compute_statistics(latencies_wall)
    total_tokens = bs * num_new

    results = {
        "batch_size": bs,
        "context_length": ctx_len,
        "num_tokens_generated": num_new,
        "total_latency_wall_mean_s": stats_wall["mean"],
        "total_latency_wall_median_s": stats_wall["median"],
        "total_latency_wall_std_s": stats_wall["std"],
        "per_token_latency_wall_ms": (stats_wall["mean"] / num_new) * 1000,
        "throughput_wall_tokens_per_s": total_tokens / stats_wall["mean"],
    }

    if latencies_cuda:
        stats_cuda = compute_statistics(latencies_cuda)
        results["total_latency_cuda_mean_s"] = stats_cuda["mean"]
        results["total_latency_cuda_median_s"] = stats_cuda["median"]
        results["total_latency_cuda_std_s"] = stats_cuda["std"]
        results["per_token_latency_cuda_ms"] = (stats_cuda["mean"] / num_new) * 1000
        results["throughput_cuda_tokens_per_s"] = total_tokens / stats_cuda["mean"]

    if config.report_percentiles:
        results["total_latency_wall_p90_s"] = stats_wall["p90"]
        results["total_latency_wall_p95_s"] = stats_wall["p95"]
        results["total_latency_wall_p99_s"] = stats_wall["p99"]
        if latencies_cuda:
            results["total_latency_cuda_p90_s"] = stats_cuda["p90"]
            results["total_latency_cuda_p95_s"] = stats_cuda["p95"]
            results["total_latency_cuda_p99_s"] = stats_cuda["p99"]

    if config.report_memory:
        results.update(get_memory_stats())

    return results
