from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import replace
import dataclasses

import torch

import globals as g
from bench.e2e.tasks.gram_approx import run_task as run_gram_approx
from bench.e2e.tasks.ose_error import run_task as run_ose_error
from bench.e2e.tasks.ridge_regression import run_task as run_ridge
from bench.e2e.tasks.sketch_and_solve_ls import run_task as run_sketch_and_solve
from data.synthetic import SyntheticDatasetConfig, make_synthetic_matrix
from data.suitesparse.load import SuiteSparseDatasetConfig, load_suitesparse_matrix
from data.llm import (
    LlmWeightsDatasetConfig,
    LlmGradientsDatasetConfig,
    load_llm_weights_matrix,
    load_llm_gradients_matrix,
)
from sketches.sketch_utils import prepare_sketch_input
from torch_utils import manual_seed


TASK_RUNNERS = {
    g.TASK_SKETCH_SOLVE_LS: run_sketch_and_solve,
    g.TASK_RIDGE_REGRESSION: run_ridge,
    g.TASK_GRAM_APPROX: run_gram_approx,
    g.TASK_OSE_ERROR: run_ose_error,
}
TASKS_NO_RHS = {
    g.TASK_GRAM_APPROX,
    g.TASK_OSE_ERROR,
}


def load_dataset(cfg, device):
    """Load a dataset based on its config type."""
    if isinstance(cfg, SyntheticDatasetConfig):
        return make_synthetic_matrix(cfg, device)
    if isinstance(cfg, SuiteSparseDatasetConfig):
        return load_suitesparse_matrix(cfg, device)
    if isinstance(cfg, LlmWeightsDatasetConfig):
        return load_llm_weights_matrix(cfg, device)
    if isinstance(cfg, LlmGradientsDatasetConfig):
        return load_llm_gradients_matrix(cfg, device)
    raise ValueError(f"Unknown dataset config type: {type(cfg)}")


def dataset_id(metadata):
    """Return a stable dataset id string from metadata."""
    if metadata["dataset"] == g.DATASET_SYNTHETIC:
        return f"{metadata['dataset']}:{metadata['name']}"
    if metadata["dataset"] == g.DATASET_SUITESPARSE:
        return f"{metadata['dataset']}:{metadata['group']}/{metadata['name']}"
    if metadata["dataset"] == g.DATASET_LLM:
        return f"{metadata['dataset']}:{metadata['group']}/{metadata['name']}"
    return metadata["dataset"]


def make_rhs(A, seed, noise):
    """Construct a right-hand side vector b for least squares tasks."""
    manual_seed(seed)
    n = A.shape[1]
    x_true = torch.randn(n, device=A.device, dtype=A.dtype)
    b = A @ x_true
    if noise > 0:
        b = b + noise * torch.randn_like(b)
    return b


def method_config_metadata(method_cfg):
    """Return optional method config fields for result rows."""
    fields = {}
    if hasattr(method_cfg, "s"):
        fields["s"] = method_cfg.s
    if hasattr(method_cfg, "kappa"):
        fields["kappa"] = method_cfg.kappa
    if hasattr(method_cfg, "use_csr"):
        fields["use_csr"] = method_cfg.use_csr
    if hasattr(method_cfg, "return_contiguous"):
        fields["return_contiguous"] = method_cfg.return_contiguous
    if hasattr(method_cfg, "block_rows"):
        fields["block_rows"] = method_cfg.block_rows
    if hasattr(method_cfg, "block_cols"):
        fields["block_cols"] = method_cfg.block_cols
    if hasattr(method_cfg, "block_size"):
        fields["block_size"] = method_cfg.block_size
    if hasattr(method_cfg, "tc"):
        fields["tc"] = method_cfg.tc
    if hasattr(method_cfg, "tr"):
        fields["tr"] = method_cfg.tr
    if hasattr(method_cfg, "br"):
        fields["br"] = method_cfg.br
    if hasattr(method_cfg, "d_block"):
        fields["d_block"] = method_cfg.d_block
    if hasattr(method_cfg, "s_block"):
        fields["s_block"] = method_cfg.s_block
    if hasattr(method_cfg, "use_fused"):
        fields["use_fused"] = method_cfg.use_fused
    if hasattr(method_cfg, "scale"):
        fields["scale"] = method_cfg.scale
    return fields


def warmup_key(dataset_id_str, A, method_cfg):
    """Return a key that groups warmups across repeats/seeds."""
    key = [
        dataset_id_str,
        tuple(A.shape),
        method_cfg.method,
        int(method_cfg.k),
        method_cfg.dtype,
    ]
    if hasattr(method_cfg, "s"):
        key.append(("s", int(method_cfg.s)))
    if hasattr(method_cfg, "use_csr"):
        key.append(("use_csr", bool(method_cfg.use_csr)))
    if hasattr(method_cfg, "return_contiguous"):
        key.append(("return_contiguous", bool(method_cfg.return_contiguous)))
    if hasattr(method_cfg, "block_rows"):
        key.append(("block_rows", int(method_cfg.block_rows)))
    if hasattr(method_cfg, "block_cols"):
        key.append(("block_cols", int(method_cfg.block_cols)))
    if hasattr(method_cfg, "block_size"):
        key.append(("block_size", int(method_cfg.block_size)))
    if hasattr(method_cfg, "kappa"):
        key.append(("kappa", int(method_cfg.kappa)))
    if hasattr(method_cfg, "tc"):
        key.append(("tc", int(method_cfg.tc)))
    if hasattr(method_cfg, "tr"):
        key.append(("tr", int(method_cfg.tr)))
    if hasattr(method_cfg, "br"):
        key.append(("br", int(method_cfg.br)))
    if hasattr(method_cfg, "d_block"):
        key.append(("d_block", int(method_cfg.d_block)))
    if hasattr(method_cfg, "s_block"):
        key.append(("s_block", int(method_cfg.s_block)))
    if hasattr(method_cfg, "use_fused"):
        key.append(("use_fused", bool(method_cfg.use_fused)))
    if hasattr(method_cfg, "scale"):
        key.append(("scale", float(method_cfg.scale)))
    return tuple(key)


def warmup_sketch(A, sketch_fn, method_cfg, reps=2, seed_offsets=(0, 1, 2)):
    """Warm up sketch kernels across multiple seeds."""
    A_work, _ = prepare_sketch_input(A, method_cfg)
    for offset in seed_offsets:
        cfg = method_cfg
        if hasattr(method_cfg, "seed"):
            cfg = dataclasses.replace(method_cfg, seed=int(method_cfg.seed) + int(offset))
        for _ in range(reps):
            _ = sketch_fn(A_work, cfg)


def format_duration(seconds):
    """Format elapsed seconds into a compact string."""
    total = int(round(seconds))
    if total < 0:
        total = 0
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def run_task(runner, A, b, sketch_fn, method_cfg, task_cfg):
    """Dispatch tasks with or without an RHS vector."""
    if task_cfg.task in TASKS_NO_RHS:
        return runner(A, sketch_fn, method_cfg, task_cfg)
    return runner(A, b, sketch_fn, method_cfg, task_cfg)


def oom_metrics(task_cfg):
    """Return NaN-filled metrics for an OOM result row."""
    nan = float("nan")
    metrics = {
        "task": task_cfg.task,
        "status": "oom",
        "error": "cuda_oom",
        "sketch_time_ms": nan,
        "sketch_time_cuda_ms": nan,
    }
    if task_cfg.task == g.TASK_GRAM_APPROX:
        metrics.update(
            {
                "gram_fro_error": nan,
                "gram_fro_rel": nan,
                "gram_spec_error": nan,
                "gram_spec_rel": nan,
            }
        )
        return metrics
    if task_cfg.task == g.TASK_OSE_ERROR:
        metrics.update(
            {
                "ose_variant": "oom",
                "ose_probes": int(getattr(task_cfg, "probes", 0)),
                "ose_r": int(getattr(task_cfg, "r", 0)),
                "ose_fro_err": nan,
                "ose_spec_err": nan,
                "ose_max_sv_dev": nan,
            }
        )
        return metrics
    if task_cfg.task in {g.TASK_RIDGE_REGRESSION, g.TASK_SKETCH_SOLVE_LS}:
        metrics.update(
            {
                "residual": nan,
                "residual_true": nan,
                "residual_proj": nan,
            }
        )
        return metrics
    return metrics
