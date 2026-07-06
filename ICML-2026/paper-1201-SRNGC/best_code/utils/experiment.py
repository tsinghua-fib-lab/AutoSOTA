"""Shared experiment helpers.

These functions keep the runnable scripts small while preserving their
original command-line behavior and random-seed flow.
"""

import os
import warnings

import torch


CUDA_CONTEXT_WARNING = (
    ".*Attempting to run cuBLAS, but there was no current CUDA context.*"
)


def suppress_cuda_context_warning() -> None:
    """Hide a benign CUDA warning emitted before the first CUDA context exists."""

    warnings.filterwarnings("ignore", message=CUDA_CONTEXT_WARNING)


def select_device(exec_idx: int) -> str:
    """Select the GPU assigned to a 1-based worker index, or CPU if unavailable."""

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        return f"cuda:{(exec_idx - 1) % n_gpu}"
    return "cpu"


def importance_type_for_penalty(penalty_type: str) -> str:
    """Map the training penalty to the corresponding importance estimator."""

    if penalty_type.startswith("Fast_Shap") or penalty_type == "Shapley":
        return "Shapley"
    if penalty_type in ["Jacob_F", "Jacob_L1"]:
        return "Jacobian"
    if penalty_type == "Layer_Weight":
        return "Layer_Weight"
    raise ValueError(f"Unknown penalty_type: {penalty_type}")


def ignore_diagonal_for_dataset(dataset: str) -> bool:
    """DREAM labels exclude self-causality during evaluation."""

    return dataset in ["DREAM3", "DREAM4"]


def batch_size_for_dataset(dataset: str) -> int:
    """Use minibatches for CausalTime and full-batch training otherwise."""

    return 512 if dataset == "CausalTime" else -1


def append_results_csv(results, save_path: str) -> None:
    """Append a grid-search result frame to its aggregate CSV file."""

    if results is None or len(results) == 0:
        return

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    write_header = (not os.path.exists(save_path)) or (os.path.getsize(save_path) == 0)
    results.to_csv(save_path, mode="a", index=False, header=write_header)
