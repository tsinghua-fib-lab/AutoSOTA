"""Lightweight notebook orchestration helpers for circuit_discovery demos."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Iterable

from .circuit import Circuit, overlap_stats
from .metrics import evaluate_good_bad_accuracy
from .models import load_circuit_model
from .utils import (
    DEVICE,
    fixed_order_dataloader,
    load_hyperparams_file,
    load_ioi_name_swap_datasets,
    load_ioi_resampled_train_datasets,
    load_task_dataset,
    plain_dataloader,
    top_k_values,
)
from .visualization import load_circuit


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIGS_PATH = PACKAGE_ROOT / "configs.yaml"
DEMO_CIRCUIT_ROOTS = (
    PROJECT_ROOT / "circuits_discovered" / "discogp_circuits",
    PROJECT_ROOT / "circuits_discovered" / "acdc_circuits",
    PROJECT_ROOT / "circuits_discovered" / "eap_circuits",
    PROJECT_ROOT / "circuits_discovered" / "edge_pruning_circuits",
)


def project_relative_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = (PROJECT_ROOT / value).resolve()
    try:
        return value.relative_to(PROJECT_ROOT)
    except ValueError:
        return value


def script_path(script_name: str) -> Path:
    configs = load_configs()
    configured_root = configs.get("paths", {}).get("experiment_root")
    if configured_root is None:
        raise KeyError("configs.yaml must define paths.experiment_root.")
    path = PROJECT_ROOT / configured_root / script_name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def command_for(script_name: str, **kwargs: Any) -> str:
    parts = ["python", str(project_relative_path(script_path(script_name)))]
    for key, value in kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parts.append(flag if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, (list, tuple)):
            parts.append(flag)
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.extend([flag, str(value)])
    return " ".join(shlex.quote(part) for part in parts)


def _script_cli_kwargs(script_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Drop notebook-only hyperparameters before building parked-script commands."""
    ignored_by_script = {
        "run_discogp_seed_overlap_iou.py": {
            "random_mode",
            "gs_temp_edge",
        },
        "run_edge_pruning_kl_seed_eval.py": {
            "objective",
            "cache_teacher_logits",
            "teacher_cache_dtype",
        },
        "run_edge_pruning_seed_iou.py": {
            "objective",
            "cache_teacher_logits",
            "teacher_cache_dtype",
        },
    }
    ignored = ignored_by_script.get(Path(script_name).name, set())
    return {key: value for key, value in params.items() if key not in ignored}


def load_configs(
    path: str | Path = CONFIGS_PATH,
) -> dict[str, Any]:
    """Load notebook command hyperparameters and artifact paths."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return load_hyperparams_file(config_path)


def notebook_config(
    notebook_name: str,
    *,
    path: str | Path = CONFIGS_PATH,
) -> dict[str, Any]:
    data = load_configs(path)
    try:
        return data["notebooks"][notebook_name]
    except KeyError as exc:
        raise KeyError(notebook_name) from exc


def command_for_notebook(
    notebook_name: str,
    *,
    path: str | Path = CONFIGS_PATH,
    parameter_key: str | None = None,
) -> str:
    config = notebook_config(notebook_name, path=path)
    script = config.get("script")
    params = config.get("hyperparams") or config.get("parameters") or {}
    if parameter_key is not None:
        scripts = config.get("scripts") or {}
        script = scripts.get(parameter_key, script)
        common = params.get("common") or {}
        keyed = params.get(parameter_key) or {}
        params = {**common, **keyed}
    if not script:
        raise KeyError(f"No script configured for {notebook_name!r}")
    return command_for(str(script), **_script_cli_kwargs(str(script), params))


def load_artifact_configs(
    *,
    algorithms: Iterable[str] | None = None,
    path: str | Path = CONFIGS_PATH,
) -> dict[str, Any]:
    configs = load_configs(path)
    artifacts = dict(configs.get("artifacts") or {})
    if algorithms is None:
        return artifacts
    wanted = set(algorithms)
    return {key: value for key, value in artifacts.items() if key in wanted}


def get_compute_device() -> str:
    """Return the device selected by ``utils.py``."""
    return str(DEVICE)


def load_model(model_name: str, *, device: str = DEVICE):
    """Load a circuit-compatible model for notebook experiments."""
    return load_circuit_model(model_name, device=device)


def load_task_dataset_from_config(params: dict[str, Any]):
    """Load the standard train/test dataloaders described by a config block."""
    return load_task_dataset(
        params["task"],
        batch_size=int(params["batch_size"]),
        train_size=int(params["train_size"]),
        test_size=int(params["test_size"]),
        random_seed=int(params.get("data_seed", 42)),
    )


def load_ioi_conditions_from_config(params: dict[str, Any]):
    """Load normal and name-swapped IOI splits using the earlier EAP transform."""
    return load_ioi_name_swap_datasets(
        task_name=str(params.get("task", "ioi")),
        train_size=int(params["train_size"]),
        test_size=int(params["test_size"]),
        random_seed=int(params.get("data_seed", 42)),
        tokenizer_name=str(params.get("tokenizer_name", "gpt2")),
        device=DEVICE,
    )


def load_eap_resampled_conditions_from_config(params: dict[str, Any]):
    """Load the normal-vs-resampled IOI splits used by the current EAP jobs."""
    return load_ioi_resampled_train_datasets(
        task_name=str(params.get("task", "ioi")),
        train_size=int(params["train_size"]),
        test_size=int(params["test_size"]),
        random_seed=int(params.get("data_seed", 42)),
        train_order_seed=int(params.get("train_order_seed", 42)),
        train_order_seed_b=int(params.get("train_order_seed_b", 43)),
        tokenizer_name=str(params.get("tokenizer_name", "gpt2")),
        device=DEVICE,
    )


def train_loader_from_config(dataset, params: dict[str, Any], *, seed: int | None = None):
    """Build the deterministic shuffled train loader used by the job scripts."""
    return fixed_order_dataloader(
        dataset,
        batch_size=int(params["batch_size"]),
        seed=int(params.get("train_order_seed", 42) if seed is None else seed),
    )


def eval_loader_from_config(dataset, params: dict[str, Any]):
    """Build the non-shuffled eval loader used by the job scripts."""
    return plain_dataloader(dataset, batch_size=int(params["batch_size"]), shuffle=False)


def top_k_values_from_config(params: dict[str, Any]) -> list[int]:
    """Return the configured inclusive descending EAP top-k sweep."""
    return top_k_values(
        top_k_max=int(params["top_k_max"]),
        top_k_min=int(params["top_k_min"]),
        top_k_step=int(params["top_k_step"]),
    )


def evaluate_circuit(model, dataloader, circuit: Circuit) -> dict[str, Any]:
    """Evaluate a finalized circuit with the repo's two-label IOI metric."""
    return evaluate_good_bad_accuracy(model=model, dataloader=dataloader, circuit=circuit)


def load_circuit_map(path_map: dict[str, str | Path]) -> dict[str, Circuit]:
    """Load all existing circuits from a label -> path mapping."""
    circuits: dict[str, Circuit] = {}
    for label, rel_path in path_map.items():
        path = Path(rel_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            circuits[label] = load_circuit(path)
    return circuits


def evaluation_rows(model, dataloader, circuits: dict[str, Circuit]) -> list[dict[str, Any]]:
    """Evaluate a mapping of finalized circuits into notebook-friendly rows."""
    rows: list[dict[str, Any]] = []
    for label, circuit in circuits.items():
        evaluation = evaluate_circuit(model, dataloader, circuit)
        rows.append(
            {
                "label": label,
                "acc": evaluation.get("acc"),
                "edge_density": evaluation.get("edge_density"),
                "node_density": evaluation.get("node_density"),
                "num_kept_edges": evaluation.get("num_kept_edges"),
                "num_kept_nodes": evaluation.get("num_kept_nodes"),
            }
        )
    return rows


def pairwise_iou_rows(circuits: dict[str, Circuit]) -> list[dict[str, Any]]:
    """Compute pairwise node/edge IoUs for loaded finalized circuits."""
    labels = list(circuits)
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            stats = overlap_stats(circuits[left], circuits[right])
            rows.append(
                {
                    "pair": f"{left}__{right}",
                    "edge_iou": stats.get("edge_jaccard"),
                    "node_iou": stats.get("node_jaccard"),
                    "edge_overlap_over_left": stats.get("edge_overlap_over_a"),
                    "edge_overlap_over_right": stats.get("edge_overlap_over_b"),
                }
            )
    return rows


__all__ = [
    "CONFIGS_PATH",
    "DEMO_CIRCUIT_ROOTS",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "command_for",
    "command_for_notebook",
    "eval_loader_from_config",
    "evaluate_circuit",
    "evaluation_rows",
    "get_compute_device",
    "load_artifact_configs",
    "load_configs",
    "load_circuit_map",
    "load_eap_resampled_conditions_from_config",
    "load_ioi_conditions_from_config",
    "load_model",
    "load_task_dataset_from_config",
    "notebook_config",
    "pairwise_iou_rows",
    "project_relative_path",
    "script_path",
    "top_k_values_from_config",
    "train_loader_from_config",
]
