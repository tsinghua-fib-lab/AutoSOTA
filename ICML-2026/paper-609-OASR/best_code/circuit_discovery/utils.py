# utils.py
# runtime helpers, reproducibility, and dataset loading.

from __future__ import annotations

import os

# must be set before importing torch / tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import random
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import torch
import yaml
from datasets import Dataset as HFDataset, load_from_disk
from torch.utils.data import DataLoader

from .circuit import Circuit, node_key

# --------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATASET_FOLDER_PATH = PACKAGE_ROOT / "datasets"
CONFIGS_PATH = PACKAGE_ROOT / "configs.yaml"
_YAML_SUFFIXES = {".yaml", ".yml"}

# --------------------------------------------------------------------------------------
# YAML hyperparameter loading
# --------------------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_yaml(path: str | Path, data: Mapping[str, Any]) -> Path:
    out = Path(path).expanduser()
    if out.suffix.lower() not in _YAML_SUFFIXES:
        raise ValueError(f"YAML output path must end in .yaml or .yml: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        yaml.safe_dump(
            dict(data),
            f,
            sort_keys=False,
            allow_unicode=True,
        )
    return out


def load_hyperparams_file(
    path: str | Path = CONFIGS_PATH,
    *,
    default: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load notebook hyperparameters from YAML.

    Hyperparameter manifests are YAML-only so they can carry inline comments.
    """
    config_path = Path(path).expanduser()
    if config_path.suffix.lower() not in _YAML_SUFFIXES:
        raise ValueError(f"Hyperparameter file must be YAML: {config_path}")

    if not config_path.exists():
        return dict(default or {"notebooks": {}, "experiments": []})

    data = load_yaml(config_path)
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected mapping in {config_path}, got {type(data).__name__}")
    return dict(data)

# --------------------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------------------

def pick_device() -> str:
    if torch.cuda.is_available():
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,nounits,noheader",
                ]
            ).decode().strip().splitlines()

            pairs = [
                (int(i), int(free_mb))
                for i, free_mb in (line.split(",") for line in out)
            ]

            device_idx = max(pairs, key=lambda x: x[1])[0]
            torch.cuda.set_device(device_idx)

            return f"cuda:{device_idx}"

        except Exception:
            return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"

DEVICE = pick_device()

# --------------------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------------------------
# circuit post-processing
# --------------------------------------------------------------------------------------

def has_kept_incoming_from(
    circuit: Circuit,
    dst: node_key,
    live_sources: set[node_key],
) -> bool:
    return any(
        edge.src in live_sources and edge.is_kept()
        for edge in circuit.incoming_edges_of(dst)
    )


def prune_dead_circuit_paths(
    circuit: Circuit,
    *,
    output_key: node_key,
    forward_sources: set[node_key],
    source_dependencies: Mapping[node_key, Iterable[node_key]],
    root_sources: Iterable[node_key] = (),
    device: str | torch.device = DEVICE,
) -> Circuit:
    """
    Return a clone with masks disabled for paths that cannot affect output.

    The caller owns architecture-specific semantics by providing:
        forward_sources: source nodes computable in the forward pass
        source_dependencies: source node -> destination nodes that compute it

    For example, a GPT attention output source depends on its Q/K/V destination
    nodes, while an MLP source depends on the MLP destination node.
    """
    out = circuit.clone()
    live_sources = set(forward_sources)
    dependency_dsts = {
        source: tuple(dsts)
        for source, dsts in source_dependencies.items()
    }

    needed_sources: set[node_key] = set()
    needed_dsts: set[node_key] = {output_key}

    for edge in out.incoming_edges_of(output_key):
        if edge.is_kept() and edge.src in live_sources:
            needed_sources.add(edge.src)

    worklist = list(needed_sources)
    expanded_sources: set[node_key] = set()

    while worklist:
        source = worklist.pop()
        if source in expanded_sources:
            continue
        expanded_sources.add(source)

        for dst in dependency_dsts.get(source, ()):
            needed_dsts.add(dst)

            for edge in out.incoming_edges_of(dst):
                if edge.is_kept() and edge.src in live_sources:
                    if edge.src not in needed_sources:
                        needed_sources.add(edge.src)
                        worklist.append(edge.src)

    false_mask = torch.tensor(False, dtype=torch.bool, device=device)
    true_mask = torch.tensor(True, dtype=torch.bool, device=device)

    for edge in out.all_edges():
        keep = (
            edge.is_kept()
            and edge.src in live_sources
            and edge.dst in needed_dsts
        )
        if not keep:
            edge.edge_mask = false_mask.clone()

    root_source_set = set(root_sources)

    for key, node in out.nodes.items():
        if key in root_source_set:
            live = True
            node.node_mask = true_mask.clone()
        elif node.is_src():
            live = (
                key in live_sources
                and key in needed_sources
            )
        else:
            live = key in needed_dsts

        if not live:
            node.node_mask = false_mask.clone()

    return out

# --------------------------------------------------------------------------------------
# model weight reinitialization
# --------------------------------------------------------------------------------------

DEFAULT_REINIT_SKIP_SUBSTRINGS = ("emb", "edge", "ln", "b_")
DEFAULT_RESIDUAL_WEIGHT_MARKERS = ("W_O", "W_out")

WeightReinitScheme = Literal[
    "scaled_gaussian",
    "residual_scaled_normal",
    "gpt2_official",
    "none",
]
ScaledGaussianStatsScope = Literal["global", "per_matrix"]


class WeightStats(NamedTuple):
    mean: float
    std: float
    min: float
    max: float
    n_params: int


def _resolve_model(target: Any) -> torch.nn.Module:
    if isinstance(target, torch.nn.Module):
        return target

    for attr in ("model", "circuit_gpt"):
        value = getattr(target, attr, None)
        if isinstance(value, torch.nn.Module):
            return value

    raise TypeError(
        "weight reinitialization expected a torch module, or an object with "
        "a .model or .circuit_gpt torch module."
    )


def _selected_named_parameters(
    model: torch.nn.Module,
    *,
    skip_substrings: tuple[str, ...] = DEFAULT_REINIT_SKIP_SUBSTRINGS,
    parameter_filter: Callable[[str, torch.nn.Parameter], bool] | None = None,
) -> list[tuple[str, torch.nn.Parameter]]:
    selected: list[tuple[str, torch.nn.Parameter]] = []

    for name, param in model.named_parameters():
        if not param.is_floating_point():
            continue
        if any(skip in name for skip in skip_substrings):
            continue
        if parameter_filter is not None and not parameter_filter(name, param):
            continue
        selected.append((name, param))

    return selected


def _update_parameter_snapshot(
    model: torch.nn.Module,
    name: str,
    param: torch.nn.Parameter,
) -> None:
    snapshot = getattr(model, "unmasked_params", None)
    if isinstance(snapshot, dict):
        snapshot[name] = param.detach().clone()


def compute_global_weight_stats(
    target: Any,
    *,
    skip_substrings: tuple[str, ...] = DEFAULT_REINIT_SKIP_SUBSTRINGS,
    parameter_filter: Callable[[str, torch.nn.Parameter], bool] | None = None,
) -> WeightStats:
    """
    Compute global stats over reinitializable model weights.

    The default filter mirrors the legacy behavior: embeddings, edge parameters,
    layer norms, and named biases are excluded. The implementation streams over
    parameters instead of concatenating them, so it does not allocate a second
    full-size model vector on GPU.
    """
    model = _resolve_model(target)
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    min_value = float("inf")
    max_value = float("-inf")

    for _, param in _selected_named_parameters(
        model,
        skip_substrings=skip_substrings,
        parameter_filter=parameter_filter,
    ):
        data = param.detach().float()
        count = data.numel()
        if count == 0:
            continue

        total_sum += float(data.sum().item())
        param_norm = float(torch.linalg.vector_norm(data).item())
        total_sq_sum += param_norm * param_norm
        total_count += count
        min_value = min(min_value, float(data.min().item()))
        max_value = max(max_value, float(data.max().item()))

    if total_count == 0:
        return WeightStats(0.0, 0.0, 0.0, 0.0, 0)

    mean = total_sum / total_count
    variance = max(total_sq_sum / total_count - mean * mean, 0.0)

    return WeightStats(
        mean=mean,
        std=math.sqrt(variance),
        min=min_value,
        max=max_value,
        n_params=total_count,
    )


def reinitialize_model_weights_scaled_gaussian(
    target: Any,
    *,
    seed: int | None = None,
    stats_scope: ScaledGaussianStatsScope = "global",
    skip_substrings: tuple[str, ...] = DEFAULT_REINIT_SKIP_SUBSTRINGS,
    parameter_filter: Callable[[str, torch.nn.Parameter], bool] | None = None,
) -> Any:
    """
    Reinitialize selected weights from a clipped Gaussian.

    By default, the Gaussian matches global mean/std/min/max across all selected
    weights. With stats_scope="per_matrix", each parameter tensor uses its own
    mean/std/min/max.
    """
    if seed is None:
        seed = 42

    model = _resolve_model(target)
    selected = _selected_named_parameters(
        model,
        skip_substrings=skip_substrings,
        parameter_filter=parameter_filter,
    )
    if not selected:
        return target

    if stats_scope == "global":
        global_stats = compute_global_weight_stats(
            model,
            skip_substrings=skip_substrings,
            parameter_filter=parameter_filter,
        )
    elif stats_scope != "per_matrix":
        raise ValueError(f"unknown scaled Gaussian stats scope: {stats_scope!r}")

    set_seed(seed)

    with torch.no_grad():
        for name, param in selected:
            if stats_scope == "per_matrix":
                data = param.detach().float()
                std = float(data.std(unbiased=False).item())
                mean = float(data.mean().item())
                min_value = float(data.min().item())
                max_value = float(data.max().item())
            else:
                mean = global_stats.mean
                std = global_stats.std
                min_value = global_stats.min
                max_value = global_stats.max

            sampled = torch.empty_like(param).normal_(
                mean=mean,
                std=std,
            )
            sampled.clamp_(min=min_value, max=max_value)
            param.copy_(sampled)
            _update_parameter_snapshot(model, name, param)

    return target


def reinitialize_model_weights_residual_scaled_normal(
    target: Any,
    *,
    seed: int | None = None,
    base_std: float = 0.02,
    skip_substrings: tuple[str, ...] = DEFAULT_REINIT_SKIP_SUBSTRINGS,
    residual_weight_markers: tuple[str, ...] = DEFAULT_RESIDUAL_WEIGHT_MARKERS,
    parameter_filter: Callable[[str, torch.nn.Parameter], bool] | None = None,
) -> Any:
    """
    Reinitialize selected weights with a GPT-style residual scaling rule.

    This stays model-class agnostic: it only uses named parameters and, if
    available, model.cfg.n_layers to scale residual-output weights.
    """
    if seed is None:
        seed = 42

    model = _resolve_model(target)
    cfg = getattr(model, "cfg", None)
    n_layers = max(int(getattr(cfg, "n_layers", 1)), 1)

    set_seed(seed)

    with torch.no_grad():
        for name, param in _selected_named_parameters(
            model,
            skip_substrings=skip_substrings,
            parameter_filter=parameter_filter,
        ):
            std = base_std
            if any(marker in name for marker in residual_weight_markers):
                std = base_std / math.sqrt(n_layers)

            torch.nn.init.normal_(param, mean=0.0, std=std)
            _update_parameter_snapshot(model, name, param)

    return target


def reinitialize_model_weights(
    target: Any,
    *,
    scheme: WeightReinitScheme = "scaled_gaussian",
    seed: int | None = None,
    scaled_gaussian_stats_scope: ScaledGaussianStatsScope = "global",
) -> Any:
    """
    Reinitialize model weights without depending on a concrete model class.
    """
    if scheme == "none":
        return target
    if scheme == "scaled_gaussian":
        return reinitialize_model_weights_scaled_gaussian(
            target,
            seed=seed,
            stats_scope=scaled_gaussian_stats_scope,
        )
    if scheme in {"residual_scaled_normal", "gpt2_official"}:
        return reinitialize_model_weights_residual_scaled_normal(target, seed=seed)

    raise ValueError(f"unknown weight reinitialization scheme: {scheme!r}")

# --------------------------------------------------------------------------------------
# dataset loading
# --------------------------------------------------------------------------------------

def load_task_dataset(
    task_name: str,
    *,
    batch_size: int,
    train_size: int = 1000,
    test_size: int = 5000,
    random_seed: int = 42,
) -> SimpleNamespace:
    """
    load a saved HuggingFace dataset and create train/test dataloaders.
    """
    set_seed(random_seed)

    dataset_path = DATASET_FOLDER_PATH / f"{task_name}_dataset"

    ds = cast(HFDataset, load_from_disk(str(dataset_path)))
    ds = ds.with_format("torch", output_all_columns=True, device=DEVICE)

    ds_split = ds.train_test_split(
        train_size=train_size,
        test_size=test_size,
        seed=random_seed,
    )

    train_loader = DataLoader(
        cast(Any, ds_split["train"]),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        cast(Any, ds_split["test"]),
        batch_size=batch_size,
        shuffle=False,
    )

    return SimpleNamespace(
        train=train_loader,
        test=test_loader,
    )


def fixed_order_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
) -> DataLoader:
    """Build the deterministic train-order loader used by the experiment scripts."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def plain_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    """Small wrapper used by notebooks when evaluating prepared HF datasets."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def top_k_values(
    *,
    top_k_max: int,
    top_k_min: int,
    top_k_step: int,
) -> list[int]:
    """Return the inclusive descending top-k sweep used by EAP demos."""
    if top_k_step <= 0:
        raise ValueError(f"top_k_step must be positive, got {top_k_step}.")
    if top_k_max < top_k_min:
        raise ValueError(
            f"top_k_max must be >= top_k_min, got {top_k_max} < {top_k_min}."
        )
    return list(range(top_k_max, top_k_min - 1, -top_k_step))


def parse_ioi_targets(targets: str) -> tuple[str, str]:
    """Parse the IOI ``'good, bad'`` target-name string."""
    pieces = [piece.strip() for piece in targets.split(",")]
    if len(pieces) != 2 or not pieces[0] or not pieces[1]:
        raise ValueError(f"expected 'good, bad' targets string, got {targets!r}.")
    return pieces[0], pieces[1]


def swap_ioi_names_in_prompt(prompt: str, good_name: str, bad_name: str) -> str:
    """
    Swap the good/bad IOI names in a prompt without touching substrings.

    This mirrors the remote EAP job script. Temporary markers avoid double
    replacement when the two names are swapped in the same string.
    """
    good_marker = "__CD_GOOD_NAME__"
    bad_marker = "__CD_BAD_NAME__"
    if good_marker in prompt or bad_marker in prompt:
        raise ValueError("prompt unexpectedly contains internal swap markers.")

    good_pattern = re.compile(rf"(?<![A-Za-z]){re.escape(good_name)}(?![A-Za-z])")
    bad_pattern = re.compile(rf"(?<![A-Za-z]){re.escape(bad_name)}(?![A-Za-z])")
    marked = good_pattern.sub(good_marker, prompt)
    marked = bad_pattern.sub(bad_marker, marked)
    return marked.replace(good_marker, bad_name).replace(bad_marker, good_name)


def make_ioi_name_swap_transform(tokenizer: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Build the IOI name-swap transform used by EAP name-sensitivity experiments.

    The prompt is re-tokenized after swapping names. The good/bad token ids and
    target text are swapped too, so evaluation remains the same functional task.
    """
    pad_token_id = tokenizer.eos_token_id

    def transform(row: dict[str, Any]) -> dict[str, Any]:
        good_name, bad_name = parse_ioi_targets(row["targets"])
        swapped_prompt = swap_ioi_names_in_prompt(row["prompt"], good_name, bad_name)
        input_ids = tokenizer.encode(swapped_prompt, add_special_tokens=False)
        original_length = len(row["input_ids"])
        if len(input_ids) > original_length:
            raise ValueError(
                "swapped prompt is longer than the original padded length: "
                f"{len(input_ids)} > {original_length}. prompt={swapped_prompt!r}"
            )

        padded = input_ids + [pad_token_id] * (original_length - len(input_ids))
        return {
            "prompt": swapped_prompt,
            "targets": f"{bad_name}, {good_name}",
            "input_ids": padded,
            "seq_lens": len(input_ids),
            "target good": row["target bad"],
            "target bad": row["target good"],
        }

    return transform


def substitute_ioi_names(prompt: str, mapping: Mapping[str, str]) -> str:
    """
    Single-pass whole-word substitution for IOI names.

    This supports resampling both names at once without accidentally replacing a
    freshly inserted name during a second pass.
    """
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(key) for key in keys) + r")\b")
    return pattern.sub(lambda match: str(mapping[match.group(0)]), prompt)


def ioi_name_token_len(tokenizer: Any, name: str) -> int:
    """Token length for a mid-prompt name, matching the EAP job script."""
    return len(tokenizer.encode(" " + name, add_special_tokens=False))


def build_ioi_name_pool(dataset: HFDataset, tokenizer: Any) -> dict[int, list[str]]:
    """Bucket deduped IOI names by leading-space token length."""
    names = sorted(
        {
            name
            for targets in dataset["targets"]
            for name in parse_ioi_targets(targets)
        }
    )
    by_len: dict[int, list[str]] = {}
    for name in names:
        by_len.setdefault(ioi_name_token_len(tokenizer, name), []).append(name)
    return by_len


def make_ioi_name_resample_transform(
    tokenizer: Any,
    names_by_len: Mapping[int, list[str]],
    *,
    base_seed: int,
    max_tries: int = 200,
) -> Callable[[dict[str, Any], int], dict[str, Any]]:
    """
    Build the training-wise IOI name-resampling transform used by EAP jobs.

    Both names are replaced with a fresh distinct pair from the dataset name
    pool. Replacement names preserve each original name's leading-space token
    length, and the re-tokenized prompt must keep the original sequence length.
    """

    def transform(row: dict[str, Any], idx: int) -> dict[str, Any]:
        good_name, bad_name = parse_ioi_targets(row["targets"])
        orig_len = int(row["seq_lens"])
        good_pool = names_by_len[ioi_name_token_len(tokenizer, good_name)]
        bad_pool = names_by_len[ioi_name_token_len(tokenizer, bad_name)]
        rng = random.Random(f"{base_seed}:{idx}")

        for _ in range(max_tries):
            new_good = rng.choice(good_pool)
            new_bad = rng.choice(bad_pool)
            if new_good == new_bad or (new_good, new_bad) == (good_name, bad_name):
                continue
            new_prompt = substitute_ioi_names(
                row["prompt"],
                {good_name: new_good, bad_name: new_bad},
            )
            input_ids = tokenizer.encode(new_prompt, add_special_tokens=False)
            if len(input_ids) != orig_len:
                continue
            return {
                "prompt": new_prompt,
                "targets": f"{new_good}, {new_bad}",
                "input_ids": input_ids,
                "seq_lens": len(input_ids),
                "target good": tokenizer.encode(
                    " " + new_good,
                    add_special_tokens=False,
                )[0],
                "target bad": tokenizer.encode(
                    " " + new_bad,
                    add_special_tokens=False,
                )[0],
            }

        raise ValueError(
            f"could not resample length-preserving names for row {idx} "
            f"within {max_tries} tries."
        )

    return transform


def make_pad_transform(
    pad_token_id: int,
    width: int,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Pad tokenized prompts to a shared width after name resampling."""

    def transform(row: dict[str, Any]) -> dict[str, Any]:
        ids = list(row["input_ids"])
        if len(ids) > width:
            raise ValueError(
                f"sequence longer than target pad width: {len(ids)} > {width}."
            )
        return {"input_ids": ids + [pad_token_id] * (width - len(ids))}

    return transform


def max_input_id_length(dataset: HFDataset) -> int:
    """Return the longest unpadded ``input_ids`` length in a HF dataset."""
    return max(len(ids) for ids in dataset["input_ids"])


def load_tokenizer_for_notebook(tokenizer_name: str) -> Any:
    """
    Load a tokenizer for notebook dataset transforms.

    This may download the tokenizer on first use, matching the notebook API's
    reproducibility path on fresh machines.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name)


def load_ioi_name_swap_datasets(
    *,
    task_name: str = "ioi",
    train_size: int,
    test_size: int,
    random_seed: int = 42,
    tokenizer_name: str = "gpt2",
    device: str | torch.device = DEVICE,
) -> dict[str, SimpleNamespace]:
    """
    Load normal and directly name-swapped IOI splits.

    This is kept as a reusable transform helper. The current EAP demo notebook
    uses ``load_ioi_resampled_train_datasets`` because that matches the
    completed job-script artifacts.
    """
    dataset_path = DATASET_FOLDER_PATH / f"{task_name}_dataset"
    ds = cast(HFDataset, load_from_disk(str(dataset_path)))
    split = ds.train_test_split(
        train_size=train_size,
        test_size=test_size,
        seed=random_seed,
    )

    tokenizer = load_tokenizer_for_notebook(tokenizer_name)
    swap_transform = make_ioi_name_swap_transform(tokenizer)

    normal_train = cast(HFDataset, split["train"])
    normal_test = cast(HFDataset, split["test"])
    swapped_train = normal_train.map(swap_transform, desc="swap train IOI names")
    swapped_test = normal_test.map(swap_transform, desc="swap test IOI names")

    return {
        "normal": SimpleNamespace(
            train=normal_train.with_format("torch", output_all_columns=True, device=device),
            test=normal_test.with_format("torch", output_all_columns=True, device=device),
        ),
        "swapped_names": SimpleNamespace(
            train=swapped_train.with_format("torch", output_all_columns=True, device=device),
            test=swapped_test.with_format("torch", output_all_columns=True, device=device),
        ),
    }


def load_ioi_resampled_train_datasets(
    *,
    task_name: str = "ioi",
    train_size: int,
    test_size: int,
    random_seed: int = 42,
    train_order_seed: int = 42,
    train_order_seed_b: int = 43,
    tokenizer_name: str = "gpt2",
    device: str | torch.device = DEVICE,
) -> dict[str, SimpleNamespace]:
    """
    Load EAP's normal-vs-resampled IOI condition datasets.

    This matches the remote EAP demo job: condition A trains on the normal split
    with ``train_order_seed``; condition B trains on a training-wise name-
    resampled split with ``train_order_seed_b``; both evaluate on the same
    untransformed test split.
    """
    dataset_path = DATASET_FOLDER_PATH / f"{task_name}_dataset"
    ds = cast(HFDataset, load_from_disk(str(dataset_path)))
    split = ds.train_test_split(
        train_size=train_size,
        test_size=test_size,
        seed=random_seed,
    )

    tokenizer = load_tokenizer_for_notebook(tokenizer_name)
    pad_token_id = tokenizer.eos_token_id
    names_by_len = build_ioi_name_pool(ds, tokenizer)
    resample_transform = make_ioi_name_resample_transform(
        tokenizer,
        names_by_len,
        base_seed=random_seed,
    )

    normal_train = cast(HFDataset, split["train"]).with_format(None)
    normal_test = cast(HFDataset, split["test"]).with_format(None)
    resampled_train = normal_train.map(
        resample_transform,
        with_indices=True,
        desc="resample train IOI names",
    )

    width = max(
        max_input_id_length(normal_train),
        max_input_id_length(resampled_train),
        max_input_id_length(normal_test),
    )
    normal_train = normal_train.map(
        make_pad_transform(pad_token_id, width),
        desc="pad normal train",
    )
    resampled_train = resampled_train.map(
        make_pad_transform(pad_token_id, width),
        desc="pad resampled train",
    )
    normal_test = normal_test.map(
        make_pad_transform(pad_token_id, width),
        desc="pad normal test",
    )

    test = normal_test.with_format("torch", output_all_columns=True, device=device)
    normal_label = f"normal_order_{train_order_seed}"
    resampled_label = f"resampled_order_{train_order_seed_b}"
    return {
        normal_label: SimpleNamespace(
            train=normal_train.with_format("torch", output_all_columns=True, device=device),
            test=test,
            train_order_seed=train_order_seed,
        ),
        resampled_label: SimpleNamespace(
            train=resampled_train.with_format("torch", output_all_columns=True, device=device),
            test=test,
            train_order_seed=train_order_seed_b,
        ),
    }
