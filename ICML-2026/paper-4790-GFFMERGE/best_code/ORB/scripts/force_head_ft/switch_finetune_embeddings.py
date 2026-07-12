"""
Fine-tune the forces head of a checkpoint using embedding swapping per sample.

Evaluation is handled separately by scripts/kaggle/evaluate_switch_embeddings.py.

This script uses ground-truth forces from the dataset (no teacher distillation).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Callable, Dict, List, Mapping, Sequence, Optional, Tuple

import ase.db
import torch
from orb_models.dataset.ase_sqlite_dataset import AseSqliteDataset
from orb_models.forcefield import base as ff_base
from orb_models.forcefield import property_definitions
from orb_models.forcefield import pretrained
from torch.utils.data import DataLoader, Subset

try:
    from ..train_orb import (
        deterministic_train_val_test_split,
        resolve_data_unit_scale,
        scale_batch_targets,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from train_orb import (  # type: ignore
        deterministic_train_val_test_split,
        resolve_data_unit_scale,
        scale_batch_targets,
    )


def load_config(path: Path) -> Dict:
    return json.loads(path.read_text())


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_model(base_model: str, device: torch.device, precision: str, compile_model: bool) -> torch.nn.Module:
    loader = pretrained.ORB_PRETRAINED_MODELS[base_model]
    model = loader(device=device, precision=precision, compile=compile_model, train=True)
    return model


def load_checkpoint(path: Path) -> Mapping[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, Mapping):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format {path}")


def to_cpu_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: tensor.detach().cpu() for key, tensor in state_dict.items()}


def make_dataset(
    dataset_name: str,
    dataset_path: Path,
    model: torch.nn.Module,
    dtype: torch.dtype,
) -> AseSqliteDataset:
    target_config = property_definitions.PropertyConfig(
        node_names=["forces"],
        graph_names=[],
    )
    return AseSqliteDataset(
        name=dataset_name,
        path=dataset_path,
        system_config=model.system_config,
        target_config=target_config,
        dtype=dtype,
    )


def extract_source_labels(db_path: Path) -> Sequence[str]:
    db = ase.db.connect(str(db_path), serial=True, type="db")
    labels: list[str] = []
    for idx in range(len(db)):
        row = db.get(idx + 1)
        labels.append(row.data.get("source_dataset", row.data.get("dataset", "")))
    return labels


def collect_layernorm_names(model: torch.nn.Module) -> Sequence[str]:
    names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            names.append(name)
    return names


def extract_ln_params(state_dict: Mapping[str, torch.Tensor], ln_names: Sequence[str]) -> Dict[str, torch.Tensor]:
    params: Dict[str, torch.Tensor] = {}
    for name in ln_names:
        weight_key = f"{name}.weight"
        bias_key = f"{name}.bias"
        if weight_key in state_dict:
            params[weight_key] = state_dict[weight_key].clone()
        if bias_key in state_dict:
            params[bias_key] = state_dict[bias_key].clone()
    return params


class FineTuneLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message)
        if self.handle is not None:
            self.handle.write(message + "\n")
            self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def swap_embeddings_inplace(
    model: torch.nn.Module,
    label: str,
    embeddings: Mapping[str, torch.Tensor],
) -> None:
    key = label.lower()
    if key not in embeddings:
        raise KeyError(f"No embedding provided for source label '{key}'. Available: {sorted(embeddings.keys())}")
    model.model.atom_emb.embeddings.weight.data.copy_(embeddings[key])


def resolve_original_index(dataset: torch.utils.data.Dataset, idx: int) -> int:
    current_dataset = dataset
    current_idx = idx
    while isinstance(current_dataset, Subset):
        current_idx = current_dataset.indices[current_idx]
        current_dataset = current_dataset.dataset
    return current_idx


def label_for_dataset(
    dataset: torch.utils.data.Dataset,
    idx: int,
    source_labels: Sequence[str],
) -> str:
    orig_idx = resolve_original_index(dataset, idx)
    return source_labels[orig_idx].lower()


def select_training_indices(
    dataset: torch.utils.data.Dataset,
    sample_graphs: int,
    seed: int,
) -> List[int]:
    total = len(dataset)
    if total == 0:
        return []
    if sample_graphs <= 0 or sample_graphs >= total:
        return list(range(total))
    rng = random.Random(seed)
    return rng.sample(range(total), sample_graphs)


def group_indices_by_label(
    dataset: torch.utils.data.Dataset,
    source_labels: Sequence[str],
) -> Dict[str, List[int]]:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        label = label_for_dataset(dataset, idx, source_labels)
        buckets[label].append(idx)
    return buckets


def collect_finetune_samples(
    dataset: torch.utils.data.Dataset,
    indices: Sequence[int],
    source_labels: Sequence[str],
    target_scale: float,
) -> tuple[List[ff_base.AtomGraphs], List[str], List[torch.Tensor]]:
    graphs: List[ff_base.AtomGraphs] = []
    labels: List[str] = []
    targets: List[torch.Tensor] = []
    for idx in indices:
        label = label_for_dataset(dataset, idx, source_labels)
        graph = dataset[idx]
        batch = ff_base.batch_graphs([graph])
        preds = batch.node_targets["forces"].detach().cpu().to(torch.float32)
        if target_scale != 1.0:
            preds = preds * target_scale
        graphs.append(graph)
        labels.append(label)
        targets.append(preds)
    return graphs, labels, targets


def freeze_all_but_forces(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if not hasattr(model, "heads") or "forces" not in model.heads:
        raise ValueError("Model does not include a 'forces' head.")
    for param in model.heads["forces"].parameters():
        param.requires_grad = True


def drop_unused_heads(model: torch.nn.Module) -> None:
    """Remove energy/stress heads so force FT runs faster."""
    if not hasattr(model, "heads"):
        return
    for head_name in ("energy", "stress"):
        if head_name in model.heads:
            del model.heads[head_name]
            if hasattr(model, "loss_weights"):
                model.loss_weights.pop(head_name, None)


@dataclass
class CachedHeadInputs:
    entries: List[Tuple[object, ...]]
    tensor_only: bool
    single_tensor: bool
    requires_graph: bool


def move_tensor_to_device(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if tensor.is_floating_point():
        return tensor.to(device=device, dtype=dtype)
    return tensor.to(device=device)


def move_structure_to_device(value: object, device: torch.device, dtype: torch.dtype) -> object:
    if isinstance(value, torch.Tensor):
        return move_tensor_to_device(value.detach(), device, dtype)
    if hasattr(value, "to"):
        try:
            moved = value.to(device=device, dtype=dtype)
        except TypeError:
            moved = value.to(device=device)
        return moved if moved is not None else value
    if isinstance(value, Mapping):
        return {key: move_structure_to_device(val, device, dtype) for key, val in value.items()}
    if isinstance(value, tuple):
        converted = tuple(move_structure_to_device(item, device, dtype) for item in value)
        if hasattr(value, "_fields"):
            try:
                return type(value)(*converted)
            except Exception:
                return converted
        return converted
    if isinstance(value, list):
        return [move_structure_to_device(item, device, dtype) for item in value]
    return value


def describe_inputs(inputs: Tuple[object, ...]) -> str:
    return ", ".join(type(item).__name__ for item in inputs)


def resolve_autocast_dtype(device: torch.device) -> Optional[torch.dtype]:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return None


def head_predict(head: torch.nn.Module, *inputs: object) -> torch.Tensor:
    if len(inputs) == 2 and hasattr(inputs[1], "n_node"):
        predict_fn = getattr(head, "predict", None)
        if callable(predict_fn):
            return predict_fn(inputs[0], inputs[1])
    return head(*inputs)


def predict_forces(
    model: torch.nn.Module,
    batch: ff_base.AtomGraphs,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    autocast_dtype = resolve_autocast_dtype(device)
    with torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype,
        enabled=use_amp,
    ):
        out = model.model(batch)
        node_features = out["node_features"]
    if use_amp:
        node_features = node_features.to(dtype=torch.float32)
    head = model.heads["forces"]
    predict_fn = getattr(head, "predict", None)
    if callable(predict_fn):
        preds = predict_fn(node_features, batch)
    else:
        preds = head(node_features)
    if getattr(model, "pair_repulsion", False):
        out_pair_repulsion = model.pair_repulsion_fn(batch)
        raw_repulsion = model._get_raw_repulsion("forces", out_pair_repulsion)
        if raw_repulsion is not None:
            preds = preds + raw_repulsion.to(dtype=preds.dtype)
    return preds.to(dtype=dtype)


def evaluate_val_loss(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    source_labels: Sequence[str],
    embeddings: Mapping[str, torch.Tensor],
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    use_amp: bool,
    val_limit: Optional[int],
    target_scale: float,
) -> float:
    was_training = model.training
    model.eval()
    mse_sum = 0.0
    count = 0
    criterion = torch.nn.MSELoss(reduction="sum")
    label_groups = group_indices_by_label(dataset, source_labels)
    processed = 0

    with torch.no_grad():
        for label, indices in label_groups.items():
            if not indices:
                continue
            subset = Subset(dataset, indices)
            loader = DataLoader(
                subset,
                batch_size=max(1, batch_size),
                shuffle=False,
                collate_fn=ff_base.batch_graphs,
            )
            for batch in loader:
                swap_embeddings_inplace(model, label, embeddings)
                batch = batch.to(device=device, dtype=dtype)
                scale_batch_targets(batch, target_scale)
                preds = predict_forces(model, batch, dtype, device, use_amp)
                target = batch.node_targets["forces"]
                mse_sum += criterion(preds, target).item()
                count += target.numel()
                processed += batch.n_node.shape[0]
                if val_limit is not None and processed >= val_limit:
                    break
            if val_limit is not None and processed >= val_limit:
                break

    if was_training:
        model.train()
    if count == 0:
        return float("nan")
    return mse_sum / count


class ForceHeadAdapter(torch.nn.Module):
    def __init__(self, in_features: int, arch: str, hidden_dim: int) -> None:
        super().__init__()
        if arch == "linear":
            self.net = torch.nn.Linear(in_features, 3)
        elif arch == "mlp":
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_features, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 3),
            )
        else:
            raise ValueError(f"Unsupported head architecture: {arch}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def cache_force_head_inputs(
    model: torch.nn.Module,
    graphs: Sequence[ff_base.AtomGraphs],
    labels: Sequence[str],
    embeddings: Mapping[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    logger: FineTuneLogger,
) -> Optional[CachedHeadInputs]:
    if not hasattr(model, "heads") or "forces" not in model.heads:
        logger.log("Force head not found; skipping cache mode.")
        return None

    cached_inputs: List[Tuple[object, ...]] = []
    captured_inputs: Optional[Tuple[object, ...]] = None
    logged_types = False
    requires_graph = False

    def hook(_module: torch.nn.Module, inputs: Tuple[object, ...], _output: torch.Tensor) -> None:
        nonlocal captured_inputs
        if not inputs:
            captured_inputs = None
            return
        captured_inputs = tuple(inputs)

    handle = model.heads["forces"].register_forward_hook(hook)
    was_training = model.training
    model.eval()
    cache_start = time.perf_counter()
    try:
        for idx, graph in enumerate(graphs):
            captured_inputs = None
            label = labels[idx]
            swap_embeddings_inplace(model, label, embeddings)
            batch = ff_base.batch_graphs([graph]).to(device=device, dtype=dtype)
            with torch.inference_mode():
                _ = model.predict(batch)
            if captured_inputs is None:
                logger.log("Force head inputs could not be captured; skipping cache mode.")
                return None
            if not logged_types:
                logger.log(f"Force head input types: {describe_inputs(captured_inputs)}")
                logged_types = True
            stored_inputs: Tuple[object, ...]
            if (
                len(captured_inputs) == 2
                and isinstance(captured_inputs[0], torch.Tensor)
                and hasattr(captured_inputs[1], "n_node")
            ):
                requires_graph = True
                stored_inputs = (move_structure_to_device(captured_inputs[0], torch.device("cpu"), dtype),)
            else:
                stored_inputs = tuple(
                    move_structure_to_device(item, torch.device("cpu"), dtype)
                    for item in captured_inputs
                )
            cached_inputs.append(stored_inputs)
    finally:
        handle.remove()
        if was_training:
            model.train()

    cache_duration = time.perf_counter() - cache_start
    logger.log(f"Cached force-head inputs for {len(cached_inputs)} graphs in {cache_duration:.2f}s.")
    tensor_only = all(all(isinstance(item, torch.Tensor) for item in entry) for entry in cached_inputs)
    single_tensor = tensor_only and all(len(entry) == 1 for entry in cached_inputs)
    if requires_graph:
        logger.log("Force head inputs require graphs; cache will reuse original graphs.")
    elif not tensor_only:
        logger.log("Force head inputs include non-tensors; cache will train existing head only.")
    return CachedHeadInputs(
        entries=cached_inputs,
        tensor_only=tensor_only,
        single_tensor=single_tensor,
        requires_graph=requires_graph,
    )


def try_batch_cached_inputs(entries: Sequence[Tuple[object, ...]]) -> Optional[Tuple[object, ...]]:
    if not entries:
        return None
    first_entry = entries[0]
    if len(first_entry) == 1:
        candidate = first_entry[0]
        if isinstance(candidate, ff_base.AtomGraphs):
            try:
                return (ff_base.batch_graphs([entry[0] for entry in entries]),)
            except Exception:
                return None
        return None
    if len(first_entry) == 2:
        tensor_candidate = first_entry[0]
        graph_candidate = first_entry[1]
        if isinstance(tensor_candidate, torch.Tensor) and hasattr(graph_candidate, "n_node"):
            try:
                batched_tensor = torch.cat([entry[0] for entry in entries], dim=0)
                batched_graph = ff_base.batch_graphs([entry[1] for entry in entries])
                return (batched_tensor, batched_graph)
            except Exception:
                return None
    return None


def train_force_head_with_cached_inputs(
    head: torch.nn.Module,
    cached: CachedHeadInputs,
    targets: Sequence[torch.Tensor],
    graphs: Optional[Sequence[ff_base.AtomGraphs]],
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    scheduler_cfg: Dict[str, object],
    dtype: torch.dtype,
    device: torch.device,
    logger: FineTuneLogger,
    use_amp: bool,
    on_epoch_end: Optional[Callable[[int], None]] = None,
) -> None:
    head.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, epochs, scheduler_cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    batch_size = max(1, batch_size)
    cached_inputs = cached.entries
    indices = list(range(len(cached_inputs)))
    if cached.requires_graph and graphs is None:
        raise ValueError("Cached head inputs require graphs, but no graphs were provided.")

    for epoch in range(1, epochs + 1):
        random.shuffle(indices)
        total_loss = 0.0
        total_nodes = 0

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)

            batch_loss_value = 0.0
            batch_nodes = 0

            if cached.requires_graph:
                batched_graphs = None
                if graphs is not None:
                    try:
                        batched_graphs = ff_base.batch_graphs([graphs[i] for i in batch_indices])
                    except Exception:
                        batched_graphs = None
                if batched_graphs is not None:
                    batched_tensor = torch.cat([cached_inputs[i][0] for i in batch_indices], dim=0).to(
                        device=device, dtype=dtype
                    )
                    batched_graphs = move_structure_to_device(batched_graphs, device, dtype)
                    batched_target = torch.cat([targets[i] for i in batch_indices], dim=0).to(
                        device=device, dtype=dtype
                    )
                    autocast_dtype = resolve_autocast_dtype(device)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=autocast_dtype,
                        enabled=use_amp,
                    ):
                        preds = head_predict(head, batched_tensor, batched_graphs)
                        loss = criterion(preds, batched_target)
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    batch_loss_value = loss.item()
                    batch_nodes = batched_target.shape[0]
                else:
                    for idx in batch_indices:
                        tensor_input = move_tensor_to_device(cached_inputs[idx][0], device, dtype)
                        graph_input = ff_base.batch_graphs([graphs[idx]]) if graphs is not None else None
                        if graph_input is None:
                            raise ValueError("Missing graphs required for cached training.")
                        graph_input = move_structure_to_device(graph_input, device, dtype)
                        target = targets[idx].to(device=device, dtype=dtype)
                        autocast_dtype = resolve_autocast_dtype(device)
                        with torch.autocast(
                            device_type=device.type,
                            dtype=autocast_dtype,
                            enabled=use_amp,
                        ):
                            preds = head_predict(head, tensor_input, graph_input)
                            loss = criterion(preds, target)
                            scaled_loss = loss / len(batch_indices)
                        if scaler.is_enabled():
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                        batch_loss_value += loss.item()
                        batch_nodes += target.shape[0]
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    batch_loss_value /= max(len(batch_indices), 1)
            elif cached.tensor_only:
                try:
                    grouped = list(zip(*(cached_inputs[i] for i in batch_indices)))
                    batched_inputs: List[torch.Tensor] = []
                    for group in grouped:
                        tensors = [move_tensor_to_device(t, device, dtype) for t in group]
                        batched_inputs.append(torch.cat(tensors, dim=0))
                    batched_target = torch.cat([targets[i] for i in batch_indices], dim=0).to(
                        device=device, dtype=dtype
                    )

                    autocast_dtype = resolve_autocast_dtype(device)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=autocast_dtype,
                        enabled=use_amp,
                    ):
                        preds = head_predict(head, *batched_inputs)
                        loss = criterion(preds, batched_target)
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    batch_loss_value = loss.item()
                    batch_nodes = batched_target.shape[0]
                except RuntimeError:
                    for idx in batch_indices:
                        inputs = tuple(
                            move_tensor_to_device(t, device, dtype) for t in cached_inputs[idx]
                        )
                        target = targets[idx].to(device=device, dtype=dtype)
                        autocast_dtype = resolve_autocast_dtype(device)
                        with torch.autocast(
                            device_type=device.type,
                            dtype=autocast_dtype,
                            enabled=use_amp,
                        ):
                            preds = head_predict(head, *inputs)
                            loss = criterion(preds, target)
                            scaled_loss = loss / len(batch_indices)
                        if scaler.is_enabled():
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                        batch_loss_value += loss.item()
                        batch_nodes += target.shape[0]

                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    batch_loss_value /= max(len(batch_indices), 1)
            else:
                batch_entries = [cached_inputs[i] for i in batch_indices]
                batched_inputs = try_batch_cached_inputs(batch_entries)
                if batched_inputs is not None:
                    inputs = tuple(
                        move_structure_to_device(item, device, dtype) for item in batched_inputs
                    )
                    batched_target = torch.cat([targets[i] for i in batch_indices], dim=0).to(
                        device=device, dtype=dtype
                    )
                    autocast_dtype = resolve_autocast_dtype(device)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=autocast_dtype,
                        enabled=use_amp,
                    ):
                        preds = head_predict(head, *inputs)
                        loss = criterion(preds, batched_target)
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    batch_loss_value = loss.item()
                    batch_nodes = batched_target.shape[0]
                else:
                    for idx in batch_indices:
                        inputs = tuple(
                            move_structure_to_device(item, device, dtype)
                            for item in cached_inputs[idx]
                        )
                        target = targets[idx].to(device=device, dtype=dtype)
                        autocast_dtype = resolve_autocast_dtype(device)
                        with torch.autocast(
                            device_type=device.type,
                            dtype=autocast_dtype,
                            enabled=use_amp,
                        ):
                            preds = head_predict(head, *inputs)
                            loss = criterion(preds, target)
                            scaled_loss = loss / len(batch_indices)
                        if scaler.is_enabled():
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                        batch_loss_value += loss.item()
                        batch_nodes += target.shape[0]
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    batch_loss_value /= max(len(batch_indices), 1)

            total_loss += batch_loss_value
            total_nodes += batch_nodes

        avg_loss = total_loss / max(math.ceil(len(indices) / batch_size), 1)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.log(f"[epoch {epoch:03d}] train_loss={avg_loss:.6f} nodes={total_nodes} lr={current_lr:.6e}")
        if scheduler is not None:
            if scheduler_cfg.get("type") == "plateau":
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        if on_epoch_end is not None:
            on_epoch_end(epoch)

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    scheduler_cfg: Dict[str, object],
) -> Optional[
    torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau
]:
    sched_type = scheduler_cfg.get("type", "none")
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    if sched_type == "step":
        step_size = int(scheduler_cfg.get("step_size", 20))
        gamma = float(scheduler_cfg.get("gamma", 0.5))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if sched_type == "plateau":
        patience = int(scheduler_cfg.get("plateau_patience", 10))
        gamma = float(scheduler_cfg.get("gamma", 0.5))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=gamma,
            patience=patience,
        )
    return None


def finetune_force_head(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: Optional[torch.utils.data.Dataset],
    source_labels: Sequence[str],
    embeddings: Mapping[str, torch.Tensor],
    sample_graphs: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    scheduler_cfg: Dict[str, object],
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    logger: FineTuneLogger,
    use_amp: bool,
    cache_head_inputs: bool,
    head_arch: str,
    head_hidden_dim: int,
    val_every: int,
    val_limit: Optional[int],
    target_scale: float,
) -> tuple[int, Optional[Dict[str, torch.Tensor]], Optional[float], Optional[int]]:
    train_start = time.perf_counter()
    indices = select_training_indices(train_dataset, sample_graphs, seed)
    if not indices:
        raise ValueError("No training graphs available for fine-tuning.")
    logger.log(f"Sampling {len(indices)} graphs from training split of size {len(train_dataset)}.")
    graphs, labels, targets = collect_finetune_samples(
        dataset=train_dataset,
        indices=indices,
        source_labels=source_labels,
        target_scale=target_scale,
    )

    freeze_all_but_forces(model)
    drop_unused_heads(model)
    if head_arch != "existing" and not cache_head_inputs:
        logger.log("Head arch override requires cached inputs; enabling cache mode.")
        cache_head_inputs = True

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss: Optional[float] = None
    best_epoch: Optional[int] = None

    def maybe_eval(epoch: int) -> None:
        nonlocal best_state, best_val_loss, best_epoch
        if val_dataset is None or val_every <= 0 or epoch % val_every != 0:
            return
        val_loss = evaluate_val_loss(
            model=model,
            dataset=val_dataset,
            source_labels=source_labels,
            embeddings=embeddings,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            use_amp=use_amp,
            val_limit=val_limit,
            target_scale=target_scale,
        )
        logger.log(f"[epoch {epoch:03d}] val_loss={val_loss:.6f}")
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = to_cpu_state_dict(model.state_dict())

    if cache_head_inputs:
        cached_inputs = cache_force_head_inputs(
            model=model,
            graphs=graphs,
            labels=labels,
            embeddings=embeddings,
            device=device,
            dtype=dtype,
            logger=logger,
        )
        if cached_inputs is not None:
            if head_arch != "existing":
                if cached_inputs.requires_graph or not cached_inputs.single_tensor:
                    logger.log(
                        "Head arch override requires a single tensor input; keeping existing force head."
                    )
                else:
                    in_features = cached_inputs.entries[0][0].shape[-1]
                    model.heads["forces"] = ForceHeadAdapter(
                        in_features=in_features,
                        arch=head_arch,
                        hidden_dim=head_hidden_dim,
                    ).to(device=device, dtype=dtype)
                    logger.log(f"Replaced force head with '{head_arch}' adapter (in_features={in_features}).")
            train_force_head_with_cached_inputs(
                head=model.heads["forces"],
                cached=cached_inputs,
                targets=targets,
                graphs=graphs,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                scheduler_cfg=scheduler_cfg,
                dtype=dtype,
                device=device,
                logger=logger,
                use_amp=use_amp,
                on_epoch_end=maybe_eval,
            )
            train_duration = time.perf_counter() - train_start
            logger.log(
                f"Fine-tuning duration: {train_duration:.2f} s ({train_duration / 60:.2f} min)"
            )
            return len(indices), best_state, best_val_loss, best_epoch
        logger.log("Falling back to full-model fine-tuning (cache unavailable).")

    model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.heads["forces"].parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, epochs, scheduler_cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    for epoch in range(1, epochs + 1):
        model.train()
        # Build batches grouped by label so we only swap embeddings once per optimizer step.
        label_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)

        batches: List[tuple[str, List[int]]] = []
        rng = random.Random(seed + epoch)
        for label, idxs in label_to_indices.items():
            rng.shuffle(idxs)
            for start in range(0, len(idxs), batch_size):
                batches.append((label, idxs[start : start + batch_size]))
        rng.shuffle(batches)

        total_loss = 0.0
        total_nodes = 0
        optimizer.zero_grad(set_to_none=True)

        for label, batch_indices in batches:
            swap_embeddings_inplace(model, label, embeddings)
            batch_graphs = [graphs[i] for i in batch_indices]
            batch = ff_base.batch_graphs(batch_graphs).to(device=device, dtype=dtype)
            target = torch.cat([targets[i] for i in batch_indices], dim=0).to(device=device, dtype=dtype)

            autocast_dtype = resolve_autocast_dtype(device)
            autocast_ctx = torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=use_amp,
            )
            with autocast_ctx:
                preds = model.predict(batch)["forces"]
                loss = criterion(preds, target)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            total_nodes += target.shape[0]

        avg_loss = total_loss / max(len(batches), 1)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.log(f"[epoch {epoch:03d}] train_loss={avg_loss:.6f} nodes={total_nodes} lr={current_lr:.6e}")
        if scheduler is not None:
            if scheduler_cfg.get("type") == "plateau":
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        maybe_eval(epoch)
    model.eval()
    train_duration = time.perf_counter() - train_start
    logger.log(
        f"Fine-tuning duration: {train_duration:.2f} s ({train_duration / 60:.2f} min)"
    )
    return len(indices), best_state, best_val_loss, best_epoch


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Config JSON (e.g., combined_tiny2 config).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to fine-tune.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to combined_sampled.db.")
    parser.add_argument("--ethanol-checkpoint", type=Path, default=None, help="(Deprecated) Checkpoint for ethanol embeddings; prefer --source-checkpoint.")
    parser.add_argument("--malonaldehyde-checkpoint", type=Path, default=None, help="(Deprecated) Checkpoint for malonaldehyde embeddings; prefer --source-checkpoint.")
    parser.add_argument("--device", type=str, default="auto", help="Torch device.")
    parser.add_argument("--batch-size", type=int, default=8, help="Graphs per optimizer step when fine-tuning.")
    parser.add_argument("--compile-model", action="store_true", help="Enable torch.compile on backbone.")
    parser.add_argument(
        "--source-checkpoint",
        action="append",
        default=None,
        help="Mapping of source label to checkpoint state (e.g., ethanol=path/to.ckpt). Repeat for each label.",
    )
    parser.add_argument(
        "--ethanol-teacher-checkpoint",
        type=Path,
        default=None,
        help="(Deprecated) Ignored; kept for backward compatibility with teacher-distillation runs.",
    )
    parser.add_argument(
        "--malonaldehyde-teacher-checkpoint",
        type=Path,
        default=None,
        help="(Deprecated) Ignored; kept for backward compatibility with teacher-distillation runs.",
    )
    parser.add_argument("--ft-sample-graphs", type=int, default=512, help="Training graphs to use when fine-tuning.")
    parser.add_argument("--ft-epochs", type=int, default=20, help="Fine-tuning epochs.")
    parser.add_argument("--ft-lr", type=float, default=5e-4, help="Fine-tuning learning rate.")
    parser.add_argument("--ft-weight-decay", type=float, default=0.0, help="Fine-tuning weight decay.")
    parser.add_argument(
        "--ft-scheduler",
        type=str,
        default="none",
        choices=["none", "cosine", "step", "plateau"],
        help="Learning-rate scheduler to use during fine-tuning.",
    )
    parser.add_argument(
        "--ft-scheduler-step-size",
        type=int,
        default=20,
        help="Step size for StepLR scheduler.",
    )
    parser.add_argument(
        "--ft-scheduler-gamma",
        type=float,
        default=0.5,
        help="Gamma for LR schedulers (step or plateau).",
    )
    parser.add_argument(
        "--ft-scheduler-plateau-patience",
        type=int,
        default=10,
        help="Plateau patience in epochs when using ReduceLROnPlateau.",
    )
    parser.add_argument("--ft-log-path", type=Path, default=None, help="Log path for fine-tuning progress.")
    parser.add_argument("--ft-output", type=Path, default=None, help="Where to write the fine-tuned checkpoint.")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.autocast/GradScaler for faster force-head training (recommended on GPU).",
    )
    parser.add_argument(
        "--ft-cache-head-inputs",
        action="store_true",
        help="Cache force-head inputs and train the head only (much faster; falls back if unsupported).",
    )
    parser.add_argument(
        "--ft-head-arch",
        type=str,
        default="existing",
        choices=["existing", "linear", "mlp"],
        help="Force-head architecture when caching inputs (existing keeps current head).",
    )
    parser.add_argument(
        "--ft-head-hidden",
        type=int,
        default=256,
        help="Hidden size for --ft-head-arch mlp (ignored otherwise).",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=1,
        help="Evaluate validation loss every N epochs (0 disables).",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=0,
        help="Limit number of graphs used for validation loss (0 uses full validation split).",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Disable validation loss measurement during fine-tuning.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    dataset_path = args.dataset
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    base_model = config["base_model"]
    precision = config.get("precision", "float32-high")
    device = resolve_device(args.device)

    model = build_model(base_model, device, precision, compile_model=args.compile_model)
    model_state = load_checkpoint(args.checkpoint)
    model.load_state_dict(model_state, strict=False)
    if config.get("force_only", False) and hasattr(model, "heads") and "stress" in model.heads:
        del model.heads["stress"]
    model.to(device)

    dtype = next(model.parameters()).dtype
    dataset = make_dataset(config["dataset_name"], dataset_path, model, dtype)
    data_units = config.get("data_units", "kcal/mol")
    target_scale = resolve_data_unit_scale(data_units)

    embed_key = "model.atom_emb.embeddings.weight"
    def parse_source_checkpoints() -> Dict[str, Path]:
        mapping: Dict[str, Path] = {}
        if args.ethanol_checkpoint:
            mapping["ethanol"] = Path(args.ethanol_checkpoint)
        if args.malonaldehyde_checkpoint:
            mapping["malonaldehyde"] = Path(args.malonaldehyde_checkpoint)
        if args.source_checkpoint:
            for entry in args.source_checkpoint:
                if "=" not in entry:
                    raise ValueError(f"--source-checkpoint expects label=path, got '{entry}'")
                label, path_str = entry.split("=", 1)
                mapping[label.strip().lower()] = Path(path_str.strip())
        if not mapping:
            raise ValueError("Provide at least one --source-checkpoint (label=path).")
        return mapping

    source_ckpts = parse_source_checkpoints()
    embeddings: Dict[str, torch.Tensor] = {}
    for label, ckpt_path in source_ckpts.items():
        state = load_checkpoint(ckpt_path)
        if embed_key not in state:
            raise KeyError(f"Checkpoint {ckpt_path} missing embedding key '{embed_key}'")
        embeddings[label] = state[embed_key].to(device)

    source_labels = extract_source_labels(dataset_path)

    val_fraction = float(config.get("val_fraction", 0.0))
    test_fraction = float(config.get("test_fraction", 0.0))
    seed = int(config.get("seed", 42))
    split_seed = config.get("split_seed")
    split_seed = int(split_seed) if split_seed is not None else seed
    val_every = int(args.val_every)
    if args.no_val:
        val_every = 0
    val_limit = args.val_limit if args.val_limit and args.val_limit > 0 else None
    train_dataset_split = None
    val_dataset_split = None
    if val_every > 0 and (val_fraction > 0.0 or test_fraction > 0.0) and len(dataset) > 1:
        train_subset, val_subset, _ = deterministic_train_val_test_split(
            dataset, val_fraction, test_fraction, split_seed
        )
        train_dataset_split = train_subset
        val_dataset_split = val_subset

    if train_dataset_split is None:
        train_dataset_split = dataset

    ckpt_path = Path(args.checkpoint)
    suffix_parts = [ckpt_path.stem, f"ft_s{args.ft_sample_graphs}", f"ep{args.ft_epochs}", f"lr{args.ft_lr}"]
    if args.ft_scheduler and args.ft_scheduler != "none":
        suffix_parts.append(f"sched-{args.ft_scheduler}")
    default_output = ckpt_path.with_name("_".join(suffix_parts) + ".ckpt")
    ft_output_path = Path(args.ft_output) if args.ft_output else default_output
    ft_log_path = args.ft_log_path or (Path("logs") / f"{ft_output_path.stem}.log")
    ft_logger = FineTuneLogger(ft_log_path)
    scheduler_cfg = {
        "type": args.ft_scheduler,
        "step_size": args.ft_scheduler_step_size,
        "gamma": args.ft_scheduler_gamma,
        "plateau_patience": args.ft_scheduler_plateau_patience,
    }
    use_amp = bool(args.amp and device.type in ("cuda", "mps"))
    cache_head_inputs = bool(args.ft_cache_head_inputs)
    head_arch = str(args.ft_head_arch)
    head_hidden_dim = int(args.ft_head_hidden)
    try:
        if args.ethanol_teacher_checkpoint or args.malonaldehyde_teacher_checkpoint:
            ft_logger.log(
                "Note: ignoring teacher checkpoints; fine-tuning uses ground-truth forces from the dataset."
            )
        ft_logger.log(
            f"Starting fine-tuning on {train_dataset_split.__class__.__name__} "
            f"(samples={args.ft_sample_graphs}, epochs={args.ft_epochs}, lr={args.ft_lr:g})."
        )
        used, best_state, best_val_loss, best_epoch = finetune_force_head(
            model=model,
            train_dataset=train_dataset_split,
            val_dataset=val_dataset_split,
            source_labels=source_labels,
            embeddings=embeddings,
            sample_graphs=args.ft_sample_graphs,
            batch_size=args.batch_size,
            epochs=args.ft_epochs,
            lr=args.ft_lr,
            weight_decay=args.ft_weight_decay,
            scheduler_cfg=scheduler_cfg,
            dtype=dtype,
            device=device,
            seed=seed,
            logger=ft_logger,
            use_amp=use_amp,
            cache_head_inputs=cache_head_inputs,
            head_arch=head_arch,
            head_hidden_dim=head_hidden_dim,
            val_every=val_every,
            val_limit=val_limit,
            target_scale=target_scale,
        )
        ft_logger.log(f"Completed fine-tuning on {used} training graphs.")
        if val_every <= 0:
            ft_logger.log("Validation disabled; saving final checkpoint.")
        elif val_dataset_split is None:
            ft_logger.log("Validation split not available; saving final checkpoint.")
        elif best_epoch is not None and best_val_loss is not None:
            ft_logger.log(f"Best val loss {best_val_loss:.6f} at epoch {best_epoch}.")
    finally:
        ft_logger.close()

    ft_output_path.parent.mkdir(parents=True, exist_ok=True)
    state_to_save = best_state if best_state is not None else to_cpu_state_dict(model.state_dict())
    checkpoint: Dict[str, object] = {"model": state_to_save, "epoch": args.ft_epochs}
    if best_epoch is not None:
        checkpoint["best_epoch"] = best_epoch
    if best_val_loss is not None:
        checkpoint["best_val_loss"] = best_val_loss
    torch.save(checkpoint, ft_output_path)
    print(f"Saved fine-tuned checkpoint to {ft_output_path}")


if __name__ == "__main__":
    main()
