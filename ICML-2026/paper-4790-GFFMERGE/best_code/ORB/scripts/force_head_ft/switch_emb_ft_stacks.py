"""
Fine-tune the forces head plus the last N GNN stacks using embedding swapping per sample.

Evaluation is handled separately by scripts/kaggle/evaluate_switch_embeddings.py.

This script uses ground-truth forces from the dataset (no teacher distillation).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence
import sys

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


class FineTuneLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
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


def group_indices_by_label(
    dataset: torch.utils.data.Dataset,
    source_labels: Sequence[str],
) -> Dict[str, List[int]]:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        label = label_for_dataset(dataset, idx, source_labels)
        buckets[label].append(idx)
    return buckets


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


def drop_unused_heads(model: torch.nn.Module) -> None:
    """Remove energy/stress heads so force FT runs faster."""
    if not hasattr(model, "heads"):
        return
    for head_name in ("energy", "stress"):
        if head_name in model.heads:
            del model.heads[head_name]
            if hasattr(model, "loss_weights"):
                model.loss_weights.pop(head_name, None)


def freeze_all_params(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_force_head(model: torch.nn.Module) -> None:
    if not hasattr(model, "heads") or "forces" not in model.heads:
        raise ValueError("Model does not include a 'forces' head.")
    for param in model.heads["forces"].parameters():
        param.requires_grad = True


def unfreeze_last_gnn_stacks(model: torch.nn.Module, count: int, logger: FineTuneLogger) -> int:
    if count <= 0:
        return 0
    stacks = None
    if hasattr(model, "model") and hasattr(model.model, "gnn_stacks"):
        stacks = model.model.gnn_stacks
    elif hasattr(model, "gnn_stacks"):
        stacks = model.gnn_stacks
    if stacks is None:
        raise ValueError("Model does not expose gnn_stacks.")
    total = len(stacks)
    if total == 0:
        return 0
    if count > total:
        logger.log(f"Requested {count} stacks, but model has {total}; unfreezing all.")
        count = total
    for stack in list(stacks)[-count:]:
        for param in stack.parameters():
            param.requires_grad = True
    return count


def resolve_autocast_dtype(device: torch.device) -> Optional[torch.dtype]:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return None


def build_grad_scaler(device: torch.device, enabled: bool) -> torch.cuda.amp.GradScaler:
    if hasattr(torch, "amp") and device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def predict_forces_with_amp(
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
    with torch.autocast(device_type=device.type, enabled=False):
        preds = model.heads["forces"].predict(node_features, batch)
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
                preds = predict_forces_with_amp(model, batch, dtype, device, use_amp)
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


def finetune_force_head_with_stacks(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: Optional[torch.utils.data.Dataset],
    source_labels: Sequence[str],
    embeddings: Mapping[str, torch.Tensor],
    sample_graphs: int,
    batch_size: int,
    epochs: int,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
    scheduler_cfg: Dict[str, object],
    stacks_to_unfreeze: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    logger: FineTuneLogger,
    use_amp: bool,
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

    freeze_all_params(model)
    drop_unused_heads(model)
    unfreeze_force_head(model)
    unfrozen = unfreeze_last_gnn_stacks(model, stacks_to_unfreeze, logger)

    head_params = list(model.heads["forces"].parameters())
    stack_params: List[torch.nn.Parameter] = []
    if unfrozen > 0:
        stacks = model.model.gnn_stacks if hasattr(model, "model") else model.gnn_stacks
        for stack in list(stacks)[-unfrozen:]:
            stack_params.extend(list(stack.parameters()))

    param_groups = [{"params": head_params, "lr": lr}]
    if stack_params:
        param_groups.append({"params": stack_params, "lr": backbone_lr})

    model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, epochs, scheduler_cfg)
    scaler = build_grad_scaler(device, enabled=use_amp)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss: Optional[float] = None
    best_epoch: Optional[int] = None

    for epoch in range(1, epochs + 1):
        model.train()
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

            preds = predict_forces_with_amp(model, batch, dtype, device, use_amp)
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
        if val_dataset is not None and val_every > 0 and epoch % val_every == 0:
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

    model.eval()
    train_duration = time.perf_counter() - train_start
    logger.log(f"Fine-tuning duration: {train_duration:.2f} s ({train_duration / 60:.2f} min)")
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
    parser.add_argument("--ft-lr", type=float, default=5e-4, help="Force-head learning rate.")
    parser.add_argument("--ft-backbone-lr", type=float, default=None, help="Learning rate for unfrozen stacks.")
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
    parser.add_argument(
        "--ft-unfreeze-stacks",
        type=int,
        default=1,
        help="Unfreeze the last N GNN stacks (0 keeps only force head trainable).",
    )
    parser.add_argument("--ft-log-path", type=Path, default=None, help="Log path for fine-tuning progress.")
    parser.add_argument("--ft-output", type=Path, default=None, help="Where to write the fine-tuned checkpoint.")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.autocast/GradScaler for faster training (recommended on GPU).",
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
        "--ft-val-fraction",
        type=float,
        default=None,
        help="Override validation fraction for fine-tuning split (default: use config). Set 0 to disable.",
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
    if args.ft_val_fraction is not None:
        val_fraction = float(args.ft_val_fraction)
    if val_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("val_fraction and test_fraction must be >= 0.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.")
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
    suffix_parts = [
        ckpt_path.stem,
        f"ftstacks{args.ft_unfreeze_stacks}",
        f"s{args.ft_sample_graphs}",
        f"ep{args.ft_epochs}",
        f"lr{args.ft_lr}",
    ]
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
    backbone_lr = args.ft_backbone_lr if args.ft_backbone_lr is not None else args.ft_lr
    try:
        if args.ethanol_teacher_checkpoint or args.malonaldehyde_teacher_checkpoint:
            ft_logger.log(
                "Note: ignoring teacher checkpoints; fine-tuning uses ground-truth forces from the dataset."
            )
        ft_logger.log(
            "Starting fine-tuning (force head + last GNN stacks) on "
            f"{train_dataset_split.__class__.__name__} "
            f"(samples={args.ft_sample_graphs}, epochs={args.ft_epochs}, lr={args.ft_lr:g}, "
            f"stacks={args.ft_unfreeze_stacks}, backbone_lr={backbone_lr:g})."
        )
        used, best_state, best_val_loss, best_epoch = finetune_force_head_with_stacks(
            model=model,
            train_dataset=train_dataset_split,
            val_dataset=val_dataset_split,
            source_labels=source_labels,
            embeddings=embeddings,
            sample_graphs=args.ft_sample_graphs,
            batch_size=args.batch_size,
            epochs=args.ft_epochs,
            lr=args.ft_lr,
            backbone_lr=backbone_lr,
            weight_decay=args.ft_weight_decay,
            scheduler_cfg=scheduler_cfg,
            stacks_to_unfreeze=args.ft_unfreeze_stacks,
            dtype=dtype,
            device=device,
            seed=seed,
            logger=ft_logger,
            use_amp=use_amp,
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
    print(f"Saved fine-tuned checkpoint to {ft_output_path}", flush=True)


if __name__ == "__main__":
    main()
