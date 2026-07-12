import argparse
import contextlib
import itertools
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import medmnist
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from medmnist import INFO
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.check_3d_equivariance import (
    CheckpointSpec,
    create_model,
    get_default_data_root,
    infer_checkpoint_stage,
    infer_run_id,
    load_checkpoint,
    load_state_dict,
    normalise_config,
    recover_config,
    resolve_checkpoint_specs,
    set_seed,
)
from experiments.util import RPPApproxConv_L2, get_irrepsmaps, get_kl_loss, get_shift_loss


RUN_DIR_PATTERN = re.compile(r"run-\d{8}_\d{6}-([a-z0-9]{8})$")
SPECIAL_NODULE_SO3_LEARN_EQ_CHANNELS = (2, 3, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark 3D checkpoints using the same checkpoint discovery and model "
            "construction path as the equivariance checker."
        )
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="*",
        help=(
            "Checkpoint files or directories. Directory inputs are scanned "
            "recursively for *_init.pth and *_final.pth files."
        ),
    )
    parser.add_argument(
        "--stages",
        choices=["init", "final"],
        nargs="+",
        default=["init", "final"],
        help="Checkpoint stages to select when an input path is a directory.",
    )
    parser.add_argument(
        "--train-split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split used for training throughput / epoch timing.",
    )
    parser.add_argument(
        "--infer-split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split used for inference throughput.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Training benchmark batch size. Defaults to the checkpoint config batch size.",
    )
    parser.add_argument(
        "--infer-batch-size",
        type=int,
        default=None,
        help="Inference benchmark batch size. Defaults to the training batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device to use. Defaults to "cuda" when available, else "cpu".',
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help='Dataset root. Defaults to checkpoint config, then "$DATA_ROOT", then "data".',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow medmnist to download missing dataset files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Load the checkpoint with strict=True.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=8,
        help="Limit the number of measured training batches. Defaults to 8.",
    )
    parser.add_argument(
        "--max-infer-batches",
        type=int,
        default=8,
        help="Limit the number of measured inference batches. Defaults to 8.",
    )
    parser.add_argument(
        "--warmup-train-batches",
        type=int,
        default=1,
        help="Number of warmup training batches before timing.",
    )
    parser.add_argument(
        "--warmup-infer-batches",
        type=int,
        default=1,
        help="Number of warmup inference batches before timing.",
    )
    parser.add_argument(
        "--projection-batches",
        type=int,
        default=4,
        help=(
            "Number of projection-penalty-only iterations to time for models exposing "
            "projection_penalty(). Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable autocast/GradScaler benchmarking when supported.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON file for the aggregated results.",
    )
    parser.add_argument(
        "--fast-init",
        action="store_true",
        help=(
            "Skip ExactKernelProjectorR3 construction for ApproxProj checkpoints. "
            "This is faster to instantiate but disables meaningful projection-overhead measurements."
        ),
    )
    parser.add_argument(
        "--include-nodule-so3-learn-eq",
        action="store_true",
        help=(
            "Also include the nodulemnist3d SO3 resnet learn_eq checkpoints for "
            "channels 2, 3, and 6 from the recent medical_mnist2d runs."
        ),
    )
    return parser.parse_args()


def _read_wandb_scalar(config_text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\n\s+value:\s+(.+)$", config_text, re.MULTILINE)
    if match is None:
        return None
    return match.group(1).strip().strip('"').strip("'")


def _extract_run_id_from_config_path(config_path: Path) -> str | None:
    match = RUN_DIR_PATTERN.match(config_path.parents[1].name)
    return None if match is None else match.group(1)


def resolve_special_checkpoint_specs() -> list[CheckpointSpec]:
    wandb_root = ROOT / "wandb"
    checkpoint_root = ROOT / "checkpoints" / "med_mnist"
    matched_run_ids: dict[int, str] = {}

    for config_path in sorted(wandb_root.glob("run-*/files/config.yaml"), reverse=True):
        text = config_path.read_text(encoding="utf-8")
        channel_value = _read_wandb_scalar(text, "channels")
        if channel_value is None:
            continue

        try:
            channels = int(channel_value)
        except ValueError:
            continue

        if channels not in SPECIAL_NODULE_SO3_LEARN_EQ_CHANNELS or channels in matched_run_ids:
            continue

        required_pairs = {
            "dataset": "nodulemnist3d",
            "group": "SO3",
            "learn_eq": "true",
            "resnet": "true",
            "activation": "gated",
            "lr": "0.0005",
            "batch_size": "32",
            "kl_div": "1",
            "kl_uniform": "1",
            "alignment_loss": "5",
        }
        if any(_read_wandb_scalar(text, key) != value for key, value in required_pairs.items()):
            continue

        run_id = _extract_run_id_from_config_path(config_path)
        if run_id is None:
            continue
        matched_run_ids[channels] = run_id

    missing_channels = sorted(set(SPECIAL_NODULE_SO3_LEARN_EQ_CHANNELS) - set(matched_run_ids))
    if missing_channels:
        raise RuntimeError(
            "Could not resolve wandb runs for the special nodule SO3 learn_eq benchmark "
            f"channels: {missing_channels}."
        )

    resolved: list[CheckpointSpec] = []
    for channels in SPECIAL_NODULE_SO3_LEARN_EQ_CHANNELS:
        run_id = matched_run_ids[channels]
        preferred_patterns = (
            f"*_{run_id}_final.pth",
            f"*_TMP_{run_id}.pth",
            f"*_{run_id}_init.pth",
        )
        checkpoint_path = None
        for pattern in preferred_patterns:
            matches = sorted(checkpoint_root.glob(pattern))
            if matches:
                checkpoint_path = matches[-1]
                break

        if checkpoint_path is None:
            raise RuntimeError(
                "Resolved special run id "
                f"{run_id} for channels={channels}, but found no matching checkpoint in "
                f"{checkpoint_root}."
            )

        resolved.append(
            CheckpointSpec(
                path=checkpoint_path,
                stage=infer_checkpoint_stage(checkpoint_path),
                discovered_from=checkpoint_root,
            )
        )

    print(
        "[discover] Added special nodule SO3 learn_eq checkpoints: "
        + ", ".join(str(spec.path.name) for spec in resolved)
    )
    return resolved


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def create_grad_scaler(device: torch.device, enabled: bool):
    scaler_enabled = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=scaler_enabled)


def unwrap_output(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if hasattr(output, "tensor"):
        output = output.tensor
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor-like model output, found {type(output).__name__}.")
    return output


def scalar_to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def build_loader(
    config: dict[str, Any],
    split: str,
    batch_size: int,
    num_workers: int,
    data_root: Path,
    download: bool,
    *,
    shuffle: bool,
    device: torch.device,
) -> tuple[torch.utils.data.DataLoader, int]:
    info = INFO[config["dataset"]]
    data_class = getattr(medmnist, info["python_class"])
    medmnist_root = Path(data_root) / "medmnist"

    if "3d" not in config["dataset"]:
        raise RuntimeError("This benchmark script only supports 3D datasets.")

    transform = transforms.Compose(
        [
            lambda x: torch.FloatTensor(x),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    dataset = data_class(
        split=split,
        transform=transform,
        download=download,
        root=str(medmnist_root),
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"[data] Loader ready for split={split}: {len(dataset)} samples, "
        f"batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}."
    )
    return loader, len(dataset)


def iter_limited(loader: torch.utils.data.DataLoader, max_batches: int | None):
    if max_batches is None:
        for batch_idx, batch in enumerate(loader):
            yield batch_idx, batch
        return

    for batch_idx, batch in enumerate(itertools.islice(itertools.cycle(loader), max_batches)):
        yield batch_idx, batch


def move_batch_to_device(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, targets = batch
    x = x.to(device, non_blocking=device.type == "cuda")
    targets = targets.to(device, non_blocking=device.type == "cuda").reshape(-1).long()
    return x, targets


def build_optimizer(config: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    name = str(config.get("optimizer", "Adam")).lower()
    lr = float(config.get("lr", 5e-4))
    weight_decay = float(config.get("basic_wd", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer setting in checkpoint config: {config.get('optimizer')}")


def collect_effective_parameter_sizes(model: torch.nn.Module) -> dict[int, int]:
    effective_sizes: dict[int, int] = {}

    for module in model.modules():
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is None:
            continue

        for param_name, chain in parametrizations.items():
            original = getattr(chain, "original", None)
            if original is None or not isinstance(original, torch.nn.Parameter):
                continue

            effective = original.numel()
            if len(chain) == 1:
                parametrization = chain[0]
                q = getattr(parametrization, "Q", None)
                if isinstance(q, torch.Tensor) and q.ndim == 2 and q.shape[0] == original.numel():
                    effective = max(0, original.numel() - int(q.shape[1]))

            effective_sizes[id(original)] = effective

    return effective_sizes


def bake_parametrized_weights_inplace(model: torch.nn.Module) -> dict[str, Any] | None:
    baked_modules: list[str] = []
    for module_name, submodule in model.named_modules():
        if hasattr(submodule, "weight") and parametrize.is_parametrized(submodule, "weight"):
            parametrize.remove_parametrizations(
                submodule,
                "weight",
                leave_parametrized=True,
            )
            baked_modules.append(module_name or "<root>")

    if not baked_modules:
        return None

    return {
        "num_modules": len(baked_modules),
        "modules": baked_modules,
    }


def compute_training_loss(
    model: torch.nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
    config: dict[str, Any],
    irrepmaps: list[Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = F.cross_entropy(logits, targets)
    metrics = {"ce_loss": float(loss.detach().cpu())}

    alignment = float(config.get("alignment_loss", 0.0))
    kl_div = float(config.get("kl_div", 0.0))
    kl_uniform = float(config.get("kl_uniform", 0.0))
    if alignment or kl_div or kl_uniform:
        shift_loss = get_shift_loss(irrepmaps)
        kl_loss, kl_uni_loss = get_kl_loss(irrepmaps)
        loss = loss + alignment * shift_loss + kl_div * kl_loss + kl_uniform * kl_uni_loss
        metrics.update(
            shift_loss=scalar_to_float(shift_loss),
            kl_loss=scalar_to_float(kl_loss),
            kl_uniform=scalar_to_float(kl_uni_loss),
        )

    if hasattr(model, "projection_penalty"):
        proj_penalty = model.projection_penalty()
        loss = loss + proj_penalty
        metrics["projection_penalty"] = scalar_to_float(proj_penalty)

    if config.get("rpp") or (config.get("approx") and not config.get("penalized_approx")):
        regularizer = RPPApproxConv_L2(
            model,
            float(config.get("conv_wd", 0.0)),
            float(config.get("basic_wd", 0.0)),
        )
        loss = loss + regularizer
        metrics["regularizer"] = scalar_to_float(regularizer)

    return loss, metrics


def count_forward_connected_parameters(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    *,
    amp_enabled: bool,
) -> dict[str, int]:
    x, _ = move_batch_to_device(batch, device)
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    effective_sizes = collect_effective_parameter_sizes(model)

    with make_autocast_context(device, amp_enabled):
        outputs = unwrap_output(model(x))
        probe_loss = outputs.float().square().mean()
    probe_loss.backward()

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    forward_connected_raw = sum(
        p.numel() for p in model.parameters() if p.requires_grad and p.grad is not None
    )
    forward_connected = sum(
        effective_sizes.get(id(p), p.numel())
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    )
    non_forward = total_trainable - forward_connected
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return {
        "trainable": int(total_trainable),
        "forward_connected_raw": int(forward_connected_raw),
        "forward_connected": int(forward_connected),
        "parametrization_reduction": int(forward_connected_raw - forward_connected),
        "not_forward_connected": int(non_forward),
    }


def warmup_inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    warmup_batches: int,
    *,
    amp_enabled: bool,
) -> None:
    if warmup_batches <= 0:
        return
    model.eval()
    with torch.inference_mode():
        for _, batch in iter_limited(loader, warmup_batches):
            x, _ = move_batch_to_device(batch, device)
            with make_autocast_context(device, amp_enabled):
                unwrap_output(model(x))
    sync_device(device)


def benchmark_inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    dataset_size: int,
    device: torch.device,
    *,
    max_batches: int | None,
    amp_enabled: bool,
) -> dict[str, Any]:
    model.eval()
    measured_batches = 0
    measured_samples = 0
    sync_device(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _, batch in iter_limited(loader, max_batches):
            x, _ = move_batch_to_device(batch, device)
            with make_autocast_context(device, amp_enabled):
                unwrap_output(model(x))
            measured_batches += 1
            measured_samples += x.shape[0]
    sync_device(device)
    elapsed = time.perf_counter() - start
    if measured_batches == 0:
        raise RuntimeError("Inference benchmark did not process any batches.")

    loader_batches = len(loader)
    extrapolated = elapsed * loader_batches / measured_batches
    is_partial = measured_batches != loader_batches

    return {
        "dataset_size": int(dataset_size),
        "num_batches_measured": int(measured_batches),
        "num_samples_measured": int(measured_samples),
        "elapsed_seconds": elapsed,
        "throughput_samples_per_second": measured_samples / elapsed,
        "latency_seconds_per_batch": elapsed / measured_batches,
        "estimated_full_split_seconds": extrapolated,
        "is_partial_measurement": is_partial,
    }


def warmup_training(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    warmup_batches: int,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    config: dict[str, Any],
    *,
    amp_enabled: bool,
) -> None:
    if warmup_batches <= 0:
        return
    model.train()
    irrepmaps = get_irrepsmaps(model)
    for _, batch in iter_limited(loader, warmup_batches):
        x, targets = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with make_autocast_context(device, amp_enabled):
            logits = unwrap_output(model(x))
            loss, _ = compute_training_loss(model, logits, targets, config, irrepmaps)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sync_device(device)


def benchmark_training(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    dataset_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    *,
    max_batches: int | None,
    amp_enabled: bool,
) -> dict[str, Any]:
    model.train()
    irrepmaps = get_irrepsmaps(model)
    scaler = create_grad_scaler(device, amp_enabled)
    measured_batches = 0
    measured_samples = 0
    loss_values: list[float] = []

    sync_device(device)
    start = time.perf_counter()
    for _, batch in iter_limited(loader, max_batches):
        x, targets = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with make_autocast_context(device, amp_enabled):
            logits = unwrap_output(model(x))
            loss, metrics = compute_training_loss(model, logits, targets, config, irrepmaps)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_values.append(float(loss.detach().cpu()))
        measured_batches += 1
        measured_samples += x.shape[0]
    sync_device(device)
    elapsed = time.perf_counter() - start
    if measured_batches == 0:
        raise RuntimeError("Training benchmark did not process any batches.")

    loader_batches = len(loader)
    extrapolated = elapsed * loader_batches / measured_batches
    is_partial = measured_batches != loader_batches

    return {
        "dataset_size": int(dataset_size),
        "num_batches_measured": int(measured_batches),
        "num_samples_measured": int(measured_samples),
        "elapsed_seconds": elapsed,
        "throughput_samples_per_second": measured_samples / elapsed,
        "latency_seconds_per_batch": elapsed / measured_batches,
        "estimated_full_epoch_seconds": extrapolated,
        "is_partial_measurement": is_partial,
        "mean_loss": float(np.mean(loss_values)),
        "final_loss": float(loss_values[-1]),
    }


def benchmark_projection_penalty(
    model: torch.nn.Module,
    device: torch.device,
    *,
    num_batches: int,
) -> dict[str, Any] | None:
    if num_batches <= 0 or not hasattr(model, "projection_penalty"):
        return None

    forward_times: list[float] = []
    backward_times: list[float] = []
    total_times: list[float] = []
    for _ in range(num_batches):
        model.zero_grad(set_to_none=True)
        sync_device(device)
        start = time.perf_counter()
        penalty = model.projection_penalty()
        sync_device(device)
        mid = time.perf_counter()
        penalty.backward()
        sync_device(device)
        end = time.perf_counter()
        forward_times.append(mid - start)
        backward_times.append(end - mid)
        total_times.append(end - start)
    model.zero_grad(set_to_none=True)

    return {
        "num_iterations": int(num_batches),
        "forward_seconds_mean": float(np.mean(forward_times)),
        "backward_seconds_mean": float(np.mean(backward_times)),
        "total_seconds_mean": float(np.mean(total_times)),
    }


def benchmark_checkpoint(
    spec: CheckpointSpec,
    args: argparse.Namespace,
    device: torch.device,
    *,
    amp_enabled: bool,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(spec.path)
    config = normalise_config(recover_config(checkpoint, spec.path))
    data_root = args.data_root or get_default_data_root(config)
    requested_train_batch = args.train_batch_size or int(config.get("batch_size", 8))
    requested_infer_batch = args.infer_batch_size or requested_train_batch
    run_id = infer_run_id(spec.path)

    init_start = time.perf_counter()
    model = create_model(config, device, full_init=not args.fast_init)
    model_init_seconds = time.perf_counter() - init_start
    missing_keys, unexpected_keys = load_state_dict(model, checkpoint, strict=args.strict)

    train_loader, train_dataset_size = build_loader(
        config,
        args.train_split,
        requested_train_batch,
        args.num_workers,
        data_root,
        args.download,
        shuffle=args.train_split == "train",
        device=device,
    )
    infer_loader, infer_dataset_size = build_loader(
        config,
        args.infer_split,
        requested_infer_batch,
        args.num_workers,
        data_root,
        args.download,
        shuffle=False,
        device=device,
    )

    first_train_batch = next(iter(train_loader))
    parameter_counts = count_forward_connected_parameters(
        model,
        first_train_batch,
        device,
        amp_enabled=amp_enabled,
    )

    warmup_inference(
        model,
        infer_loader,
        device,
        args.warmup_infer_batches,
        amp_enabled=amp_enabled,
    )
    optimizer = build_optimizer(config, model)
    train_scaler = create_grad_scaler(device, amp_enabled)
    warmup_training(
        model,
        train_loader,
        device,
        args.warmup_train_batches,
        optimizer,
        train_scaler,
        config,
        amp_enabled=amp_enabled,
    )
    training_metrics = benchmark_training(
        model,
        train_loader,
        train_dataset_size,
        device,
        optimizer,
        config,
        max_batches=args.max_train_batches,
        amp_enabled=amp_enabled,
    )

    model.eval()
    warmup_inference(
        model,
        infer_loader,
        device,
        args.warmup_infer_batches,
        amp_enabled=amp_enabled,
    )
    inference_metrics = benchmark_inference(
        model,
        infer_loader,
        infer_dataset_size,
        device,
        max_batches=args.max_infer_batches,
        amp_enabled=amp_enabled,
    )

    baked_parametrizations = bake_parametrized_weights_inplace(model)
    baked_inference_metrics = None
    if baked_parametrizations is not None:
        print(
            f"[model] Baked and removed weight parametrizations from "
            f"{baked_parametrizations['num_modules']} module(s) for inference."
        )
        warmup_inference(
            model,
            infer_loader,
            device,
            args.warmup_infer_batches,
            amp_enabled=amp_enabled,
        )
        baked_inference_metrics = benchmark_inference(
            model,
            infer_loader,
            infer_dataset_size,
            device,
            max_batches=args.max_infer_batches,
            amp_enabled=amp_enabled,
        )

    projection_metrics = benchmark_projection_penalty(
        model,
        device,
        num_batches=args.projection_batches,
    )
    if projection_metrics is not None:
        projection_metrics["fraction_of_train_step"] = (
            projection_metrics["total_seconds_mean"] / training_metrics["latency_seconds_per_batch"]
        )

    return {
        "checkpoint": str(spec.path),
        "checkpoint_stage": spec.stage,
        "discovered_from": None if spec.discovered_from is None else str(spec.discovered_from),
        "run_id": run_id,
        "device": str(device),
        "model_class": type(model).__name__,
        "config": config,
        "model_init_seconds": model_init_seconds,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "parameter_counts": parameter_counts,
        "training": training_metrics,
        "inference": inference_metrics,
        "baked_parametrizations": baked_parametrizations,
        "inference_baked": baked_inference_metrics,
        "projection_penalty_overhead": projection_metrics,
    }


def summarize_result(result: dict[str, Any]) -> str:
    train_tp = result["training"]["throughput_samples_per_second"]
    infer_tp = result["inference"]["throughput_samples_per_second"]
    params = result["parameter_counts"]["forward_connected"]
    summary = (
        f"[summary] {Path(result['checkpoint']).name}: "
        f"forward_params={params}, "
        f"train_tp={train_tp:.2f} samples/s, "
        f"infer_tp={infer_tp:.2f} samples/s, "
        f"epoch_est={result['training']['estimated_full_epoch_seconds']:.2f}s"
    )
    baked = result.get("inference_baked")
    if baked is not None:
        summary += f", infer_tp_baked={baked['throughput_samples_per_second']:.2f} samples/s"
    proj = result.get("projection_penalty_overhead")
    if proj is not None:
        summary += (
            f", projection_overhead={proj['total_seconds_mean']:.4f}s "
            f"({proj['fraction_of_train_step'] * 100:.1f}% of train step)"
        )
    return summary


def main() -> None:
    args = parse_args()
    if not args.inputs and not args.include_nodule_so3_learn_eq:
        raise RuntimeError(
            "Provide at least one checkpoint input or enable --include-nodule-so3-learn-eq."
        )
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type in {"cuda", "cpu"})
    print(f"[setup] Using device {device}.")
    print(f"[setup] AMP enabled: {amp_enabled}.")

    specs: list[CheckpointSpec] = []
    seen: set[Path] = set()
    if args.inputs:
        for spec in resolve_checkpoint_specs(args.inputs, args.stages):
            if spec.path not in seen:
                specs.append(spec)
                seen.add(spec.path)
    if args.include_nodule_so3_learn_eq:
        for spec in resolve_special_checkpoint_specs():
            if spec.path not in seen:
                specs.append(spec)
                seen.add(spec.path)

    payload: dict[str, Any] = {
        "device": str(device),
        "amp_enabled": amp_enabled,
        "results": [],
    }

    for index, spec in enumerate(specs, start=1):
        print("=" * 100)
        print(f"[main] Benchmarking checkpoint {index}/{len(specs)}: {spec.path}")
        print("=" * 100)
        try:
            result = benchmark_checkpoint(spec, args, device, amp_enabled=amp_enabled)
        except Exception as exc:
            result = {
                "checkpoint": str(spec.path),
                "checkpoint_stage": spec.stage,
                "discovered_from": None if spec.discovered_from is None else str(spec.discovered_from),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[error] {result['error']}")
        payload["results"].append(result)
        if "error" not in result:
            print(summarize_result(result))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[output] Wrote JSON results to {args.json_out}.")


if __name__ == "__main__":
    main()
