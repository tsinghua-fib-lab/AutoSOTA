from __future__ import annotations

import json
import time
from collections import defaultdict

import torch

from meisenmeister.training.registry import get_trainer_class
from meisenmeister.training.splits import get_fold_sample_ids
from meisenmeister.utils import (
    autocast_context,
    find_dataset_dir,
    maybe_compile_model,
    verify_required_global_paths_set,
)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _build_cuda_events(device: torch.device) -> dict[str, torch.cuda.Event] | None:
    if device.type != "cuda":
        return None
    return {
        name: torch.cuda.Event(enable_timing=True)
        for name in (
            "transfer_start",
            "transfer_end",
            "forward_start",
            "forward_end",
            "backward_start",
            "backward_end",
            "optimizer_start",
            "optimizer_end",
        )
    }


def _elapsed_seconds(
    events: dict[str, torch.cuda.Event] | None,
    start_name: str,
    end_name: str,
    start_time: float,
    end_time: float,
) -> float:
    if events is None:
        return float(end_time - start_time)
    return float(events[start_name].elapsed_time(events[end_name]) / 1000.0)


def _next_batch(iterator, dataloader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        return next(iterator), iterator


def _benchmark_training_phase(
    trainer,
    *,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, float | int]:
    architecture = trainer.get_architecture()
    optimizer = trainer.get_optimizer()
    loss_fn = trainer.get_loss()
    dataloader = trainer.get_train_dataloader()
    architecture.train()

    timing_buckets: dict[str, list[float]] = defaultdict(list)
    first_step_time_sec: float | None = None
    total_samples = 0
    iterator = iter(dataloader)

    for step_idx in range(warmup_steps + measured_steps):
        data_start_time = time.perf_counter()
        batch, iterator = _next_batch(iterator, dataloader)
        data_end_time = time.perf_counter()

        events = _build_cuda_events(trainer.device)
        transfer_start_time = time.perf_counter()
        if events is not None:
            events["transfer_start"].record()
        images = batch["image"].to(
            trainer.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        labels = batch["label"].to(
            trainer.device,
            dtype=torch.long,
            non_blocking=True,
        )
        if events is not None:
            events["transfer_end"].record()
        transfer_end_time = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        forward_start_time = time.perf_counter()
        if events is not None:
            events["forward_start"].record()
        with autocast_context(trainer.device, trainer.amp_dtype):
            logits = architecture(images)
            loss = loss_fn(logits, labels)
        if events is not None:
            events["forward_end"].record()
        forward_end_time = time.perf_counter()

        backward_start_time = time.perf_counter()
        if events is not None:
            events["backward_start"].record()
        if trainer.grad_scaler is not None:
            trainer.grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        if events is not None:
            events["backward_end"].record()
        backward_end_time = time.perf_counter()

        optimizer_start_time = time.perf_counter()
        if events is not None:
            events["optimizer_start"].record()
        if trainer.grad_scaler is not None:
            trainer.grad_scaler.step(optimizer)
            trainer.grad_scaler.update()
        else:
            optimizer.step()
        if events is not None:
            events["optimizer_end"].record()

        step_end_time = time.perf_counter()
        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)
            step_end_time = time.perf_counter()

        step_time_sec = float(step_end_time - transfer_start_time)
        if first_step_time_sec is None:
            first_step_time_sec = step_time_sec

        if step_idx < warmup_steps:
            continue

        batch_size = int(labels.shape[0])
        total_samples += batch_size
        timing_buckets["data_time_sec"].append(float(data_end_time - data_start_time))
        timing_buckets["host_to_device_time_sec"].append(
            _elapsed_seconds(
                events,
                "transfer_start",
                "transfer_end",
                transfer_start_time,
                transfer_end_time,
            )
        )
        timing_buckets["forward_time_sec"].append(
            _elapsed_seconds(
                events,
                "forward_start",
                "forward_end",
                forward_start_time,
                forward_end_time,
            )
        )
        timing_buckets["backward_time_sec"].append(
            _elapsed_seconds(
                events,
                "backward_start",
                "backward_end",
                backward_start_time,
                backward_end_time,
            )
        )
        timing_buckets["optimizer_step_time_sec"].append(
            _elapsed_seconds(
                events,
                "optimizer_start",
                "optimizer_end",
                optimizer_start_time,
                step_end_time,
            )
        )
        timing_buckets["step_time_sec"].append(step_time_sec)

    steady_step_time_sec = _mean(timing_buckets["step_time_sec"])
    return {
        "warmup_steps": int(warmup_steps),
        "measured_steps": int(measured_steps),
        "train_first_step_time_sec": float(first_step_time_sec or 0.0),
        "train_steady_step_time_sec": steady_step_time_sec,
        "train_compile_warmup_cost_sec": max(
            0.0,
            float((first_step_time_sec or 0.0) - steady_step_time_sec),
        ),
        "train_data_time_sec": _mean(timing_buckets["data_time_sec"]),
        "train_host_to_device_time_sec": _mean(
            timing_buckets["host_to_device_time_sec"]
        ),
        "train_forward_time_sec": _mean(timing_buckets["forward_time_sec"]),
        "train_backward_time_sec": _mean(timing_buckets["backward_time_sec"]),
        "train_optimizer_step_time_sec": _mean(
            timing_buckets["optimizer_step_time_sec"]
        ),
        "train_throughput_samples_per_sec": (
            float(total_samples / sum(timing_buckets["step_time_sec"]))
            if timing_buckets["step_time_sec"]
            else 0.0
        ),
    }


def _benchmark_validation_phase(
    trainer,
    *,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, float | int]:
    architecture = trainer.get_architecture()
    loss_fn = trainer.get_loss()
    dataloader = trainer.get_val_dataloader()
    architecture.eval()

    timing_buckets: dict[str, list[float]] = defaultdict(list)
    total_samples = 0
    iterator = iter(dataloader)

    with torch.inference_mode():
        for step_idx in range(warmup_steps + measured_steps):
            data_start_time = time.perf_counter()
            batch, iterator = _next_batch(iterator, dataloader)
            data_end_time = time.perf_counter()

            events = _build_cuda_events(trainer.device)
            transfer_start_time = time.perf_counter()
            if events is not None:
                events["transfer_start"].record()
            images = batch["image"].to(
                trainer.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            labels = batch["label"].to(
                trainer.device,
                dtype=torch.long,
                non_blocking=True,
            )
            if events is not None:
                events["transfer_end"].record()
            transfer_end_time = time.perf_counter()

            forward_start_time = time.perf_counter()
            if events is not None:
                events["forward_start"].record()
            with autocast_context(trainer.device, trainer.amp_dtype):
                logits = architecture(images)
                loss_fn(logits, labels)
            if events is not None:
                events["forward_end"].record()
            step_end_time = time.perf_counter()
            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
                step_end_time = time.perf_counter()

            if step_idx < warmup_steps:
                continue

            batch_size = int(labels.shape[0])
            total_samples += batch_size
            timing_buckets["data_time_sec"].append(
                float(data_end_time - data_start_time)
            )
            timing_buckets["host_to_device_time_sec"].append(
                _elapsed_seconds(
                    events,
                    "transfer_start",
                    "transfer_end",
                    transfer_start_time,
                    transfer_end_time,
                )
            )
            timing_buckets["forward_time_sec"].append(
                _elapsed_seconds(
                    events,
                    "forward_start",
                    "forward_end",
                    forward_start_time,
                    step_end_time,
                )
            )
            timing_buckets["step_time_sec"].append(
                float(step_end_time - transfer_start_time)
            )

    return {
        "val_warmup_steps": int(warmup_steps),
        "val_measured_steps": int(measured_steps),
        "val_step_time_sec": _mean(timing_buckets["step_time_sec"]),
        "val_data_time_sec": _mean(timing_buckets["data_time_sec"]),
        "val_host_to_device_time_sec": _mean(timing_buckets["host_to_device_time_sec"]),
        "val_forward_time_sec": _mean(timing_buckets["forward_time_sec"]),
        "val_throughput_samples_per_sec": (
            float(total_samples / sum(timing_buckets["step_time_sec"]))
            if timing_buckets["step_time_sec"]
            else 0.0
        ),
    }


def benchmark_train(
    d: int,
    *,
    fold: int,
    trainer_name: str = "mmTrainer",
    num_workers: int | None = None,
    experiment_postfix: str | None = None,
    compile_enabled: bool = True,
    train_warmup_steps: int = 3,
    train_steps: int = 10,
    val_warmup_steps: int = 2,
    val_steps: int = 5,
) -> dict[str, object]:
    if train_warmup_steps < 0 or train_steps < 1:
        raise ValueError("train warmup must be >= 0 and train steps must be >= 1")
    if val_warmup_steps < 0 or val_steps < 1:
        raise ValueError("val warmup must be >= 0 and val steps must be >= 1")

    dataset_id = f"{d:03d}"
    paths = verify_required_global_paths_set(("mm_preprocessed", "mm_results"))
    preprocessed_dataset_dir = find_dataset_dir(paths["mm_preprocessed"], dataset_id)
    dataset_dir = preprocessed_dataset_dir
    get_fold_sample_ids(preprocessed_dataset_dir, fold)

    trainer_class = get_trainer_class(trainer_name)
    architecture_name = getattr(trainer_class, "ARCHITECTURE_NAME", "ResNet3D18")
    trainer = trainer_class(
        dataset_id=dataset_id,
        fold=fold,
        dataset_dir=dataset_dir,
        preprocessed_dataset_dir=preprocessed_dataset_dir,
        results_dir=paths["mm_results"],
        architecture_name=architecture_name,
        num_workers=num_workers,
        experiment_postfix=experiment_postfix,
        compile_enabled=compile_enabled,
    )

    architecture = trainer.get_architecture()
    architecture, compile_applied, compile_status_message = maybe_compile_model(
        architecture,
        device=trainer.device,
        enabled=trainer.compile_enabled,
    )
    trainer._architecture = architecture
    trainer.get_optimizer()
    trainer.get_scheduler()

    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)

    result = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_dir.name,
        "trainer_name": trainer_class.__name__,
        "architecture_name": architecture_name,
        "device": str(trainer.device),
        "amp_enabled": bool(trainer.amp_enabled),
        "amp_dtype": None if trainer.amp_dtype is None else str(trainer.amp_dtype),
        "compile_enabled": bool(trainer.compile_enabled),
        "compile_applied": bool(compile_applied),
        "compile_status": compile_status_message,
        "batch_size": int(trainer.batch_size),
        "num_workers": int(trainer.num_workers),
        "train_dataset_size": len(trainer.get_train_dataset()),
        "val_dataset_size": len(trainer.get_val_dataset()),
    }
    result.update(
        _benchmark_training_phase(
            trainer,
            warmup_steps=train_warmup_steps,
            measured_steps=train_steps,
        )
    )
    result.update(
        _benchmark_validation_phase(
            trainer,
            warmup_steps=val_warmup_steps,
            measured_steps=val_steps,
        )
    )
    result["peak_gpu_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(trainer.device))
        if trainer.device.type == "cuda"
        else 0
    )
    trainer.fold_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = trainer.fold_dir / "benchmark.json"
    result["benchmark_path"] = str(benchmark_path)
    benchmark_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result
