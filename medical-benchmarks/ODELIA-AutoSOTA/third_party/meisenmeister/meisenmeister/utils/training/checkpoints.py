from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def load_resume_checkpoint(
    *,
    last_checkpoint_path: Path,
    best_checkpoint_path: Path,
    log_fn,
) -> dict:
    if not last_checkpoint_path.is_file() and not best_checkpoint_path.is_file():
        raise FileNotFoundError(
            "Cannot continue training because neither model_last.pt nor model_best.pt exists "
            f"in {last_checkpoint_path.parent}"
        )

    candidate_paths = [last_checkpoint_path, best_checkpoint_path]
    errors: list[str] = []
    for index, candidate_path in enumerate(candidate_paths):
        if not candidate_path.is_file():
            continue
        try:
            checkpoint = torch.load(
                candidate_path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as error:
            errors.append(f"{candidate_path}: {error}")
            continue

        if index == 1:
            log_fn(
                f"WARNING: model_last.pt is unavailable or unreadable; resuming from fallback checkpoint {candidate_path}"
            )
        return checkpoint

    error_message = "\n".join(errors) if errors else "<no readable checkpoints>"
    raise RuntimeError(
        "Cannot continue training because no readable checkpoint was found.\n"
        f"{error_message}"
    )


def validate_resume_state(
    checkpoint: dict,
    *,
    dataset_id: str,
    dataset_name: str,
    fold: int | str,
    trainer_name: str,
    architecture_name: str,
    experiment_postfix: str | None,
    architecture_kwargs: dict | None = None,
) -> None:
    config = checkpoint["trainer_config"]
    expected = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "fold": fold,
        "trainer_name": trainer_name,
        "architecture_name": architecture_name,
        "experiment_postfix": experiment_postfix,
    }
    for key, expected_value in expected.items():
        actual_value = config.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Resume checkpoint mismatch for '{key}': expected {expected_value!r}, got {actual_value!r}"
            )
    if architecture_kwargs is not None:
        actual_architecture_kwargs = config.get("architecture_kwargs", {})
        if actual_architecture_kwargs != architecture_kwargs:
            raise ValueError(
                "Resume checkpoint mismatch for 'architecture_kwargs': "
                f"expected {architecture_kwargs!r}, got {actual_architecture_kwargs!r}"
            )


def restore_checkpoint_payload(trainer, checkpoint: dict) -> None:
    trainer.get_architecture().load_state_dict(checkpoint["model_state_dict"])
    trainer.get_optimizer().load_state_dict(checkpoint["optimizer_state_dict"])
    trainer.get_scheduler().load_state_dict(checkpoint["scheduler_state_dict"])
    grad_scaler_state_dict = checkpoint.get("grad_scaler_state_dict")
    if trainer.grad_scaler is not None and grad_scaler_state_dict is not None:
        trainer.grad_scaler.load_state_dict(grad_scaler_state_dict)
    trainer._history = checkpoint["history"]
    trainer._best_state = checkpoint["best_state"]
    restore_rng_state(checkpoint.get("rng_state"))


def restore_rng_state(rng_state: dict | None) -> None:
    if rng_state is None:
        return
    random_state = rng_state.get("python")
    if random_state is not None:
        random.setstate(random_state)
    numpy_state = rng_state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    torch_state = rng_state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_state = rng_state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(
    *,
    path: Path,
    epoch_idx: int,
    trainer_config: dict,
    history: dict,
    best_state: dict,
    model_state_dict: dict,
    optimizer_state_dict: dict,
    scheduler_state_dict: dict,
    grad_scaler_state_dict: dict | None = None,
) -> None:
    checkpoint = {
        "trainer_config": trainer_config,
        "last_completed_epoch": int(epoch_idx),
        "history": history,
        "best_state": best_state,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "grad_scaler_state_dict": grad_scaler_state_dict,
        "rng_state": capture_rng_state(),
    }
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def capture_rng_state() -> dict:
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    return rng_state


def build_trainer_config(
    *,
    dataset_id: str,
    dataset_name: str,
    fold: int | str,
    trainer_name: str,
    architecture_name: str,
    experiment_postfix: str | None,
    source_weights_path: str | None,
    results_dir: Path,
    experiment_dir: Path,
    fold_dir: Path,
    num_epochs: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    initial_lr: float,
    weight_decay: float,
    device: torch.device,
    architecture,
) -> dict:
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "fold": fold,
        "trainer_name": trainer_name,
        "architecture_name": architecture_name,
        "experiment_postfix": experiment_postfix,
        "source_weights_path": source_weights_path,
        "results_dir": str(results_dir),
        "experiment_dir": str(experiment_dir),
        "fold_dir": str(fold_dir),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "initial_lr": initial_lr,
        "weight_decay": weight_decay,
        "device": str(device),
        "num_classes": getattr(architecture, "num_classes", None),
        "in_channels": getattr(architecture, "in_channels", None),
        "architecture_kwargs": getattr(architecture, "get_init_kwargs", lambda: {})(),
    }
