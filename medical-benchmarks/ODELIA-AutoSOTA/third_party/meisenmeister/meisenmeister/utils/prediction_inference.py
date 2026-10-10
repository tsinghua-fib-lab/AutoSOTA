from __future__ import annotations

from pathlib import Path

import torch

from meisenmeister.architectures import get_architecture_class
from meisenmeister.utils.training.performance import maybe_compile_model


def resolve_checkpoint_path(*, fold_dir: Path, checkpoint: str) -> Path:
    return fold_dir / ("model_best.pt" if checkpoint == "best" else "model_last.pt")


def load_checkpoint_payload(checkpoint_path: Path) -> dict:
    return torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


def load_model_from_checkpoint_payload(
    *,
    checkpoint_payload: dict,
    architecture_name: str,
    device: torch.device,
    compile_model: bool,
) -> tuple[torch.nn.Module, bool, str]:
    trainer_config = checkpoint_payload.get("trainer_config", {})
    in_channels = trainer_config.get("in_channels")
    num_classes = trainer_config.get("num_classes")
    if in_channels is None or num_classes is None:
        raise ValueError(
            "Checkpoint is missing architecture metadata required for portable inference"
        )
    architecture_kwargs = trainer_config.get("architecture_kwargs", {})
    architecture_class = get_architecture_class(architecture_name)
    model = architecture_class(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        **architecture_kwargs,
    ).to(device)
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.eval()
    model, compile_applied, compile_status_message = maybe_compile_model(
        model,
        device=device,
        enabled=compile_model,
    )
    return model, compile_applied, compile_status_message


def load_fold_predictors_from_experiment_dir(
    *,
    experiment_dir: Path,
    architecture_name: str,
    folds: list[int | str],
    checkpoint: str,
    compile_model: bool,
) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictors = []
    for fold in folds:
        fold_dir = experiment_dir / f"fold_{fold}"
        checkpoint_path = resolve_checkpoint_path(
            fold_dir=fold_dir,
            checkpoint=checkpoint,
        )
        checkpoint_payload = load_checkpoint_payload(checkpoint_path)
        model, compile_applied, compile_status_message = (
            load_model_from_checkpoint_payload(
                checkpoint_payload=checkpoint_payload,
                architecture_name=architecture_name,
                device=device,
                compile_model=compile_model,
            )
        )
        predictors.append(
            {
                "fold": fold,
                "device": device,
                "checkpoint_path": str(checkpoint_path),
                "compile_applied": compile_applied,
                "compile_status_message": compile_status_message,
                "model": model,
            }
        )
    return predictors
