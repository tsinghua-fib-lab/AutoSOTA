from __future__ import annotations

import random
from pathlib import Path

from meisenmeister.training.registry import get_trainer_class
from meisenmeister.training.splits import get_fold_sample_ids
from meisenmeister.utils import (
    build_experiment_paths,
    find_dataset_dir,
    save_da_preview,
    verify_required_global_paths_set,
)


def preview_da(
    d: int,
    fold: int,
    trainer_name: str = "mmTrainer",
    num_workers: int | None = None,
    experiment_postfix: str | None = None,
    num_samples: int = 3,
    output_path: str | None = None,
) -> Path:
    if not 0 <= d <= 999:
        raise ValueError(f"Dataset id must be between 0 and 999, got {d}")
    if fold < 0:
        raise ValueError(f"Fold must be non-negative, got {fold}")
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}")

    dataset_id = f"{d:03d}"
    paths = verify_required_global_paths_set(("mm_preprocessed", "mm_results"))
    mm_preprocessed = paths["mm_preprocessed"]
    mm_results = paths["mm_results"]

    preprocessed_dataset_dir = find_dataset_dir(mm_preprocessed, dataset_id)
    dataset_dir = preprocessed_dataset_dir

    get_fold_sample_ids(preprocessed_dataset_dir, fold)
    trainer_class = get_trainer_class(trainer_name)
    architecture_name = getattr(trainer_class, "ARCHITECTURE_NAME", "ResNet3D18")
    trainer = trainer_class(
        dataset_id=dataset_id,
        fold=fold,
        dataset_dir=dataset_dir,
        preprocessed_dataset_dir=preprocessed_dataset_dir,
        results_dir=mm_results,
        architecture_name=architecture_name,
        num_workers=num_workers,
        continue_training=False,
        weights_path=None,
        experiment_postfix=experiment_postfix,
        compile_enabled=False,
    )

    experiment_paths = build_experiment_paths(
        results_dir=mm_results,
        dataset_name=dataset_dir.name,
        trainer_name=trainer_class.__name__,
        architecture_name=architecture_name,
        experiment_postfix=experiment_postfix,
        fold=fold,
    )
    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else experiment_paths["da_preview_path"]
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    train_dataset = trainer.get_train_dataset()
    num_available_samples = len(train_dataset)
    if num_available_samples == 0:
        raise ValueError("Training dataset is empty; cannot generate DA preview")

    preview_indices = random.sample(
        range(num_available_samples),
        k=min(num_samples, num_available_samples),
    )
    preview_samples = [train_dataset[index] for index in preview_indices]
    save_da_preview(preview_samples, resolved_output_path)
    return resolved_output_path
