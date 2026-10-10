from pathlib import Path

import torch

from meisenmeister.training.registry import get_trainer_class
from meisenmeister.training.splits import get_fold_sample_ids
from meisenmeister.utils import (
    build_experiment_paths,
    find_dataset_dir,
    log_message,
    maybe_compile_model,
    run_final_validation_evaluation,
    verify_required_global_paths_set,
)


def train(
    d: int,
    fold: int | str,
    trainer_name: str = "mmTrainer",
    num_workers: int | None = None,
    continue_training: bool = False,
    weights_path: str | None = None,
    experiment_postfix: str | None = None,
    val: str | None = None,
    compile_enabled: bool = True,
    grad_cam_enabled: bool = False,
) -> None:
    if not 0 <= d <= 999:
        raise ValueError(f"Dataset id must be between 0 and 999, got {d}")
    if isinstance(fold, int):
        if fold < 0:
            raise ValueError(f"Fold must be non-negative, got {fold}")
    elif fold != "all":
        raise ValueError(f"Fold must be a non-negative integer or 'all', got {fold!r}")
    if continue_training and weights_path is not None:
        raise ValueError("Cannot use --continue-training and --weights together")
    if val is not None and val not in {"last", "best"}:
        raise ValueError(f"val must be one of ('last', 'best'), got {val!r}")
    if val is not None and continue_training:
        raise ValueError("Cannot use --val together with --continue-training")
    if val is not None and weights_path is not None:
        raise ValueError("Cannot use --val together with --weights")

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
        continue_training=continue_training,
        weights_path=None if weights_path is None else Path(weights_path),
        experiment_postfix=experiment_postfix,
        compile_enabled=compile_enabled,
        grad_cam_enabled=grad_cam_enabled,
    )
    if grad_cam_enabled:
        trainer.ensure_grad_cam_available()
    if val is None:
        trainer.fit()
        return

    experiment_paths = build_experiment_paths(
        results_dir=mm_results,
        dataset_name=dataset_dir.name,
        trainer_name=trainer_class.__name__,
        architecture_name=architecture_name,
        experiment_postfix=experiment_postfix,
        fold=fold,
    )
    checkpoint_path = (
        experiment_paths["last_checkpoint_path"]
        if val == "last"
        else experiment_paths["best_checkpoint_path"]
    )
    output_path = (
        experiment_paths["eval_last_path"]
        if val == "last"
        else experiment_paths["eval_best_path"]
    )
    experiment_paths["fold_dir"].mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    architecture = trainer.get_architecture()
    architecture.load_state_dict(checkpoint["model_state_dict"])
    architecture, compile_applied, compile_status_message = maybe_compile_model(
        architecture,
        device=trainer.device,
        enabled=getattr(trainer, "compile_enabled", True),
    )
    trainer._architecture = architecture
    trainer.compile_applied = compile_applied
    trainer.compile_status_message = compile_status_message
    log_message(
        f"Loaded {'last' if val == 'last' else 'best'} checkpoint from {checkpoint_path}",
        experiment_paths["log_path"],
    )
    log_message(
        f"Torch compile applied: {compile_applied} ({compile_status_message})",
        experiment_paths["log_path"],
    )
    run_final_validation_evaluation(
        trainer,
        output_path=output_path,
        log_path=experiment_paths["log_path"],
        n_bootstrap=getattr(trainer, "FINAL_EVAL_N_BOOTSTRAP", 2000),
        confidence_level=getattr(trainer, "FINAL_EVAL_CONFIDENCE_LEVEL", 0.95),
        seed=getattr(trainer, "FINAL_EVAL_BOOTSTRAP_SEED", 0),
        log_fn=log_message,
        grad_cam_output_dir=(
            experiment_paths["grad_cam_last_dir"]
            if grad_cam_enabled and val == "last"
            else (experiment_paths["grad_cam_best_dir"] if grad_cam_enabled else None)
        ),
        grad_cam_checkpoint_kind=val,
    )
