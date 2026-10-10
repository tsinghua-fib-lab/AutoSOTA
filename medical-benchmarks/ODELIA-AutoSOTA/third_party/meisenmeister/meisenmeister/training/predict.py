from __future__ import annotations

import json
import tempfile
from pathlib import Path

from meisenmeister.plan_and_preprocess.preprocessing_utils import load_mm_plans
from meisenmeister.training.prediction_pipeline import (
    get_experiment_metadata,
    load_fold_predictors,
    normalize_prediction_folds,
    resolve_trainer_architecture_name,
    run_prediction,
)
from meisenmeister.utils.file_utils import load_dataset_json
from meisenmeister.utils.path_utils import (
    find_dataset_dir,
    require_global_paths_set,
    verify_required_global_paths_set,
)
from meisenmeister.utils.prediction_inference import (
    load_fold_predictors_from_experiment_dir,
)
from meisenmeister.utils.prediction_utils import (
    build_prediction_dataset_json,
    resolve_prediction_file_ending_from_paths,
    stage_prediction_case_file,
)
from meisenmeister.utils.training.artifacts import build_experiment_paths


def predict_case_from_files(
    model_folder: str,
    pre_path: str,
    post1_path: str,
    post2_path: str,
    *,
    folds: list[int | str],
    checkpoint: str = "best",
    use_tta: bool = True,
    compile_model: bool = True,
    num_workers: int = 8,
) -> dict:
    experiment_dir = Path(model_folder)
    dataset_json = load_dataset_json(experiment_dir, resolve_training_cases=False)
    case_id = "case_000"
    source_paths = [
        Path(path).expanduser().resolve()
        for path in (pre_path, post1_path, post2_path)
    ]
    expected_num_channels = len(dataset_json["channel_names"])
    if len(source_paths) != expected_num_channels:
        raise ValueError(
            "Single-case prediction requires exactly "
            f"{expected_num_channels} input files, got {len(source_paths)}"
        )
    missing_paths = [str(path) for path in source_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Single-case prediction input files not found: "
            + ", ".join(missing_paths)
        )
    input_file_ending = resolve_prediction_file_ending_from_paths(source_paths)
    run_root = Path(tempfile.mkdtemp(prefix=".mm_predict_case_"))
    input_dir = run_root / "input"
    output_dir = run_root / "output"
    concise_path = run_root / "concise_predictions.json"
    input_dir.mkdir()
    output_dir.mkdir()

    for channel_index, source in enumerate(source_paths):
        staged_path = input_dir / f"{case_id}_{channel_index:04d}{input_file_ending}"
        stage_prediction_case_file(source, staged_path)

    predictions_path = predict_from_modelfolder(
        str(experiment_dir),
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        folds=folds,
        checkpoint=checkpoint,
        use_tta=use_tta,
        compile_model=compile_model,
        num_workers=num_workers,
        concise_output_path=str(concise_path),
    )

    output_mask_path = output_dir / f"{case_id}_breast_mask{input_file_ending}"
    if not output_mask_path.is_file():
        raise FileNotFoundError(
            f"Expected breast mask output missing: {output_mask_path}"
        )

    return {
        "mask_path": str(output_mask_path),
        "predictions_path": str(predictions_path),
        "concise_predictions": json.loads(concise_path.read_text(encoding="utf-8")),
        "case_id": case_id,
        "run_directory": str(run_root),
    }


@require_global_paths_set
def predict(
    d: int,
    input_dir: str,
    output_dir: str,
    folds: list[int | str],
    trainer_name: str = "mmTrainer",
    experiment_postfix: str | None = None,
    checkpoint: str = "best",
    use_tta: bool = True,
    compile_model: bool = True,
    num_workers: int = 8,
    concise_output_path: str | None = None,
) -> Path:
    if not 0 <= d <= 999:
        raise ValueError(f"Dataset id must be between 0 and 999, got {d}")
    if checkpoint not in {"best", "last"}:
        raise ValueError(
            f"checkpoint must be one of ('best', 'last'), got {checkpoint!r}"
        )

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    dataset_id = f"{d:03d}"
    paths = verify_required_global_paths_set()
    dataset_dir = find_dataset_dir(paths["mm_raw"], dataset_id)
    preprocessed_dataset_dir = paths["mm_preprocessed"] / dataset_dir.name
    results_dir = paths["mm_results"]
    architecture_name = resolve_trainer_architecture_name(trainer_name)
    dataset_json = build_prediction_dataset_json(
        input_path,
        load_dataset_json(dataset_dir),
    )
    plans = load_mm_plans(preprocessed_dataset_dir / "mmPlans.json")
    experiment_dir = build_experiment_paths(
        results_dir=results_dir,
        dataset_name=dataset_dir.name,
        trainer_name=trainer_name,
        architecture_name=architecture_name,
        experiment_postfix=experiment_postfix,
        fold=0,
    )["experiment_dir"]
    fold_values = normalize_prediction_folds(folds, experiment_dir=experiment_dir)
    fold_predictors = load_fold_predictors(
        dataset_dir=dataset_dir,
        results_dir=results_dir,
        trainer_name=trainer_name,
        architecture_name=architecture_name,
        experiment_postfix=experiment_postfix,
        folds=fold_values,
        checkpoint=checkpoint,
        compile_model=compile_model,
    )
    return run_prediction(
        dataset_id=dataset_id,
        dataset_name=dataset_dir.name,
        input_path=input_path,
        output_path=output_path,
        dataset_json=dataset_json,
        plans=plans,
        fold_predictors=fold_predictors,
        trainer_name=trainer_name,
        architecture_name=architecture_name,
        experiment_postfix=experiment_postfix,
        folds=fold_values,
        checkpoint=checkpoint,
        use_tta=use_tta,
        num_workers=num_workers,
        concise_output_path=concise_output_path,
    )


def predict_from_modelfolder(
    model_folder: str,
    input_dir: str,
    output_dir: str,
    folds: list[int | str],
    checkpoint: str = "best",
    use_tta: bool = True,
    compile_model: bool = True,
    num_workers: int = 8,
    concise_output_path: str | None = None,
) -> Path:
    if checkpoint not in {"best", "last"}:
        raise ValueError(
            f"checkpoint must be one of ('best', 'last'), got {checkpoint!r}"
        )

    experiment_dir = Path(model_folder)
    fold_values = normalize_prediction_folds(folds, experiment_dir=experiment_dir)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    dataset_json = build_prediction_dataset_json(
        input_path,
        load_dataset_json(experiment_dir, resolve_training_cases=False),
    )
    plans = load_mm_plans(experiment_dir / "mmPlans.json")
    metadata = get_experiment_metadata(experiment_dir, fold_values, checkpoint)
    fold_predictors = load_fold_predictors_from_experiment_dir(
        experiment_dir=experiment_dir,
        architecture_name=metadata["architecture_name"],
        folds=fold_values,
        checkpoint=checkpoint,
        compile_model=compile_model,
    )
    return run_prediction(
        dataset_id=metadata["dataset_id"],
        dataset_name=metadata["dataset_name"],
        input_path=input_path,
        output_path=output_path,
        dataset_json=dataset_json,
        plans=plans,
        fold_predictors=fold_predictors,
        trainer_name=metadata["trainer_name"],
        architecture_name=metadata["architecture_name"],
        experiment_postfix=metadata["experiment_postfix"],
        folds=fold_values,
        checkpoint=checkpoint,
        use_tta=use_tta,
        num_workers=num_workers,
        concise_output_path=concise_output_path,
    )
