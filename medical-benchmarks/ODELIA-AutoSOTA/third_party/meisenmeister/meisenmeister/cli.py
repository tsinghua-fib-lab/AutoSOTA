import argparse

from meisenmeister.plan_and_preprocess.create_breast_seg import (
    create_breast_segmentations,
)
from meisenmeister.plan_and_preprocess.extract_dataset_fingerprint import (
    extract_dataset_fingerprint,
)
from meisenmeister.plan_and_preprocess.homogenize import homogenize
from meisenmeister.plan_and_preprocess.plan_and_preprocess import plan_and_preprocess
from meisenmeister.plan_and_preprocess.plan_experiment import plan_experiment
from meisenmeister.plan_and_preprocess.preprocess import preprocess
from meisenmeister.training.benchmark import benchmark_train
from meisenmeister.training.evaluate_predictions import evaluate_predictions
from meisenmeister.training.predict import predict, predict_from_modelfolder
from meisenmeister.training.preview_da import preview_da
from meisenmeister.training.splits import create_five_fold_splits
from meisenmeister.training.train import train
from meisenmeister.utils import (
    find_dataset_dir,
    require_global_paths_set,
    verify_required_global_paths_set,
)


def _parse_fold(value: str) -> int | str:
    if value == "all":
        return value
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid fold {value!r}. Use a non-negative integer or 'all'."
        ) from error


def mm_extract_dataset_fingerprint() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_extract_dataset_fingerprint",
        description="Run the extract_dataset_fingerprint entrypoint.",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads for ROI fingerprint extraction.",
    )
    args = parser.parse_args()
    extract_dataset_fingerprint(args.d, num_workers=args.num_workers)


def mm_create_breast_segmentations() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_create_breast_segmentations",
        description="Create breast segmentations in masksTr next to imagesTr.",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    args = parser.parse_args()
    create_breast_segmentations(args.d)


def mm_homogenize() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_homogenize",
        description="Resample all non-_0000 images to the _0000 space and overwrite the raw NIfTI files.",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    args = parser.parse_args()

    confirmation = input(
        "ARE YOU SURE YOU WANT TO OVERWRITE YOUR RAW DATA? Type 'yes' to continue: "
    ).strip()
    if confirmation != "yes":
        raise SystemExit("Aborted.")

    homogenize(args.d)


def mm_plan_experiment() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_plan_experiment",
        description="Create mmPlans.json from dataset_fingerprint.json.",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    args = parser.parse_args()
    plan_experiment(args.d)


def mm_preprocess() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_preprocess",
        description="Preprocess ROI crops into blosc2 b2nd format from mmPlans.json.",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads for preprocessing.",
    )
    args = parser.parse_args()
    preprocess(args.d, num_workers=args.num_workers)


def mm_plan_and_preprocess() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_plan_and_preprocess",
        description="Run planning and preprocessing for a dataset.",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads for preprocessing.",
    )
    args = parser.parse_args()
    plan_and_preprocess(args.d, num_workers=args.num_workers)


def mm_train() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_train",
        description="Resolve a trainer, build its training components, and call fit().",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=_parse_fold,
        required=True,
        help="Fold index from splits.json, or 'all' to use all samples for both train and val.",
    )
    parser.add_argument(
        "--trainer",
        default="mmTrainer",
        help="Trainer class name registered under meisenmeister.training.trainers.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of DataLoader worker processes. Defaults to a CPU-based heuristic capped at 8 workers.",
    )
    parser.add_argument(
        "-c",
        "--continue-training",
        action="store_true",
        help="Continue training from model_last.pt in the computed experiment fold directory.",
    )
    parser.add_argument(
        "-w",
        "--weights",
        help="Path to pretrained weights for starting a fresh fine-tuning run.",
    )
    parser.add_argument(
        "--postfix",
        help="Optional suffix appended to the experiment name.",
    )
    parser.add_argument(
        "--val",
        nargs="?",
        const="last",
        choices=("last", "best"),
        help="Skip training and run final validation only. `--val` uses model_last.pt and writes eval_last.json; `--val best` uses model_best.pt and writes eval_best.json.",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile for training.",
    )
    parser.add_argument(
        "--grad-cam",
        action="store_true",
        help="Export Grad-CAM++ masks during final validation only.",
    )
    args = parser.parse_args()
    train(
        args.d,
        fold=args.fold,
        trainer_name=args.trainer,
        num_workers=args.num_workers,
        continue_training=args.continue_training,
        weights_path=args.weights,
        experiment_postfix=args.postfix,
        val=args.val,
        compile_enabled=not args.disable_compile,
        grad_cam_enabled=args.grad_cam,
    )


def mm_preview_da() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_preview_da",
        description="Load the training dataset with a selected trainer's DA pipeline and write a preview PNG.",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        required=True,
        help="Fold index from splits.json.",
    )
    parser.add_argument(
        "--trainer",
        default="mmTrainer",
        help="Trainer class name whose training dataset and augmentation pipeline should be used.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Optional worker-count setting passed into the selected trainer.",
    )
    parser.add_argument(
        "--postfix",
        help="Optional suffix appended to the experiment name when resolving the default output path.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of random training samples to visualize.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output PNG path. Defaults to the trainer fold directory da_preview.png.",
    )
    args = parser.parse_args()
    output_path = preview_da(
        args.d,
        fold=args.fold,
        trainer_name=args.trainer,
        num_workers=args.num_workers,
        experiment_postfix=args.postfix,
        num_samples=args.num_samples,
        output_path=args.output,
    )
    print(output_path)


def mm_benchmark_train() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_benchmark_train",
        description="Benchmark the current training stack with warmup and steady-state timing.",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        required=True,
        help="Fold index from splits.json.",
    )
    parser.add_argument(
        "--trainer",
        default="mmTrainer",
        help="Trainer class name registered under meisenmeister.training.trainers.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of DataLoader worker processes. Defaults to a CPU-based heuristic capped at 8 workers.",
    )
    parser.add_argument(
        "--postfix",
        help="Optional suffix appended to the experiment name.",
    )
    parser.add_argument(
        "--train-warmup-steps",
        type=int,
        default=3,
        help="Number of train iterations to exclude as warmup.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=10,
        help="Number of steady-state train iterations to measure.",
    )
    parser.add_argument(
        "--val-warmup-steps",
        type=int,
        default=2,
        help="Number of validation iterations to exclude as warmup.",
    )
    parser.add_argument(
        "--val-steps",
        type=int,
        default=5,
        help="Number of steady-state validation iterations to measure.",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile for benchmarking.",
    )
    args = parser.parse_args()
    benchmark_train(
        args.d,
        fold=args.fold,
        trainer_name=args.trainer,
        num_workers=args.num_workers,
        experiment_postfix=args.postfix,
        compile_enabled=not args.disable_compile,
        train_warmup_steps=args.train_warmup_steps,
        train_steps=args.train_steps,
        val_warmup_steps=args.val_warmup_steps,
        val_steps=args.val_steps,
    )


def mm_predict() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_predict",
        description="Run ROI-level inference on one or more cases from an imagesTr-style input folder.",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="Input folder containing one or more cases in imagesTr naming format.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output folder for masks and predictions.json.",
    )
    parser.add_argument(
        "-f",
        "--folds",
        type=_parse_fold,
        nargs="+",
        required=True,
        help="One or more fold indices to use for ensembling, or 'all' to use fold_all.",
    )
    parser.add_argument(
        "--trainer",
        default="mmTrainer",
        help="Trainer class name registered under meisenmeister.training.trainers.",
    )
    parser.add_argument(
        "--postfix",
        help="Optional suffix appended to the experiment name.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        choices=("best", "last"),
        help="Checkpoint variant to load for each fold.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable default flip test-time augmentation.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile for inference.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of CPU worker threads for ROI preparation during inference.",
    )
    parser.add_argument(
        "--concise-output",
        help="Optional path for a concise single-case ROI probability JSON derived from predictions.json.",
    )
    args = parser.parse_args()
    predict(
        args.d,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        folds=args.folds,
        trainer_name=args.trainer,
        experiment_postfix=args.postfix,
        checkpoint=args.checkpoint,
        use_tta=not args.no_tta,
        compile_model=not args.no_compile,
        num_workers=args.num_workers,
        concise_output_path=args.concise_output,
    )


def mm_predict_from_modelfolder() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_predict_from_modelfolder",
        description="Run ROI-level inference from a portable experiment folder containing checkpoints, dataset.json, and mmPlans.json.",
    )
    parser.add_argument(
        "model_folder",
        help="Experiment folder containing dataset.json, mmPlans.json, and fold_<n> checkpoints.",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="Input folder containing one or more cases in imagesTr naming format.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output folder for masks and predictions.json.",
    )
    parser.add_argument(
        "-f",
        "--folds",
        type=_parse_fold,
        nargs="+",
        required=True,
        help="One or more fold indices to use for ensembling, or 'all' to use fold_all.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        choices=("best", "last"),
        help="Checkpoint variant to load for each fold.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable default flip test-time augmentation.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile for inference.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of CPU worker threads for ROI preparation during inference.",
    )
    parser.add_argument(
        "--concise-output",
        help="Optional path for a concise single-case ROI probability JSON derived from predictions.json.",
    )
    args = parser.parse_args()
    predict_from_modelfolder(
        args.model_folder,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        folds=args.folds,
        checkpoint=args.checkpoint,
        use_tta=not args.no_tta,
        compile_model=not args.no_compile,
        num_workers=args.num_workers,
        concise_output_path=args.concise_output,
    )


def mm_evaluate_predictions() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_evaluate_predictions",
        description="Evaluate a 3-class predictions.json file against a labelsTr.json-style target mapping.",
    )
    parser.add_argument(
        "-t",
        "--targets",
        required=True,
        help="Target JSON path with labelsTr.json semantics keyed by ROI sample id.",
    )
    parser.add_argument(
        "-p",
        "--predictions",
        required=True,
        help="Path to predictions.json generated by mm_predict or mm_predict_from_modelfolder.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output directory or evaluation JSON path. Defaults to predictions.json's directory.",
    )
    args = parser.parse_args()
    evaluate_predictions(
        targets_path=args.targets,
        predictions_path=args.predictions,
        output_path=args.output,
    )


@require_global_paths_set
def mm_create_5fold() -> None:
    parser = argparse.ArgumentParser(
        prog="mm_create_5fold",
        description="Create a leakage-safe 5-fold splits.json in the preprocessed dataset folder.",
    )
    parser.add_argument(
        "-d",
        type=int,
        required=True,
        help="Integer dataset identifier.",
    )
    args = parser.parse_args()

    if not 0 <= args.d <= 999:
        raise ValueError(f"Dataset id must be between 0 and 999, got {args.d}")

    dataset_id = f"{args.d:03d}"
    paths = verify_required_global_paths_set()
    dataset_dir = find_dataset_dir(paths["mm_raw"], dataset_id)
    preprocessed_dataset_dir = paths["mm_preprocessed"] / dataset_dir.name
    output_path = create_five_fold_splits(preprocessed_dataset_dir)
    print(output_path)


if __name__ == "__main__":
    mm_extract_dataset_fingerprint()
