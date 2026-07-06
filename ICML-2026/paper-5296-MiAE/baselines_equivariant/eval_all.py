import argparse
import statistics
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=(
        "The TorchScript type system doesn't support instance-level annotations "
        "on empty non-base types in `__init__`.*"
    ),
    category=UserWarning,
)

import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch_geometric.loader import DataLoader as PygDataLoader

from data import resolve_existing_path
from formatting import format_table
from test_model import (
    SCRIPT_DIR,
    build_model,
    device,
    extract_state_dict,
    infer_num_classes,
    load_model_weights,
    load_or_build_test_graphs,
    run_inference,
)


MODEL_NAMES = ("gotennet", "bige3nn", "mace")
METRIC_KEYS = ("test_loss", "accuracy", "balanced_accuracy", "f1_macro")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_NAMES),
        choices=MODEL_NAMES,
        help="Models to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Evaluate multiple seeds and report per-model mean/std.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=".",
        help="Directory containing best_model_<model>_seed<seed>.pt files.",
    )
    parser.add_argument("--gotennet_checkpoint", type=str, default=None)
    parser.add_argument("--bige3nn_checkpoint", type=str, default=None)
    parser.add_argument("--mace_checkpoint", type=str, default=None)

    parser.add_argument("--dataset_root", type=str, default="./datasets/afdb_FS_plddt80")
    parser.add_argument("--dataset_name", type=str, default="ted")
    parser.add_argument("--dataset_batch_size", type=int, default=512)
    parser.add_argument(
        "--tedbench_path",
        type=str,
        default=None,
        help="Optional path containing the tedbench package for raw TED preprocessing.",
    )
    parser.add_argument("--test_graphs_path", type=str, default="test_graphs.pt")
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--feature_mode", type=str, default="constant", choices=["constant", "token"])
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--n_classes", type=int, default=None)
    parser.add_argument("--allow_partial_load", action="store_true")

    parser.add_argument("--external_dataset_root", type=str, default=None)
    parser.add_argument("--external_dataset_name", type=str, default="cath4.4")
    parser.add_argument("--external_split", type=str, default="test")
    parser.add_argument("--fail_fast", action="store_true")

    return parser.parse_args()


def get_seeds(args):
    return args.seeds if args.seeds is not None else [args.seed]


def default_checkpoint_path(model_name, seed, args):
    override = getattr(args, f"{model_name}_checkpoint")
    if override:
        return override
    return str(Path(args.checkpoint_dir) / f"best_model_{model_name}_seed{seed}.pt")


def resolve_checkpoint(model_name, seed, args):
    checkpoint_path = resolve_existing_path(default_checkpoint_path(model_name, seed, args), SCRIPT_DIR)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for {model_name} seed {seed}: {checkpoint_path}. "
            f"Pass --{model_name}_checkpoint explicitly if it is elsewhere."
        )
    return checkpoint_path


def evaluate_one(model_name, seed, checkpoint_path, test_loader, F_in, args):
    print(f"\n=== {model_name} seed {seed} ===")
    print("Checkpoint:", checkpoint_path)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(state)
    n_classes = args.n_classes or infer_num_classes(state_dict)
    if n_classes is None:
        raise ValueError(f"Could not infer n_classes for {model_name}; pass --n_classes.")

    model = build_model(model_name, F_in, n_classes, args.cutoff, state_dict).to(device)
    load_model_weights(
        model,
        checkpoint_path,
        state=state,
        allow_partial=args.allow_partial_load,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred = run_inference(model, test_loader, criterion)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    metrics = {
        "model": model_name,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "test_loss": test_loss,
        "accuracy": test_acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1,
    }
    print(
        "Result: "
        f"loss={test_loss:.4f}, "
        f"acc={test_acc:.4f}, "
        f"bal_acc={bal_acc:.4f}, "
        f"f1_macro={f1:.4f}"
    )
    return metrics


def print_summary(results):
    if not results:
        return

    print("\nSummary")
    print(
        format_table(
            ["model", "seed", "loss", "acc", "bal_acc", "f1_macro"],
            [
                [
                    row["model"],
                    row["seed"],
                    f"{row['test_loss']:.4f}",
                    f"{row['accuracy']:.4f}",
                    f"{row['balanced_accuracy']:.4f}",
                    f"{row['f1_macro']:.4f}",
                ]
                for row in results
            ],
            right_align={1, 2, 3, 4, 5},
        )
    )


def mean_std(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def print_mean_std(results):
    if not results:
        return

    print("\nMean/std by model")
    rows_out = []
    for model_name in MODEL_NAMES:
        rows = [row for row in results if row["model"] == model_name]
        if not rows:
            continue

        values = {}
        for key in METRIC_KEYS:
            values[f"{key}_mean"], values[f"{key}_std"] = mean_std([row[key] for row in rows])

        rows_out.append(
            [
                model_name,
                len(rows),
                f"{values['test_loss_mean']:.4f} +/- {values['test_loss_std']:.4f}",
                f"{values['accuracy_mean']:.4f} +/- {values['accuracy_std']:.4f}",
                f"{values['balanced_accuracy_mean']:.4f} +/- {values['balanced_accuracy_std']:.4f}",
                f"{values['f1_macro_mean']:.4f} +/- {values['f1_macro_std']:.4f}",
            ]
        )

    print(
        format_table(
            ["model", "n", "loss", "acc", "bal_acc", "f1_macro"],
            rows_out,
            right_align={1, 2, 3, 4, 5},
        )
    )


def main():
    args = parse_args()
    seeds = get_seeds(args)
    if len(seeds) > 1:
        overridden = [model for model in MODEL_NAMES if getattr(args, f"{model}_checkpoint")]
        if overridden:
            raise ValueError(
                "Per-model checkpoint overrides are only supported for single-seed eval. "
                f"Remove overrides for multi-seed eval: {', '.join(overridden)}"
            )

    test_graphs = load_or_build_test_graphs(args)
    test_loader = PygDataLoader(
        test_graphs,
        batch_size=args.test_batch_size,
        shuffle=False,
    )
    F_in = test_graphs[0].x.shape[1]

    results = []
    failures = []
    for model_name in args.models:
        for seed in seeds:
            try:
                checkpoint_path = resolve_checkpoint(model_name, seed, args)
                results.append(evaluate_one(model_name, seed, checkpoint_path, test_loader, F_in, args))
            except Exception as exc:
                failures.append((model_name, seed, exc))
                print(f"\nFAILED {model_name} seed {seed}: {exc}")
                if args.fail_fast:
                    raise

    print_summary(results)
    print_mean_std(results)

    if failures:
        print("\nFailures")
        for model_name, seed, exc in failures:
            print(f"{model_name} seed {seed}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
