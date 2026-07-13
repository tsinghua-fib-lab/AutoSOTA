import argparse
import os
import pickle
import random
import time
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from jaxcld.models.asr_model import ASRModel


class RunResults(NamedTuple):
    estimator: str
    train_acc: float
    eval_acc: float
    eval_split: str
    n_train: int
    n_eval: int
    n_classes: int
    model_path: str
    csv_path: str
    log_path: str
    training_time_s: float


def _maybe_init_wandb(enable: bool, run_name: str, config: dict):
    if not enable:
        return None
    try:
        import wandb  # type: ignore
    except Exception:
        print("wandb not available; continuing without wandb logging.")
        return None
    wandb.init(project="CLD", name=run_name, config=config)
    return wandb


def _load_split_embeddings(
    asr_model: ASRModel,
    data_dir: str,
    dataset_split: str,
    data_seed: int,
    shuffle: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """
    Returns:
        A: (N, D) float array of pooled encoder embeddings
        y: (N,) int labels for language index
        n_classes: optional int (returned by MMS backend)
    """
    out = asr_model.load_data(
        data_dir,
        data_seed=data_seed,
        dataset_split=dataset_split,
        shuffle=shuffle,
    )
    if isinstance(out, tuple) and len(out) == 2:
        A, y = out
        return np.asarray(A), np.asarray(y), None
    if isinstance(out, tuple) and len(out) == 3:
        A, y, n_classes = out
        return np.asarray(A), np.asarray(y), int(n_classes)
    raise ValueError(f"Unexpected return from ASRModel.load_data: type={type(out)} value={out}")


def run(
    model_name: str,
    data_dir: str,
    output_dir: str,
    data_seed: int,
    eval_split: str,
    estimator: str,
    c: float,
    max_iter: int,
    class_weight: Optional[str],
    dual: str,
    kernel: str,
    gamma: str,
    degree: int,
    n_neighbors: int,
    knn_weights: str,
    knn_metric: str,
    wandb_enable: bool,
    languages: List[str],
) -> RunResults:
    # Lazy import sklearn so importing this module doesn't require it.
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC, SVC
    from sklearn.metrics import accuracy_score

    asr_model = ASRModel.from_pretrained(model_name, config={"languages": languages})

    print(f"Loading embeddings from {data_dir} (train split)...")
    X_train, y_train, n_classes_train = _load_split_embeddings(
        asr_model=asr_model,
        data_dir=data_dir,
        dataset_split="train",
        data_seed=data_seed,
        shuffle=True,
    )

    print(f"Loading embeddings from {data_dir} (eval split: {eval_split})...")
    X_eval, y_eval, n_classes_eval = _load_split_embeddings(
        asr_model=asr_model,
        data_dir=data_dir,
        dataset_split=eval_split,
        data_seed=data_seed,
        shuffle=False,
    )

    n_classes = int(
        n_classes_train
        or n_classes_eval
        or len(np.unique(np.concatenate([np.asarray(y_train), np.asarray(y_eval)], axis=0)))
    )

    if estimator not in {"linear_svm", "kernel_svm", "knn"}:
        raise ValueError("--estimator must be one of: linear_svm, kernel_svm, knn")

    dual_value = None
    if estimator == "linear_svm":
        if dual not in {"auto", "true", "false"}:
            raise ValueError("--dual must be one of: auto, true, false")
        dual_value = "auto" if dual == "auto" else (dual == "true")

    gamma_value: object = gamma
    if estimator == "kernel_svm" and gamma not in {"scale", "auto"}:
        gamma_value = float(gamma)

    steps = [("scaler", StandardScaler())]
    if estimator == "linear_svm":
        steps.append(
            (
                "svm",
                LinearSVC(
                    C=float(c),
                    max_iter=int(max_iter),
                    class_weight=class_weight,
                    dual=dual_value,
                    random_state=int(data_seed),
                ),
            )
        )
    elif estimator == "kernel_svm":
        steps.append(
            (
                "svm",
                SVC(
                    C=float(c),
                    kernel=kernel,
                    gamma=gamma_value,
                    degree=int(degree),
                    class_weight=class_weight,
                    random_state=int(data_seed),
                ),
            )
        )
    else:
        steps.append(
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=int(n_neighbors),
                    weights=knn_weights,
                    metric=knn_metric,
                ),
            )
        )

    clf = Pipeline(steps=steps)

    config = {
        "model_name": model_name,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "data_seed": data_seed,
        "eval_split": eval_split,
        "estimator": estimator,
        "C": c,
        "max_iter": max_iter,
        "class_weight": class_weight,
        "dual": dual,
        "kernel": kernel,
        "gamma": gamma,
        "degree": degree,
        "n_neighbors": n_neighbors,
        "knn_weights": knn_weights,
        "knn_metric": knn_metric,
        "n_train": int(X_train.shape[0]),
        "n_eval": int(X_eval.shape[0]),
        "n_classes": n_classes,
        "embedding_dim": int(X_train.shape[1]) if X_train.ndim == 2 else None,
    }

    wandb = _maybe_init_wandb(wandb_enable, run_name=f"{estimator}_{model_name}", config=config)

    if estimator == "linear_svm":
        print("Training Linear SVM (sklearn LinearSVC) ...")
    elif estimator == "kernel_svm":
        print(f"Training Kernel SVM (sklearn SVC, kernel={kernel}) ...")
    else:
        print("Training KNN (sklearn KNeighborsClassifier) ...")
    start = time.time()
    clf.fit(X_train, y_train)
    training_time_s = time.time() - start

    yhat_train = clf.predict(X_train)
    yhat_eval = clf.predict(X_eval)

    train_acc = float(accuracy_score(y_train, yhat_train))
    eval_acc = float(accuracy_score(y_eval, yhat_eval))

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"{eval_split} accuracy: {eval_acc:.4f}")

    # Prepare output directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    # NOTE: model_name may include slashes (e.g. "openai/whisper-small"). We want to keep
    # those slashes for directories (model_dir), but sanitize them in the filename only.
    safe_model_name = model_name.replace("/", "_")
    model_path = os.path.join(model_dir, f"{safe_model_name}_{estimator}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    # Save metrics CSV (tab-separated to match train_cvxnn.py)
    csv_path = os.path.join(model_dir, f"{estimator}_metrics.csv")
    metrics_df = pd.DataFrame(
        {
            "train_acc": [train_acc],
            "eval_acc": [eval_acc],
            "eval_split": [eval_split],
            "n_train": [int(X_train.shape[0])],
            "n_eval": [int(X_eval.shape[0])],
            "n_classes": [n_classes],
            "embedding_dim": [int(X_train.shape[1]) if X_train.ndim == 2 else None],
            "training_time_s": [training_time_s],
            "model_path": [model_path],
        }
    )
    metrics_df.to_csv(csv_path, sep="\t", encoding="utf-8", index=False, header=True)

    # Save summary text file
    log_path = os.path.join(model_dir, f"{estimator}_results.txt")
    with open(log_path, "w") as f:
        for k, v in metrics_df.iloc[0].to_dict().items():
            f.write(f"{k}: {v}\n")

    if wandb is not None:
        try:
            wandb.log(
                {
                    "train_acc": train_acc,
                    "eval_acc": eval_acc,
                    "training_time_s": training_time_s,
                    "n_train": int(X_train.shape[0]),
                    "n_eval": int(X_eval.shape[0]),
                    "n_classes": n_classes,
                    "model_path": model_path,
                }
            )
            wandb.finish()
        except Exception as e:
            print(f"wandb logging failed (continuing): {e}")

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {csv_path}")

    return RunResults(
        estimator=estimator,
        train_acc=train_acc,
        eval_acc=eval_acc,
        eval_split=eval_split,
        n_train=int(X_train.shape[0]),
        n_eval=int(X_eval.shape[0]),
        n_classes=n_classes,
        model_path=model_path,
        csv_path=csv_path,
        log_path=log_path,
        training_time_s=training_time_s,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sklearn classifier on ASR encoder embeddings for language classification")

    # Required Arguments
    parser.add_argument("--model_name", type=str, required=True, help="ASR model name (e.g. facebook/mms-1b, openai/whisper-small)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to HF dataset directory (load_from_disk), with splits train/valid/test")
    parser.add_argument("--languages", type=str, required=True, help="Comma-separated list of languages to train on")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory (models + metrics will be saved here)")

    # Optional Arguments
    parser.add_argument("--data_seed", type=int, default=None, help="Data seed (default: random 1-10)")
    parser.add_argument("--eval_split", type=str, default="valid", help="Which split to evaluate on: valid or test (default: valid)")

    # Classifier Selection
    parser.add_argument(
        "--estimator",
        type=str,
        default="linear_svm",
        choices=["linear_svm", "kernel_svm", "knn"],
        help="Classifier type to train (default: linear_svm)",
    )

    # SVM Hyperparameters
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength (default: 1.0)")
    parser.add_argument("--max_iter", type=int, default=5000, help="Max iterations for LinearSVC (default: 5000; only for linear_svm)")
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["linear", "poly", "rbf", "sigmoid"],
        help="Kernel for kernel_svm (default: rbf)",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help='Gamma for kernel_svm SVC (e.g., "scale", "auto", "0.1"; default: scale)',
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Polynomial degree for kernel_svm with poly kernel (default: 3)",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        default=None,
        help='Class weighting; use "balanced" for inverse-frequency weights (default: None)',
    )
    parser.add_argument(
        "--dual",
        type=str,
        default="auto",
        help='LinearSVC dual formulation: "auto", "true", or "false" (default: auto; only for linear_svm)',
    )
    parser.add_argument("--n_neighbors", type=int, default=5, help="Number of neighbors for knn (default: 5)")
    parser.add_argument(
        "--knn_weights",
        type=str,
        default="uniform",
        choices=["uniform", "distance"],
        help="Weight function for knn (default: uniform)",
    )
    parser.add_argument("--knn_metric", type=str, default="minkowski", help="Distance metric for knn (default: minkowski)")

    # Logging
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")

    args = parser.parse_args()

    if args.data_seed is None:
        args.data_seed = random.randint(1, 10)

    # Normalize class_weight
    class_weight = args.class_weight
    if isinstance(class_weight, str) and class_weight.lower() == "none":
        class_weight = None

    results = run(
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        data_seed=int(args.data_seed),
        eval_split=args.eval_split,
        estimator=args.estimator,
        c=float(args.C),
        max_iter=int(args.max_iter),
        class_weight=class_weight,
        dual=args.dual,
        kernel=args.kernel,
        gamma=args.gamma,
        degree=int(args.degree),
        n_neighbors=int(args.n_neighbors),
        knn_weights=args.knn_weights,
        knn_metric=args.knn_metric,
        wandb_enable=not args.no_wandb,
        languages=args.languages.split(","),
    )

    # Print a small final summary (human-readable)
    print("\nFinal Results:")
    print(
        pd.DataFrame(
            {
                "estimator": [results.estimator],
                "train_acc": [results.train_acc],
                f"{results.eval_split}_acc": [results.eval_acc],
                "n_train": [results.n_train],
                "n_eval": [results.n_eval],
                "n_classes": [results.n_classes],
                "training_time_s": [results.training_time_s],
                "model_path": [results.model_path],
            }
        )
    )
