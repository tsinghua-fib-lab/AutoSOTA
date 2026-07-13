#!/usr/bin/env python3
"""
Train a small NN language detection head on top of pooled ASR embeddings.

This script trains on the **pooled encoder embeddings** returned by:
  `jaxcld.models.asr_model.ASRModel.load_data()`  -> (N, D) embeddings, (N,) labels

It produces an artifact compatible with:
  `jaxcld.models.lang_detect_head.NNLangDetectHead.load()`

Example:
  python train_nn.py \
    --dataset_path data/multiclass \
    --model_name openai/whisper-small \
    --languages en,hi,id,ms,zh \
    --output_dir data/test/test_nn
"""

import argparse
import itertools
import os
import pickle
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from jaxcld.models.asr_model import ASRModel
from jaxcld.models.lang_detect_head import NNLangDetectHeadModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a tiny NN language detector head (binary or multiclass) on pooled ASR embeddings."
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the ingested DatasetDict directory.")
    parser.add_argument("--model_name", type=str, required=True, help="ASR model name (e.g. facebook/mms-1b-all, openai/whisper-small)")
    parser.add_argument("--languages", type=str, required=True, help="Comma-separated list of language codes.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the model artifact and metrics.")

    parser.add_argument("--eval_split", type=str, default="valid", help="Dataset split to evaluate on (default: valid).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay (default: 0.0).")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Epochs (default: 10).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=256, help="Train batch size (default: 256).")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=512, help="Eval batch size (default: 512).")
    parser.add_argument("--fp16", action="store_true", help="Train the head in FP16 when supported (default: False).")
    parser.add_argument(
        "--tuning_strategy",
        type=str,
        default="none",
        choices=["none", "grid", "random"],
        help="Hyperparameter tuning strategy: none, grid, or random (default: none).",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=12,
        help="Number of random trials when --tuning_strategy=random (default: 12).",
    )
    parser.add_argument(
        "--tune_learning_rates",
        type=str,
        default="1e-4,3e-4,1e-3,3e-3",
        help="Candidate learning rates (comma-separated). Used by grid/random tuning.",
    )
    parser.add_argument(
        "--tune_weight_decays",
        type=str,
        default="0.0,1e-5,1e-4,1e-3",
        help="Candidate weight decays (comma-separated). Used by grid/random tuning.",
    )
    parser.add_argument(
        "--tune_num_train_epochs",
        type=str,
        default="5,10,20",
        help="Candidate epoch counts (comma-separated). Used by grid/random tuning.",
    )
    parser.add_argument(
        "--tune_train_batch_sizes",
        type=str,
        default="128,256,512",
        help="Candidate train batch sizes (comma-separated). Used by grid/random tuning.",
    )

    return parser.parse_args()


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def _parse_float_list(csv_values: str, arg_name: str) -> List[float]:
    vals = [x.strip() for x in str(csv_values).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{arg_name} must contain at least one value.")
    out = [float(x) for x in vals]
    return out


def _parse_int_list(csv_values: str, arg_name: str) -> List[int]:
    vals = [x.strip() for x in str(csv_values).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{arg_name} must contain at least one value.")
    out = [int(x) for x in vals]
    return out


def _load_split_embeddings(
    asr_model: ASRModel,
    dataset_path: str,
    dataset_split: str,
    data_seed: int,
    shuffle: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """
    Returns:
        X: (N, D) pooled embeddings
        y: (N,) int labels
        n_classes: optional int (some backends return it, e.g. MMS)
    """
    out = asr_model.load_data(
        dataset_path,
        data_seed=data_seed,
        dataset_split=dataset_split,
        shuffle=shuffle,
    )
    if isinstance(out, tuple) and len(out) == 2:
        X, y = out
        return np.asarray(X), np.asarray(y), None
    if isinstance(out, tuple) and len(out) == 3:
        X, y, n_classes = out
        return np.asarray(X), np.asarray(y), int(n_classes)
    raise ValueError(f"Unexpected return from ASRModel.load_data: type={type(out)} value={out}")


@torch.no_grad()
def _predict_batches(model: nn.Module, loader: DataLoader, device: torch.device, fp16: bool) -> np.ndarray:
    model.eval()
    preds: List[int] = []
    for xb, _ in loader:
        xb = xb.to(device=device)
        if fp16:
            xb = xb.to(dtype=torch.float16)
        # pooled (B, D) -> hidden_states (B, 1, D) so NNLangDetectHeadModule pooling is a no-op
        logits = model(xb.unsqueeze(1))
        pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        preds.extend(pred.tolist())
    return np.asarray(preds, dtype=int)


def _train_and_eval(
    d_model: int,
    num_classes: int,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    device: torch.device,
    fp16: bool,
    learning_rate: float,
    weight_decay: float,
    num_train_epochs: int,
) -> Dict[str, object]:
    model = NNLangDetectHeadModule(d_model=d_model, n_classes=int(num_classes)).to(device=device)
    if fp16:
        model = model.to(dtype=torch.float16)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))

    start = time.time()
    train_acc = eval_acc = float("nan")
    last_avg_loss = float("nan")
    for _ in range(int(num_train_epochs)):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device=device)
            yb = yb.to(device=device)
            if fp16:
                xb = xb.to(dtype=torch.float16)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb.unsqueeze(1))
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = int(xb.shape[0])
            running_loss += float(loss.detach().cpu().item()) * bs
            n_seen += bs

        train_preds = _predict_batches(model, train_loader, device=device, fp16=bool(fp16))
        eval_preds = _predict_batches(model, eval_loader, device=device, fp16=bool(fp16))
        train_acc = _accuracy(y_train, train_preds)
        eval_acc = _accuracy(y_eval, eval_preds)
        last_avg_loss = running_loss / max(1, n_seen)

    training_time_s = time.time() - start
    return {
        "model": model,
        "train_acc": float(train_acc),
        "eval_acc": float(eval_acc),
        "avg_loss": float(last_avg_loss),
        "training_time_s": float(training_time_s),
    }


def main():
    args = parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    os.makedirs(args.output_dir, exist_ok=True)

    languages = [x.strip() for x in str(args.languages).split(",") if x.strip()]
    if not languages:
        raise ValueError("--languages must be a non-empty comma-separated list (e.g. en,zh)")

    asr = ASRModel.from_pretrained(args.model_name, config={"languages": languages})

    print(f"[Info] Loading pooled embeddings from {args.dataset_path} ...")
    X_train, y_train, n_classes_train = _load_split_embeddings(
        asr_model=asr,
        dataset_path=args.dataset_path,
        dataset_split="train",
        data_seed=int(args.seed),
        shuffle=True,
    )

    eval_split = str(args.eval_split)
    try:
        X_eval, y_eval, n_classes_eval = _load_split_embeddings(
            asr_model=asr,
            dataset_path=args.dataset_path,
            dataset_split=eval_split,
            data_seed=int(args.seed),
            shuffle=False,
        )
    except Exception as e:
        # Common mismatch: users pass "eval" but dataset uses "valid"
        if eval_split == "eval":
            print(f"[Warn] Failed to load eval split='eval' ({e}); retrying with split='valid' ...")
            eval_split = "valid"
            X_eval, y_eval, n_classes_eval = _load_split_embeddings(
                asr_model=asr,
                dataset_path=args.dataset_path,
                dataset_split=eval_split,
                data_seed=int(args.seed),
                shuffle=False,
            )
        else:
            raise

    # Optional test split
    X_test = y_test = None
    try:
        X_test, y_test, _ = _load_split_embeddings(
            asr_model=asr,
            dataset_path=args.dataset_path,
            dataset_split="test",
            data_seed=int(args.seed),
            shuffle=False,
        )
    except Exception:
        pass

    inferred_n_classes = int(
        n_classes_train
        or n_classes_eval
        or len(np.unique(np.concatenate([np.asarray(y_train), np.asarray(y_eval)], axis=0)))
    )
    num_classes = max(inferred_n_classes, len(languages))

    d_model = int(asr.get_dimensions())
    if X_train.ndim != 2 or int(X_train.shape[1]) != d_model:
        raise ValueError(f"Unexpected train embedding shape {X_train.shape}; expected (N, {d_model})")

    device = asr.get_device()
    if not isinstance(device, torch.device):
        device = torch.device(str(device))

    print(
        f"[Info] d_model={d_model} num_classes={num_classes} device={device} "
        f"(train={int(X_train.shape[0])} eval={int(X_eval.shape[0])} eval_split={eval_split})"
    )

    xtr = torch.from_numpy(np.asarray(X_train)).to(dtype=torch.float32)
    ytr = torch.from_numpy(np.asarray(y_train)).to(dtype=torch.long)
    xev = torch.from_numpy(np.asarray(X_eval)).to(dtype=torch.float32)
    yev = torch.from_numpy(np.asarray(y_eval)).to(dtype=torch.long)

    eval_loader = DataLoader(
        TensorDataset(xev, yev),
        batch_size=int(args.per_device_eval_batch_size),
        shuffle=False,
        drop_last=False,
    )
    trial_rows: List[Dict[str, object]] = []
    model = None
    training_time_s = 0.0

    if args.tuning_strategy == "none":
        train_loader = DataLoader(
            TensorDataset(xtr, ytr),
            batch_size=int(args.per_device_train_batch_size),
            shuffle=True,
            drop_last=False,
        )
        print("[Info] Starting training ...")
        result = _train_and_eval(
            d_model=d_model,
            num_classes=num_classes,
            train_loader=train_loader,
            eval_loader=eval_loader,
            y_train=np.asarray(y_train),
            y_eval=np.asarray(y_eval),
            device=device,
            fp16=bool(args.fp16),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            num_train_epochs=int(args.num_train_epochs),
        )
        model = result["model"]
        train_acc = float(result["train_acc"])
        eval_acc = float(result["eval_acc"])
        avg_loss = float(result["avg_loss"])
        training_time_s = float(result["training_time_s"])
        print(
            f"[Result] loss={avg_loss:.4f} train_acc={train_acc:.4f} {eval_split}_acc={eval_acc:.4f} "
            f"(lr={float(args.learning_rate)} wd={float(args.weight_decay)} "
            f"epochs={int(args.num_train_epochs)} train_bs={int(args.per_device_train_batch_size)})"
        )
        trial_rows.append(
            {
                "trial": 1,
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "num_train_epochs": int(args.num_train_epochs),
                "train_batch_size": int(args.per_device_train_batch_size),
                "avg_loss": avg_loss,
                "train_acc": train_acc,
                "eval_acc": eval_acc,
                "training_time_s": training_time_s,
                "is_best": 1,
            }
        )
        best_cfg = {
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "num_train_epochs": int(args.num_train_epochs),
            "train_batch_size": int(args.per_device_train_batch_size),
        }
    else:
        lrs = _parse_float_list(args.tune_learning_rates, "--tune_learning_rates")
        wds = _parse_float_list(args.tune_weight_decays, "--tune_weight_decays")
        epochs_list = _parse_int_list(args.tune_num_train_epochs, "--tune_num_train_epochs")
        train_bs_list = _parse_int_list(args.tune_train_batch_sizes, "--tune_train_batch_sizes")
        all_candidates = list(itertools.product(lrs, wds, epochs_list, train_bs_list))
        if not all_candidates:
            raise ValueError("No hyperparameter candidates were generated for tuning.")

        rng = random.Random(int(args.seed))
        if args.tuning_strategy == "grid":
            candidates = all_candidates
        else:
            k = min(int(args.num_trials), len(all_candidates))
            if k <= 0:
                raise ValueError("--num_trials must be > 0 when --tuning_strategy=random.")
            candidates = rng.sample(all_candidates, k=k)

        print(
            f"[Info] Starting {args.tuning_strategy} search with {len(candidates)} trial(s) "
            f"(candidate pool={len(all_candidates)}) ..."
        )

        best_eval_acc = -float("inf")
        best_train_acc = float("nan")
        best_cfg = None
        best_model = None
        for idx, (lr, wd, n_epochs, train_bs) in enumerate(candidates, start=1):
            train_loader = DataLoader(
                TensorDataset(xtr, ytr),
                batch_size=int(train_bs),
                shuffle=True,
                drop_last=False,
            )
            result = _train_and_eval(
                d_model=d_model,
                num_classes=num_classes,
                train_loader=train_loader,
                eval_loader=eval_loader,
                y_train=np.asarray(y_train),
                y_eval=np.asarray(y_eval),
                device=device,
                fp16=bool(args.fp16),
                learning_rate=float(lr),
                weight_decay=float(wd),
                num_train_epochs=int(n_epochs),
            )
            train_acc = float(result["train_acc"])
            eval_acc = float(result["eval_acc"])
            avg_loss = float(result["avg_loss"])
            trial_time_s = float(result["training_time_s"])
            is_best = int(eval_acc > best_eval_acc)
            if is_best:
                best_eval_acc = eval_acc
                best_train_acc = train_acc
                best_cfg = {
                    "learning_rate": float(lr),
                    "weight_decay": float(wd),
                    "num_train_epochs": int(n_epochs),
                    "train_batch_size": int(train_bs),
                }
                best_model = result["model"]
            trial_rows.append(
                {
                    "trial": idx,
                    "learning_rate": float(lr),
                    "weight_decay": float(wd),
                    "num_train_epochs": int(n_epochs),
                    "train_batch_size": int(train_bs),
                    "avg_loss": avg_loss,
                    "train_acc": train_acc,
                    "eval_acc": eval_acc,
                    "training_time_s": trial_time_s,
                    "is_best": is_best,
                }
            )
            print(
                f"[Trial {idx}/{len(candidates)}] loss={avg_loss:.4f} train_acc={train_acc:.4f} "
                f"{eval_split}_acc={eval_acc:.4f} (lr={lr} wd={wd} epochs={n_epochs} train_bs={train_bs})"
            )

        if best_cfg is None or best_model is None:
            raise RuntimeError("Tuning completed without selecting a best model.")
        model = best_model
        train_acc = float(best_train_acc)
        eval_acc = float(best_eval_acc)
        training_time_s = float(sum(float(row["training_time_s"]) for row in trial_rows))
        print(
            f"[Info] Best config from tuning: lr={best_cfg['learning_rate']} wd={best_cfg['weight_decay']} "
            f"epochs={best_cfg['num_train_epochs']} train_bs={best_cfg['train_batch_size']} "
            f"with {eval_split}_acc={eval_acc:.4f}"
        )

    test_acc = None
    if X_test is not None and y_test is not None:
        xt = torch.from_numpy(np.asarray(X_test)).to(dtype=torch.float32)
        yt = torch.from_numpy(np.asarray(y_test)).to(dtype=torch.long)
        test_loader = DataLoader(TensorDataset(xt, yt), batch_size=int(args.per_device_eval_batch_size), shuffle=False)
        test_preds = _predict_batches(model, test_loader, device=device, fp16=bool(args.fp16))
        test_acc = _accuracy(y_test, test_preds)
        print(f"Test accuracy: {test_acc:.4f}")

    # Save artifact compatible with NNLangDetectHead.load(): pickle with keys "classifier.*"
    safe_model_name = str(args.model_name).replace("/", "_")
    model_dir = os.path.join(args.output_dir, str(args.model_name))
    os.makedirs(model_dir, exist_ok=True)
    artifact_path = os.path.join(model_dir, f"{safe_model_name}_nn_head.pkl")

    state = {}
    for k, v in model.classifier.state_dict().items():
        state[f"classifier.{k}"] = v.detach().cpu().numpy()
    with open(artifact_path, "wb") as f:
        pickle.dump(state, f)

    metrics_path = os.path.join(model_dir, "nn_metrics.csv")
    with open(metrics_path, "w") as f:
        f.write(
            "train_acc\teval_acc\teval_split\tn_train\tn_eval\tn_classes\tembedding_dim\ttraining_time_s\ttest_acc\t"
            "tuning_strategy\tlearning_rate\tweight_decay\tnum_train_epochs\ttrain_batch_size\tmodel_path\n"
        )
        f.write(
            f"{train_acc}\t{eval_acc}\t{eval_split}\t{int(X_train.shape[0])}\t{int(X_eval.shape[0])}\t"
            f"{int(num_classes)}\t{int(d_model)}\t{training_time_s}\t{test_acc}\t"
            f"{args.tuning_strategy}\t{best_cfg['learning_rate']}\t{best_cfg['weight_decay']}\t"
            f"{best_cfg['num_train_epochs']}\t{best_cfg['train_batch_size']}\t{artifact_path}\n"
        )

    trials_path = os.path.join(model_dir, "nn_tuning_trials.csv")
    with open(trials_path, "w") as f:
        f.write(
            "trial\tlearning_rate\tweight_decay\tnum_train_epochs\ttrain_batch_size\tavg_loss\t"
            f"train_acc\t{eval_split}_acc\ttraining_time_s\tis_best\n"
        )
        for row in trial_rows:
            f.write(
                f"{row['trial']}\t{row['learning_rate']}\t{row['weight_decay']}\t{row['num_train_epochs']}\t"
                f"{row['train_batch_size']}\t{row['avg_loss']}\t{row['train_acc']}\t{row['eval_acc']}\t"
                f"{row['training_time_s']}\t{row['is_best']}\n"
            )

    print(f"[Info] Saved NN head artifact to: {artifact_path}")
    print(f"[Info] Saved metrics to: {metrics_path}")
    print(f"[Info] Saved tuning trials to: {trials_path}")


if __name__ == "__main__":
    main()

