"""Minimal, reproducible pipeline for LLM self-recognition experiments."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

from activation_extraction import extract_activations_for_texts
from evaluation import (
    bootstrap_ci_clustered,
    evaluate_model_outputs,
    train_and_evaluate,
)
from helpers import (
    build_base_name,
    convert_metrics_for_json,
    flatten_yaml_config,
    get_default_parameters,
    resolve_unique_base_name,
    write_eval,
)
from model_utils import load_model_and_tokenizer
from text_generation import generate_texts


def _maybe_hf_login() -> None:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("HF token not found in .env. Proceeding without login.")


def _ensure_default_config(cfg_path: Path) -> None:
    if cfg_path.exists():
        return
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    default_cfg = get_default_parameters()
    cfg_path.write_text(
        yaml.safe_dump(default_cfg, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Created default config at {cfg_path}")


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_by_prompt(ai_df, human_df, test_size: float, seed: int):
    unique_prompt_ids = ai_df["prompt_idx"].unique()
    train_prompt_ids, test_prompt_ids = train_test_split(
        unique_prompt_ids,
        test_size=test_size,
        random_state=seed,
    )
    train_ids = set(train_prompt_ids)
    test_ids = set(test_prompt_ids)
    train_ai = ai_df[ai_df["prompt_idx"].isin(train_ids)].copy()
    train_human = human_df[human_df["prompt_idx"].isin(train_ids)].copy()
    test_ai = ai_df[ai_df["prompt_idx"].isin(test_ids)].copy()
    test_human = human_df[human_df["prompt_idx"].isin(test_ids)].copy()
    return train_ai, train_human, test_ai, test_human


def _evaluate_mass_mean(
    train_df,
    test_df,
    standardize: bool,
    bootstrap_B: int,
    bootstrap_seed: int,
):
    X_train = np.vstack(train_df["activations"].to_numpy())
    y_train = train_df["label"].to_numpy()
    X_test = np.vstack(test_df["activations"].to_numpy())
    y_test = test_df["label"].to_numpy()
    prompt_ids = test_df["prompt_idx"].to_numpy() if "prompt_idx" in test_df.columns else None

    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = NearestCentroid()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    centroids = model.centroids_
    y_scores = None
    if centroids.shape[0] == 2:
        distances = np.linalg.norm(X_test[:, None, :] - centroids[None, :, :], axis=2)
        y_scores = -distances[:, 1]  # higher = closer to human centroid

    metrics, _, _ = evaluate_model_outputs(
        y_test, y_pred, None, y_scores=y_scores
    )
    ci = None
    if y_scores is not None and prompt_ids is not None and len(np.unique(y_test)) > 1:
        auroc = float(roc_auc_score(y_test, y_scores))
        metrics["roc_auc_human"] = auroc
        ci_result = bootstrap_ci_clustered(
            scores=y_scores,
            labels=y_test,
            prompt_ids=prompt_ids,
            B=bootstrap_B,
            seed=bootstrap_seed,
            metric="auroc",
        )
        ci = {"ci_low": ci_result["ci_low"], "ci_high": ci_result["ci_high"]}
    return metrics, ci


def main(config_path: str | None = None) -> None:
    cfg_dir = Path("configs")
    default_cfg_path = cfg_dir / "default.yaml"
    _ensure_default_config(default_cfg_path)

    load_path = Path(config_path) if config_path else default_cfg_path
    with load_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}
    config = flatten_yaml_config(raw_cfg)

    _maybe_hf_login()
    _set_seeds(config["RANDOM_SEED"])

    root_dir = Path(__file__).resolve().parent
    model_dir_name = str(config["MODEL_NAME"]).rsplit("/", 1)[-1]
    output_dir = root_dir / "evaluation_data" / config["DATASET_NAME"] / model_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    custom_name = config.get("SAVE_NAME")
    base_name = str(custom_name).strip() if custom_name else build_base_name(config)
    final_base_name = resolve_unique_base_name(output_dir, base_name)
    run_dir = output_dir / final_base_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")
    print(f"Loaded config from {load_path}")

    model, tokenizer, device = load_model_and_tokenizer(config["MODEL_NAME"])

    texts = generate_texts(config, model, tokenizer, device)
    ai_activations, human_activations = extract_activations_for_texts(
        texts, config, model, tokenizer, device
    )

    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_ai, train_human, test_ai, test_human = _split_by_prompt(
        ai_activations,
        human_activations,
        test_size=config["TEST_SIZE"],
        seed=config["RANDOM_SEED"],
    )

    results = train_and_evaluate(
        train_ai,
        train_human,
        test_ai,
        test_human,
        config,
    )

    best_layer = results["best_layer"]
    best_result = results["per_layer_results"].get(best_layer)

    mass_mean_metrics = None
    mass_mean_ci = None
    if config.get("COMPUTE_MASS_MEAN", False) and best_layer is not None and best_result is not None:
        train_df_best = pd.concat(
            [
                train_ai[train_ai["layer_idx"] == best_layer],
                train_human[train_human["layer_idx"] == best_layer],
            ],
            ignore_index=True,
        )
        test_df_best = pd.concat(
            [
                test_ai[test_ai["layer_idx"] == best_layer],
                test_human[test_human["layer_idx"] == best_layer],
            ],
            ignore_index=True,
        )
        mass_mean_metrics, mass_mean_ci = _evaluate_mass_mean(
            train_df_best,
            test_df_best,
            standardize=config.get("STANDARDIZE_ACTIVATIONS", True),
            bootstrap_B=config["BOOTSTRAP_B"],
            bootstrap_seed=config["BOOTSTRAP_SEED"] or config["RANDOM_SEED"],
        )

    eval_payload = {
        "dataset": config["DATASET_NAME"],
        "model_name": config["MODEL_NAME"],
        "num_samples": int(config["NUM_SAMPLES"]),
        "prompt_tokens": int(config["TARGET_PROMPT_TOKENS"]),
        "completion_tokens": int(config["TARGET_COMPLETION_TOKENS"]),
        "extraction_mode": config["EXTRACTION_MODE"],
        "layers_analyzed": [int(x) for x in config["LAYERS_TO_ANALYZE"]],
        "tokens_to_analyze": [int(x) for x in config["TOKENS_TO_ANALYZE"]],
        "batch_size": int(config["BATCH_SIZE"]),
        "compute_perplexity": bool(config["COMPUTE_PERPLEXITY"]),
        "compute_mass_mean": bool(config["COMPUTE_MASS_MEAN"]),
        "classifier": {
            "type": "lda",
            "params": config.get("CLASSIFIER_PARAMS", {}).get("lda", {}),
        },
        "best_layer_idx": best_layer,
        "activations_metrics": convert_metrics_for_json(best_result["metrics_vec"]) if best_result else None,
        "per_layer_metrics": {
            str(k): {
                "vector": convert_metrics_for_json(v["metrics_vec"]),
                "auroc_ci": results["per_layer_ci"].get(k),
            }
            for k, v in results["per_layer_results"].items()
        },
        "per_layer_roc_auc": results["per_layer_roc_auc"],
        "per_layer_roc_auc_ci": results["per_layer_ci"],
        "perplexity_metrics": convert_metrics_for_json(results["perplexity_metrics"])
        if results["perplexity_metrics"]
        else None,
        "mass_mean_metrics": convert_metrics_for_json(mass_mean_metrics) if mass_mean_metrics else None,
        "bootstrap_config": results["bootstrap_config"],
    }
    if mass_mean_ci is not None and eval_payload["mass_mean_metrics"] is not None:
        eval_payload["mass_mean_metrics"]["auroc_ci"] = mass_mean_ci

    write_eval(run_dir, eval_payload)

    print("\nSummary")
    print(f"Best layer: {best_layer}")
    if best_result is not None:
        mv = best_result["metrics_vec"]
        print(f"Accuracy: {mv['accuracy']:.4f} | AUROC(H): {mv['roc_auc_human']:.4f}")
    if results["perplexity_metrics"] is not None:
        mp = results["perplexity_metrics"]
        print(f"Perplexity baseline AUROC(H): {mp['roc_auc_human']:.4f}")
    if mass_mean_metrics is not None:
        mm = mass_mean_metrics
        print(f"Mass-mean baseline AUROC(H): {mm['roc_auc_human']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Self-Recognition: run the minimal pipeline."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to a YAML configuration file (defaults to configs/default.yaml).",
    )
    args = parser.parse_args()
    main(config_path=args.config)
