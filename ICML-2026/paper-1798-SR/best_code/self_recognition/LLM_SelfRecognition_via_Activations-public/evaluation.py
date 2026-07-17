"""Utilities for evaluating binary classifiers."""

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)


def compute_tpr_at_fpr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_fprs: Sequence[float] = (0.01, 0.05, 0.10),
) -> Dict[str, Optional[float]]:
    """Compute TPR at fixed FPR thresholds."""
    result = {f"tpr_at_fpr_{fpr:.2f}".replace(".", "_"): None for fpr in target_fprs}
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_scores)
    for target_fpr in target_fprs:
        key = f"tpr_at_fpr_{target_fpr:.2f}".replace(".", "_")
        valid_mask = fpr_arr <= target_fpr
        if np.any(valid_mask):
            result[key] = float(tpr_arr[valid_mask][-1])
    return result


def bootstrap_ci_clustered(
    scores: np.ndarray,
    labels: np.ndarray,
    prompt_ids: np.ndarray,
    B: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    metric: str = "auroc",
) -> Dict[str, Any]:
    """Compute a cluster bootstrap confidence interval."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    prompt_ids = np.asarray(prompt_ids)

    if len(scores) != len(labels) or len(scores) != len(prompt_ids):
        raise ValueError("scores, labels, and prompt_ids must have the same length")

    unique_prompts = np.unique(prompt_ids)
    n_prompts = len(unique_prompts)

    prompt_to_indices = {}
    for prompt_id in unique_prompts:
        mask = prompt_ids == prompt_id
        prompt_to_indices[prompt_id] = np.where(mask)[0]

    def compute_metric(y_true, y_scores):
        if len(np.unique(y_true)) < 2:
            return np.nan
        if metric == "auroc":
            return roc_auc_score(y_true, y_scores)
        elif metric == "auprc":
            return average_precision_score(y_true, y_scores)
        else:
            raise ValueError(f"Unknown metric: {metric}. Expected 'auroc' or 'auprc'.")

    point_estimate = compute_metric(labels, scores)

    rng = np.random.default_rng(seed)
    bootstrap_metrics = np.zeros(B)

    for b in range(B):
        sampled_prompts = rng.choice(unique_prompts, size=n_prompts, replace=True)

        boot_indices = []
        for prompt_id in sampled_prompts:
            boot_indices.extend(prompt_to_indices[prompt_id])
        boot_indices = np.array(boot_indices)

        boot_scores = scores[boot_indices]
        boot_labels = labels[boot_indices]

        bootstrap_metrics[b] = compute_metric(boot_labels, boot_scores)

    valid_metrics = bootstrap_metrics[~np.isnan(bootstrap_metrics)]

    ci_low = float(np.percentile(valid_metrics, 100 * alpha / 2)) if len(valid_metrics) > 0 else np.nan
    ci_high = float(np.percentile(valid_metrics, 100 * (1 - alpha / 2))) if len(valid_metrics) > 0 else np.nan

    return {
        "point_estimate": float(point_estimate),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_dist": bootstrap_metrics,
        "B": B,
        "seed": seed,
        "metric": metric,
    }


def train_classifier(X_train, y_train, classifier_params: dict):
    """Train an LDA classifier with provided parameters."""
    params_by_type = classifier_params or {}
    default_params = {"solver": "svd"}
    default_params.update(params_by_type.get("lda", {}) or {})
    model = LinearDiscriminantAnalysis(**default_params)
    model.fit(X_train, y_train)
    return model


def masked_mean_confidence(probas: Optional[np.ndarray], mask: Optional[np.ndarray]) -> Optional[float]:
    """Average max class probability over rows selected by `mask`."""
    if probas is None or mask is None or probas.shape[0] == 0:
        return None
    if not np.any(mask):
        return None
    selected = probas[mask]
    per_row_max = np.max(selected, axis=1)
    return float(np.mean(per_row_max))


def evaluate_model_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    y_scores: Optional[np.ndarray] = None,
):
    """Compute core metrics and confidence summaries."""
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    prec = float(precision_score(y_true, y_pred))
    rec = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    roc_auc = float("nan")
    if y_proba is not None and len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_proba[:, 1]))
    report = classification_report(
        y_true,
        y_pred,
        target_names=["AI-generated", "Human-authored"],
        output_dict=True,
    )
    correct_mask = (y_true == y_pred)
    avg_conf_correct = masked_mean_confidence(y_proba, correct_mask)
    avg_conf_incorrect = masked_mean_confidence(y_proba, ~correct_mask)

    scores_for_tpr = None
    if y_proba is not None:
        scores_for_tpr = y_proba[:, 1]
    elif y_scores is not None:
        scores_for_tpr = y_scores
    tpr_at_fpr = compute_tpr_at_fpr(y_true, scores_for_tpr) if scores_for_tpr is not None else {}

    metrics = {
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "precision_human": prec,
        "recall_human": rec,
        "f1_human": f1,
        "roc_auc_human": roc_auc,
        "classification_report": report,
        "avg_confidence_correct": avg_conf_correct,
        "avg_confidence_incorrect": avg_conf_incorrect,
        **tpr_at_fpr,
    }
    return metrics, float(acc), cm


def train_and_evaluate(
    train_ai_activations: pd.DataFrame,
    train_human_activations: pd.DataFrame,
    test_ai_activations: pd.DataFrame,
    test_human_activations: pd.DataFrame,
    config: dict,
) -> Dict[str, Any]:
    """Train classifiers and evaluate on test data."""
    print("\n>>> Preparing data for classification")

    bootstrap_B = config.get("BOOTSTRAP_B", 1000)
    bootstrap_seed = config.get("BOOTSTRAP_SEED") or config.get("RANDOM_SEED", 42)

    train_df = pd.concat([train_ai_activations, train_human_activations], ignore_index=True)
    test_df = pd.concat([test_ai_activations, test_human_activations], ignore_index=True)

    total_train = len(train_df)
    total_test = len(test_df)
    feature_dim = len(train_df["activations"].iloc[0])

    print(f"Training samples (vectors): {total_train}")
    print(f"Test samples (vectors): {total_test}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Train AI/Human: {(train_df['label'] == 0).sum()}/{(train_df['label'] == 1).sum()}")
    print(f"Test AI/Human: {(test_df['label'] == 0).sum()}/{(test_df['label'] == 1).sum()}")

    perplexity_metrics = None
    perplexity_ci = None
    if config.get("COMPUTE_PERPLEXITY", False):
        group_col = "prompt_idx" if "prompt_idx" in train_human_activations.columns else "text"

        human_ppl = train_human_activations.dropna(subset=["perplexity"]).groupby(group_col)["perplexity"].first()
        ai_ppl = train_ai_activations.dropna(subset=["perplexity"]).groupby(group_col)["perplexity"].first()

        if len(human_ppl) > 0 and len(ai_ppl) > 0:
            X_train_ppl = np.vstack([human_ppl.values.reshape(-1, 1), ai_ppl.values.reshape(-1, 1)])
            median_ppl = float(np.median(X_train_ppl))

            test_human_ppl_df = test_human_activations.dropna(subset=["perplexity"]).groupby(group_col).agg({
                "perplexity": "first",
            }).reset_index()
            test_ai_ppl_df = test_ai_activations.dropna(subset=["perplexity"]).groupby(group_col).agg({
                "perplexity": "first",
            }).reset_index()

            X_test_ppl = np.vstack([
                test_human_ppl_df["perplexity"].values.reshape(-1, 1),
                test_ai_ppl_df["perplexity"].values.reshape(-1, 1),
            ])
            y_test_ppl = np.hstack([np.ones(len(test_human_ppl_df)), np.zeros(len(test_ai_ppl_df))])

            if group_col == "prompt_idx":
                ppl_test_prompt_ids = np.hstack([
                    test_human_ppl_df["prompt_idx"].values,
                    test_ai_ppl_df["prompt_idx"].values,
                ])
            else:
                ppl_test_prompt_ids = np.arange(len(y_test_ppl))

            y_pred_ppl = (X_test_ppl[:, 0] >= median_ppl).astype(int)
            perplexity_metrics, _, _ = evaluate_model_outputs(
                y_test_ppl, y_pred_ppl, None,
                y_scores=X_test_ppl[:, 0],
            )
            perplexity_auroc = float(roc_auc_score(y_test_ppl, X_test_ppl[:, 0]))
            perplexity_metrics["roc_auc_human"] = perplexity_auroc

            if group_col == "prompt_idx":
                perplexity_ci_result = bootstrap_ci_clustered(
                    scores=X_test_ppl[:, 0],
                    labels=y_test_ppl,
                    prompt_ids=ppl_test_prompt_ids,
                    B=bootstrap_B,
                    seed=bootstrap_seed,
                    metric="auroc",
                )
                perplexity_ci = {
                    "ci_low": perplexity_ci_result["ci_low"],
                    "ci_high": perplexity_ci_result["ci_high"],
                }
                perplexity_metrics["auroc_ci"] = perplexity_ci
                print(f"Perplexity baseline: AUROC={perplexity_auroc:.4f} [{perplexity_ci['ci_low']:.3f}, {perplexity_ci['ci_high']:.3f}]")

    print(f"\n>>> Training and evaluating {config.get('CLASSIFIER_TYPE', 'lda')} classifier")
    unique_layers = sorted(train_df["layer_idx"].unique().tolist())

    per_layer_results = {}
    per_layer_ci = {}
    per_layer_roc_auc = {}
    best_layer = None
    best_score = -1.0

    has_prompt_idx = "prompt_idx" in test_df.columns

    for lyr in tqdm(unique_layers, desc="Per-layer classifiers"):
        train_df_l = train_df[train_df["layer_idx"] == lyr]
        X_train_l = np.vstack(train_df_l["activations"].to_numpy())
        y_train_l = train_df_l["label"].to_numpy()

        scaler = None
        if config.get("STANDARDIZE_ACTIVATIONS", True):
            scaler = StandardScaler()
            X_train_l = scaler.fit_transform(X_train_l)

        model_l = train_classifier(
            X_train_l, y_train_l,
            config.get("CLASSIFIER_PARAMS", {}),
        )

        test_df_l = test_df[test_df["layer_idx"] == lyr]
        X_test_l = np.vstack(test_df_l["activations"].to_numpy())
        y_test_l = test_df_l["label"].to_numpy()
        prompt_ids_l = test_df_l["prompt_idx"].to_numpy() if has_prompt_idx else None

        if scaler is not None:
            X_test_l = scaler.transform(X_test_l)

        y_pred_l = model_l.predict(X_test_l)
        y_pred_proba_l = getattr(model_l, "predict_proba", None)
        y_pred_proba_l = y_pred_proba_l(X_test_l) if y_pred_proba_l is not None else None
        metrics_vec_l, accuracy_l, cm_l = evaluate_model_outputs(y_test_l, y_pred_l, y_pred_proba_l)

        auroc_ci_l = None
        if y_pred_proba_l is not None and has_prompt_idx:
            scores_for_ci = y_pred_proba_l[:, 1]
            auroc_ci_l = bootstrap_ci_clustered(
                scores=scores_for_ci,
                labels=y_test_l,
                prompt_ids=prompt_ids_l,
                B=bootstrap_B,
                seed=bootstrap_seed,
                metric="auroc",
            )
            per_layer_ci[int(lyr)] = {
                "ci_low": auroc_ci_l["ci_low"],
                "ci_high": auroc_ci_l["ci_high"],
            }

        per_layer_results[int(lyr)] = {
            "metrics_vec": metrics_vec_l,
            "cm": cm_l,
            "accuracy_for_summary": float(accuracy_l),
            "auroc_ci": auroc_ci_l,
        }
        per_layer_roc_auc[int(lyr)] = metrics_vec_l.get("roc_auc_human", float("nan"))

        auroc_l = metrics_vec_l.get("roc_auc_human", float("nan"))
        ci_str = ""
        if auroc_ci_l is not None:
            ci_str = f" [{auroc_ci_l['ci_low']:.3f}, {auroc_ci_l['ci_high']:.3f}]"
        print(f"Layer {lyr}: AUROC={auroc_l:.4f}{ci_str}, accuracy={metrics_vec_l['accuracy']:.4f}")

        if accuracy_l > best_score:
            best_score = float(accuracy_l)
            best_layer = int(lyr)

    n_test_prompts = len(test_df["prompt_idx"].unique()) if has_prompt_idx else None

    return {
        "per_layer_results": per_layer_results,
        "per_layer_ci": per_layer_ci,
        "per_layer_roc_auc": per_layer_roc_auc,
        "best_layer": best_layer,
        "best_score": best_score,
        "perplexity_metrics": perplexity_metrics,
        "perplexity_ci": perplexity_ci,
        "unique_layers": unique_layers,
        "bootstrap_config": {
            "B": bootstrap_B,
            "seed": bootstrap_seed,
            "resampling_unit": "prompt/article" if has_prompt_idx else "sample",
            "n_test_prompts": n_test_prompts,
        },
    }
