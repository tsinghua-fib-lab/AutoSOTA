from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from rlie.datasets.data_loader import _normalize_label
from rlie.core.metrics import compute_accuracy_and_macro_f1


def select_seed_samples(df: pd.DataFrame, num: int, seed: int = 42) -> pd.DataFrame:
    if num >= len(df):
        return df.copy().reset_index(drop=True)
    return df.sample(n=num, random_state=seed).reset_index(drop=True)


def select_hard_samples(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num: int,
    texts: Optional[List[str]] = None,
    cluster_seed: int = 42,
) -> List[int]:
    """Select hard samples for refinement, optionally with diversity clustering.

    When `texts` is provided, misclassified samples are stratified by true
    class and clustered via TF-IDF + k-means to ensure diverse failure modes
    are represented. Falls back to original margin-based selection when texts
    are unavailable, num is very small, or insufficient misclassified samples
    exist.
    """
    # ---- identify misclassified samples ----
    deltas = []
    for idx, (prob, label) in enumerate(zip(probabilities, labels)):
        predicted = 1 if prob >= 0.5 else 0
        correct = int(label)
        misclassified = 1 if predicted != correct else 0
        margin_error = abs(1.0 - float(prob)) if correct == 1 else abs(float(prob))
        deltas.append((misclassified, margin_error, idx))

    misclassified_items = [(m, me, idx) for m, me, idx in deltas if m == 1]
    correctly_classified_items = [(m, me, idx) for m, me, idx in deltas if m == 0]

    # ---- fallback: no texts, too few misclassified, or num too small ----
    if (
        texts is None
        or num < 4
        or len(misclassified_items) < 4
    ):
        deltas.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
        return [idx for _, _, idx in deltas[:num]]

    # ---- build per-class lists ----
    class_samples: Dict[int, List[tuple]] = {}
    class_texts: Dict[int, List[str]] = {}
    for m, me, idx in misclassified_items:
        c = int(labels[idx])
        class_samples.setdefault(c, []).append((me, idx))
        class_texts.setdefault(c, []).append(texts[idx])

    n_classes = len(class_samples)
    if n_classes == 0:
        deltas.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
        return [idx for _, _, idx in deltas[:num]]

    # ---- allocate budget across classes: equal share, remainder to largest ----
    per_class = max(1, num // n_classes)
    selected: List[int] = []
    budgets = {c: per_class for c in class_samples}
    remaining = num - per_class * n_classes
    if remaining > 0:
        sorted_by_size = sorted(class_samples, key=lambda c: len(class_samples[c]), reverse=True)
        for c in sorted_by_size[:remaining]:
            budgets[c] += 1

    # ---- within each class, cluster and pick hardest per cluster ----
    for class_id, samples in class_samples.items():
        budget = budgets.get(class_id, 1)
        if budget <= 0:
            continue
        # sort by margin_error descending (hardest first)
        samples.sort(key=lambda item: item[0], reverse=True)
        if len(samples) <= budget:
            selected.extend([idx for _, idx in samples])
            continue

        # TF-IDF + KMeans clustering
        class_text_list = class_texts.get(class_id, [])
        if len(class_text_list) < 3 or budget < 2:
            # too few to cluster, pick top by margin
            selected.extend([idx for _, idx in samples[:budget]])
            continue

        n_clusters = min(budget, len(class_text_list))
        try:
            vectorizer = TfidfVectorizer(
                max_features=500, stop_words="english", sublinear_tf=True
            )
            tfidf_matrix = vectorizer.fit_transform(class_text_list)
            if tfidf_matrix.shape[0] < n_clusters:
                n_clusters = tfidf_matrix.shape[0]
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=cluster_seed,
                n_init="auto",
            )
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
        except Exception:
            # clustering failed, fall back to top-N by margin
            selected.extend([idx for _, idx in samples[:budget]])
            continue

        # from each cluster, pick the sample with highest margin_error
        cluster_best: Dict[int, tuple] = {}
        for i, (me, idx) in enumerate(samples):
            cl = int(cluster_labels[i])
            if cl not in cluster_best or me > cluster_best[cl][0]:
                cluster_best[cl] = (me, idx)
        # take top `budget` clusters by hardness
        cluster_picks = sorted(cluster_best.values(), key=lambda x: x[0], reverse=True)
        selected.extend([idx for _, idx in cluster_picks[:budget]])

    # ---- fill remaining budget with correctly classified hard examples if needed ----
    if len(selected) < num and correctly_classified_items:
        correctly_classified_items.sort(key=lambda item: item[1], reverse=True)
        for _, _, idx in correctly_classified_items:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= num:
                break

    # ---- deduplicate while preserving order ----
    seen = set()
    result = []
    for idx in selected:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    return result[:num]


def sort_samples_by_label(df: pd.DataFrame, label_column: str, label_mapping: Dict[str, int]) -> pd.DataFrame:
    if df.empty or label_column not in df:
        return df

    def sort_key(value):
        normalized = _normalize_label(value)
        return label_mapping.get(normalized, float("inf"))

    sorted_df = df.copy()
    sorted_df["__label_sort_key"] = sorted_df[label_column].apply(sort_key)
    sorted_df = sorted_df.sort_values("__label_sort_key", kind="stable").drop(columns="__label_sort_key")
    return sorted_df.reset_index(drop=True)


def compute_rule_metrics(vectors: List[List[int]], labels: List[int]) -> List[Dict[str, float]]:
    metrics: List[Dict[str, float]] = []
    label_count = len(labels)
    for predictions in vectors:
        if label_count == 0 or len(predictions) != label_count:
            metrics.append({"accuracy": float("nan"), "f1": float("nan"), "coverage": 0.0})
            continue
        effective = [(label, pred) for label, pred in zip(labels, predictions) if pred != 0]
        coverage = len(effective) / label_count if label_count else 0.0
        if not effective:
            metrics.append({"accuracy": float("nan"), "f1": float("nan"), "coverage": coverage})
            continue
        y_true = [label for label, _ in effective]
        y_pred = [1 if pred == 1 else 0 for _, pred in effective]
        acc, f1 = compute_accuracy_and_macro_f1(y_true, y_pred, labels=[0, 1])
        metrics.append({"accuracy": acc, "f1": f1, "coverage": coverage})
    return metrics


def filter_rules_by_coverage(
    rules: List[str],
    train_vectors: List[List[int]],
    train_metrics: List[Dict[str, float]],
    threshold: float,
    *,
    val_vectors: Optional[List[List[int]]] = None,
    val_metrics: Optional[List[Dict[str, float]]] = None,
    ensure_one: bool = False,
    context: str = "",
    logger=None,
):
    if threshold <= 0 or not rules:
        return rules, train_vectors, train_metrics, val_vectors, val_metrics, 0

    coverages = []
    keep_indices: List[int] = []
    for idx, metric in enumerate(train_metrics):
        coverage = float(metric.get("coverage", 0.0) or 0.0)
        if np.isnan(coverage):
            coverage = 0.0
        coverages.append(coverage)
        if coverage >= threshold:
            keep_indices.append(idx)

    if not keep_indices:
        if ensure_one and coverages:
            best_idx = int(np.argmax(coverages))
            keep_indices = [best_idx]
            if logger is not None:
                logger.info(
                    "%s: 所有规则覆盖率低于阈值 %.3f，仅保留覆盖率最高的规则 (%.3f)",
                    context or "coverage_filter",
                    threshold,
                    coverages[best_idx],
                )
        else:
            removed = len(rules)
            if removed and logger is not None:
                logger.info(
                    "%s: 移除了 %d 条覆盖率低于 %.3f 的规则",
                    context or "coverage_filter",
                    removed,
                    threshold,
                )
            return [], [], [], None if val_vectors is None else [], None if val_metrics is None else [], removed

    removed = len(rules) - len(keep_indices)
    if removed > 0 and logger is not None:
        logger.info(
            "%s: 移除了 %d 条覆盖率低于 %.3f 的规则",
            context or "coverage_filter",
            removed,
            threshold,
        )

    filtered_rules = [rules[idx] for idx in keep_indices]
    filtered_train_vectors = [train_vectors[idx] for idx in keep_indices]
    filtered_train_metrics = [train_metrics[idx] for idx in keep_indices]
    filtered_val_vectors = None if val_vectors is None else [val_vectors[idx] for idx in keep_indices]
    filtered_val_metrics = None if val_metrics is None else [val_metrics[idx] for idx in keep_indices]

    return (
        filtered_rules,
        filtered_train_vectors,
        filtered_train_metrics,
        filtered_val_vectors,
        filtered_val_metrics,
        removed,
    )


def format_messages_for_logging(messages: Sequence[Dict[str, str]]) -> str:
    formatted: List[str] = []
    for idx, message in enumerate(messages, start=1):
        role = str(message.get("role", "")).upper() or "UNKNOWN"
        content = str(message.get("content", "")).strip()
        formatted.append(f"[{idx}] {role}: {content}")
    return "\n".join(formatted)


def annotate_label_feedback(
    samples: pd.DataFrame,
    sample_indices: List[int],
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    label_column: str,
    label_lookup: Dict[int, str],
) -> pd.DataFrame:
    if samples.empty or label_column not in samples.columns:
        return samples
    if probabilities.size == 0 or true_labels.size == 0:
        return samples

    annotated = samples.copy()
    predicted_labels = (probabilities >= 0.5).astype(int)
    max_rows = len(annotated)
    for row_pos, sample_idx in enumerate(sample_indices):
        if row_pos >= max_rows:
            break
        if sample_idx >= len(predicted_labels) or sample_idx >= len(true_labels):
            continue

        original_value = annotated.iloc[row_pos][label_column]
        true_display = str(original_value)
        if not true_display or true_display.lower() == "nan":
            true_idx = int(true_labels[sample_idx])
            true_display = label_lookup.get(true_idx, str(true_idx))

        predicted_idx = int(predicted_labels[sample_idx])
        predicted_display = label_lookup.get(predicted_idx, str(predicted_idx))
        true_key = _normalize_label(true_display)
        predicted_key = _normalize_label(predicted_display)
        if true_key == predicted_key:
            feedback = f"True Label: {true_display} (prediction matched)"
        else:
            feedback = f"True Label: {true_display} (misclassified as {predicted_display})"
        annotated.iat[row_pos, annotated.columns.get_loc(label_column)] = feedback

    return annotated


def rank_rule_contributions(
    feature_store,
    model,
    sample_indices: List[int],
    labels: List[int],
    limit: int = 10,
) -> List[Dict[str, object]]:
    if limit <= 0 or not feature_store.rules:
        return []

    _ = (model, sample_indices, labels)
    ranked = []
    for idx, info in enumerate(feature_store.rules):
        primary = info.train_f1
        secondary = info.train_accuracy
        if primary is None and secondary is None:
            continue
        score = primary if primary is not None else secondary
        ranked.append((float(score), float(secondary or 0.0), idx))

    if not ranked:
        return []

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    results: List[Dict[str, object]] = []
    for score, _, idx in ranked[: min(limit, len(ranked))]:
        rule_info = feature_store.rules[idx]
        results.append({"text": rule_info.text, "score": float(score), "round": rule_info.source_round})
    return results


def format_hypotheses_text(ranked_rules: List[Dict[str, object]]) -> str:
    if not ranked_rules:
        return "No prior hypotheses available."
    return "\n".join(f"{pos}. {info.get('text', '')}" for pos, info in enumerate(ranked_rules, start=1))
