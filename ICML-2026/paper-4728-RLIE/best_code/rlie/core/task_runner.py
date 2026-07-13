import copy
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rlie.llm.prompt import BasePrompt
from rlie.utils.logger_config import LoggerConfig
from rlie.datasets.data_loader import TaskDataLoader, _normalize_label
from rlie.llm.rule_generator import RuleGenerator
from rlie.llm.rule_scorer import RuleScorer
from rlie.datasets.labels import map_labels
from rlie.core.feature_manager import RuleFeatureStore
from rlie.core.modeling import SoftmaxTrainer
from rlie.core.convergence import ConvergenceMonitor, ConvergenceState
from rlie.core.metrics import compute_accuracy_and_macro_f1
from rlie.core.reporting import (
    resolve_metric_name,
    save_dataset_csv,
    save_json,
    save_rule_snapshot,
    should_update_best_snapshot,
)
from rlie.core.rule_utils import (
    annotate_label_feedback,
    compute_rule_metrics,
    filter_rules_by_coverage,
    format_hypotheses_text,
    format_messages_for_logging,
    rank_rule_contributions,
    select_hard_samples,
    select_seed_samples,
    sort_samples_by_label,
)
from rlie.utils import eval_utils
from rlie import run_eval_only


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
LOGGER = logging.getLogger("rlie")



def _set_global_info_enabled(enabled: bool) -> None:
    level = logging.INFO if enabled else logging.WARNING
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)
    if LoggerConfig.console_handler is not None:
        LoggerConfig.console_handler.setLevel(level)
    if LoggerConfig.file_handler is not None:
        LoggerConfig.file_handler.setLevel(level)

def run_single_task(
    config: Dict,
    evaluation_modes: List[str],
    task_name: str,
    base_output_dir: str,
    repeat_index: int,
    total_repeats: int,
    config_path: str,
):
    logging_cfg = config.get("logging", {}) or {}
    run_name = logging_cfg.get("run_name")
    timestamped = bool(logging_cfg.get("timestamped"))

    subdir_parts = []
    if run_name:
        subdir_parts.append(str(run_name))
    elif timestamped:
        subdir_parts.append(datetime.now().strftime("run_%Y%m%d_%H%M%S"))

    if task_name:
        subdir_parts.append(task_name.replace("/", "_"))

    final_output_dir = base_output_dir
    if total_repeats > 1:
        subdir_parts.append(f"repeat{repeat_index + 1}")

    if subdir_parts:
        final_output_dir = os.path.join(base_output_dir, "_".join(subdir_parts))

    os.makedirs(final_output_dir, exist_ok=True)
    logging_cfg["output_dir"] = final_output_dir
    config["logging"] = logging_cfg
    output_dir = logging_cfg.get("output_dir")
    LOGGER.info("[%s] Output directory: %s", task_name or "default", final_output_dir)

    print_generation_prompts = bool(logging_cfg.get("print_generation_prompts", True))

    coverage_threshold = float(config.get("rule_coverage_threshold", 0.2) or 0.0)
    if coverage_threshold < 0:
        coverage_threshold = 0.0

    data_loader = TaskDataLoader(config)
    dataset = data_loader.load_dataset()

    generation_cfg = config.get("generation", {})
    sample_size = int(generation_cfg.get("sample_size", config.get("num_seed_examples", 3)))
    rules_per_call = int(generation_cfg.get("rules_per_call", sample_size))
    if sample_size <= 0:
        raise ValueError("generation.sample_size must be positive")
    if rules_per_call <= 0:
        raise ValueError("generation.rules_per_call must be positive")

    prompt = BasePrompt(data_loader.task)
    llm_cfg = config.get("llms", {}) or {}
    llm_generation_cfg = copy.deepcopy(llm_cfg.get("generation") or {})
    llm_evaluation_cfg = copy.deepcopy(llm_cfg.get("evaluation") or {})
    shared_client_cfg = copy.deepcopy(llm_cfg.get("client") or {})

    rule_learning_model = llm_generation_cfg.pop("rule_learning_model", llm_generation_cfg.get("model"))
    if not rule_learning_model:
        raise ValueError("Config.llms.generation.rule_learning_model must be defined")

    evaluation_model_names = llm_generation_cfg.pop("evaluation_models", None)
    if evaluation_model_names is None:
        evaluation_model_names = []
    elif not isinstance(evaluation_model_names, list):
        raise ValueError("Config.llms.generation.evaluation_models must be a list")
    evaluation_model_names = [str(name) for name in evaluation_model_names if str(name).strip()]

    if not llm_evaluation_cfg:
        raise ValueError("Config.llms.evaluation must be defined")

    generation_llm_cfg = eval_utils.prepare_llm_config(
        llm_generation_cfg,
        rule_learning_model,
        shared_client=shared_client_cfg,
    )
    rule_learning_checker_cfg = eval_utils.prepare_llm_config(
        llm_evaluation_cfg,
        rule_learning_model,
        shared_client=generation_llm_cfg.get("client"),
    )
    if output_dir:
        generation_llm_cfg.setdefault("output_dir", output_dir)
        rule_learning_checker_cfg.setdefault("output_dir", output_dir)

    if not evaluation_model_names:
        fallback_model = llm_evaluation_cfg.get("model")
        if fallback_model:
            if isinstance(fallback_model, list):
                evaluation_model_names = [str(name) for name in fallback_model if str(name).strip()]
            else:
                evaluation_model_names = [str(fallback_model)]

    rule_generator = RuleGenerator(generation_llm_cfg)
    rule_checker = RuleScorer(
        rule_learning_checker_cfg,
        prompt,
        data_loader.label_mapping,
        label_values=getattr(data_loader, "label_values", None),
    )

    evaluation_model_cfgs: List[Tuple[str, Dict]] = []
    for model_name in evaluation_model_names:
        per_model_cfg = eval_utils.prepare_llm_config(
            llm_evaluation_cfg,
            model_name,
            shared_client=generation_llm_cfg.get("client"),
        )
        if output_dir:
            per_model_cfg.setdefault("output_dir", output_dir)
        evaluation_model_cfgs.append((model_name, per_model_cfg))

    train_df = dataset.train.reset_index(drop=True)
    val_df = dataset.val.reset_index(drop=True)
    test_df = dataset.test.reset_index(drop=True)

    feature_store = RuleFeatureStore(
        samples=train_df,
        label_column=data_loader.task.label_name,
        val_samples=val_df if len(val_df) > 0 else None,
    )
    trainer = SoftmaxTrainer(config)
    convergence = ConvergenceMonitor(config)
    rule_cap_value = config.get("max_rules")
    max_rules = int(rule_cap_value) if rule_cap_value is not None else None
    prune_metric = config.get("rule_prune_metric", "f1")
    best_model_metric = resolve_metric_name(config.get("best_model_metric", "accuracy"))
    best_metric_display = "F1" if best_model_metric == "f1" else "Accuracy"
    train_metric_name = "f1" if best_model_metric == "f1" else "accuracy"

    best_metric_value = float("-inf")
    best_snapshot = None

    label_mapping = data_loader.label_mapping
    metric_labels = sorted(set(label_mapping.values()))
    train_labels = np.array(map_labels(train_df, label_mapping, data_loader.task.label_name))
    train_label_list = train_labels.tolist()
    val_labels = np.array(map_labels(val_df, label_mapping, data_loader.task.label_name)) if len(val_df) > 0 else np.array([])
    val_label_list = val_labels.tolist()
    test_labels = np.array(map_labels(test_df, label_mapping, data_loader.task.label_name)) if len(test_df) > 0 else np.array([])
    test_label_list = test_labels.tolist()
    # Build a reverse lookup that preserves the original label casing for prompt display.
    label_lookup: Dict[int, str] = {idx: key for key, idx in label_mapping.items()}
    label_column = data_loader.task.label_name
    for df in (train_df, val_df, test_df):
        if label_column not in df:
            continue
        for value in df[label_column].dropna().unique():
            normalized = _normalize_label(value)
            idx = label_mapping.get(normalized)
            if idx is not None:
                label_lookup[idx] = str(value)

    round_idx = 0
    seed_samples = select_seed_samples(train_df, sample_size, seed=42)
    seed_samples = sort_samples_by_label(seed_samples, label_column, label_mapping)
    LOGGER.info(
        "Round %d: requesting %d hypotheses (sample_size=%d) from generation model %s",
        round_idx,
        rules_per_call,
        len(seed_samples),
        generation_llm_cfg.get("model"),
    )
    generation_messages = prompt.batched_generation(seed_samples, rules_per_call)
    if print_generation_prompts:
        LOGGER.info(
            "Generation prompt (round %d initial):\n%s",
            round_idx,
            format_messages_for_logging(generation_messages),
        )
    try:
        rules = rule_generator.generate(
            generation_messages,
            rules_per_call,
        )
    except Exception as exc:
        LOGGER.error("Failed to generate initial hypotheses: %s", exc)
        return
    LOGGER.info("Round %d: generated %d initial rules", round_idx, len(rules))

    LOGGER.info("Round %d: scoring %d rules locally (train)", round_idx, len(rules))
    rule_vectors = rule_checker.batch_score(rules, train_df, labels=train_label_list)
    rule_metrics = compute_rule_metrics(rule_vectors, train_label_list)
    if len(val_df) > 0:
        LOGGER.info("Round %d: scoring %d rules locally (val)", round_idx, len(rules))
        val_rule_vectors = rule_checker.batch_score(rules, val_df, labels=val_label_list)
        val_rule_metrics = compute_rule_metrics(val_rule_vectors, val_label_list)
    else:
        val_rule_vectors = None
        val_rule_metrics = None

    (
        rules,
        rule_vectors,
        rule_metrics,
        val_rule_vectors,
        val_rule_metrics,
        removed_low_cov,
    ) = filter_rules_by_coverage(
        rules,
        rule_vectors,
        rule_metrics,
        coverage_threshold,
        val_vectors=val_rule_vectors,
        val_metrics=val_rule_metrics,
        ensure_one=True,
        context=f"round {round_idx} initial",
        logger=LOGGER,
    )
    if not rules:
        LOGGER.error(
            "Round %d: 无规则满足覆盖率阈值 %.3f，终止任务 %s",
            round_idx,
            coverage_threshold,
            task_name,
        )
        return {}

    save_json(
        os.path.join(config["logging"]["output_dir"], f"round_{round_idx}_rules.json"),
        rules,
    )
    feature_store.add_rules(
        rules,
        rule_vectors,
        round_idx,
        val_vectors=val_rule_vectors,
        train_metrics=rule_metrics,
        val_metrics=val_rule_metrics,
    )
    removed = feature_store.prune_rules(max_rules, prune_metric)
    if removed:
        LOGGER.info(
            "Pruned %d rules using metric '%s' to enforce max=%s",
            removed,
            prune_metric,
            max_rules,
        )

    features = feature_store.to_sample_feature_matrix()
    LOGGER.info(
        "Round %d: training logistic regression on %d samples x %d rules",
        round_idx,
        features.shape[0],
        features.shape[1],
    )
    result = trainer.train(features, train_labels)
    LOGGER.info(
        "Round %d: 超参搜索选择 params=%s (metric=%s cv_best=%.4f)",
        round_idx,
        result.best_params,
        result.selection_metric,
        result.cv_best_score,
    )
    LOGGER.info("Round %d training accuracy=%.3f f1=%.3f", round_idx, result.accuracy, result.f1)
    LOGGER.info(
        "Round %d: 一共有 %d 个假设, 保留 %d 个假设",
        round_idx,
        result.total_hypotheses,
        result.retained_hypotheses,
    )

    avg_delta = convergence.compute_confidence_delta(result.probabilities, train_labels)
    LOGGER.info("Round %d average confidence delta %.4f", round_idx, avg_delta)

    if len(val_df) > 0:
        val_features = feature_store.to_sample_feature_matrix("val")
        val_probs = result.model.predict_proba(val_features)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        val_accuracy, val_f1 = compute_accuracy_and_macro_f1(
            val_labels,
            val_preds,
            labels=metric_labels,
        )
    else:
        val_accuracy = float("nan")
        val_f1 = float("nan")
        val_probs = np.array([])

    LOGGER.info("Round %d validation accuracy=%.3f f1=%.3f", round_idx, val_accuracy, val_f1)

    improvement_threshold = convergence.min_improvement
    current_metric = val_f1 if best_model_metric == "f1" else val_accuracy
    current_train_metric = result.f1 if best_model_metric == "f1" else result.accuracy
    improved_this_round = should_update_best_snapshot(
        current_metric,
        current_train_metric,
        best_snapshot,
        best_metric_value,
        improvement_threshold,
        train_metric_name,
    )
    last_val_f1 = val_f1
    last_val_metric = current_metric
    last_round_improved = improved_this_round
    if improved_this_round:
        best_metric_value = current_metric
        best_snapshot = {
            "round": round_idx,
            "model": result.model,
            "probabilities": result.probabilities.copy(),
            "best_params": dict(result.best_params),
            "cv_best_score": result.cv_best_score,
            "selection_metric": result.selection_metric,
            "train_metrics": {
                "accuracy": result.accuracy,
                "f1": result.f1,
                "avg_conf_delta": avg_delta,
                "total_hypotheses": result.total_hypotheses,
                "retained_hypotheses": result.retained_hypotheses,
            },
            "val_metrics": {
                "accuracy": val_accuracy,
                "f1": val_f1,
            },
            "rule_store": {
                "rules": copy.deepcopy(feature_store.rules),
                "train_features": feature_store.features.copy()
                if feature_store.features.size > 0
                else np.zeros((0, len(train_df))),
                "val_features": feature_store.val_features.copy()
                if feature_store.val_features.size > 0
                else np.zeros((0, len(val_df))),
            },
            "metric_value": current_metric,
        }
        LOGGER.info(
            "Round %d: 更新最佳验证%s=%.3f",
            round_idx,
            best_metric_display,
            current_metric,
        )

    save_json(
        os.path.join(config["logging"]["output_dir"], f"round_{round_idx}_metrics.json"),
        {
            "accuracy": result.accuracy,
            "f1": result.f1,
            "train_accuracy": result.accuracy,
            "train_f1": result.f1,
            "avg_conf_delta": avg_delta,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "total_hypotheses": result.total_hypotheses,
            "retained_hypotheses": result.retained_hypotheses,
            "model_selection_metric": result.selection_metric,
            "model_selection_best_score": result.cv_best_score,
            "model_selection_params": result.best_params,
        },
    )

    dataset_dir = config["logging"]["output_dir"]
    train_csv_path = os.path.join(dataset_dir, f"round_{round_idx}_train_dataset.csv")
    save_dataset_csv(features, train_label_list, train_csv_path)
    if len(val_df) > 0:
        val_features_matrix = feature_store.to_sample_feature_matrix("val")
        val_csv_path = os.path.join(dataset_dir, f"round_{round_idx}_val_dataset.csv")
        save_dataset_csv(val_features_matrix, val_label_list, val_csv_path)
    save_rule_snapshot(
        round_idx,
        feature_store,
        result.model,
        dataset_dir,
    )

    while True:
        round_idx += 1
        monitored_value = last_val_f1 if best_model_metric == "f1" else last_val_metric
        state = ConvergenceState(
            round_idx=round_idx,
            avg_confidence_delta=avg_delta,
            monitored_metric=monitored_value,
            improved=last_round_improved,
        )
        if convergence.should_stop(state):
            LOGGER.info("Convergence reached at round %d", round_idx - 1)
            break

        LOGGER.info("Round %d: selecting hard samples", round_idx)
        hard_indices = select_hard_samples(result.probabilities, train_labels, sample_size, texts=train_df["text"].tolist())
        hard_samples = train_df.iloc[hard_indices].reset_index(drop=True)
        hard_samples = sort_samples_by_label(hard_samples, label_column, label_mapping)
        hard_samples = annotate_label_feedback(
            hard_samples,
            hard_indices,
            result.probabilities,
            train_labels,
            label_column,
            label_lookup,
        )

        top_rule_data = rank_rule_contributions(
            feature_store,
            result.model,
            hard_indices,
            train_label_list,
            limit=10,
        )
        hypotheses_text = format_hypotheses_text(top_rule_data)
        if top_rule_data:
            preview_entries = []
            for info in top_rule_data[:3]:
                score = float(info.get("score", 0.0))
                text = str(info.get("text", ""))[:40]
                preview_entries.append(f"{score:.3f}:{text}")
            preview = ", ".join(preview_entries)
            LOGGER.info(
                "Round %d: top rules for refinement (score:text) %s",
                round_idx,
                preview,
            )
        else:
            LOGGER.info(
                "Round %d: no prior rules available for refinement context",
                round_idx,
            )

        refine_messages = prompt.batched_generation_refine(
            hard_samples,
            rules_per_call,
            hypotheses_text,
        )
        if print_generation_prompts:
            LOGGER.info(
                "Generation prompt (round %d refine):\n%s",
                round_idx,
                format_messages_for_logging(refine_messages),
            )
        try:
            new_rules = rule_generator.generate(refine_messages, rules_per_call)
        except Exception as exc:
            LOGGER.error("Failed to generate new hypotheses at round %d: %s", round_idx, exc)
            break
        LOGGER.info("Round %d: generated %d new rules", round_idx, len(new_rules))

        LOGGER.info("Round %d: scoring %d new rules locally (train)", round_idx, len(new_rules))
        new_rule_vectors = rule_checker.batch_score(new_rules, train_df, labels=train_label_list)
        new_rule_metrics = compute_rule_metrics(new_rule_vectors, train_label_list)
        LOGGER.info("Round %d: scoring %d new rules locally (val)", round_idx, len(new_rules))
        if len(val_df) > 0:
            new_val_rule_vectors = rule_checker.batch_score(new_rules, val_df, labels=val_label_list)
            new_val_rule_metrics = compute_rule_metrics(new_val_rule_vectors, val_label_list)
        else:
            new_val_rule_vectors = None
            new_val_rule_metrics = None

        (
            new_rules,
            new_rule_vectors,
            new_rule_metrics,
            new_val_rule_vectors,
            new_val_rule_metrics,
            removed_low_cov,
        ) = filter_rules_by_coverage(
            new_rules,
            new_rule_vectors,
            new_rule_metrics,
            coverage_threshold,
            val_vectors=new_val_rule_vectors,
            val_metrics=new_val_rule_metrics,
            ensure_one=False,
            context=f"round {round_idx} new",
            logger=LOGGER,
        )
        if not new_rules:
            LOGGER.info(
                "Round %d: 所有新规则覆盖率低于 %.3f，跳过本轮更新",
                round_idx,
                coverage_threshold,
            )
            round_idx -= 1
            continue

        save_json(
            os.path.join(config["logging"]["output_dir"], f"round_{round_idx}_rules.json"),
            new_rules,
        )
        feature_store.add_rules(
            new_rules,
            new_rule_vectors,
            round_idx,
            val_vectors=new_val_rule_vectors,
            train_metrics=new_rule_metrics,
            val_metrics=new_val_rule_metrics,
        )
        removed = feature_store.prune_rules(max_rules, prune_metric)
        if removed:
            LOGGER.info(
                "After round %d additions, pruned %d rules using metric '%s' to enforce max=%s",
                round_idx,
                removed,
                prune_metric,
                max_rules,
            )

        features = feature_store.to_sample_feature_matrix()
        LOGGER.info(
            "Round %d: training logistic regression on %d samples x %d rules",
            round_idx,
            features.shape[0],
            features.shape[1],
        )
        result = trainer.train(features, train_labels)
        avg_delta = convergence.compute_confidence_delta(result.probabilities, train_labels)
        LOGGER.info(
            "Round %d: 超参搜索选择 params=%s (metric=%s cv_best=%.4f)",
            round_idx,
            result.best_params,
            result.selection_metric,
            result.cv_best_score,
        )
        LOGGER.info(
            "Round %d training accuracy=%.3f f1=%.3f avg_delta=%.4f",
            round_idx,
            result.accuracy,
            result.f1,
            avg_delta,
        )
        LOGGER.info(
            "Round %d: 一共有 %d 个假设, 保留 %d 个假设",
            round_idx,
            result.total_hypotheses,
            result.retained_hypotheses,
        )

        if len(val_df) > 0:
            val_features = feature_store.to_sample_feature_matrix("val")
            val_probs = result.model.predict_proba(val_features)[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)
            val_accuracy, val_f1 = compute_accuracy_and_macro_f1(
                val_labels,
                val_preds,
                labels=metric_labels,
            )
        else:
            val_accuracy = float("nan")
            val_f1 = float("nan")

        LOGGER.info("Round %d validation accuracy=%.3f f1=%.3f", round_idx, val_accuracy, val_f1)

        current_metric = val_f1 if best_model_metric == "f1" else val_accuracy
        current_train_metric = result.f1 if best_model_metric == "f1" else result.accuracy
        improved_this_round = should_update_best_snapshot(
            current_metric,
            current_train_metric,
            best_snapshot,
            best_metric_value,
            improvement_threshold,
            train_metric_name,
        )
        last_val_f1 = val_f1
        last_val_metric = current_metric
        last_round_improved = improved_this_round
        if improved_this_round:
            best_metric_value = current_metric
            best_snapshot = {
                "round": round_idx,
                "model": result.model,
                "probabilities": result.probabilities.copy(),
                "best_params": dict(result.best_params),
                "cv_best_score": result.cv_best_score,
                "selection_metric": result.selection_metric,
                "train_metrics": {
                    "accuracy": result.accuracy,
                    "f1": result.f1,
                    "avg_conf_delta": avg_delta,
                    "total_hypotheses": result.total_hypotheses,
                    "retained_hypotheses": result.retained_hypotheses,
                },
                "val_metrics": {
                    "accuracy": val_accuracy,
                    "f1": val_f1,
                },
                "rule_store": {
                    "rules": copy.deepcopy(feature_store.rules),
                    "train_features": feature_store.features.copy()
                    if feature_store.features.size > 0
                    else np.zeros((0, len(train_df))),
                    "val_features": feature_store.val_features.copy()
                    if feature_store.val_features.size > 0
                    else np.zeros((0, len(val_df))),
                },
                "metric_value": current_metric,
            }
            LOGGER.info(
                "Round %d: 更新最佳验证%s=%.3f",
                round_idx,
                best_metric_display,
                current_metric,
            )

        save_json(
            os.path.join(config["logging"]["output_dir"], f"round_{round_idx}_metrics.json"),
            {
                "accuracy": result.accuracy,
                "f1": result.f1,
                "train_accuracy": result.accuracy,
                "train_f1": result.f1,
                "avg_conf_delta": avg_delta,
                "val_accuracy": val_accuracy,
                "val_f1": val_f1,
                "total_hypotheses": result.total_hypotheses,
                "retained_hypotheses": result.retained_hypotheses,
                "model_selection_metric": result.selection_metric,
                "model_selection_best_score": result.cv_best_score,
                "model_selection_params": result.best_params,
            },
        )

        train_csv_path = os.path.join(dataset_dir, f"round_{round_idx}_train_dataset.csv")
        save_dataset_csv(features, train_label_list, train_csv_path)
        if len(val_df) > 0:
            val_features_matrix = feature_store.to_sample_feature_matrix("val")
            val_csv_path = os.path.join(dataset_dir, f"round_{round_idx}_val_dataset.csv")
            save_dataset_csv(val_features_matrix, val_label_list, val_csv_path)
        save_rule_snapshot(
            round_idx,
            feature_store,
            result.model,
            dataset_dir,
        )

    if best_snapshot is None:
        LOGGER.warning("未找到有效的验证集指标，使用最后一轮模型作为最佳模型")
        best_snapshot = {
            "round": round_idx,
            "model": result.model,
            "probabilities": result.probabilities.copy(),
            "best_params": dict(result.best_params),
            "cv_best_score": result.cv_best_score,
            "selection_metric": result.selection_metric,
            "train_metrics": {
                "accuracy": result.accuracy,
                "f1": result.f1,
                "avg_conf_delta": avg_delta,
                "total_hypotheses": result.total_hypotheses,
                "retained_hypotheses": result.retained_hypotheses,
            },
            "val_metrics": {
                "accuracy": float("nan"),
                "f1": float("nan"),
            },
            "rule_store": {
                "rules": copy.deepcopy(feature_store.rules),
                "train_features": feature_store.features.copy()
                if feature_store.features.size > 0
                else np.zeros((0, len(train_df))),
                "val_features": feature_store.val_features.copy()
                if feature_store.val_features.size > 0
                else np.zeros((0, len(val_df))),
            },
            "metric_value": float("nan"),
        }
        best_metric_value = float("nan")

    best_round = best_snapshot["round"]
    LOGGER.info(
        "Best validation %s=%.3f at round %d",
        best_metric_display,
        best_snapshot.get("metric_value", float("nan")),
        best_round,
    )
    LOGGER.info(
        "Best hyperparameters params=%s (metric=%s cv_best=%.4f)",
        best_snapshot["best_params"],
        best_snapshot["selection_metric"],
        best_snapshot["cv_best_score"],
    )

    feature_store.rules = copy.deepcopy(best_snapshot["rule_store"]["rules"])
    feature_store.features = best_snapshot["rule_store"]["train_features"].copy()
    if feature_store.val_samples is not None:
        feature_store.val_features = best_snapshot["rule_store"].get(
            "val_features",
            np.zeros((0, len(feature_store.val_samples))),
        ).copy()

    final_model = best_snapshot["model"]

    # Identify active rules (non-zero coefficients) to avoid redundant scoring.
    try:
        coef = final_model.named_steps["clf"].coef_.ravel()
    except KeyError as exc:  # pragma: no cover
        raise RuntimeError("Pipeline is expected to contain a 'clf' step with coefficients") from exc
    nonzero_mask = np.abs(coef) > 1e-12
    active_indices = [idx for idx, flag in enumerate(nonzero_mask) if flag]
    LOGGER.info(
        "Best round %d uses %d/%d active rules",
        best_round,
        len(active_indices),
        len(nonzero_mask),
    )
    active_texts = [feature_store.rules[idx].text for idx in active_indices]

    final_train_matrix = feature_store.to_sample_feature_matrix()
    if final_train_matrix.size > 0:
        final_train_matrix[:, ~nonzero_mask] = 0
    save_dataset_csv(
        final_train_matrix,
        train_label_list,
        os.path.join(config["logging"]["output_dir"], "final_train_dataset.csv"),
    )
    if len(val_df) > 0:
        final_val_matrix = feature_store.to_sample_feature_matrix("val")
        if final_val_matrix.size > 0:
            final_val_matrix[:, ~nonzero_mask] = 0
        save_dataset_csv(
            final_val_matrix,
            val_label_list,
            os.path.join(config["logging"]["output_dir"], "final_val_dataset.csv"),
        )

    summary_payload = {
        "round": best_round,
        "train": best_snapshot["train_metrics"],
        "val": best_snapshot["val_metrics"],
        "hyperparameters": {
            "params": best_snapshot["best_params"],
            "selection_metric": best_snapshot["selection_metric"],
            "cv_best_score": best_snapshot["cv_best_score"],
        },
        "best_metric": {
            "name": best_model_metric,
            "value": best_snapshot.get("metric_value", float("nan")),
        },
    }
    summary_payload["rule_snapshot_dir"] = os.path.join(
        config["logging"]["output_dir"],
        "rule_snapshots",
    )
    save_json(
        os.path.join(config["logging"]["output_dir"], "final_round_metrics.json"),
        summary_payload,
    )

    output_dir = config["logging"]["output_dir"]
    # 使用 run_eval_only 的评估逻辑：实验A+B（基于已保存的快照）
    eval_payload = run_eval_only.run_eval_only(
        config_path=config_path,
        run_dir=output_dir,
        round_id=None,
        task_root=task_name,
        repeat_index=repeat_index,
    )
    # 汇总：实验A+B结果
    summaries: Dict[str, Dict[str, float]] = {}
    for block in ("experiment_A", "experiment_B"):
        block_data = eval_payload.get(block, {})
        for model_name, modes in block_data.items():
            for mode_key, metrics in modes.items():
                safe_name = eval_utils.sanitize_identifier(model_name)
                summaries[f"{block}.{mode_key}__{safe_name}"] = metrics
    return summaries
