import copy
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from rlie.llm.rule_scorer import RuleScorer
from rlie.core.metrics import compute_accuracy_and_macro_f1
from rlie.utils import concurrent_config


LOGGER = logging.getLogger("rlie.eval_utils")


def save_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def sanitize_identifier(value: str, default: str = "model") -> str:
    text = str(value or "").strip()
    if not text:
        return default
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    return safe or default


def parse_evaluation_modes(config: Dict) -> List[str]:
    raw_modes = config.get("evaluation", {}).get("modes")
    if raw_modes is None:
        return []
    if isinstance(raw_modes, str):
        modes = [chunk.strip() for chunk in raw_modes.split(",") if chunk.strip()]
    elif isinstance(raw_modes, (list, tuple)):
        modes = [str(item).strip() for item in raw_modes if str(item).strip()]
    else:
        raise ValueError("evaluation.modes 必须是字符串或列表")
    valid = {
        "regression_only",
        "llm_evaluation",
        "llm_evaluation_with_linear_regression",
        "llm_evaluation_with_linear_regression_and_label",
    }
    normalized = []
    for mode in modes:
        key = mode.lower()
        if key not in valid:
            raise ValueError(f"未知的 evaluation mode: {mode}")
        normalized.append(key)
    return normalized


def prepare_llm_config(base_cfg: Dict, model_name: str, shared_client: Optional[Dict]) -> Dict:
    cfg = copy.deepcopy(base_cfg) if base_cfg else {}
    cfg["model"] = model_name
    disable_list = cfg.pop("disable_thinking_models", None)
    if disable_list and model_name in disable_list:
        cfg.setdefault("disable_thinking", True)
    effort_map = cfg.pop("reasoning_effort_models", None)
    if effort_map:
        effort = effort_map.get(model_name)
        if effort:
            cfg.setdefault("reasoning_effort", effort)
    template_kwargs_map = cfg.pop("chat_template_kwargs_models", None)
    if template_kwargs_map:
        template_kwargs = template_kwargs_map.get(model_name)
        if template_kwargs:
            cfg.setdefault("chat_template_kwargs", template_kwargs)
    client_cfg = copy.deepcopy(cfg.get("client") or {})
    if shared_client:
        client_cfg.setdefault("base_url", shared_client.get("base_url"))
        client_cfg.setdefault("api_key", shared_client.get("api_key"))
    if not client_cfg:
        raise ValueError("LLM config missing client credentials (set llms.client.base_url/api_key)")
    cfg["client"] = client_cfg
    return cfg


def _evaluate_regression_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: float,
    metric_labels: Sequence[int],
) -> Tuple[float, float, np.ndarray]:
    if weights.size < features.shape[1]:
        pad = np.zeros(features.shape[1] - weights.size, dtype=float)
        weights = np.concatenate([weights, pad])
    elif weights.size > features.shape[1]:
        weights = weights[: features.shape[1]]
    logits = np.dot(features, weights) + bias
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    acc, f1 = compute_accuracy_and_macro_f1(labels, preds, labels=metric_labels)
    return acc, f1, preds


def evaluate_models(
    *,
    run_dir: str,
    data_loader,
    prompt,
    evaluation_model_cfgs: List[Tuple[str, Dict]],
    evaluation_modes: Sequence[str],
    rule_texts: List[str],
    weights: np.ndarray,
    bias: float,
    active_texts: List[str],
    weighted_hypotheses: List[Tuple[str, float]],
    final_test_preds: Optional[np.ndarray],
    test_df: pd.DataFrame,
    test_label_list: List[int],
    test_labels: np.ndarray,
    metric_labels: Sequence[int],
    label_lookup: Dict[int, str],
    best_round: int,
    skip_regression_model: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    if not evaluation_model_cfgs:
        LOGGER.info("No evaluation models configured; skipping.")
        return results

    should_run_regression = "regression_only" in evaluation_modes
    should_run_llm = "llm_evaluation" in evaluation_modes
    should_run_llm_with_lr = "llm_evaluation_with_linear_regression" in evaluation_modes
    should_run_llm_with_label = (
        "llm_evaluation_with_linear_regression_and_label" in evaluation_modes
    )

    can_run_llm_modes = not test_df.empty and bool(active_texts)
    can_run_llm_lr = can_run_llm_modes and bool(weighted_hypotheses)
    can_run_llm_label = can_run_llm_lr and final_test_preds is not None

    llm_modes: List[str] = []
    if should_run_llm and can_run_llm_modes:
        llm_modes.append("llm_evaluation")
    if should_run_llm_with_lr and can_run_llm_lr:
        llm_modes.append("llm_evaluation_with_linear_regression")
    if should_run_llm_with_label and can_run_llm_label:
        llm_modes.append("llm_evaluation_with_linear_regression_and_label")

    metric_file_bases = {
        "llm_evaluation": "final_test_metrics_llm_evaluation",
        "llm_evaluation_with_linear_regression": "final_test_metrics_llm_evaluation_with_linear_regression",
        "llm_evaluation_with_linear_regression_and_label": "final_test_metrics_llm_evaluation_with_linear_regression_and_label",
    }

    # 模型级并发预算分配
    model_configs_with_budget = []
    for model_name, cfg in evaluation_model_cfgs:
        cfg_copy = copy.deepcopy(cfg)

        # 如果有 repeat 级别注入的预算，分配给各个模型
        if "_concurrent_budget" in cfg_copy:
            repeat_budget = cfg_copy["_concurrent_budget"]
            num_models = len(evaluation_model_cfgs)

            # 使用 allocate_concurrent_budget 分配
            # 假设每个模型内部有多个样本需要评估
            _, per_model_budget = concurrent_config.allocate_concurrent_budget(
                total_limit=repeat_budget,
                outer_parallelism=num_models,
                inner_parallelism=100,  # 假设样本数量（实际会根据样本数动态调整）
                min_inner=5,
            )
            cfg_copy["_concurrent_budget"] = per_model_budget
            LOGGER.debug(
                "模型 %s 分配并发预算: %d (repeat预算=%d, 模型数=%d)",
                model_name,
                per_model_budget,
                repeat_budget,
                num_models,
            )

        model_configs_with_budget.append((model_name, cfg_copy))

    def _score_regression_model(model_name: str, cfg: Dict):
        if skip_regression_model and model_name == skip_regression_model:
            LOGGER.info(
                "Skip regression_only for model %s because it matches rule_learning_model",
                model_name,
            )
            return None
        checker = RuleScorer(
            cfg,
            prompt,
            data_loader.label_mapping,
            label_values=getattr(data_loader, "label_values", None),
            disable_progress=True,
        )
        LOGGER.info("Recomputing regression features using model %s", model_name)
        rule_vectors = checker.batch_score(rule_texts, test_df, labels=test_label_list)
        if not rule_vectors:
            LOGGER.warning("Regression rescore (%s) returned no vectors; skipping", model_name)
            return None
        feature_matrix = np.array(rule_vectors, dtype=int).T
        acc_reg, f1_reg, _ = _evaluate_regression_metrics(
            feature_matrix,
            test_labels,
            weights,
            bias,
            metric_labels,
        )
        safe_name = sanitize_identifier(model_name)
        save_json(
            os.path.join(run_dir, f"final_test_metrics_regression_only__{safe_name}.json"),
            {
                "accuracy": acc_reg,
                "f1": f1_reg,
                "round": best_round,
                "mode": "regression_only",
                "coverage": 1.0,
            },
        )
        results.setdefault(model_name, {})["regression_only"] = {
            "accuracy": acc_reg,
            "f1": f1_reg,
            "coverage": 1.0,
        }
        return feature_matrix

    if should_run_regression:
        if not model_configs_with_budget:
            LOGGER.info("实验A：无评估模型可用，跳过")
        else:
            first_model_name, first_cfg = model_configs_with_budget[0]
            LOGGER.info("实验A：回归重打分 - 只运行第一个评估模型: %s", first_model_name)
            try:
                _score_regression_model(first_model_name, first_cfg)
                LOGGER.info("实验A：模型 %s 回归重打分完成", first_model_name)
            except Exception as exc:
                LOGGER.error("实验A：模型 %s 回归重打分失败: %s", first_model_name, exc)

    if not llm_modes or not model_configs_with_budget:
        LOGGER.info("未启用 LLM 评估模式或无模型可跑，跳过。")
        return results

    def _run_llm_modes(model_name: str, cfg: Dict):
        checker = RuleScorer(
            cfg,
            prompt,
            data_loader.label_mapping,
            label_values=getattr(data_loader, "label_values", None),
            disable_progress=len(model_configs_with_budget) > 1,
        )
        safe_name = sanitize_identifier(model_name)

        if "llm_evaluation" in llm_modes:
            predictions = checker.evaluate_rule_set(active_texts, test_df)
            if predictions:
                is_three_state = getattr(checker, "_use_three_state", False)
                if is_three_state:
                    effective = [
                        (label, pred)
                        for label, pred in zip(test_label_list, predictions)
                        if pred != 0
                    ]
                else:
                    effective = list(zip(test_label_list, predictions))
                total = len(test_label_list)
                coverage = len(effective) / total if total else 0.0
                if effective:
                    y_true = [label for label, _ in effective]
                    y_pred = [1 if pred == 1 else 0 for _, pred in effective]
                    acc, f1 = compute_accuracy_and_macro_f1(y_true, y_pred, labels=metric_labels)
                else:
                    acc = float("nan")
                    f1 = float("nan")
                save_json(
                    os.path.join(run_dir, f"{metric_file_bases['llm_evaluation']}__{safe_name}.json"),
                    {
                        "accuracy": acc,
                        "f1": f1,
                        "coverage": coverage,
                        "round": best_round,
                        "mode": "llm_evaluation",
                        "active_rules": len(active_texts),
                    },
                )
                results.setdefault(model_name, {})["llm_evaluation"] = {
                    "accuracy": acc,
                    "f1": f1,
                    "coverage": coverage,
                }
            else:
                LOGGER.warning("LLM evaluation (%s) 未返回任何预测", model_name)

        if "llm_evaluation_with_linear_regression" in llm_modes:
            predictions_lr = checker.evaluate_rule_set_with_linear_regression(
                weighted_hypotheses,
                bias,
                test_df,
            )
            if predictions_lr:
                effective_lr = list(zip(test_label_list, predictions_lr))
                total_lr = len(test_label_list)
                coverage_lr = len(effective_lr) / total_lr if total_lr else 0.0
                if effective_lr:
                    y_true = [label for label, _ in effective_lr]
                    y_pred = [1 if pred == 1 else 0 for _, pred in effective_lr]
                    acc_lr, f1_lr = compute_accuracy_and_macro_f1(y_true, y_pred, labels=metric_labels)
                else:
                    acc_lr = float("nan")
                    f1_lr = float("nan")
                save_json(
                    os.path.join(
                        run_dir,
                        f"{metric_file_bases['llm_evaluation_with_linear_regression']}__{safe_name}.json",
                    ),
                    {
                        "accuracy": acc_lr,
                        "f1": f1_lr,
                        "coverage": coverage_lr,
                        "round": best_round,
                        "mode": "llm_evaluation_with_linear_regression",
                        "active_rules": len(weighted_hypotheses),
                    },
                )
                results.setdefault(model_name, {})["llm_evaluation_with_linear_regression"] = {
                    "accuracy": acc_lr,
                    "f1": f1_lr,
                    "coverage": coverage_lr,
                }
            else:
                LOGGER.warning("LLM evaluation with linear regression (%s) 未返回任何预测", model_name)

        if "llm_evaluation_with_linear_regression_and_label" in llm_modes:
            if final_test_preds is None:
                LOGGER.warning(
                    "LLM evaluation with linear regression + label (%s) 依赖回归预测，当前不可用",
                    model_name,
                )
            else:
                predicted_texts = [
                    label_lookup.get(int(pred), str(int(pred)))
                    for pred in final_test_preds
                ]
                predictions_label = checker.evaluate_rule_set_with_label(
                    weighted_hypotheses,
                    bias,
                    test_df,
                    predicted_texts,
                )
                if predictions_label:
                    effective_label = list(zip(test_label_list, predictions_label))
                    total_label = len(test_label_list)
                    coverage_label = len(effective_label) / total_label if total_label else 0.0
                    if effective_label:
                        y_true = [label for label, _ in effective_label]
                        y_pred = [1 if pred == 1 else 0 for _, pred in effective_label]
                        acc_label, f1_label = compute_accuracy_and_macro_f1(
                            y_true,
                            y_pred,
                            labels=metric_labels,
                        )
                    else:
                        acc_label = float("nan")
                        f1_label = float("nan")
                    save_json(
                        os.path.join(
                            run_dir,
                            f"{metric_file_bases['llm_evaluation_with_linear_regression_and_label']}__{safe_name}.json",
                        ),
                        {
                            "accuracy": acc_label,
                            "f1": f1_label,
                            "coverage": coverage_label,
                            "round": best_round,
                            "mode": "llm_evaluation_with_linear_regression_and_label",
                            "active_rules": len(active_texts),
                        },
                    )
                    results.setdefault(model_name, {})[
                        "llm_evaluation_with_linear_regression_and_label"
                    ] = {
                        "accuracy": acc_label,
                        "f1": f1_label,
                        "coverage": coverage_label,
                    }
                else:
                    LOGGER.warning(
                        "LLM evaluation with linear regression + label (%s) 未返回任何预测",
                        model_name,
                    )

    LOGGER.info("实验B：使用基准模型的特征，遍历所有评估模型跑 LLM 策略 - 并行执行 %d 个评估模型", len(model_configs_with_budget))
    with ThreadPoolExecutor(max_workers=len(model_configs_with_budget)) as executor:
        futures = {
            executor.submit(_run_llm_modes, model_name, cfg): model_name
            for model_name, cfg in model_configs_with_budget
        }
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                future.result()
                LOGGER.info("实验B：模型 %s LLM评估完成", model_name)
            except Exception as exc:
                LOGGER.error("实验B：模型 %s LLM评估失败: %s", model_name, exc)

    LOGGER.info("评估阶段完成，结果保存在 %s", run_dir)
    return results
