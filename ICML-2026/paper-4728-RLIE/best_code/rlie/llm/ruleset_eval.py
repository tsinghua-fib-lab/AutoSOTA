import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm.auto import tqdm

from rlie.core.metrics import compute_accuracy_and_macro_f1
from rlie.llm.parsing import allowed_answers, append_answer_constraint
from rlie.utils import concurrent_config


LOGGER = logging.getLogger("rlie.rule_scorer.ruleset_eval")


def batch_score_serial(
    checker,
    rules: List[str],
    samples: pd.DataFrame,
    labels: Optional[Sequence[int]] = None,
) -> List[List[int]]:
    matrix: List[List[int]] = []
    total_samples = len(samples)

    if total_samples == 0:
        if not checker._disable_tqdm:
            tqdm.write("No samples provided for scoring; skipping metrics.")
        return [[] for _ in rules]

    labels_array: Optional[List[int]] = list(labels) if labels is not None else None

    for idx, rule in enumerate(rules, start=1):
        bar_desc = f"Rule {idx}/{len(rules)}"
        bar = tqdm(
            total=total_samples,
            desc=bar_desc,
            unit="sample",
            leave=True,
            disable=checker._disable_tqdm,
        )
        predictions = checker.score_rule(rule, samples, progress_callback=bar.update)

        if labels_array is not None:
            effective_pairs = [
                (label, pred)
                for label, pred in zip(labels_array, predictions)
                if pred != 0
            ]
            coverage = len(effective_pairs) / len(labels_array) if labels_array else 0.0
            if effective_pairs:
                y_true = [label for label, _ in effective_pairs]
                y_pred = [1 if pred == 1 else 0 for _, pred in effective_pairs]
                acc, f1 = compute_accuracy_and_macro_f1(
                    y_true,
                    y_pred,
                    labels=checker.metric_labels,
                )
                bar.set_postfix(
                    {
                        "acc": f"{acc:.3f}",
                        "f1": f"{f1:.3f}",
                        "cov": f"{coverage:.2f}",
                    }
                )
            else:
                bar.set_postfix({"acc": "nan", "f1": "nan", "cov": f"{coverage:.2f}"})

        bar.close()
        matrix.append(predictions)

    return matrix


def batch_score_parallel(
    checker,
    rules: List[str],
    samples: pd.DataFrame,
    labels: Optional[Sequence[int]] = None,
    max_rule_workers: Optional[int] = None,
) -> List[List[int]]:
    if not rules or len(samples) == 0:
        if not checker._disable_tqdm and len(samples) == 0:
            tqdm.write("No samples provided for parallel scoring; skipping.")
        return [[] for _ in rules]

    original_max_concurrent = checker.max_concurrent
    if max_rule_workers is not None:
        rule_workers, adjusted_concurrent = max_rule_workers, original_max_concurrent
    else:
        rule_workers, adjusted_concurrent = concurrent_config.allocate_concurrent_budget(
            total_limit=original_max_concurrent,
            outer_parallelism=len(rules),
            inner_parallelism=len(samples),
            min_inner=5,
        )

    if rule_workers > 1:
        checker.max_concurrent = adjusted_concurrent
        LOGGER.info(
            "规则间并行: %d 规则 × %d 样本并发 = %d 实际总并发 (预算=%d)",
            rule_workers,
            adjusted_concurrent,
            rule_workers * adjusted_concurrent,
            original_max_concurrent,
        )

    labels_array = list(labels) if labels is not None else None

    def score_single_rule(args):
        idx, rule = args
        bar_desc = f"Rule {idx}/{len(rules)}"
        bar = tqdm(
            total=len(samples),
            desc=bar_desc,
            unit="sample",
            leave=True,
            disable=checker._disable_tqdm or rule_workers > 1,
        )

        try:
            predictions = checker.score_rule(rule, samples, progress_callback=bar.update)
            if labels_array is not None:
                effective_pairs = [
                    (label, pred)
                    for label, pred in zip(labels_array, predictions)
                    if pred != 0
                ]
                coverage = len(effective_pairs) / len(labels_array) if labels_array else 0.0
                if effective_pairs:
                    y_true = [label for label, _ in effective_pairs]
                    y_pred = [1 if pred == 1 else 0 for _, pred in effective_pairs]
                    acc, f1 = compute_accuracy_and_macro_f1(
                        y_true,
                        y_pred,
                        labels=checker.metric_labels,
                    )
                    bar.set_postfix(
                        {"acc": f"{acc:.3f}", "f1": f"{f1:.3f}", "cov": f"{coverage:.2f}"}
                    )
                else:
                    bar.set_postfix({"acc": "nan", "f1": "nan", "cov": f"{coverage:.2f}"})

            return predictions
        finally:
            bar.close()

    matrix: List[List[int]] = []
    try:
        if rule_workers == 1:
            for args in enumerate(rules, start=1):
                matrix.append(score_single_rule(args))
        else:
            with ThreadPoolExecutor(max_workers=rule_workers) as executor:
                matrix = list(executor.map(score_single_rule, enumerate(rules, start=1)))
    finally:
        checker.max_concurrent = original_max_concurrent

    return matrix


def batch_score(
    checker,
    rules: List[str],
    samples: pd.DataFrame,
    labels: Optional[Sequence[int]] = None,
) -> List[List[int]]:
    if len(rules) <= 2:
        LOGGER.debug("使用串行评分模式 (规则数=%d)", len(rules))
        return batch_score_serial(checker, rules, samples, labels)
    LOGGER.debug("使用并行评分模式 (规则数=%d)", len(rules))
    return batch_score_parallel(checker, rules, samples, labels)


def evaluate_rule_set(
    checker,
    hypotheses: Sequence[str],
    samples: pd.DataFrame,
) -> List[int]:
    if not hypotheses:
        return []

    total_samples = len(samples)
    if total_samples == 0:
        return []

    bar = tqdm(
        total=total_samples,
        desc="LLM Eval",
        unit="sample",
        leave=True,
        disable=checker._disable_tqdm,
    )

    if checker._provider == "remote" and checker._use_async_remote:
        try:
            return _score_ruleset_remote(
                checker,
                hypotheses,
                samples,
                list(range(total_samples)),
                progress_callback=bar.update,
            )
        finally:
            bar.close()

    def _evaluate(row_idx: int) -> int:
        messages = checker._format_multiple_messages(hypotheses, samples, row_idx)
        return checker._call_and_parse(messages, context={"sample_idx": row_idx})

    results: List[int] = []
    try:
        if checker.max_concurrent == 1:
            for idx in range(total_samples):
                results.append(_evaluate(idx))
                bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=checker.max_concurrent) as executor:
                results = [0] * total_samples
                future_to_pos = {
                    executor.submit(_evaluate, idx): idx
                    for idx in range(total_samples)
                }
                for future in as_completed(future_to_pos):
                    pos = future_to_pos[future]
                    try:
                        results[pos] = future.result()
                    except Exception:
                        results[pos] = 0
                    bar.update(1)
    finally:
        bar.close()

    return results


def _score_ruleset_remote(
    checker,
    hypotheses: Sequence[str],
    samples: pd.DataFrame,
    indices: List[int],
    progress_callback: Optional[Callable[[int], None]] = None,
) -> List[int]:
    async def _gather() -> List[int]:
        semaphore = asyncio.Semaphore(checker.max_concurrent)
        results: List[int] = [0 for _ in indices]

        async def _evaluate(position: int, row_idx: int):
            messages = checker._format_multiple_messages(hypotheses, samples, row_idx)
            results[position] = await checker._call_and_parse_async(
                messages,
                semaphore,
                context={"sample_idx": row_idx},
            )
            if progress_callback is not None:
                progress_callback(1)

        tasks = [
            asyncio.create_task(_evaluate(pos, row_idx))
            for pos, row_idx in enumerate(indices)
        ]
        for coro in asyncio.as_completed(tasks):
            await coro
        return results

    return checker._run_async(_gather())


def evaluate_rule_set_with_label(
    checker,
    weighted_hypotheses: Sequence[Tuple[str, float]],
    bias: float,
    samples: pd.DataFrame,
    predicted_labels: Sequence[str],
) -> List[int]:
    if not weighted_hypotheses or len(samples) == 0:
        return []
    if len(predicted_labels) != len(samples):
        raise ValueError("Predicted labels length must match number of samples for LLM evaluation")

    total_samples = len(samples)
    bar = tqdm(
        total=total_samples,
        desc="LLM Eval (with label)",
        unit="sample",
        leave=True,
        disable=checker._disable_tqdm,
    )

    if checker._provider == "remote" and checker._use_async_remote:
        try:
            return _score_ruleset_with_label_remote(
                checker,
                weighted_hypotheses,
                bias,
                samples,
                list(range(total_samples)),
                predicted_labels,
                progress_callback=bar.update,
            )
        finally:
            bar.close()

    def _evaluate(row_idx: int) -> int:
        messages = checker._format_multiple_with_label_messages(
            weighted_hypotheses,
            bias,
            samples,
            row_idx,
            predicted_labels[row_idx],
        )
        return checker._call_and_parse(messages, context={"sample_idx": row_idx})

    results: List[int] = []
    try:
        if checker.max_concurrent == 1:
            for idx in range(total_samples):
                results.append(_evaluate(idx))
                bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=checker.max_concurrent) as executor:
                results = [0] * total_samples
                future_to_pos = {
                    executor.submit(_evaluate, idx): idx
                    for idx in range(total_samples)
                }
                for future in as_completed(future_to_pos):
                    pos = future_to_pos[future]
                    try:
                        results[pos] = future.result()
                    except Exception:
                        results[pos] = 0
                    bar.update(1)
    finally:
        bar.close()

    return results


def _score_ruleset_with_label_remote(
    checker,
    weighted_hypotheses: Sequence[Tuple[str, float]],
    bias: float,
    samples: pd.DataFrame,
    indices: List[int],
    predicted_labels: Sequence[str],
    progress_callback: Optional[Callable[[int], None]] = None,
) -> List[int]:
    async def _gather() -> List[int]:
        semaphore = asyncio.Semaphore(checker.max_concurrent)
        results: List[int] = [0 for _ in indices]

        async def _evaluate(position: int, row_idx: int):
            messages = checker._format_multiple_with_label_messages(
                weighted_hypotheses,
                bias,
                samples,
                row_idx,
                predicted_labels[row_idx],
            )
            results[position] = await checker._call_and_parse_async(
                messages,
                semaphore,
                context={"sample_idx": row_idx},
            )
            if progress_callback is not None:
                progress_callback(1)

        tasks = [
            asyncio.create_task(_evaluate(pos, row_idx))
            for pos, row_idx in enumerate(indices)
        ]
        for coro in asyncio.as_completed(tasks):
            await coro
        return results

    return checker._run_async(_gather())


def evaluate_rule_set_with_linear_regression(
    checker,
    weighted_hypotheses: Sequence[Tuple[str, float]],
    bias: float,
    samples: pd.DataFrame,
) -> List[int]:
    if not weighted_hypotheses or len(samples) == 0:
        return []

    total_samples = len(samples)
    bar = tqdm(
        total=total_samples,
        desc="LLM Eval (linear regression)",
        unit="sample",
        leave=True,
        disable=checker._disable_tqdm,
    )

    if checker._provider == "remote" and checker._use_async_remote:
        try:
            return _score_ruleset_with_linear_regression_remote(
                checker,
                weighted_hypotheses,
                bias,
                samples,
                list(range(total_samples)),
                progress_callback=bar.update,
            )
        finally:
            bar.close()

    def _evaluate(row_idx: int) -> int:
        messages = checker.prompt.multiple_hypotheses_inference_with_linear_regression(
            list(weighted_hypotheses),
            bias,
            samples,
            row_idx,
        )
        messages = append_answer_constraint(
            messages,
            allowed_answers(
                checker.canonical_positive,
                checker.canonical_negative,
                include_not_applicable=False,
            ),
        )
        return checker._call_and_parse(messages, context={"sample_idx": row_idx})

    results: List[int] = []
    try:
        if checker.max_concurrent == 1:
            for idx in range(total_samples):
                results.append(_evaluate(idx))
                bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=checker.max_concurrent) as executor:
                results = [0] * total_samples
                future_to_pos = {
                    executor.submit(_evaluate, idx): idx
                    for idx in range(total_samples)
                }
                for future in as_completed(future_to_pos):
                    pos = future_to_pos[future]
                    try:
                        results[pos] = future.result()
                    except Exception:
                        results[pos] = 0
                    bar.update(1)
    finally:
        bar.close()

    return results


def _score_ruleset_with_linear_regression_remote(
    checker,
    weighted_hypotheses: Sequence[Tuple[str, float]],
    bias: float,
    samples: pd.DataFrame,
    indices: List[int],
    progress_callback: Optional[Callable[[int], None]] = None,
) -> List[int]:
    async def _gather() -> List[int]:
        semaphore = asyncio.Semaphore(checker.max_concurrent)
        results: List[int] = [0 for _ in indices]

        async def _evaluate(position: int, row_idx: int):
            messages = checker.prompt.multiple_hypotheses_inference_with_linear_regression(
                list(weighted_hypotheses),
                bias,
                samples,
                row_idx,
            )
            messages = append_answer_constraint(
                messages,
                allowed_answers(
                    checker.canonical_positive,
                    checker.canonical_negative,
                    include_not_applicable=False,
                ),
            )
            results[position] = await checker._call_and_parse_async(
                messages,
                semaphore,
                context={"sample_idx": row_idx},
            )
            if progress_callback is not None:
                progress_callback(1)

        tasks = [
            asyncio.create_task(_evaluate(pos, row_idx))
            for pos, row_idx in enumerate(indices)
        ]
        for coro in asyncio.as_completed(tasks):
            await coro
        return results

    return checker._run_async(_gather())
