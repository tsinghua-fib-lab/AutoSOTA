from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from collections import Counter

import numpy as np

from constants import (
    CRITIQUE_VERDICT_CORRECT,
    CRITIQUE_VERDICT_UNKNOWN,
    STATUS_FAILED,
    STATUS_ILL_POSED,
    STATUS_SUCCEEDED,
)
from data_models import AnswerEntry, load_answer_entries, load_benchmark_entries
from model_config import load_registry
from victory import VictorySide, resolve_automated_victory

from compute_bt_elo import (
    collect_decisions,
    find_critique_verdict,
    final_question,
    load_critique_verdicts,
    resolve_model,
)
from utils import critique_key, format_key, question_key


GameRecord = Tuple[str, str, str, int]


@dataclass
class ItemizedDataset:
    answerers: List[str]
    questioners: List[str]
    question_keys: List[str]
    questioner_by_question: List[int]
    edges: List[Tuple[int, int, int]]


@dataclass
class BTParams:
    beta: List[float]
    alpha: List[float]
    delta: List[float]


@dataclass
class Sigmas:
    sigma_beta: float
    sigma_alpha: float
    sigma_delta: float


def _as_float_if_scalar(value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return float(value)
    return value


def sigmoid(x):
    x_arr = np.asarray(x, dtype=float)
    out = np.empty_like(x_arr)
    pos = x_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return _as_float_if_scalar(out)


def log1pexp(x):
    x_arr = np.asarray(x, dtype=float)
    out = np.logaddexp(0.0, x_arr)
    return _as_float_if_scalar(out)


def collect_itemized_games(
    benchmarks_dir: Path,
    answers_dir: Path,
    critiques_dir: Path,
    auto_eval_dir: Path,
    registry_path: Optional[Path],
    answer_critique_mode: str,
    self_answer_critique_mode: str,
    fallback_any_mode: bool,
    log_automated_disagreements: bool = True,
) -> Tuple[List[GameRecord], Counter]:
    registry = load_registry(str(registry_path)) if registry_path and registry_path.exists() else None
    critique_verdicts = load_critique_verdicts(critiques_dir)
    decisions_by_claim = collect_decisions(auto_eval_dir)
    skip_counts: Counter = Counter()
    games: List[GameRecord] = []

    for bench_path in benchmarks_dir.glob("*.json"):
        q_slug = bench_path.stem
        benchmarks = load_benchmark_entries(bench_path)
        q_name = resolve_model(registry, q_slug) or q_slug
        answers_root = answers_dir / q_slug
        if not answers_root.exists():
            continue
        for answer_file in answers_root.glob("*.json"):
            a_slug = answer_file.stem
            answers = load_answer_entries(answer_file)
            answers_by_key: Dict[Tuple[Optional[str], Optional[str]], Tuple[int, AnswerEntry]] = {}
            for idx, answer_entry in enumerate(answers):
                if not answer_entry:
                    continue
                answer_key = question_key(answer_entry.question_model or q_slug, answer_entry.run_id)
                if not answer_key:
                    skip_counts["answer_missing_key"] += 1
                    continue
                prior = answers_by_key.get(answer_key)
                if not prior or (
                    prior[1].status != STATUS_SUCCEEDED
                    and answer_entry.status == STATUS_SUCCEEDED
                ):
                    answers_by_key[answer_key] = (idx, answer_entry)

            for idx, bench_entry in enumerate(benchmarks):
                if not bench_entry or bench_entry.status != STATUS_SUCCEEDED:
                    skip_counts["question_missing_or_failed"] += 1
                    continue
                question_text = final_question(bench_entry)
                if not question_text:
                    skip_counts["question_missing_or_failed"] += 1
                    continue
                bench_key = question_key(q_slug, bench_entry.run_id)
                if not bench_key:
                    skip_counts["question_missing_key"] += 1
                    continue
                answer_match = answers_by_key.get(bench_key)
                if not answer_match:
                    skip_counts["answer_missing"] += 1
                    continue
                answer_idx, answer_entry = answer_match

                question_key_str = format_key(bench_key)

                answer_name = resolve_model(registry, answer_entry.answer_model) or resolve_model(registry, a_slug) or a_slug

                self_mode, self_info = find_critique_verdict(
                    critique_verdicts,
                    q_slug,
                    a_slug,
                    q_slug,
                    answer_idx,
                    question_key(answer_entry.question_model or q_slug, answer_entry.run_id),
                    self_answer_critique_mode,
                    fallback_any_mode,
                )
                if self_info:
                    self_verdict = self_info.get("verdict")
                    if self_verdict == CRITIQUE_VERDICT_CORRECT:
                        pass
                    elif self_verdict == CRITIQUE_VERDICT_UNKNOWN:
                        skip_counts["self_answer_unknown"] += 1
                        continue
                    else:
                        claim_key = critique_key(
                            q_slug,
                            q_slug,
                            a_slug,
                            self_mode,
                            self_info.get("run_id"),
                        )
                        outcome = resolve_automated_victory(
                            "critique",
                            decisions_by_claim.get(claim_key, []),
                            context=format_key(claim_key or ()),
                            log_automated_disagreements=log_automated_disagreements,
                        )
                        if outcome == VictorySide.ALICE:
                            skip_counts["self_answer_invalid"] += 1
                            continue
                        if outcome in {None, VictorySide.DROP}:
                            skip_counts["self_answer_no_majority"] += 1
                            continue

                if answer_entry.status == STATUS_FAILED:
                    games.append((answer_name, q_name, question_key_str, 0))
                    continue

                if answer_entry.status == STATUS_ILL_POSED:
                    claim_key = critique_key(
                        q_slug,
                        a_slug,
                        None,
                        None,
                        answer_entry.run_id,
                    )
                    outcome = resolve_automated_victory(
                        "illposed",
                        decisions_by_claim.get(claim_key, []),
                        context=format_key(claim_key or ()),
                        log_automated_disagreements=log_automated_disagreements,
                    )
                    if outcome == VictorySide.ALICE:
                        skip_counts["illposed_validated"] += 1
                        continue
                    if outcome == VictorySide.BOB:
                        games.append((answer_name, q_name, question_key_str, 0))
                        continue
                    skip_counts["illposed_no_majority"] += 1
                    continue

                if answer_entry.status != STATUS_SUCCEEDED:
                    skip_counts["answer_invalid_status"] += 1
                    continue

                mode, verdict_info = find_critique_verdict(
                    critique_verdicts,
                    q_slug,
                    q_slug,
                    a_slug,
                    answer_idx,
                    question_key(answer_entry.question_model or q_slug, answer_entry.run_id),
                    answer_critique_mode,
                    fallback_any_mode,
                )
                if not verdict_info:
                    skip_counts["critique_missing"] += 1
                    continue
                verdict = verdict_info.get("verdict")
                if verdict == CRITIQUE_VERDICT_UNKNOWN:
                    skip_counts["critique_unknown"] += 1
                    continue
                if verdict == CRITIQUE_VERDICT_CORRECT:
                    games.append((answer_name, q_name, question_key_str, 1))
                    continue

                claim_key = critique_key(
                    q_slug,
                    a_slug,
                    q_slug,
                    mode,
                    verdict_info.get("run_id"),
                )
                outcome = resolve_automated_victory(
                    "critique",
                    decisions_by_claim.get(claim_key, []),
                    context=format_key(claim_key or ()),
                    log_automated_disagreements=log_automated_disagreements,
                )
                if outcome == VictorySide.BOB:
                    games.append((answer_name, q_name, question_key_str, 1))
                    continue
                if outcome == VictorySide.ALICE:
                    games.append((answer_name, q_name, question_key_str, 0))
                    continue
                skip_counts["critique_no_majority"] += 1

    return games, skip_counts


def build_itemized_dataset(games: Sequence[GameRecord]) -> ItemizedDataset:
    answerers = sorted({game[0] for game in games})
    questioners = sorted({game[1] for game in games})
    question_keys = sorted({game[2] for game in games})

    answerer_index = {name: idx for idx, name in enumerate(answerers)}
    questioner_index = {name: idx for idx, name in enumerate(questioners)}

    question_owner: Dict[str, str] = {}
    for _, questioner_name, question_key_str, _ in games:
        existing = question_owner.get(question_key_str)
        if existing and existing != questioner_name:
            raise ValueError(
                f"Question key {question_key_str} has multiple questioners: {existing} vs {questioner_name}"
            )
        question_owner[question_key_str] = questioner_name

    questioner_by_question = [questioner_index[question_owner[qkey]] for qkey in question_keys]
    question_key_index = {qkey: idx for idx, qkey in enumerate(question_keys)}

    edges = []
    for answerer_name, _, question_key_str, outcome in games:
        edges.append((question_key_index[question_key_str], answerer_index[answerer_name], outcome))

    return ItemizedDataset(
        answerers=answerers,
        questioners=questioners,
        question_keys=question_keys,
        questioner_by_question=questioner_by_question,
        edges=edges,
    )


def _beta_from_free(beta_free: Sequence[float]) -> List[float]:
    if len(beta_free) == 0:
        return [0.0]
    # Enforce sum(beta)=0 by setting the final entry as the negative sum.
    beta_last = -sum(beta_free)
    return list(beta_free) + [beta_last]


def fit_itemized_bt(
    dataset: ItemizedDataset,
    sigmas: Sigmas,
    max_iter: int = 3000,
    lr: float = 0.05,
    tol: float = 1e-6,
    init: Optional[BTParams] = None,
) -> BTParams:
    num_answerers = len(dataset.answerers)
    num_questioners = len(dataset.questioners)
    num_questions = len(dataset.question_keys)

    beta_free = np.zeros(max(0, num_answerers - 1), dtype=float)
    alpha = np.zeros(num_questioners, dtype=float)
    delta = np.zeros(num_questions, dtype=float)

    if init:
        if num_answerers > 1 and len(init.beta) == num_answerers:
            beta_free = np.asarray(init.beta[:-1], dtype=float)
        if len(init.alpha) == num_questioners:
            alpha = np.asarray(init.alpha, dtype=float)
        if len(init.delta) == num_questions:
            delta = np.asarray(init.delta, dtype=float)

    inv_beta_var = 1.0 / (sigmas.sigma_beta * sigmas.sigma_beta)
    inv_alpha_var = 1.0 / (sigmas.sigma_alpha * sigmas.sigma_alpha)
    inv_delta_var = 1.0 / (sigmas.sigma_delta * sigmas.sigma_delta)

    edges = np.asarray(dataset.edges, dtype=np.int64)
    if edges.size == 0:
        return BTParams(
            beta=_beta_from_free(beta_free.tolist()),
            alpha=alpha.tolist(),
            delta=delta.tolist(),
        )
    q_idx = edges[:, 0]
    b_idx = edges[:, 1]
    outcome = edges[:, 2].astype(float)
    questioner_by_question = np.asarray(dataset.questioner_by_question, dtype=np.int64)
    a_idx = questioner_by_question[q_idx]

    last_obj = None
    for step_idx in range(max_iter):
        if beta_free.size == 0:
            beta_full = np.array([0.0])
        else:
            beta_full = np.concatenate([beta_free, [-beta_free.sum()]])
        beta_last = beta_full[-1]
        grad_beta = np.zeros_like(beta_free)
        grad_alpha = np.zeros_like(alpha)
        grad_delta = np.zeros_like(delta)

        logit = beta_full[b_idx] - alpha[a_idx] - delta[q_idx]
        p = sigmoid(logit)
        ll = float(np.dot(outcome, logit) - np.sum(log1pexp(logit)))
        err = outcome - p

        if grad_beta.size:
            mask = b_idx < grad_beta.size
            # Repeated indices require scatter-add to accumulate gradients.
            np.add.at(grad_beta, b_idx[mask], err[mask])
            if np.any(~mask):
                grad_beta -= err[~mask].sum()
        np.add.at(grad_alpha, a_idx, -err)
        np.add.at(grad_delta, q_idx, -err)

        # L2 priors
        if grad_beta.size:
            grad_beta -= inv_beta_var * (beta_free - beta_last)
        grad_alpha -= inv_alpha_var * alpha
        grad_delta -= inv_delta_var * delta

        prior = -0.5 * inv_beta_var * float(np.sum(beta_full * beta_full))
        prior -= 0.5 * inv_alpha_var * float(np.sum(alpha * alpha))
        prior -= 0.5 * inv_delta_var * float(np.sum(delta * delta))
        obj = ll + prior

        scale = lr / math.sqrt(step_idx + 1.0)
        max_step = 0.0
        if grad_beta.size:
            step = scale * grad_beta
            beta_free += step
            max_step = max(max_step, float(np.max(np.abs(step))))
        if grad_alpha.size:
            step = scale * grad_alpha
            alpha += step
            max_step = max(max_step, float(np.max(np.abs(step))))
        if grad_delta.size:
            step = scale * grad_delta
            delta += step
            max_step = max(max_step, float(np.max(np.abs(step))))

        if last_obj is not None and abs(obj - last_obj) < tol and max_step < tol:
            break
        last_obj = obj

    beta_full = _beta_from_free(beta_free.tolist())
    return BTParams(beta=beta_full, alpha=alpha.tolist(), delta=delta.tolist())


def fit_itemized_bt_weighted(
    dataset: ItemizedDataset,
    weighted_edges: Sequence[Tuple[int, int, int, float]],
    sigmas: Sigmas,
    max_iter: int = 3000,
    lr: float = 0.05,
    tol: float = 1e-6,
    init: Optional[BTParams] = None,
) -> BTParams:
    num_answerers = len(dataset.answerers)
    num_questioners = len(dataset.questioners)
    num_questions = len(dataset.question_keys)

    beta_free = np.zeros(max(0, num_answerers - 1), dtype=float)
    alpha = np.zeros(num_questioners, dtype=float)
    delta = np.zeros(num_questions, dtype=float)

    if init:
        if num_answerers > 1 and len(init.beta) == num_answerers:
            beta_free = np.asarray(init.beta[:-1], dtype=float)
        if len(init.alpha) == num_questioners:
            alpha = np.asarray(init.alpha, dtype=float)
        if len(init.delta) == num_questions:
            delta = np.asarray(init.delta, dtype=float)

    inv_beta_var = 1.0 / (sigmas.sigma_beta * sigmas.sigma_beta)
    inv_alpha_var = 1.0 / (sigmas.sigma_alpha * sigmas.sigma_alpha)
    inv_delta_var = 1.0 / (sigmas.sigma_delta * sigmas.sigma_delta)

    edges = np.asarray(weighted_edges, dtype=float)
    if edges.size == 0:
        return BTParams(
            beta=_beta_from_free(beta_free.tolist()),
            alpha=alpha.tolist(),
            delta=delta.tolist(),
        )
    q_idx = edges[:, 0].astype(np.int64)
    b_idx = edges[:, 1].astype(np.int64)
    outcome = edges[:, 2].astype(float)
    weight = edges[:, 3].astype(float)
    questioner_by_question = np.asarray(dataset.questioner_by_question, dtype=np.int64)
    a_idx = questioner_by_question[q_idx]

    last_obj = None
    for step_idx in range(max_iter):
        if beta_free.size == 0:
            beta_full = np.array([0.0])
        else:
            beta_full = np.concatenate([beta_free, [-beta_free.sum()]])
        beta_last = beta_full[-1]
        grad_beta = np.zeros_like(beta_free)
        grad_alpha = np.zeros_like(alpha)
        grad_delta = np.zeros_like(delta)

        use_mask = weight > 0.0
        if not np.any(use_mask):
            break
        q_sub = q_idx[use_mask]
        b_sub = b_idx[use_mask]
        a_sub = a_idx[use_mask]
        outcome_sub = outcome[use_mask]
        weight_sub = weight[use_mask]

        logit = beta_full[b_sub] - alpha[a_sub] - delta[q_sub]
        p = sigmoid(logit)
        ll = float(np.sum(weight_sub * (outcome_sub * logit - log1pexp(logit))))
        err = weight_sub * (outcome_sub - p)

        if grad_beta.size:
            mask = b_sub < grad_beta.size
            np.add.at(grad_beta, b_sub[mask], err[mask])
            if np.any(~mask):
                grad_beta -= err[~mask].sum()
        np.add.at(grad_alpha, a_sub, -err)
        np.add.at(grad_delta, q_sub, -err)

        if grad_beta.size:
            grad_beta -= inv_beta_var * (beta_free - beta_last)
        grad_alpha -= inv_alpha_var * alpha
        grad_delta -= inv_delta_var * delta

        prior = -0.5 * inv_beta_var * float(np.sum(beta_full * beta_full))
        prior -= 0.5 * inv_alpha_var * float(np.sum(alpha * alpha))
        prior -= 0.5 * inv_delta_var * float(np.sum(delta * delta))
        obj = ll + prior

        scale = lr / math.sqrt(step_idx + 1.0)
        max_step = 0.0
        if grad_beta.size:
            step = scale * grad_beta
            beta_free += step
            max_step = max(max_step, float(np.max(np.abs(step))))
        if grad_alpha.size:
            step = scale * grad_alpha
            alpha += step
            max_step = max(max_step, float(np.max(np.abs(step))))
        if grad_delta.size:
            step = scale * grad_delta
            delta += step
            max_step = max(max_step, float(np.max(np.abs(step))))

        if last_obj is not None and abs(obj - last_obj) < tol and max_step < tol:
            break
        last_obj = obj

    beta_full = _beta_from_free(beta_free.tolist())
    return BTParams(beta=beta_full, alpha=alpha.tolist(), delta=delta.tolist())


def log_posterior(
    params: BTParams,
    dataset: ItemizedDataset,
    sigmas: Sigmas,
) -> float:
    if not dataset.edges:
        ll = 0.0
    else:
        edges = np.asarray(dataset.edges, dtype=np.int64)
        q_idx = edges[:, 0]
        b_idx = edges[:, 1]
        outcome = edges[:, 2].astype(float)
        questioner_by_question = np.asarray(dataset.questioner_by_question, dtype=np.int64)
        a_idx = questioner_by_question[q_idx]
        params_beta = np.asarray(params.beta, dtype=float)
        params_alpha = np.asarray(params.alpha, dtype=float)
        params_delta = np.asarray(params.delta, dtype=float)
        logit = params_beta[b_idx] - params_alpha[a_idx] - params_delta[q_idx]
        ll = float(np.dot(outcome, logit) - np.sum(log1pexp(logit)))

    inv_beta_var = 1.0 / (sigmas.sigma_beta * sigmas.sigma_beta)
    inv_alpha_var = 1.0 / (sigmas.sigma_alpha * sigmas.sigma_alpha)
    inv_delta_var = 1.0 / (sigmas.sigma_delta * sigmas.sigma_delta)

    params_beta = np.asarray(params.beta, dtype=float)
    params_alpha = np.asarray(params.alpha, dtype=float)
    params_delta = np.asarray(params.delta, dtype=float)
    prior = -0.5 * inv_beta_var * float(np.sum(params_beta * params_beta))
    prior -= 0.5 * inv_alpha_var * float(np.sum(params_alpha * params_alpha))
    prior -= 0.5 * inv_delta_var * float(np.sum(params_delta * params_delta))

    beta_free_count = max(0, len(params.beta) - 1)
    prior -= beta_free_count * math.log(sigmas.sigma_beta)
    prior -= len(params.alpha) * math.log(sigmas.sigma_alpha)
    prior -= len(params.delta) * math.log(sigmas.sigma_delta)

    return ll + prior


def _negative_hessian(
    params: BTParams,
    dataset: ItemizedDataset,
    sigmas: Sigmas,
) -> np.ndarray:
    num_answerers = len(params.beta)
    num_questioners = len(params.alpha)
    num_questions = len(params.delta)

    beta_free_count = max(0, num_answerers - 1)
    total_params = beta_free_count + num_questioners + num_questions
    hessian = np.zeros((total_params, total_params), dtype=float)

    if dataset.edges:
        edges = np.asarray(dataset.edges, dtype=np.int64)
        q_edges = edges[:, 0]
        b_edges = edges[:, 1]
        questioner_by_question = np.asarray(dataset.questioner_by_question, dtype=np.int64)
        a_edges = questioner_by_question[q_edges]
        params_beta = np.asarray(params.beta, dtype=float)
        params_alpha = np.asarray(params.alpha, dtype=float)
        params_delta = np.asarray(params.delta, dtype=float)
        logit = params_beta[b_edges] - params_alpha[a_edges] - params_delta[q_edges]
        p = sigmoid(logit)
        w = p * (1.0 - p)

        for q_idx, b_idx, a_idx, weight in zip(q_edges, b_edges, a_edges, w):
            if weight == 0.0:
                continue
            coeffs: List[Tuple[int, float]] = []
            if beta_free_count:
                if b_idx < beta_free_count:
                    coeffs.append((int(b_idx), 1.0))
                else:
                    for j in range(beta_free_count):
                        coeffs.append((j, -1.0))
            coeffs.append((beta_free_count + int(a_idx), -1.0))
            coeffs.append((beta_free_count + num_questioners + int(q_idx), -1.0))

            for i, (idx_i, coeff_i) in enumerate(coeffs):
                for idx_j, coeff_j in coeffs[i:]:
                    value = weight * coeff_i * coeff_j
                    hessian[idx_i, idx_j] += value
                    if idx_i != idx_j:
                        hessian[idx_j, idx_i] += value

    inv_beta_var = 1.0 / (sigmas.sigma_beta * sigmas.sigma_beta)
    inv_alpha_var = 1.0 / (sigmas.sigma_alpha * sigmas.sigma_alpha)
    inv_delta_var = 1.0 / (sigmas.sigma_delta * sigmas.sigma_delta)

    if beta_free_count:
        hessian[:beta_free_count, :beta_free_count] += inv_beta_var
        diag_idx = np.arange(beta_free_count)
        hessian[diag_idx, diag_idx] += inv_beta_var

    if num_questioners:
        diag_idx = np.arange(num_questioners)
        hessian[beta_free_count + diag_idx, beta_free_count + diag_idx] += inv_alpha_var

    if num_questions:
        diag_idx = np.arange(num_questions)
        offset = beta_free_count + num_questioners
        hessian[offset + diag_idx, offset + diag_idx] += inv_delta_var

    return hessian


def logdet_cholesky(matrix, jitter: float = 1e-8, max_tries: int = 5) -> float:
    mat = np.asarray(matrix, dtype=float)
    if mat.size == 0:
        return 0.0
    attempt = 0
    eye = np.eye(mat.shape[0])
    while True:
        try:
            chol = np.linalg.cholesky(mat)
            return 2.0 * float(np.log(np.diag(chol)).sum())
        except np.linalg.LinAlgError:
            attempt += 1
            if attempt > max_tries:
                raise
            mat = mat + eye * jitter
            jitter *= 10.0


def laplace_log_marginal(
    params: BTParams,
    dataset: ItemizedDataset,
    sigmas: Sigmas,
) -> float:
    log_post = log_posterior(params, dataset, sigmas)
    hessian = _negative_hessian(params, dataset, sigmas)
    log_det = logdet_cholesky(hessian)
    return log_post - 0.5 * log_det


def evaluate_sigmas(
    dataset: ItemizedDataset,
    sigmas: Sigmas,
    max_iter: int,
    lr: float,
    tol: float,
    init: Optional[BTParams] = None,
) -> Tuple[float, BTParams]:
    params = fit_itemized_bt(
        dataset,
        sigmas,
        max_iter=max_iter,
        lr=lr,
        tol=tol,
        init=init,
    )
    score = laplace_log_marginal(params, dataset, sigmas)
    return score, params
