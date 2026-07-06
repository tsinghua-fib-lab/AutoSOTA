#!/usr/bin/env python3
"""
5-fold cross-validation predictive validity experiment for CRB paper (Table 2).

Reproduces: Accuracy, Log-loss, Brier score for itemized Bradley-Terry model
vs. a baserate predictor, using question-level 5-fold CV.
"""
import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

# Add repo to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import (
    CRITIQUE_VERDICT_CORRECT,
    CRITIQUE_VERDICT_UNKNOWN,
    STATUS_FAILED,
    STATUS_ILL_POSED,
    STATUS_SUCCEEDED,
)
from data_models import (
    AnswerEntry,
    BenchmarkEntry,
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    load_evaluation_entries,
)
from model_config import load_registry
from utils import (
    collect_invalid_self_answer_questions,
    collect_self_answer_adjudications,
    format_key,
    is_latest_outer_attempt,
    judging_task_key,
    latest_outer_attempt_by_run,
    normalize_outer_attempt,
    question_key,
    task_key_from_prefix,
)
from victory import VictorySide, resolve_automated_victory


QuestionKey = Tuple[Optional[str], Optional[str], Optional[str]]
GameRecord = Tuple[str, str, str, int]  # answerer, questioner, question_key_str, outcome


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def log1pexp(x: float) -> float:
    if x > 0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def final_question(entry: BenchmarkEntry) -> Optional[str]:
    generations = entry.generation_rounds or []
    if not generations:
        return None
    refinements = generations[-1].refinement_rounds or []
    if not refinements:
        return None
    return refinements[-1].question


def resolve_model(registry, name_or_slug: Optional[str]) -> Optional[str]:
    if not name_or_slug:
        return None
    if registry is None:
        return name_or_slug
    return registry.resolve_model_name(name_or_slug)


def collect_decisions(auto_eval_dir: Path) -> Dict[Tuple, List]:
    from data_models import AutomatedEvaluation
    decisions_by_claim: Dict[Tuple, List[AutomatedEvaluation]] = defaultdict(list)
    if not auto_eval_dir.exists():
        return decisions_by_claim
    for eval_file in auto_eval_dir.glob("*.json"):
        data = load_evaluation_entries(eval_file)
        for decision in data.decisions:
            key = judging_task_key(decision) if decision else None
            if key:
                decisions_by_claim[key].append(decision)
    return decisions_by_claim


def load_critique_verdicts(
    critiques_dir: Path,
    *,
    latest_by_question: Optional[Dict[str, Dict[str, int]]] = None,
    invalid_questions: Optional[Set[QuestionKey]] = None,
) -> Dict[Tuple, Dict[str, Dict[str, Optional[str]]]]:
    verdicts: Dict[Tuple, Dict[str, Dict[str, Optional[str]]]] = defaultdict(dict)
    if not critiques_dir.exists():
        return verdicts
    for mode_dir in critiques_dir.glob("*"):
        mode = mode_dir.name
        for q_dir in mode_dir.glob("*"):
            q_slug = q_dir.name
            latest_by_run = latest_by_question.get(q_slug, {}) if latest_by_question else {}
            for crit_file in q_dir.glob("*.json"):
                parts = crit_file.stem.split("__", 1)
                if len(parts) != 2:
                    continue
                critic_slug, answer_slug = parts
                entries = load_critique_entries(crit_file)
                for idx, entry in enumerate(entries):
                    if not entry or entry.status != STATUS_SUCCEEDED:
                        continue
                    outer_attempt = normalize_outer_attempt(entry.outer_attempt)
                    if entry.run_id is not None and latest_by_run:
                        if not is_latest_outer_attempt(entry.run_id, outer_attempt, latest_by_run):
                            continue
                    attempts = entry.attempts or []
                    if not attempts:
                        continue
                    verdict = attempts[-1].verdict
                    if not verdict:
                        continue
                    info = {
                        "verdict": verdict,
                        "run_id": entry.run_id,
                        "outer_attempt": outer_attempt,
                        "topic_slug": entry.topic_slug,
                        "question": entry.question,
                    }
                    q_key = question_key(entry.question_author, entry.run_id, outer_attempt)
                    if invalid_questions and q_key in invalid_questions:
                        continue
                    if q_key:
                        verdicts[(q_slug, critic_slug, answer_slug, q_key)][mode] = info
                    verdicts[(q_slug, critic_slug, answer_slug, idx)][mode] = info
    return verdicts


def find_critique_verdict(
    verdicts, q_slug, critic_slug, answer_slug, idx, q_key,
    preferred_mode, fallback_any,
):
    modes = {}
    if q_key is not None:
        modes = verdicts.get((q_slug, critic_slug, answer_slug, q_key), {})
    if not modes:
        modes = verdicts.get((q_slug, critic_slug, answer_slug, idx), {})
    if preferred_mode and preferred_mode in modes:
        return preferred_mode, modes[preferred_mode]
    if not fallback_any:
        return None, None
    if not modes:
        return None, None
    mode = sorted(modes.keys())[0]
    return mode, modes[mode]


def collect_games(
    benchmarks_dir, answers_dir, critiques_dir, auto_eval_dir,
    registry_path, answer_critique_mode, self_answer_critique_mode,
    fallback_any_mode, human_eval_dir=None,
    log_automated_disagreements=True,
):
    registry = load_registry(str(registry_path)) if registry_path and registry_path.exists() else None
    if human_eval_dir is None:
        human_eval_dir = Path("evaluations")

    latest_by_question = {}
    for bench_path in benchmarks_dir.glob("*.json"):
        q_slug = bench_path.stem
        entries = load_benchmark_entries(bench_path)
        latest_by_question[q_slug] = latest_outer_attempt_by_run(entries)

    invalid_questions = collect_invalid_self_answer_questions(
        critiques_dir, auto_eval_dir, human_eval_dir, registry,
        log_automated_disagreements=log_automated_disagreements,
    )
    self_answer_outcomes = collect_self_answer_adjudications(
        critiques_dir, auto_eval_dir, human_eval_dir, registry,
        log_automated_disagreements=log_automated_disagreements,
    )
    critique_verdicts = load_critique_verdicts(
        critiques_dir, latest_by_question=latest_by_question,
        invalid_questions=invalid_questions,
    )
    decisions_by_claim = collect_decisions(auto_eval_dir)

    skip_counts = Counter()
    games = []

    for bench_path in benchmarks_dir.glob("*.json"):
        q_slug = bench_path.stem
        benchmarks = load_benchmark_entries(bench_path)
        latest_by_run = latest_by_question.get(q_slug, {})
        q_name = resolve_model(registry, q_slug) or q_slug
        answers_root = answers_dir / q_slug
        if not answers_root.exists():
            continue

        for answer_file in answers_root.glob("*.json"):
            a_slug = answer_file.stem
            answers = load_answer_entries(answer_file)
            answers_by_key = {}
            for idx, answer_entry in enumerate(answers):
                if not answer_entry:
                    continue
                outer_attempt = normalize_outer_attempt(answer_entry.outer_attempt)
                if answer_entry.run_id is not None and latest_by_run:
                    if not is_latest_outer_attempt(answer_entry.run_id, outer_attempt, latest_by_run):
                        continue
                answer_key_value = question_key(
                    answer_entry.question_model or q_slug,
                    answer_entry.run_id, outer_attempt,
                )
                if not answer_key_value:
                    skip_counts["answer_missing_key"] += 1
                    continue
                if answer_key_value in invalid_questions:
                    skip_counts["question_invalid_self_answer"] += 1
                    continue
                prior = answers_by_key.get(answer_key_value)
                if not prior or (
                    prior[1].status != STATUS_SUCCEEDED
                    and answer_entry.status == STATUS_SUCCEEDED
                ):
                    answers_by_key[answer_key_value] = (idx, answer_entry)

            for idx, bench_entry in enumerate(benchmarks):
                if not bench_entry or bench_entry.status != STATUS_SUCCEEDED:
                    skip_counts["question_missing_or_failed"] += 1
                    continue
                outer_attempt = normalize_outer_attempt(bench_entry.outer_attempt)
                if bench_entry.run_id is not None and latest_by_run:
                    if not is_latest_outer_attempt(bench_entry.run_id, outer_attempt, latest_by_run):
                        continue
                question_text = final_question(bench_entry)
                if not question_text:
                    skip_counts["question_missing_or_failed"] += 1
                    continue
                bench_key = question_key(q_slug, bench_entry.run_id, outer_attempt)
                if not bench_key:
                    skip_counts["question_missing_key"] += 1
                    continue
                if bench_key in invalid_questions:
                    skip_counts["question_invalid_self_answer"] += 1
                    continue
                answer_match = answers_by_key.get(bench_key)
                if not answer_match:
                    skip_counts["answer_missing"] += 1
                    continue
                answer_idx, answer_entry = answer_match

                answer_name = (
                    resolve_model(registry, answer_entry.answer_model)
                    or resolve_model(registry, a_slug)
                    or a_slug
                )

                # Skip self-answers (q_slug == a_slug); used only for feasibility gating
                if q_slug == a_slug:
                    continue

                # Self-answer validation
                self_mode, self_info = find_critique_verdict(
                    critique_verdicts, q_slug, a_slug, q_slug, answer_idx,
                    question_key(answer_entry.question_model or q_slug,
                                 answer_entry.run_id, outer_attempt),
                    self_answer_critique_mode, fallback_any_mode,
                )
                if self_info:
                    self_verdict = self_info.get("verdict")
                    if self_verdict == CRITIQUE_VERDICT_CORRECT:
                        pass
                    elif self_verdict == CRITIQUE_VERDICT_UNKNOWN:
                        skip_counts["self_answer_unknown"] += 1
                        continue
                    else:
                        q_key = question_key(q_slug, self_info.get("run_id"),
                                            self_info.get("outer_attempt"))
                        outcome = self_answer_outcomes.get(q_key)
                        if outcome == VictorySide.ALICE:
                            skip_counts["self_answer_invalid"] += 1
                            continue
                        if outcome in {None, VictorySide.DROP}:
                            skip_counts["self_answer_no_majority"] += 1
                            continue

                if answer_entry.status == STATUS_FAILED:
                    games.append((answer_name, q_name, format_key(bench_key), 0))
                    continue

                if answer_entry.status == STATUS_ILL_POSED:
                    prefix = f"illposed/{q_slug}/{a_slug}"
                    claim_key = task_key_from_prefix(
                        prefix, answer_entry.run_id, answer_entry.outer_attempt,
                    )
                    outcome = resolve_automated_victory(
                        "illposed", decisions_by_claim.get(claim_key, []),
                        context=format_key(claim_key or ()),
                        log_automated_disagreements=log_automated_disagreements,
                    )
                    if outcome == VictorySide.ALICE:
                        skip_counts["illposed_validated"] += 1
                        continue
                    if outcome == VictorySide.BOB:
                        games.append((answer_name, q_name, format_key(bench_key), 0))
                        continue
                    skip_counts["illposed_no_majority"] += 1
                    continue

                if answer_entry.status != STATUS_SUCCEEDED:
                    skip_counts["answer_invalid_status"] += 1
                    continue

                # Answer critique
                mode, verdict_info = find_critique_verdict(
                    critique_verdicts, q_slug, q_slug, a_slug, answer_idx,
                    question_key(answer_entry.question_model or q_slug,
                                 answer_entry.run_id, outer_attempt),
                    answer_critique_mode, fallback_any_mode,
                )
                if not verdict_info:
                    skip_counts["critique_missing"] += 1
                    continue
                verdict = verdict_info.get("verdict")
                if verdict == CRITIQUE_VERDICT_UNKNOWN:
                    skip_counts["critique_unknown"] += 1
                    continue
                if verdict == CRITIQUE_VERDICT_CORRECT:
                    games.append((answer_name, q_name, format_key(bench_key), 1))
                    continue

                prefix = f"critique/{mode}/{q_slug}/{q_slug}__{a_slug}"
                claim_key = task_key_from_prefix(
                    prefix, verdict_info.get("run_id"),
                    verdict_info.get("outer_attempt"),
                )
                outcome = resolve_automated_victory(
                    "critique", decisions_by_claim.get(claim_key, []),
                    context=format_key(claim_key or ()),
                    log_automated_disagreements=log_automated_disagreements,
                )
                if outcome == VictorySide.BOB:
                    games.append((answer_name, q_name, format_key(bench_key), 1))
                    continue
                if outcome == VictorySide.ALICE:
                    games.append((answer_name, q_name, format_key(bench_key), 0))
                    continue
                skip_counts["critique_no_majority"] += 1

    return games, skip_counts


# ---------------------------------------------------------------------------
# Itemized Bradley-Terry fit (from analysis/bt_utils.py, self-contained)
# ---------------------------------------------------------------------------

def build_itemized_dataset(games):
    """Build dataset from game records."""
    answerers = sorted({g[0] for g in games})
    questioners = sorted({g[1] for g in games})
    question_keys = sorted({g[2] for g in games})

    answerer_index = {name: idx for idx, name in enumerate(answerers)}
    questioner_index = {name: idx for idx, name in enumerate(questioners)}
    question_key_index = {qkey: idx for idx, qkey in enumerate(question_keys)}

    # Map each question_key to its questioner
    question_owner = {}
    for _, qname, qkey, _ in games:
        if qkey not in question_owner:
            question_owner[qkey] = qname

    questioner_by_question = [questioner_index[question_owner[qkey]] for qkey in question_keys]

    edges = []
    for aname, _, qkey, outcome in games:
        edges.append((question_key_index[qkey], answerer_index[aname], outcome))

    return {
        "answerers": answerers,
        "questioners": questioners,
        "question_keys": question_keys,
        "questioner_by_question": questioner_by_question,
        "edges": edges,
    }


def fit_itemized_bt(dataset, sigmas, max_iter=3000, lr=0.05, tol=1e-6):
    """Fit bipartite Bradley-Terry with item-level deltas."""
    num_answerers = len(dataset["answerers"])
    num_questioners = len(dataset["questioners"])
    num_questions = len(dataset["question_keys"])

    beta_free = np.zeros(max(0, num_answerers - 1), dtype=float)
    alpha = np.zeros(num_questioners, dtype=float)
    delta = np.zeros(num_questions, dtype=float)

    inv_beta_var = 1.0 / (sigmas["sigma_beta"] ** 2)
    inv_alpha_var = 1.0 / (sigmas["sigma_alpha"] ** 2)
    inv_delta_var = 1.0 / (sigmas["sigma_delta"] ** 2)

    edges_arr = np.asarray(dataset["edges"], dtype=np.int64)
    if edges_arr.size == 0:
        return {"beta": [0.0] if num_answerers == 0 else [0.0] * num_answerers,
                "alpha": alpha.tolist(), "delta": delta.tolist()}

    q_idx = edges_arr[:, 0]
    b_idx = edges_arr[:, 1]
    outcome = edges_arr[:, 2].astype(float)
    questioner_by_question = np.asarray(dataset["questioner_by_question"], dtype=np.int64)
    a_idx = questioner_by_question[q_idx]

    last_obj = None
    for step_idx in range(max_iter):
        if beta_free.size == 0:
            beta_full = np.array([0.0])
        else:
            beta_full = np.concatenate([beta_free, [-beta_free.sum()]])
        beta_last = beta_full[-1]

        logit = beta_full[b_idx] - alpha[a_idx] - delta[q_idx]
        x_arr = np.asarray(logit, dtype=float)
        p = np.empty_like(x_arr)
        pos_mask = x_arr >= 0
        p[pos_mask] = 1.0 / (1.0 + np.exp(-x_arr[pos_mask]))
        exp_x = np.exp(x_arr[~pos_mask])
        p[~pos_mask] = exp_x / (1.0 + exp_x)

        ll = float(np.dot(outcome, logit) - np.sum(np.logaddexp(0.0, logit)))
        err = outcome - p

        grad_beta = np.zeros_like(beta_free)
        grad_alpha = np.zeros_like(alpha)
        grad_delta = np.zeros_like(delta)

        if grad_beta.size:
            mask = b_idx < grad_beta.size
            np.add.at(grad_beta, b_idx[mask], err[mask])
            if np.any(~mask):
                grad_beta -= err[~mask].sum()
        np.add.at(grad_alpha, a_idx, -err)
        np.add.at(grad_delta, q_idx, -err)

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

    beta_full = (beta_free.tolist() + [-float(beta_free.sum())]) if beta_free.size else [0.0]
    return {"beta": beta_full, "alpha": alpha.tolist(), "delta": delta.tolist()}


# ---------------------------------------------------------------------------
# Predictive validity: 5-fold CV
# ---------------------------------------------------------------------------

def run_predictive_validity(games, n_folds=5, seed=42):
    """Run question-level 5-fold CV and compute Accuracy, Log-loss, Brier."""
    dataset = build_itemized_dataset(games)
    question_keys = dataset["question_keys"]
    edges = np.asarray(dataset["edges"], dtype=np.int64)
    questioner_by_question = np.asarray(dataset["questioner_by_question"], dtype=np.int64)
    answerers = dataset["answerers"]
    questioners = dataset["questioners"]

    if edges.size == 0:
        print("No edges to evaluate.")
        return

    q_idx = edges[:, 0]
    b_idx = edges[:, 1]
    outcomes = edges[:, 2].astype(float)
    a_idx = questioner_by_question[q_idx]

    # Baserate: overall win rate
    baserate_prob = float(np.mean(outcomes))
    print(f"\nBaserate win probability: {baserate_prob:.4f}")

    # Baserate metrics
    baserate_accuracy = max(baserate_prob, 1 - baserate_prob)
    baserate_logloss = -float(np.mean(
        outcomes * math.log(max(baserate_prob, 1e-15)) +
        (1 - outcomes) * math.log(max(1 - baserate_prob, 1e-15))
    ))
    baserate_brier = float(np.mean((outcomes - baserate_prob) ** 2))
    print(f"Baserate Accuracy: {baserate_accuracy:.4f}")
    print(f"Baserate Log-loss:  {baserate_logloss:.4f}")
    print(f"Baserate Brier:     {baserate_brier:.4f}")

    # 5-fold CV over questions
    np.random.seed(seed)
    unique_questions = np.unique(q_idx)
    np.random.shuffle(unique_questions)
    fold_size = len(unique_questions) // n_folds

    sigmas = {"sigma_beta": 5.0, "sigma_alpha": 5.0, "sigma_delta": 1.0}

    all_probs = np.zeros(len(outcomes))
    all_folds = np.zeros(len(outcomes), dtype=int)

    print(f"\n{n_folds}-fold CV over {len(unique_questions)} questions with {len(outcomes)} interactions:")

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(unique_questions)
        test_q = set(unique_questions[start:end])
        train_q = set(unique_questions) - test_q

        train_mask = np.isin(q_idx, list(train_q))
        test_mask = np.isin(q_idx, list(test_q))

        # Build training dataset
        train_edges = edges[train_mask].tolist()
        train_dataset = {
            "answerers": answerers,
            "questioners": questioners,
            "question_keys": question_keys,
            "questioner_by_question": questioner_by_question.tolist(),
            "edges": train_edges,
        }

        params = fit_itemized_bt(train_dataset, sigmas, max_iter=3000, lr=0.05, tol=1e-6)
        beta = np.asarray(params["beta"], dtype=float)
        alpha = np.asarray(params["alpha"], dtype=float)
        delta = np.asarray(params["delta"], dtype=float)

        # Predict test set
        test_q_idx = q_idx[test_mask]
        test_b_idx = b_idx[test_mask]
        test_a_idx = a_idx[test_mask]
        test_outcomes = outcomes[test_mask]

        logit = beta[test_b_idx] - alpha[test_a_idx] - delta[test_q_idx]
        probs = 1.0 / (1.0 + np.exp(-logit))

        all_probs[test_mask] = probs
        all_folds[test_mask] = fold

        fold_acc = np.mean((probs >= 0.5) == test_outcomes)
        fold_n = len(test_outcomes)
        print(f"  Fold {fold+1}: {fold_n} test interactions, accuracy={fold_acc:.4f}")

    # Overall metrics
    pred_outcomes = (all_probs >= 0.5).astype(float)
    accuracy = float(np.mean(pred_outcomes == outcomes))
    eps = 1e-15
    logloss = -float(np.mean(
        outcomes * np.log(np.clip(all_probs, eps, 1 - eps)) +
        (1 - outcomes) * np.log(np.clip(1 - all_probs, eps, 1 - eps))
    ))
    brier = float(np.mean((all_probs - outcomes) ** 2))

    print(f"\n{'='*60}")
    print(f"Table 2 Results (Itemized BT, {n_folds}-fold CV over questions)")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Baserate':>10} {'Model':>10} {'Delta':>10}")
    print(f"{'-'*45}")
    print(f"{'Accuracy':<15} {baserate_accuracy:>10.4f} {accuracy:>10.4f} {accuracy - baserate_accuracy:>+10.4f}")
    print(f"{'Log-loss':<15} {baserate_logloss:>10.4f} {logloss:>10.4f} {logloss - baserate_logloss:>+10.4f}")
    print(f"{'Brier':<15} {baserate_brier:>10.4f} {brier:>10.4f} {brier - baserate_brier:>+10.4f}")
    print(f"{'='*60}")

    return {
        "accuracy": accuracy,
        "logloss": logloss,
        "brier": brier,
        "baserate_accuracy": baserate_accuracy,
        "baserate_logloss": baserate_logloss,
        "baserate_brier": baserate_brier,
        "n_games": len(outcomes),
        "n_questions": len(unique_questions),
        "n_folds": n_folds,
    }


def main():
    parser = argparse.ArgumentParser(description="Predictive validity CV for CRB")
    parser.add_argument("--benchmarks-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--automated-dir", type=Path, default=Path("automated_evaluations"))
    parser.add_argument("--human-evals-dir", type=Path, default=Path("evaluations"))
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--answer-critique-mode", type=str, default="evaluator")
    parser.add_argument("--self-answer-critique-mode", type=str, default="contradictor")
    parser.add_argument("--fallback-any-mode", action="store_true")
    parser.add_argument("--disable-disagreement-logs", action="store_true", default=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Collecting games...")
    games, skips = collect_games(
        args.benchmarks_dir, args.answers_dir, args.critiques_dir,
        args.automated_dir, args.config,
        args.answer_critique_mode, args.self_answer_critique_mode,
        args.fallback_any_mode, args.human_evals_dir,
        log_automated_disagreements=not args.disable_disagreement_logs,
    )

    total_games = len(games)
    wins = sum(1 for g in games if g[3] == 1)
    print(f"Games: {total_games} (wins={wins}, losses={total_games - wins})")
    if skips:
        print("Skips:")
        for reason, count in sorted(skips.items()):
            print(f"  {reason}: {count}")

    if not games:
        print("No valid games found. Aborting.")
        return 1

    result = run_predictive_validity(games, n_folds=args.n_folds, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
