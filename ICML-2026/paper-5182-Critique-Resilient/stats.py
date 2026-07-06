from collections import Counter, defaultdict
from pathlib import Path
import argparse
import logging

from typing import Dict, List, Optional, Set, Tuple, Union

from data_models import (
    AutomatedEvaluation,
    BenchmarkEntry,
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    load_debate_entries,
    load_evaluation_entries,
    load_human_evaluation_entries,
)
from constants import (
    CRITIQUE_VERDICT_CORRECT,
    CRITIQUE_VERDICT_INCORRECT,
    CRITIQUE_VERDICT_INSUFFICIENT,
    CRITIQUE_VERDICT_OBSCURE,
    CRITIQUE_VERDICT_UNKNOWN,
    STATUS_FAILED,
    STATUS_ILL_POSED,
    STATUS_SUCCEEDED,
)
from victory import (
    VictorySide,
    resolve_automated_victory,
    verdict_to_victory_side,
)
from model_config import load_registry
from utils import (
    collect_invalid_questions,
    format_key,
    human_evaluation_key_from_entry,
    is_latest_outer_attempt,
    judging_task_key,
    latest_outer_attempt_by_run,
    normalize_outer_attempt,
    question_key,
    task_key_from_prefix,
)

logger = logging.getLogger(__name__)

QuestionKey = Tuple[Optional[str], Optional[str], Optional[str]]
CritiqueKey = Tuple[str, str, str, QuestionKey]


def count_human_labels(
    evaluations_dir: Path,
    latest_by_question: Optional[Dict[str, Dict[str, int]]] = None,
):
    label_counts = defaultdict(int)
    for eval_file in evaluations_dir.glob("*.json"):
        data = load_human_evaluation_entries(eval_file)
        for dec in data.decisions:
            key = human_evaluation_key_from_entry(dec)
            if key:
                if latest_by_question:
                    run_id, question_model, _answer_model, _critic_model, _mode, outer_attempt = key
                    latest_by_run = latest_by_question.get(question_model or "", {})
                    if run_id is not None and latest_by_run:
                        if not is_latest_outer_attempt(run_id, normalize_outer_attempt(outer_attempt), latest_by_run):
                            continue
                label_counts[key] += 1
    return label_counts


def _task_key(
    prefix: str, run_id: Optional[str], outer_attempt: Optional[str], _topic_slug: Optional[str], _question: Optional[str]
):
    return task_key_from_prefix(prefix, run_id, outer_attempt)


_CRITIQUE_INCORRECT_EQUIV = {
    CRITIQUE_VERDICT_INCORRECT,
    CRITIQUE_VERDICT_INSUFFICIENT,
    CRITIQUE_VERDICT_OBSCURE,
}


def normalize_critique_verdict(verdict: Optional[str]) -> Optional[str]:
    if verdict in _CRITIQUE_INCORRECT_EQUIV:
        return CRITIQUE_VERDICT_INCORRECT
    return verdict


def final_question(entry: BenchmarkEntry) -> Optional[str]:
    generations = entry.generation_rounds or []
    if not generations:
        return None
    refinements = generations[-1].refinement_rounds or []
    if not refinements:
        return None
    return refinements[-1].question


def latest_outer_attempts_by_question(benchmarks_dir: Path) -> Dict[str, Dict[str, int]]:
    latest_by_question: Dict[str, Dict[str, int]] = {}
    for bench_path in benchmarks_dir.glob("*.json"):
        q_slug = bench_path.stem
        entries = load_benchmark_entries(bench_path)
        latest_by_question[q_slug] = latest_outer_attempt_by_run(entries)
    return latest_by_question


def self_answer_attempt_distribution(
    benchmarks_dir: Path,
    invalid_questions: Set[QuestionKey],
) -> Tuple[Dict[str, Counter], Dict[str, int], Dict[str, int]]:
    per_model: Dict[str, Counter] = {}
    totals: Dict[str, int] = {}
    missing: Dict[str, int] = {}
    for bench_path in benchmarks_dir.glob("*.json"):
        q_slug = bench_path.stem
        entries = load_benchmark_entries(bench_path)
        attempts_by_run: Dict[str, List[Tuple[int, BenchmarkEntry]]] = defaultdict(list)
        for entry in entries:
            if not entry or entry.run_id is None:
                continue
            attempt = normalize_outer_attempt(entry.outer_attempt) or 1
            attempts_by_run[str(entry.run_id)].append((attempt, entry))
        totals[q_slug] = len(attempts_by_run)
        counts: Counter = Counter()
        missing_count = 0
        for run_id, attempts in attempts_by_run.items():
            attempts.sort(key=lambda item: item[0])
            found = False
            for attempt, entry in attempts:
                if entry.status != STATUS_SUCCEEDED:
                    continue
                if not final_question(entry):
                    continue
                q_key = question_key(q_slug, entry.run_id, attempt)
                if q_key and q_key in invalid_questions:
                    continue
                counts[attempt] += 1
                found = True
                break
            if not found:
                missing_count += 1
        per_model[q_slug] = counts
        missing[q_slug] = missing_count
    return per_model, totals, missing


def load_critique_verdicts(
    critiques_dir: Path,
    *,
    latest_by_question: Optional[Dict[str, Dict[str, int]]] = None,
    invalid_questions: Optional[Set[QuestionKey]] = None,
) -> Dict[CritiqueKey, Dict[str, Dict[str, Optional[str]]]]:
    verdicts: Dict[CritiqueKey, Dict[str, Dict[str, Optional[str]]]] = defaultdict(dict)
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
                    verdict = normalize_critique_verdict(attempts[-1].verdict)
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
                    if not q_key:
                        raise ValueError(
                            "Missing critique question key (run_id/outer_attempt/question_author) "
                            f"for {crit_file} index {idx}"
                        )
                    verdicts[(q_slug, critic_slug, answer_slug, q_key)][mode] = info
    return verdicts


def find_critique_verdict(
    verdicts: Dict[CritiqueKey, Dict[str, Dict[str, Optional[str]]]],
    q_slug: str,
    critic_slug: str,
    answer_slug: str,
    idx: int,
    q_key: Optional[QuestionKey],
    preferred_mode: Optional[str],
    fallback_any: bool,
) -> Tuple[Optional[str], Optional[Dict[str, Optional[str]]]]:
    if q_key is None:
        raise ValueError(
            "Missing critique question key (run_id/outer_attempt/question_author) "
            f"for {q_slug}/{critic_slug}__{answer_slug} index {idx}"
        )
    modes = verdicts.get((q_slug, critic_slug, answer_slug, q_key), {})
    if preferred_mode and preferred_mode in modes:
        return preferred_mode, modes[preferred_mode]
    if not fallback_any:
        return None, None
    if not modes:
        return None, None
    mode = sorted(modes.keys())[0]
    return mode, modes[mode]


def count_items(
    path: Path,
    kind: str,
    latest_by_question: Optional[Dict[str, Dict[str, int]]] = None,
):
    counts = 0
    if kind == "questions":
        for file in path.glob("*.json"):
            q_slug = file.stem
            entries = load_benchmark_entries(file)
            latest_by_run = latest_by_question.get(q_slug, {}) if latest_by_question else latest_outer_attempt_by_run(entries)
            for entry in entries:
                if not entry:
                    continue
                outer_attempt = normalize_outer_attempt(entry.outer_attempt)
                if entry.run_id is not None and latest_by_run:
                    if not is_latest_outer_attempt(entry.run_id, outer_attempt, latest_by_run):
                        continue
                counts += 1
    elif kind == "answers":
        for q_dir in path.glob("*"):
            q_slug = q_dir.name
            for ans_file in q_dir.glob("*.json"):
                entries = load_answer_entries(ans_file)
                latest_by_run = latest_by_question.get(q_slug, {}) if latest_by_question else latest_outer_attempt_by_run(entries)
                for entry in entries:
                    if not entry:
                        continue
                    outer_attempt = normalize_outer_attempt(entry.outer_attempt)
                    if entry.run_id is not None and latest_by_run:
                        if not is_latest_outer_attempt(entry.run_id, outer_attempt, latest_by_run):
                            continue
                    counts += 1
    elif kind in {"critiques", "illposed"}:
        base = "critiques" if kind == "critiques" else "debates/illposed"
        base_path = Path(base)
        for sub in base_path.glob("**/*.json"):
            q_slug = sub.parent.name
            entries = load_debate_entries(sub) if kind == "illposed" else load_critique_entries(sub)
            latest_by_run = latest_by_question.get(q_slug, {}) if latest_by_question else latest_outer_attempt_by_run(entries)
            for entry in entries:
                if not entry:
                    continue
                outer_attempt = normalize_outer_attempt(getattr(entry, "outer_attempt", None))
                if getattr(entry, "run_id", None) is not None and latest_by_run:
                    if not is_latest_outer_attempt(entry.run_id, outer_attempt, latest_by_run):
                        continue
                counts += 1
    return counts


def collect_claim_keys(
    critiques_dir: Path,
    debates_dir: Path,
    latest_by_question: Optional[Dict[str, Dict[str, int]]] = None,
):
    claim_keys = set()
    for mode_dir in critiques_dir.glob("*"):
        for q_dir in mode_dir.glob("*"):
            for crit_file in q_dir.glob("*.json"):
                q_slug = q_dir.name
                crit_ids = load_critique_entries(crit_file)
                latest_by_run = latest_by_question.get(q_slug, {}) if latest_by_question else {}
                for entry in crit_ids:
                    if not entry:
                        continue
                    outer_attempt = normalize_outer_attempt(entry.outer_attempt)
                    if entry.run_id is not None and latest_by_run:
                        if not is_latest_outer_attempt(entry.run_id, outer_attempt, latest_by_run):
                            continue
                    prefix = f"critique/{mode_dir.name}/{q_slug}/{crit_file.stem}"
                    claim_key = _task_key(
                        prefix, entry.run_id, outer_attempt, entry.topic_slug, entry.question
                    )
                    if claim_key:
                        claim_keys.add(claim_key)
    for debate_file in (debates_dir / "illposed").glob("*/*.json"):
        q_slug = debate_file.parent.name
        a_slug = debate_file.stem
        debates = load_debate_entries(debate_file)
        latest_by_run = latest_by_question.get(q_slug, {}) if latest_by_question else {}
        for entry in debates:
            if not entry:
                continue
            outer_attempt = normalize_outer_attempt(entry.outer_attempt)
            if entry.run_id is not None and latest_by_run:
                if not is_latest_outer_attempt(entry.run_id, outer_attempt, latest_by_run):
                    continue
            prefix = f"illposed/{q_slug}/{a_slug}"
            claim_key = _task_key(prefix, entry.run_id, outer_attempt, entry.topic_slug, entry.question)
            if claim_key:
                claim_keys.add(claim_key)
    return claim_keys


def critique_verdicts(
    critiques_dir: Path,
    latest_by_question: Optional[Dict[str, Dict[str, int]]] = None,
):
    verdict_counts = Counter()
    for mode_dir in critiques_dir.glob("*"):
        for q_dir in mode_dir.glob("*"):
            for crit_file in q_dir.glob("*.json"):
                latest_by_run = latest_by_question.get(q_dir.name, {}) if latest_by_question else {}
                entries = load_critique_entries(crit_file)
                for entry in entries:
                    if not entry:
                        verdict_counts["missing"] += 1
                        continue
                    outer_attempt = normalize_outer_attempt(entry.outer_attempt)
                    if entry.run_id is not None and latest_by_run:
                        if not is_latest_outer_attempt(entry.run_id, outer_attempt, latest_by_run):
                            continue
                    attempts = entry.attempts or []
                    if not attempts:
                        verdict_counts["missing"] += 1
                        continue
                    verdict = normalize_critique_verdict(attempts[-1].verdict)
                    if not verdict:
                        verdict_counts["missing"] += 1
                    else:
                        verdict_counts[verdict] += 1
    return verdict_counts


def count_illposed_answers(
    answers_dir: Path,
    latest_by_question: Optional[Dict[str, Dict[str, int]]] = None,
):
    """Count answers with status='ill-posed'"""
    count = 0
    for q_dir in answers_dir.glob("*"):
        q_slug = q_dir.name
        latest_by_run = latest_by_question.get(q_slug, {}) if latest_by_question else None
        for ans_file in q_dir.glob("*.json"):
            entries = load_answer_entries(ans_file)
            for entry in entries:
                if not entry:
                    continue
                outer_attempt = normalize_outer_attempt(entry.outer_attempt)
                if latest_by_run and entry.run_id is not None:
                    if not is_latest_outer_attempt(entry.run_id, outer_attempt, latest_by_run):
                        continue
                if entry.status == STATUS_ILL_POSED:
                    count += 1
    return count


def collect_decisions_by_claim(auto_eval_dir: Path) -> Dict[Tuple, List[AutomatedEvaluation]]:
    decisions_by_claim: Dict[Tuple, List[AutomatedEvaluation]] = defaultdict(list)
    for decision in collect_automated_evaluations(auto_eval_dir):
        key = judging_task_key(decision)
        if key:
            decisions_by_claim[key].append(decision)
    return decisions_by_claim


def _format_count(count: int, total: int) -> str:
    if total <= 0:
        return f"{count} (0.00%)"
    pct = 100.0 * count / total
    return f"{count} ({pct:.2f}%)"


def _drop_reason_from_verdicts(drop_verdicts: Counter) -> str:
    if not drop_verdicts:
        return "unanimous_drop"
    if len(drop_verdicts) == 1:
        verdict = next(iter(drop_verdicts))
        return f"unanimous_drop_{verdict}"
    return "unanimous_drop_mixed_verdicts"


def _format_drop_reason(reason: str) -> str:
    if reason.startswith("unanimous_drop_"):
        suffix = reason[len("unanimous_drop_") :]
        if suffix == "mixed_verdicts":
            return "unanimous drop: mixed/unknown"
        return f"unanimous drop: {suffix}"
    mapping = {
        "no_decisions": "no automated decisions",
        "no_valid_verdicts": "all judgments invalid",
        "no_unanimity": "no unanimity among judges",
    }
    return mapping.get(reason, reason)


def _analyze_automated_victory(
    claim_type: str,
    decisions: List[AutomatedEvaluation],
    *,
    context: Optional[str] = None,
    log_automated_disagreements: bool = True,
) -> Tuple[Optional[VictorySide], str, Counter, Counter]:
    verdicts = [
        decision.verdict
        for decision in decisions
        if decision and decision.verdict and decision.status != STATUS_FAILED
    ]
    if not verdicts:
        return None, "no_decisions", Counter(), Counter()

    invalid_verdicts: Counter = Counter()
    drop_verdicts: Counter = Counter()
    sides: List[VictorySide] = []
    for verdict in verdicts:
        side = verdict_to_victory_side(claim_type, verdict)
        if side is None:
            invalid_verdicts[verdict] += 1
            continue
        sides.append(side)
        if side == VictorySide.DROP:
            drop_verdicts[verdict] += 1

    if not sides:
        return None, "no_valid_verdicts", invalid_verdicts, drop_verdicts

    unique = set(sides)
    if len(unique) != 1:
        if log_automated_disagreements:
            logger.error(
                "Automated judgments disagree for %s: %s",
                context or "unknown task",
                sorted(side.value for side in unique),
            )
        return None, "no_unanimity", invalid_verdicts, drop_verdicts

    outcome = unique.pop()
    if outcome == VictorySide.DROP:
        reason = _drop_reason_from_verdicts(drop_verdicts)
        return outcome, reason, invalid_verdicts, drop_verdicts

    return outcome, "unanimous_win", invalid_verdicts, drop_verdicts


def _diagnose_missing_critique(
    critiques_dir: Path,
    mode: str,
    q_slug: str,
    critic_slug: str,
    answer_slug: str,
    q_key: Optional[QuestionKey],
) -> Tuple[str, str]:
    crit_path = critiques_dir / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
    if not crit_path.exists():
        return "file_missing", str(crit_path)
    entries = load_critique_entries(crit_path)
    if not q_key:
        return "question_key_missing", str(crit_path)
    for entry in entries:
        if not entry:
            continue
        outer_attempt = normalize_outer_attempt(entry.outer_attempt)
        entry_key = question_key(entry.question_author, entry.run_id, outer_attempt)
        if entry_key != q_key:
            continue
        if entry.status != STATUS_SUCCEEDED:
            return f"status_{entry.status}", str(crit_path)
        attempts = entry.attempts or []
        if not attempts:
            return "no_attempts", str(crit_path)
        verdict = attempts[-1].verdict
        if not verdict:
            return "verdict_missing", str(crit_path)
        return "unexpected_missing", str(crit_path)
    return "entry_missing", str(crit_path)


def print_protocol_stats(
    benchmarks_dir: Path,
    answers_dir: Path,
    critiques_dir: Path,
    auto_eval_dir: Path,
    answer_critique_mode: str = "evaluator",
    self_answer_critique_mode: str = "contradictor",
    fallback_any_mode: bool = False,
    log_automated_disagreements: bool = True,
    list_missing_critiques: bool = False,
    latest_by_question: Optional[Dict[str, Dict[str, int]]] = None,
    invalid_questions: Optional[Set[QuestionKey]] = None,
) -> None:
    critique_verdicts = load_critique_verdicts(
        critiques_dir,
        latest_by_question=latest_by_question,
    )
    decisions_by_claim = collect_decisions_by_claim(auto_eval_dir)

    counts = Counter()
    candidate_games = 0
    no_victory_reasons = {
        "self_critique": Counter(),
        "illposed": Counter(),
        "answer_critique": Counter(),
    }
    invalid_verdicts = {
        "self_critique": Counter(),
        "illposed": Counter(),
        "answer_critique": Counter(),
    }
    missing_critique_reasons = Counter()
    missing_critique_cases: List[Dict[str, str]] = []

    def _should_log_disagreements(q_key: Optional[QuestionKey]) -> bool:
        if not log_automated_disagreements:
            return False
        if invalid_questions and q_key in invalid_questions:
            return False
        return True

    for bench_path in benchmarks_dir.glob("*.json"):
        q_slug = bench_path.stem
        benchmarks = load_benchmark_entries(bench_path)
        answers_root = answers_dir / q_slug
        if not answers_root.exists():
            continue
        for answer_file in answers_root.glob("*.json"):
            a_slug = answer_file.stem
            answers = load_answer_entries(answer_file)
            latest_by_run = latest_by_question.get(q_slug, {}) if latest_by_question else latest_outer_attempt_by_run(benchmarks)
            answers_by_key: Dict[QuestionKey, object] = {}
            for answer_entry in answers:
                if not answer_entry:
                    continue
                answer_outer_attempt = normalize_outer_attempt(answer_entry.outer_attempt)
                if answer_entry.run_id is not None and latest_by_run:
                    if not is_latest_outer_attempt(answer_entry.run_id, answer_outer_attempt, latest_by_run):
                        continue
                answer_key = question_key(
                    answer_entry.question_model or q_slug,
                    answer_entry.run_id,
                    answer_outer_attempt,
                )
                if not answer_key:
                    continue
                prior = answers_by_key.get(answer_key)
                if not prior or (
                    prior.status != STATUS_SUCCEEDED
                    and answer_entry.status == STATUS_SUCCEEDED
                ):
                    answers_by_key[answer_key] = answer_entry

            for bench_entry in benchmarks:
                if bench_entry:
                    outer_attempt = normalize_outer_attempt(bench_entry.outer_attempt)
                    if bench_entry.run_id is not None and latest_by_run:
                        if not is_latest_outer_attempt(bench_entry.run_id, outer_attempt, latest_by_run):
                            continue
                counts["pairs_total"] += 1
                if not bench_entry or bench_entry.status != STATUS_SUCCEEDED:
                    counts["drop_question_missing_or_failed"] += 1
                    continue
                question_text = final_question(bench_entry)
                if not question_text:
                    counts["drop_question_missing_or_failed"] += 1
                    continue
                bench_key = question_key(q_slug, bench_entry.run_id, outer_attempt)
                if not bench_key:
                    counts["drop_question_missing_or_failed"] += 1
                    continue
                answer_entry = answers_by_key.get(bench_key)
                if not answer_entry:
                    counts["drop_answer_missing"] += 1
                    continue

                candidate_games += 1
                log_for_question = _should_log_disagreements(bench_key)

                # Step 3: Bob critiques Alice's self-answer (question validity check)
                self_mode, self_info = find_critique_verdict(
                    critique_verdicts,
                    q_slug,
                    a_slug,
                    q_slug,
                    0,
                    bench_key,
                    self_answer_critique_mode,
                    fallback_any_mode,
                )
                if not self_info:
                    counts["drop_self_critique_missing"] += 1
                    counts["final_drop"] += 1
                    continue
                counts["self_critique_present"] += 1
                self_verdict = normalize_critique_verdict(self_info.get("verdict"))

                if self_verdict == CRITIQUE_VERDICT_UNKNOWN:
                    counts["drop_self_critique_unknown"] += 1
                    counts["final_drop"] += 1
                    continue

                if self_verdict == CRITIQUE_VERDICT_CORRECT:
                    counts["self_critique_correct"] += 1
                else:
                    prefix = f"critique/{self_mode}/{q_slug}/{a_slug}__{q_slug}"
                    claim_key = _task_key(
                        prefix,
                        self_info.get("run_id"),
                        self_info.get("outer_attempt"),
                        self_info.get("topic_slug"),
                        self_info.get("question"),
                    )
                    outcome, reason, invalid_counts, _ = _analyze_automated_victory(
                        "critique",
                        decisions_by_claim.get(claim_key, []),
                        context=format_key(claim_key or ()),
                        log_automated_disagreements=log_for_question,
                    )
                    if outcome == VictorySide.ALICE:
                        counts["drop_self_critique_upheld"] += 1
                        counts["final_drop"] += 1
                        continue
                    if outcome == VictorySide.BOB:
                        counts["self_critique_rejected"] += 1
                    else:
                        counts["drop_self_critique_no_victory"] += 1
                        counts["final_drop"] += 1
                        no_victory_reasons["self_critique"][reason] += 1
                        invalid_verdicts["self_critique"].update(invalid_counts)
                        continue

                # Step 4: Bob answers Alice's question
                if answer_entry.status == STATUS_FAILED:
                    counts["answer_failed"] += 1
                    counts["final_alice_wins"] += 1
                    continue

                if answer_entry.status == STATUS_ILL_POSED:
                    counts["answer_illposed_claimed"] += 1
                    prefix = f"illposed/{q_slug}/{a_slug}"
                    claim_key = _task_key(
                        prefix,
                        answer_entry.run_id,
                        answer_entry.outer_attempt,
                        answer_entry.topic_slug,
                        answer_entry.question,
                    )
                    outcome, reason, invalid_counts, _ = _analyze_automated_victory(
                        "illposed",
                        decisions_by_claim.get(claim_key, []),
                        context=format_key(claim_key or ()),
                        log_automated_disagreements=log_for_question,
                    )
                    if outcome == VictorySide.ALICE:
                        counts["drop_illposed_upheld"] += 1
                        counts["final_drop"] += 1
                        continue
                    if outcome == VictorySide.BOB:
                        counts["illposed_rejected"] += 1
                        counts["final_alice_wins"] += 1
                        continue
                    counts["drop_illposed_no_victory"] += 1
                    counts["final_drop"] += 1
                    no_victory_reasons["illposed"][reason] += 1
                    invalid_verdicts["illposed"].update(invalid_counts)
                    continue

                if answer_entry.status != STATUS_SUCCEEDED:
                    counts["drop_answer_invalid_status"] += 1
                    counts["final_drop"] += 1
                    continue

                counts["answer_succeeded"] += 1

                # Step 5: Alice critiques Bob's answer
                mode, verdict_info = find_critique_verdict(
                    critique_verdicts,
                    q_slug,
                    q_slug,
                    a_slug,
                    0,
                    bench_key,
                    answer_critique_mode,
                    fallback_any_mode,
                )
                if not verdict_info:
                    counts["drop_critique_missing"] += 1
                    counts["final_drop"] += 1
                    if list_missing_critiques:
                        reason, path = _diagnose_missing_critique(
                            critiques_dir,
                            answer_critique_mode,
                            q_slug,
                            q_slug,
                            a_slug,
                            bench_key,
                        )
                        missing_critique_reasons[reason] += 1
                        missing_critique_cases.append(
                            {
                                "question": q_slug,
                                "answerer": a_slug,
                                "question_key": format_key(bench_key or ()),
                                "reason": reason,
                                "path": path,
                            }
                        )
                    continue
                verdict = normalize_critique_verdict(verdict_info.get("verdict"))
                if verdict == CRITIQUE_VERDICT_UNKNOWN:
                    counts["drop_critique_unknown"] += 1
                    counts["final_drop"] += 1
                    continue
                if verdict == CRITIQUE_VERDICT_CORRECT:
                    counts["critique_says_correct"] += 1
                    counts["final_bob_wins"] += 1
                    continue

                prefix = f"critique/{mode}/{q_slug}/{q_slug}__{a_slug}"
                claim_key = _task_key(
                    prefix,
                    verdict_info.get("run_id"),
                    verdict_info.get("outer_attempt"),
                    verdict_info.get("topic_slug"),
                    verdict_info.get("question"),
                )
                outcome, reason, invalid_counts, _ = _analyze_automated_victory(
                    "critique",
                    decisions_by_claim.get(claim_key, []),
                    context=format_key(claim_key or ()),
                    log_automated_disagreements=log_for_question,
                )
                if outcome == VictorySide.BOB:
                    counts["critique_incorrect_defender_wins"] += 1
                    counts["final_bob_wins"] += 1
                    continue
                if outcome == VictorySide.ALICE:
                    counts["critique_incorrect_claimant_wins"] += 1
                    counts["final_alice_wins"] += 1
                    continue
                counts["critique_incorrect_no_victory"] += 1
                counts["final_drop"] += 1
                no_victory_reasons["answer_critique"][reason] += 1
                invalid_verdicts["answer_critique"].update(invalid_counts)

    pairs_total = counts["pairs_total"]

    print("\nProtocol flow (percentages of candidate games):")
    print(f"  Total pairs considered: {pairs_total}")
    print(f"  Candidate games (valid question + answer entry): {_format_count(candidate_games, pairs_total)}")
    print(f"  Dropped before candidate (question missing/failed): {_format_count(counts['drop_question_missing_or_failed'], pairs_total)}")
    print(f"  Dropped before candidate (answer missing): {_format_count(counts['drop_answer_missing'], pairs_total)}")

    print(f"  Bob critiques Alice's question: {_format_count(counts['self_critique_present'], candidate_games)}")
    print(f"  Bob's critique missing: {_format_count(counts['drop_self_critique_missing'], candidate_games)}")
    print(f"  Bob's critique says correct: {_format_count(counts['self_critique_correct'], candidate_games)}")
    print(f"  Bob's critique unknown: {_format_count(counts['drop_self_critique_unknown'], candidate_games)}")
    print(f"  Bob's critique upheld (drop): {_format_count(counts['drop_self_critique_upheld'], candidate_games)}")
    print(f"  Bob's critique rejected: {_format_count(counts['self_critique_rejected'], candidate_games)}")
    print(f"  Bob's critique no victory (drop): {_format_count(counts['drop_self_critique_no_victory'], candidate_games)}")

    print(f"  Bob fails to answer: {_format_count(counts['answer_failed'], candidate_games)}")
    print(f"  Bob claims ill-posed: {_format_count(counts['answer_illposed_claimed'], candidate_games)}")
    print(f"  Ill-posed upheld (drop): {_format_count(counts['drop_illposed_upheld'], candidate_games)}")
    print(f"  Ill-posed rejected (Alice wins): {_format_count(counts['illposed_rejected'], candidate_games)}")
    print(f"  Ill-posed no victory (drop): {_format_count(counts['drop_illposed_no_victory'], candidate_games)}")
    print(f"  Bob answers successfully: {_format_count(counts['answer_succeeded'], candidate_games)}")

    print(f"  Alice's critique missing: {_format_count(counts['drop_critique_missing'], candidate_games)}")
    print(f"  Alice's critique unknown: {_format_count(counts['drop_critique_unknown'], candidate_games)}")
    print(f"  Alice says answer correct (Bob wins): {_format_count(counts['critique_says_correct'], candidate_games)}")
    print(f"  Alice says incorrect, Bob wins: {_format_count(counts['critique_incorrect_defender_wins'], candidate_games)}")
    print(f"  Alice says incorrect, Alice wins: {_format_count(counts['critique_incorrect_claimant_wins'], candidate_games)}")
    print(f"  Alice says incorrect, no victory (drop): {_format_count(counts['critique_incorrect_no_victory'], candidate_games)}")

    print(f"  Final Bob wins: {_format_count(counts['final_bob_wins'], candidate_games)}")
    print(f"  Final Alice wins: {_format_count(counts['final_alice_wins'], candidate_games)}")
    print(f"  Final dropped: {_format_count(counts['final_drop'], candidate_games)}")

    print("\nNo-victory breakdowns (percentages of each drop category):")
    breakdowns = [
        ("Bob's critique no victory (drop)", "self_critique", counts["drop_self_critique_no_victory"]),
        ("Ill-posed no victory (drop)", "illposed", counts["drop_illposed_no_victory"]),
        ("Alice says incorrect, no victory (drop)", "answer_critique", counts["critique_incorrect_no_victory"]),
    ]
    for label, key, total in breakdowns:
        if total <= 0:
            continue
        print(f"  {label}:")
        for reason in sorted(no_victory_reasons[key].keys()):
            count = no_victory_reasons[key][reason]
            print(f"    {_format_drop_reason(reason)}: {_format_count(count, total)}")
        if invalid_verdicts[key]:
            print("    invalid judgments:")
            for verdict in sorted(invalid_verdicts[key].keys()):
                print(f"      {verdict}: {invalid_verdicts[key][verdict]}")

    if list_missing_critiques and missing_critique_cases:
        print("\nAlice critique missing reasons:")
        for reason, count in missing_critique_reasons.most_common():
            print(f"  {reason}: {count}")
        print("\nAlice critique missing cases:")
        for case in missing_critique_cases:
            print(
                f"  {case['question']} -> {case['answerer']} "
                f"[{case['index']}]: {case['reason']} ({case['path']})"
            )


def collect_automated_evaluations(auto_eval_dir: Path) -> List[AutomatedEvaluation]:
    """Collect all automated evaluation decisions from all judge files."""
    all_decisions = []
    if not auto_eval_dir.exists():
        return all_decisions

    for eval_file in auto_eval_dir.glob("*.json"):
        data = load_evaluation_entries(eval_file)
        all_decisions.extend(data.decisions)

    return all_decisions


def _critique_target_key(task_key: Tuple) -> Optional[Tuple[str, str, str]]:
    if not task_key or len(task_key) < 3:
        return None
    run_id, question_model, answer_model = task_key[:3]
    if not run_id or not question_model or not answer_model:
        return None
    return (run_id, question_model, answer_model)


def _illposed_target_key(task_key: Tuple) -> Optional[Tuple[str, str]]:
    if not task_key or len(task_key) < 2:
        return None
    run_id, question_model, *_rest = task_key
    if not run_id or not question_model:
        return None
    return (run_id, question_model)


def count_inter_judge_disagreements(
    auto_eval_dir: Path,
    latest_by_question: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[int, int]:
    """Count claims with non-unanimous judge verdicts."""
    decisions = collect_automated_evaluations(auto_eval_dir)
    decisions_by_claim = defaultdict(list)
    for decision in decisions:
        key = judging_task_key(decision)
        if key:
            decisions_by_claim[key].append(decision)

    naive_count = 0
    dedup_targets = set()

    for claim_key, claim_decisions in decisions_by_claim.items():
        if not claim_decisions:
            continue
        claim_type = claim_decisions[0].type
        if claim_type not in {"critique", "critique_debate", "illposed"}:
            continue
        run_id, question_model, _answer_model, _critic_model, _mode, outer_attempt = claim_key
        if latest_by_question:
            latest_by_run = latest_by_question.get(question_model or "", {})
            if run_id is not None and latest_by_run:
                if not is_latest_outer_attempt(run_id, normalize_outer_attempt(outer_attempt), latest_by_run):
                    continue
        verdicts = [d.verdict for d in claim_decisions if d.verdict]
        if len(verdicts) < 2:
            continue
        if len(set(verdicts)) <= 1:
            continue

        naive_count += 1
        if claim_type in {"critique", "critique_debate"}:
            target_key = _critique_target_key(claim_key)
        else:
            target_key = _illposed_target_key(claim_key)
        dedup_targets.add(target_key or (claim_type, format_key(claim_key)))

    return naive_count, len(dedup_targets)


def build_critique_verdict_map(
    critiques_dir: Path,
    latest_by_question: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Dict]:
    """
    Build a map of critique IDs to verdict metadata.
    This is used by both compute_model_stats and main to avoid duplication.

    Returns:
        Dict mapping critique IDs to dicts containing:
        - verdict: str (correct/incorrect/insufficient/obscure)
        - answer_author: str (model name)
        - question_author: str (model name)
        - critic: str (model name)
    """
    critique_verdict_map = {}
    for mode_dir in critiques_dir.glob("*"):
        for q_dir in mode_dir.glob("*"):
            for crit_file in q_dir.glob("*.json"):
                q_slug = q_dir.name
                entries = load_critique_entries(crit_file)
                latest_by_run = latest_by_question.get(q_slug, {}) if latest_by_question else {}
                for entry in entries:
                    if not entry:
                        continue
                    outer_attempt = normalize_outer_attempt(entry.outer_attempt)
                    if entry.run_id is not None and latest_by_run:
                        if not is_latest_outer_attempt(entry.run_id, outer_attempt, latest_by_run):
                            continue
                    attempts = entry.attempts if entry else None
                    verdict = normalize_critique_verdict(attempts[-1].verdict) if attempts else None
                    # Extract answer author and question author from entry
                    answer_author = entry.answer_author if entry else None
                    question_author = entry.question_author if entry else None
                    critic_model_name = entry.critic if entry else None

                    prefix = f"critique/{mode_dir.name}/{q_slug}/{crit_file.stem}"
                    task_key = _task_key(
                        prefix, entry.run_id, outer_attempt, entry.topic_slug, entry.question
                    )
                    if not task_key:
                        continue
                    critique_verdict_map[task_key] = {
                        "verdict": verdict,
                        "answer_author": answer_author,
                        "question_author": question_author,
                        "critic": critic_model_name
                    }
    return critique_verdict_map


def compute_model_stats(
    auto_eval_dir: Path,
    critiques_dir: Path,
    log_automated_disagreements: bool = True,
    latest_by_question: Optional[Dict[str, Dict[str, int]]] = None,
    invalid_questions: Optional[Set[QuestionKey]] = None,
):
    """Compute agreement statistics for various model roles."""
    decisions = collect_automated_evaluations(auto_eval_dir)

    # Track all encountered model names for validation
    encountered_models = set()

    # Group decisions by claim key
    decisions_by_claim = defaultdict(list)
    for decision in decisions:
        claim_key = judging_task_key(decision)
        if claim_key:
            decisions_by_claim[claim_key].append(decision)
            # Track model names
            for key in ["question_model", "answer_model", "critic_model", "judge_model"]:
                model_name = getattr(decision, key)
                if model_name:
                    encountered_models.add(model_name)

    if latest_by_question:
        filtered = defaultdict(list)
        for claim_key, claim_decisions in decisions_by_claim.items():
            run_id, question_model, _answer_model, _critic_model, _mode, outer_attempt = claim_key
            latest_by_run = latest_by_question.get(question_model or "", {})
            if run_id is not None and latest_by_run:
                if not is_latest_outer_attempt(run_id, normalize_outer_attempt(outer_attempt), latest_by_run):
                    continue
            filtered[claim_key] = claim_decisions
        decisions_by_claim = filtered

    # For each model, track self-answer correctness victory rates.
    model_self_answers = defaultdict(list)

    # Track self-answers declared "correct" by critics (no debate needed)
    model_self_answers_no_debate = defaultdict(int)

    # For each defender model (answer author), track their victory rate when defending answers.
    model_as_defender_success_rate = defaultdict(list)

    # For each claimant model (critic), track their victory rate when making claims.
    model_as_claimant_success_rate = defaultdict(list)

    # For cross-model answers, track three categories per (answer_model, question_model) pair
    cross_model_answers = defaultdict(lambda: defaultdict(lambda: {
        "declared_correct": 0,  # Critic said "correct" (no debate)
        "critiqued_correct": 0,  # Critic said wrong, but victory rules sided with defender
        "critiqued_wrong": 0,    # Critic said wrong, and victory rules sided with claimant
        "total": 0
    }))

    # Build critique verdict map to identify "correct" verdicts (no debate)
    critique_verdict_map = build_critique_verdict_map(critiques_dir, latest_by_question=latest_by_question)

    for claim_key, claim_decisions in decisions_by_claim.items():
        if not claim_decisions:
            continue

        # Extract metadata from first decision (all should have same metadata)
        first = claim_decisions[0]
        claim_type = first.type
        question_model = first.question_model
        answer_model = first.answer_model
        critic_model = first.critic_model
        mode = first.mode

        if claim_type == "critique":
            run_id, question_model_key, _answer_model, _critic_model, _mode, outer_attempt = claim_key
            q_key = question_key(question_model_key, run_id, outer_attempt)
            log_for_claim = (
                log_automated_disagreements
                and not (invalid_questions and q_key in invalid_questions)
            )
            outcome = resolve_automated_victory(
                claim_type,
                claim_decisions,
                context=format_key(claim_key),
                log_automated_disagreements=log_for_claim,
            )
            if outcome in {None, VictorySide.DROP}:
                continue

            # Determine if this is a self-answer (question_model == answer_model)
            is_self_answer = (question_model == answer_model)

            if is_self_answer:
                win_rate = 100.0 if outcome == VictorySide.BOB else 0.0
                model_self_answers[answer_model].append(win_rate)
            else:
                if outcome == VictorySide.BOB:
                    cross_model_answers[answer_model][question_model]["critiqued_correct"] += 1
                else:
                    cross_model_answers[answer_model][question_model]["critiqued_wrong"] += 1
                cross_model_answers[answer_model][question_model]["total"] += 1

            # Defender stats (Bob is the answer_model defending)
            defender_rate = 100.0 if outcome == VictorySide.BOB else 0.0
            model_as_defender_success_rate[answer_model].append(defender_rate)

            # Claimant stats (Alice is the critic_model claiming error)
            claimant_rate = 100.0 if outcome == VictorySide.ALICE else 0.0
            model_as_claimant_success_rate[critic_model].append(claimant_rate)

    # Now add answers declared "correct" by critics (these don't appear in automated evaluations)
    for cid, crit_info in critique_verdict_map.items():
        if crit_info["verdict"] == CRITIQUE_VERDICT_CORRECT:
            answer_author = crit_info["answer_author"]
            question_author = crit_info["question_author"]

            if answer_author and question_author:
                if answer_author == question_author:
                    # Self-answer declared correct (no debate needed)
                    model_self_answers_no_debate[answer_author] += 1
                else:
                    # Cross-model answer declared correct
                    cross_model_answers[answer_author][question_author]["declared_correct"] += 1
                    cross_model_answers[answer_author][question_author]["total"] += 1

    # Log all model names encountered for verification
    if encountered_models:
        logger.info(f"Encountered {len(encountered_models)} unique model names in data: {sorted(encountered_models)}")

    return model_self_answers, model_self_answers_no_debate, model_as_defender_success_rate, model_as_claimant_success_rate, cross_model_answers


def print_agreement_stats(model_stats, title):
    """Print average victory rate across claims."""
    if not model_stats:
        print(f"\n{title}: No data available")
        return

    print(f"\n{title}:")
    for model in sorted(model_stats.keys()):
        percentages = model_stats[model]
        if not percentages:
            continue

        # Safe division with explicit check
        avg_percentage = sum(percentages) / len(percentages) if percentages else 0.0
        print(f"  {model}: {avg_percentage:.1f}% victory rate (across {len(percentages)} critiques)")


def print_cross_model_stats(cross_model_stats):
    """Print statistics for models answering other models' questions."""
    if not cross_model_stats:
        print("\nCross-model answer correctness: No data available")
        return

    print("\nCross-model answer correctness (by question maker):")
    for answer_model in sorted(cross_model_stats.keys()):
        print(f"\n  {answer_model} answering:")
        by_q_model = cross_model_stats[answer_model]
        for q_model in sorted(by_q_model.keys()):
            stats = by_q_model[q_model]
            total = stats["total"]
            if total > 0:
                declared_pct = 100 * stats["declared_correct"] / total
                critiqued_correct_pct = 100 * stats["critiqued_correct"] / total
                critiqued_wrong_pct = 100 * stats["critiqued_wrong"] / total

                print(f"    {q_model}'s questions ({total} total):")
                print(f"      {declared_pct:.1f}% declared correct by critic (no debate)")
                print(f"      {critiqued_correct_pct:.1f}% critiqued but victory rules say correct")
                print(f"      {critiqued_wrong_pct:.1f}% critiqued and victory rules say wrong")


def main():
    parser = argparse.ArgumentParser(description="Report aggregate stats for benchmarks.")
    parser.add_argument("--disable-disagreement-logs", action="store_true")
    parser.add_argument("--list-missing-critiques", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    args, _ = parser.parse_known_args()
    log_automated_disagreements = not args.disable_disagreement_logs

    benchmarks_dir = Path("benchmarks")
    answers_dir = Path("answers")
    critiques_dir = Path("critiques")
    debates_dir = Path("debates")
    evaluations_dir = Path("evaluations")
    auto_eval_dir = Path("automated_evaluations")
    registry = load_registry(str(args.config))

    latest_by_question = latest_outer_attempts_by_question(benchmarks_dir)
    invalid_questions = collect_invalid_questions(
        critiques_dir,
        answers_dir,
        auto_eval_dir,
        evaluations_dir,
        registry,
        log_automated_disagreements=False,
    )

    questions = count_items(benchmarks_dir, "questions", latest_by_question)
    answers = count_items(answers_dir, "answers", latest_by_question)
    critiques_count = count_items(critiques_dir, "critiques", latest_by_question)
    illposed_debate_count = count_items(Path(debates_dir), "illposed", latest_by_question)
    illposed_answer_count = count_illposed_answers(answers_dir, latest_by_question)

    labels = count_human_labels(evaluations_dir, latest_by_question)
    claim_keys = collect_claim_keys(critiques_dir, debates_dir, latest_by_question)

    # Only count labels for critique claims with non-correct verdicts and all illposed claims
    filtered_claim_keys = set()
    verdicts = critique_verdicts(critiques_dir, latest_by_question)

    # Build shared verdict map (reused by compute_model_stats)
    critique_verdict_map = build_critique_verdict_map(critiques_dir, latest_by_question)

    for cid in claim_keys:
        is_critique = bool(cid and (cid[3] or cid[4]))
        if is_critique:
            crit_info = critique_verdict_map.get(cid, {})
            v = crit_info.get("verdict") if isinstance(crit_info, dict) else crit_info
            if v == CRITIQUE_VERDICT_CORRECT:
                continue
        filtered_claim_keys.add(cid)

    label_hist = Counter(labels.get(cid, 0) for cid in filtered_claim_keys)

    print("Counts:")
    print(f"- Questions: {questions}")
    print(f"- Answers: {answers}")
    print(f"- Answers claiming ill-posed: {illposed_answer_count}")
    print(f"- Critiques: {critiques_count}")
    print(f"- Ill-posed debates: {illposed_debate_count}")
    v_counts = critique_verdicts(critiques_dir, latest_by_question)
    print("\nCritiques by final verdict (including missing/unknown):")
    for verdict, count in v_counts.items():
        print(f"  {verdict}: {count}")
    print("\nLabel histogram (number of labels -> count of claims):")
    for n_labels in sorted(label_hist):
        print(f"  {n_labels}: {label_hist[n_labels]}")

    attempt_counts, totals_by_model, missing_by_model = self_answer_attempt_distribution(
        benchmarks_dir,
        invalid_questions,
    )
    print("\nValid self-answer by outer_attempt (percent of run IDs):")
    for model in sorted(attempt_counts.keys()):
        total = totals_by_model.get(model, 0)
        if total == 0:
            continue
        print(f"  {model}:")
        for attempt in sorted(attempt_counts[model].keys()):
            count = attempt_counts[model][attempt]
            pct = 100.0 * count / total
            print(f"    outer_attempt {attempt}: {count} ({pct:.1f}%)")
        missing = missing_by_model.get(model, 0)
        if missing:
            pct = 100.0 * missing / total
            print(f"    missing valid self-answer: {missing} ({pct:.1f}%)")

    inter_judge_naive, inter_judge_dedup = count_inter_judge_disagreements(auto_eval_dir, latest_by_question)
    print("\nInter-judge disagreements (critiques/ill-posed claims):")
    print(f"  naive: {inter_judge_naive}")
    print(f"  deduped by answer/question: {inter_judge_dedup}")

    print_protocol_stats(
        benchmarks_dir,
        answers_dir,
        critiques_dir,
        auto_eval_dir,
        log_automated_disagreements=log_automated_disagreements,
        list_missing_critiques=args.list_missing_critiques,
        latest_by_question=latest_by_question,
        invalid_questions=invalid_questions,
    )

    # Compute and print automated evaluation statistics
    model_self_answers, model_self_answers_no_debate, model_as_defender_success_rate, model_as_claimant_success_rate, cross_model_stats = compute_model_stats(
        auto_eval_dir,
        critiques_dir,
        log_automated_disagreements=log_automated_disagreements,
        latest_by_question=latest_by_question,
        invalid_questions=invalid_questions,
    )

    print_agreement_stats(
        model_self_answers,
        "Self-answer correctness (debated victory rate for answer correctness)"
    )

    # Print self-answers declared correct without debate
    if model_self_answers_no_debate:
        print("\nSelf-answers declared correct by critics (no debate needed):")
        for model in sorted(model_self_answers_no_debate.keys()):
            count = model_self_answers_no_debate[model]
            print(f"  {model}: {count} answers")

    print_agreement_stats(
        model_as_defender_success_rate,
        "Defender success (victory rate for defender/answer)"
    )

    print_agreement_stats(
        model_as_claimant_success_rate,
        "Claimant success (victory rate for claimant/critic)"
    )

    print_cross_model_stats(cross_model_stats)


if __name__ == "__main__":
    main()
