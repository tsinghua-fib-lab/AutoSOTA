import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

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
    CritiqueEntry,
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    load_debate_entries,
    load_evaluation_entries,
)
from model_config import ModelSpec, _slugify, load_registry
from utils import benchmark_answers_from_entries, format_key, judging_task_key, question_key, task_key_from_prefix


def _load_runs(path: Path, limit: Optional[int]) -> List[str]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        run_ids = [str(k) for k in data.keys()]
    elif isinstance(data, list):
        run_ids = []
        for idx, item in enumerate(data):
            if isinstance(item, dict) and "run_id" in item:
                run_ids.append(str(item["run_id"]))
            else:
                run_ids.append(str(idx))
    else:
        raise ValueError(f"Unsupported runs file format: {type(data).__name__}")
    if limit is not None:
        run_ids = run_ids[:limit]
    return run_ids


def _entries_by_run(entries: Sequence[Optional[Union[BenchmarkEntry, AnswerEntry]]]) -> Dict[str, object]:
    mapping: Dict[str, object] = {}
    for entry in entries:
        if entry and getattr(entry, "run_id", None) is not None:
            mapping[str(entry.run_id)] = entry
    return mapping


def _pick_entry(entries_by_run: Dict[str, object], run_id: str):
    return entries_by_run.get(run_id)


QuestionKey = Tuple[Optional[str], Optional[str]]


def _question_key_for_answer(entry: Optional[AnswerEntry]) -> Optional[QuestionKey]:
    if not entry:
        return None
    return question_key(entry.question_model, entry.run_id)


def _question_key_for_critique(entry: Optional[CritiqueEntry]) -> Optional[QuestionKey]:
    if not entry:
        return None
    return question_key(entry.question_author, entry.run_id)


def _map_by_key(entries: Sequence[Optional[object]], key_fn) -> Dict[QuestionKey, object]:
    mapping: Dict[QuestionKey, object] = {}
    for entry in entries:
        if not entry:
            continue
        key = key_fn(entry)
        if not key or key in mapping:
            continue
        mapping[key] = entry
    return mapping


def _map_by_key_prefer_succeeded(
    entries: Sequence[Optional[object]],
    key_fn,
) -> Dict[QuestionKey, object]:
    mapping: Dict[QuestionKey, object] = {}
    for entry in entries:
        if not entry:
            continue
        key = key_fn(entry)
        if not key:
            continue
        existing = mapping.get(key)
        if not existing:
            mapping[key] = entry
            continue
        existing_status = getattr(existing, "status", None)
        entry_status = getattr(entry, "status", None)
        if existing_status != STATUS_SUCCEEDED and entry_status == STATUS_SUCCEEDED:
            mapping[key] = entry
    return mapping


def _question_key_for_debate(entry: Optional[object], question_model: Optional[str]) -> Optional[QuestionKey]:
    if not entry:
        return None
    return question_key(question_model, getattr(entry, "run_id", None))


def _task_key(prefix: str, run_id: Optional[str], _topic_slug: Optional[str], _question: Optional[str]):
    return task_key_from_prefix(prefix, run_id)


def _expected_illposed_by_pair(answers_dir: Path, allowed_slugs: Set[str]) -> Dict[Tuple[str, str], int]:
    expected: Dict[Tuple[str, str], int] = {}
    for q_dir in answers_dir.glob("*"):
        q_slug = q_dir.name
        if allowed_slugs and q_slug not in allowed_slugs:
            continue
        for answer_file in q_dir.glob("*.json"):
            a_slug = answer_file.stem
            entries = load_answer_entries(answer_file)
            keys: Set[QuestionKey] = set()
            for entry in entries:
                if entry and entry.status == STATUS_ILL_POSED:
                    key = _question_key_for_answer(entry)
                    if key:
                        keys.add(key)
            if keys:
                expected[(q_slug, a_slug)] = len(keys)
    return expected


def _expected_critique_debates_by_pair(
    critiques_dir: Path,
    answers_dir: Path,
    benchmark_dir: Path,
    critique_modes: Sequence[str],
    allowed_slugs: Set[str],
    expected_runs_by_slug: Dict[str, Set[str]],
    assume_missing_critiques: bool,
) -> Dict[Tuple[str, str, str, str], int]:
    expected: Dict[Tuple[str, str, str, str], int] = {}
    for mode in critique_modes:
        mode_dir = critiques_dir / mode
        if not mode_dir.exists():
            continue
        for q_dir in mode_dir.glob("*"):
            q_slug = q_dir.name
            if allowed_slugs and q_slug not in allowed_slugs:
                continue
            benchmark_entries = load_benchmark_entries(benchmark_dir / f"{q_slug}.json")
            expected_runs = expected_runs_by_slug.get(q_slug, set())
            for crit_file in q_dir.glob("*.json"):
                parts = crit_file.stem.split("__")
                if len(parts) != 2:
                    continue
                critic_slug, answer_slug = parts
                answers = _load_answer_records(
                    answers_dir,
                    q_slug,
                    answer_slug,
                    benchmark_entries,
                )
                answer_by_run = _entries_by_run(answers)
                critiques = load_critique_entries(crit_file)
                key = (mode, q_slug, critic_slug, answer_slug)
                if assume_missing_critiques:
                    expected_for_file = 0
                    for run_id in expected_runs:
                        answer_entry = answer_by_run.get(run_id)
                        if answer_entry and answer_entry.status in {STATUS_FAILED, STATUS_ILL_POSED}:
                            continue
                        expected_for_file += 1
                    correct_count = 0
                    for crit_entry in critiques:
                        if not crit_entry or crit_entry.status != STATUS_SUCCEEDED:
                            continue
                        verdict = _final_critique_verdict(crit_entry)
                        if verdict == CRITIQUE_VERDICT_CORRECT:
                            correct_count += 1
                    expected[key] = max(0, expected_for_file - correct_count)
                else:
                    answer_map = _map_by_key(answers, _question_key_for_answer)
                    count = 0
                    for crit_entry in critiques:
                        if not crit_entry or crit_entry.status != STATUS_SUCCEEDED:
                            continue
                        verdict = _final_critique_verdict(crit_entry)
                        if verdict in {None, CRITIQUE_VERDICT_UNKNOWN, CRITIQUE_VERDICT_CORRECT}:
                            continue
                        entry_key_value = _question_key_for_critique(crit_entry)
                        if not entry_key_value or entry_key_value not in answer_map:
                            continue
                        count += 1
                    expected[key] = count
    return expected


def _expected_eval_count(
    registry,
    judge_specs: Sequence[ModelSpec],
    illposed_counts: Dict[Tuple[str, str], int],
    critique_counts: Dict[Tuple[str, str, str, str], int],
) -> Tuple[int, int]:
    judge_names = [spec.name for spec in judge_specs]
    expected = 0
    expected_tasks = 0
    for (q_slug, a_slug), count in illposed_counts.items():
        if not count:
            continue
        alice_model = registry.resolve_model_name(a_slug)
        bob_model = registry.resolve_model_name(q_slug)
        participants = {alice_model, bob_model}
        expected += count * sum(1 for judge in judge_names if judge not in participants)
        expected_tasks += count
    for (_mode, _q_slug, critic_slug, answer_slug), count in critique_counts.items():
        if not count:
            continue
        alice_model = registry.resolve_model_name(critic_slug)
        bob_model = registry.resolve_model_name(answer_slug)
        participants = {alice_model, bob_model}
        expected += count * sum(1 for judge in judge_names if judge not in participants)
        expected_tasks += count
    return expected, expected_tasks


def _format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return f"{numerator}/0 (n/a)"
    pct = 100.0 * numerator / denominator
    return f"{numerator}/{denominator} ({pct:.1f}%)"


def _final_question(entry: Optional[BenchmarkEntry]) -> Optional[str]:
    if not entry:
        return None
    generations = entry.generation_rounds or []
    if not generations:
        return None
    refinements = generations[-1].refinement_rounds or []
    if not refinements:
        return None
    return refinements[-1].question


def _final_critique_verdict(entry: Optional[CritiqueEntry]) -> Optional[str]:
    if not entry:
        return None
    attempts = entry.attempts or []
    if not attempts:
        return None
    return attempts[-1].verdict


def _has_critique_text(entry: Optional[CritiqueEntry]) -> bool:
    if not entry:
        return False
    attempts = entry.attempts or []
    if not attempts:
        return False
    last = attempts[-1]
    if not last.verdict:
        return False
    if last.notes is None or not isinstance(last.notes, str):
        return False
    return True


def _load_answer_records(
    answers_dir: Path,
    q_slug: str,
    answer_slug: str,
    benchmark_entries: Sequence[Optional[BenchmarkEntry]],
) -> List[Optional[AnswerEntry]]:
    answer_path = answers_dir / q_slug / f"{answer_slug}.json"
    if answer_path.exists():
        return load_answer_entries(answer_path)
    if answer_slug == q_slug:
        return benchmark_answers_from_entries(q_slug, list(benchmark_entries))
    return []


def _count_question_stats(
    benchmark_dir: Path,
    benchmark_specs: Sequence[ModelSpec],
    run_ids: Sequence[str],
) -> Tuple[Counter, int, Dict[str, Set[str]]]:
    counts: Counter = Counter()
    expected_runs_by_slug: Dict[str, Set[str]] = {}
    for spec in benchmark_specs:
        q_slug = spec.slug
        entries = load_benchmark_entries(benchmark_dir / f"{q_slug}.json")
        entries_by_run = _entries_by_run(entries)
        expected_runs: Set[str] = set()
        for run_id in run_ids:
            entry = _pick_entry(entries_by_run, run_id)
            if entry:
                counts["generated"] += 1
                if entry.status == STATUS_SUCCEEDED:
                    counts["succeeded"] += 1
                    expected_runs.add(run_id)
                elif entry.status == STATUS_ILL_POSED:
                    counts["ill_posed"] += 1
                elif entry.status == STATUS_FAILED:
                    counts["failed"] += 1
            else:
                counts["missing"] += 1
                expected_runs.add(run_id)
        expected_runs_by_slug[q_slug] = expected_runs
    expected = len(run_ids) * len(benchmark_specs)
    return counts, expected, expected_runs_by_slug


def _count_answer_stats(
    answers_dir: Path,
    answer_specs: Sequence[ModelSpec],
    benchmark_specs: Sequence[ModelSpec],
    run_ids: Sequence[str],
    expected_runs_by_slug: Dict[str, Set[str]],
    allow_self_answering: bool,
) -> Tuple[Counter, int]:
    counts: Counter = Counter()
    expected = 0
    for q_spec in benchmark_specs:
        q_slug = q_spec.slug
        expected_runs = expected_runs_by_slug.get(q_slug, set())
        eligible_answers = [
            spec for spec in answer_specs
            if allow_self_answering or spec.slug != q_slug
        ]
        expected += len(expected_runs) * len(eligible_answers)
        for answer_spec in eligible_answers:
            answer_path = answers_dir / q_slug / f"{answer_spec.slug}.json"
            entries = load_answer_entries(answer_path) if answer_path.exists() else []
            entries_by_run = _entries_by_run(entries)
            for run_id in run_ids:
                if run_id not in expected_runs:
                    continue
                entry = _pick_entry(entries_by_run, run_id)
                if entry:
                    counts["generated"] += 1
                    if entry.status == STATUS_SUCCEEDED:
                        counts["succeeded"] += 1
                    elif entry.status == STATUS_ILL_POSED:
                        counts["ill_posed"] += 1
                    elif entry.status == STATUS_FAILED:
                        counts["failed"] += 1
                else:
                    counts["missing"] += 1
    return counts, expected


def _expected_critiques_for_question(
    mode: str,
    question_model: ModelSpec,
    benchmark_entries: Sequence[Optional[BenchmarkEntry]],
    answer_specs: Sequence[ModelSpec],
    critic_names: Set[str],
    answers_dir: Path,
    registry,
    limit: Optional[int],
    custom_pairs: Sequence[Tuple[str, str, str]],
    expected_runs: Set[str],
) -> int:
    q_slug = question_model.slug
    question_author_answers = _load_answer_records(
        answers_dir,
        q_slug,
        q_slug,
        benchmark_entries,
    )
    answer_by_run = _entries_by_run(question_author_answers)

    def has_budget(current: int) -> bool:
        return limit is None or current < limit

    count = 0
    if mode == "custom":
        if not custom_pairs:
            return 0
        for question_author, answer_author, critic in custom_pairs:
            if question_author != question_model.name:
                continue
            if critic_names and critic not in critic_names:
                continue
            critic_spec = registry.models.get(critic)
            if not critic_spec or "critique" not in critic_spec.roles:
                continue
            answer_slug = _slugify(answer_author)
            answer_records = _load_answer_records(
                answers_dir,
                q_slug,
                answer_slug,
                benchmark_entries,
            )
            answer_by_run = _entries_by_run(answer_records)
            for run_id in expected_runs:
                if not has_budget(count):
                    return count
                answer_entry = answer_by_run.get(run_id)
                if answer_entry and answer_entry.status in {STATUS_FAILED, STATUS_ILL_POSED}:
                    continue
                count += 1
        return count

    if mode == "contradictor":
        critic_list = sorted(critic_names) if critic_names else [spec.name for spec in answer_specs]
        for critic_model in critic_list:
            if critic_model == question_model.name:
                continue
            critic_spec = registry.models.get(critic_model)
            if not critic_spec or "critique" not in critic_spec.roles:
                continue
            for run_id in expected_runs:
                if not has_budget(count):
                    return count
                answer_entry = answer_by_run.get(run_id)
                if answer_entry and answer_entry.status in {STATUS_FAILED, STATUS_ILL_POSED}:
                    continue
                count += 1
        return count

    if mode == "evaluator" and critic_names and question_model.name not in critic_names:
        return 0

    for answer_spec in answer_specs:
        if mode in {"all", "custom"} and critic_names and answer_spec.name not in critic_names:
            continue
        answer_records = _load_answer_records(
            answers_dir,
            q_slug,
            answer_spec.slug,
            benchmark_entries,
        )
        if answer_spec.name == question_model.name and not answer_records:
            answer_records = question_author_answers

        answer_by_run = _entries_by_run(answer_records)
        for run_id in expected_runs:
            if not has_budget(count):
                return count
            if mode == "evaluator" and answer_spec.name == question_model.name:
                continue
            answer_entry = answer_by_run.get(run_id)
            if answer_entry and answer_entry.status in {STATUS_FAILED, STATUS_ILL_POSED}:
                continue
            count += 1

    return count


def _count_critique_stats(
    critiques_dir: Path,
    benchmark_dir: Path,
    benchmark_specs: Sequence[ModelSpec],
    answer_specs: Sequence[ModelSpec],
    critic_specs: Sequence[ModelSpec],
    critique_modes: Sequence[str],
    answers_dir: Path,
    registry,
    limit: Optional[int],
    custom_pairs: Sequence[Tuple[str, str, str]],
    expected_runs_by_slug: Dict[str, Set[str]],
) -> Tuple[Dict[str, Counter], Dict[str, int]]:
    expected_by_mode: Dict[str, int] = {}
    actual_by_mode: Dict[str, Counter] = {}
    critic_names = {spec.name for spec in critic_specs}

    for mode in critique_modes:
        expected = 0
        counts: Counter = Counter()
        tasks_by_file: Dict[Tuple[str, str, str], List[Tuple[str, QuestionKey]]] = {}

        for spec in benchmark_specs:
            benchmark_entries = load_benchmark_entries(benchmark_dir / f"{spec.slug}.json")
            tasks = _expected_critique_tasks_for_question(
                mode,
                spec,
                benchmark_entries,
                answer_specs,
                critic_names,
                answers_dir,
                registry,
                limit,
                custom_pairs,
                expected_runs_by_slug.get(spec.slug, set()),
            )
            expected += len(tasks)
            for critic_slug, answer_slug, run_id, key in tasks:
                file_key = (spec.slug, critic_slug, answer_slug)
                tasks_by_file.setdefault(file_key, []).append((run_id, key))

        for (q_slug, critic_slug, answer_slug), tasks in tasks_by_file.items():
            crit_file = critiques_dir / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
            critiques = load_critique_entries(crit_file)
            critique_map = _map_by_key_prefer_succeeded(critiques, _question_key_for_critique)
            for _run_id, key in tasks:
                entry = critique_map.get(key)
                if entry:
                    counts["generated"] += 1
                    if entry.status == STATUS_SUCCEEDED:
                        counts["succeeded"] += 1
                    else:
                        counts["failed"] += 1
                else:
                    counts["missing"] += 1

        expected_by_mode[mode] = expected
        actual_by_mode[mode] = counts

    return actual_by_mode, expected_by_mode


def _count_debate_stats(
    debates_dir: Path,
    critiques_dir: Path,
    answers_dir: Path,
    benchmark_dir: Path,
    critique_modes: Sequence[str],
    benchmark_specs: Sequence[ModelSpec],
    expected_critiques_by_mode: Dict[str, int],
    assume_missing_critiques: bool,
) -> Tuple[Counter, Counter, Dict[str, int], Dict[str, int]]:
    actual_illposed = Counter()
    actual_critique_by_mode: Counter = Counter()
    expected_illposed = Counter()
    expected_critique_by_mode: Dict[str, int] = {}
    allowed_slugs = {spec.slug for spec in benchmark_specs}

    for q_dir in answers_dir.glob("*"):
        q_slug = q_dir.name
        if allowed_slugs and q_slug not in allowed_slugs:
            continue
        for answer_file in q_dir.glob("*.json"):
            entries = load_answer_entries(answer_file)
            seen_keys: Set[QuestionKey] = set()
            for entry in entries:
                if entry and entry.status == STATUS_ILL_POSED:
                    key = _question_key_for_answer(entry)
                    if key:
                        seen_keys.add(key)
            expected_illposed["expected"] += len(seen_keys)

    for debate_file in (debates_dir / "illposed").glob("*/*.json"):
        entries = load_debate_entries(debate_file)
        actual_illposed["generated"] += sum(1 for entry in entries if entry)

    for mode in critique_modes:
        expected = 0
        if assume_missing_critiques:
            correct_count = 0
            mode_dir = critiques_dir / mode
            if mode_dir.exists():
                for q_dir in mode_dir.glob("*"):
                    q_slug = q_dir.name
                    if allowed_slugs and q_slug not in allowed_slugs:
                        continue
                    for crit_file in q_dir.glob("*.json"):
                        critiques = load_critique_entries(crit_file)
                        for crit_entry in critiques:
                            if not crit_entry or crit_entry.status != STATUS_SUCCEEDED:
                                continue
                            verdict = _final_critique_verdict(crit_entry)
                            if verdict == CRITIQUE_VERDICT_CORRECT:
                                correct_count += 1
            expected = max(0, expected_critiques_by_mode.get(mode, 0) - correct_count)
        else:
            mode_dir = critiques_dir / mode
            if mode_dir.exists():
                for q_dir in mode_dir.glob("*"):
                    q_slug = q_dir.name
                    if allowed_slugs and q_slug not in allowed_slugs:
                        continue
                    benchmark_entries = load_benchmark_entries(benchmark_dir / f"{q_slug}.json")
                    for crit_file in q_dir.glob("*.json"):
                        parts = crit_file.stem.split("__")
                        if len(parts) != 2:
                            continue
                        _critic_slug, answer_slug = parts
                        answers = _load_answer_records(
                            answers_dir,
                            q_slug,
                            answer_slug,
                            benchmark_entries,
                        )
                        critiques = load_critique_entries(crit_file)
                        answer_map = _map_by_key(answers, _question_key_for_answer)
                        for crit_entry in critiques:
                            if not crit_entry or crit_entry.status != STATUS_SUCCEEDED:
                                continue
                            verdict = _final_critique_verdict(crit_entry)
                            if verdict in {None, CRITIQUE_VERDICT_UNKNOWN, CRITIQUE_VERDICT_CORRECT}:
                                continue
                            key = _question_key_for_critique(crit_entry)
                            if not key or key not in answer_map:
                                continue
                            expected += 1
        expected_critique_by_mode[mode] = expected

        debate_mode_dir = debates_dir / "critiques" / mode
        if debate_mode_dir.exists():
            for q_dir in debate_mode_dir.glob("*"):
                if q_dir.name not in allowed_slugs:
                    continue
                for debate_file in q_dir.glob("*.json"):
                    entries = load_debate_entries(debate_file)
                    actual_critique_by_mode[mode] += sum(1 for entry in entries if entry)

    return actual_illposed, actual_critique_by_mode, expected_illposed, expected_critique_by_mode


def _collect_judging_tasks(
    debates_dir: Path,
    critiques_dir: Path,
    answers_dir: Path,
    benchmark_dir: Path,
    registry,
    critique_modes: Sequence[str],
    allow_no_debate: bool,
    force_correct_critiques: bool,
) -> Dict[str, Tuple[str, str]]:
    tasks: Dict[str, Tuple[str, str]] = {}

    for debate_file in (debates_dir / "illposed").glob("*/*.json"):
        q_slug = debate_file.parent.name
        a_slug = debate_file.stem
        benchmark_entries = load_benchmark_entries(benchmark_dir / f"{q_slug}.json")
        answers = _load_answer_records(
            answers_dir,
            q_slug,
            a_slug,
            benchmark_entries,
        )
        debates = load_debate_entries(debate_file)
        debate_map = _map_by_key(debates, lambda entry, q_slug=q_slug: _question_key_for_debate(entry, q_slug))
        answer_map = _map_by_key(answers, _question_key_for_answer)
        keys = set(debate_map) | set(answer_map)
        for key in keys:
            debate = debate_map.get(key)
            answer_entry = answer_map.get(key)
            status = answer_entry.status if answer_entry else None
            if status and status != STATUS_ILL_POSED:
                continue
            history = debate.history if debate else []
            if not history and not allow_no_debate:
                continue
            if not debate and not answer_entry:
                continue
            alice_model = registry.resolve_model_name((debate.alice_model if debate else None) or a_slug)
            bob_model = registry.resolve_model_name((debate.bob_model if debate else None) or q_slug)
            question = (debate.question if debate else None) or (answer_entry.question if answer_entry else None)
            run_id = (debate.run_id if debate else None) or (answer_entry.run_id if answer_entry else None)
            topic_slug = (debate.topic_slug if debate else None) or (answer_entry.topic_slug if answer_entry else None)
            prefix = f"illposed/{q_slug}/{a_slug}"
            task_key = _task_key(prefix, run_id, topic_slug, question)
            if task_key:
                tasks[task_key] = (alice_model, bob_model)

    for mode in critique_modes:
        mode_dir = critiques_dir / mode
        if not mode_dir.exists():
            continue
        for q_dir in mode_dir.glob("*"):
            q_slug = q_dir.name
            for crit_file in q_dir.glob("*.json"):
                parts = crit_file.stem.split("__")
                if len(parts) != 2:
                    continue
                critic_slug, answer_slug = parts
                critiques = load_critique_entries(crit_file)
                debate_file = debates_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
                debates = load_debate_entries(debate_file)
                debate_map = _map_by_key(debates, lambda entry, q_slug=q_slug: _question_key_for_debate(entry, q_slug))
                for crit_entry in critiques:
                    if not crit_entry or crit_entry.status != STATUS_SUCCEEDED:
                        continue
                    verdict = _final_critique_verdict(crit_entry)
                    if verdict in {None, CRITIQUE_VERDICT_UNKNOWN}:
                        continue
                    if verdict == CRITIQUE_VERDICT_CORRECT and not force_correct_critiques:
                        continue
                    key = _question_key_for_critique(crit_entry)
                    debate = debate_map.get(key)
                    history = debate.history if debate else []
                    if not history and not allow_no_debate:
                        continue
                    if not _has_critique_text(crit_entry):
                        continue
                    alice_model = registry.resolve_model_name((debate.alice_model if debate else None) or critic_slug)
                    bob_model = registry.resolve_model_name((debate.bob_model if debate else None) or answer_slug)
                    question = (debate.question if debate else None) or crit_entry.question
                    run_id = (debate.run_id if debate else None) or crit_entry.run_id
                    topic_slug = (debate.topic_slug if debate else None) or crit_entry.topic_slug
                    prefix = f"critique/{mode}/{q_slug}/{critic_slug}__{answer_slug}"
                    task_key = _task_key(prefix, run_id, topic_slug, question)
                    if task_key:
                        tasks[task_key] = (alice_model, bob_model)

    return tasks


def _count_automated_evaluations(
    automated_dir: Path,
    tasks: Dict[Tuple, Tuple[str, str]],
    judge_specs: Sequence[ModelSpec],
) -> Tuple[int, int]:
    judge_names = [spec.name for spec in judge_specs]
    expected = 0
    participants_by_task = {}
    for task_key, (alice_model, bob_model) in tasks.items():
        participants = {alice_model, bob_model}
        participants_by_task[task_key] = participants
        expected += sum(1 for judge in judge_names if judge not in participants)

    actual = 0
    task_keys = set(tasks.keys())
    for eval_file in automated_dir.glob("*.json"):
        payload = load_evaluation_entries(eval_file)
        for decision in payload.decisions:
            decision_key = judging_task_key(decision)
            if decision_key not in task_keys:
                continue
            participants = participants_by_task.get(decision_key, set())
            if decision.judge_model and decision.judge_model in participants:
                continue
            actual += 1
    return actual, expected


def _parse_custom_map(path: Optional[Path]) -> List[Tuple[str, str, str]]:
    if not path:
        return []
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    pairs = []
    for item in payload:
        question_author = item.get("question_author") or item.get("question_owner")
        answer_author = item.get("answer_author") or item.get("answerer")
        critic = item.get("critic")
        if not question_author or not answer_author or not critic:
            continue
        pairs.append((question_author, answer_author, critic))
    return pairs


def _remaining_question_tasks(
    benchmark_dir: Path,
    benchmark_specs: Sequence[ModelSpec],
    run_ids: Sequence[str],
) -> List[str]:
    lines: List[str] = []
    for spec in benchmark_specs:
        entries = load_benchmark_entries(benchmark_dir / f"{spec.slug}.json")
        entries_by_run = _entries_by_run(entries)
        for run_id in run_ids:
            entry = _pick_entry(entries_by_run, run_id)
            if not entry:
                lines.append(f"question/{spec.slug}/{run_id} status=missing")
            elif entry.status == STATUS_FAILED:
                lines.append(f"question/{spec.slug}/{run_id} status=failed")
    return lines


def _remaining_answer_tasks(
    answers_dir: Path,
    answer_specs: Sequence[ModelSpec],
    benchmark_specs: Sequence[ModelSpec],
    expected_runs_by_slug: Dict[str, Set[str]],
    allow_self_answering: bool,
) -> List[str]:
    lines: List[str] = []
    for q_spec in benchmark_specs:
        q_slug = q_spec.slug
        expected_runs = expected_runs_by_slug.get(q_slug, set())
        eligible_answers = [
            spec for spec in answer_specs
            if allow_self_answering or spec.slug != q_slug
        ]
        for answer_spec in eligible_answers:
            answer_path = answers_dir / q_slug / f"{answer_spec.slug}.json"
            entries = load_answer_entries(answer_path) if answer_path.exists() else []
            entries_by_run = _entries_by_run(entries)
            for run_id in expected_runs:
                entry = _pick_entry(entries_by_run, run_id)
                if not entry:
                    lines.append(f"answer/{q_slug}/{answer_spec.slug}/{run_id} status=missing")
                elif entry.status == STATUS_FAILED:
                    lines.append(f"answer/{q_slug}/{answer_spec.slug}/{run_id} status=failed")
    return lines


def _expected_critique_tasks_for_question(
    mode: str,
    question_model: ModelSpec,
    benchmark_entries: Sequence[Optional[BenchmarkEntry]],
    answer_specs: Sequence[ModelSpec],
    critic_names: Set[str],
    answers_dir: Path,
    registry,
    limit: Optional[int],
    custom_pairs: Sequence[Tuple[str, str, str]],
    expected_runs: Set[str],
) -> List[Tuple[str, str, str, QuestionKey]]:
    tasks: List[Tuple[str, str, str, QuestionKey]] = []
    q_slug = question_model.slug
    question_author_answers = _load_answer_records(
        answers_dir,
        q_slug,
        q_slug,
        benchmark_entries,
    )
    question_author_by_run = _entries_by_run(question_author_answers)
    def task_key(run_id: str, answer_entry: Optional[AnswerEntry]) -> QuestionKey:
        if answer_entry:
            key = _question_key_for_answer(answer_entry)
            if key:
                return key
        return question_key(q_slug, run_id)

    def has_budget(current: int) -> bool:
        return limit is None or current < limit

    count = 0
    if mode == "custom":
        if not custom_pairs:
            return tasks
        for question_author, answer_author, critic in custom_pairs:
            if question_author != question_model.name:
                continue
            if critic_names and critic not in critic_names:
                continue
            critic_spec = registry.models.get(critic)
            if not critic_spec or "critique" not in critic_spec.roles:
                continue
            answer_slug = _slugify(answer_author)
            answer_records = _load_answer_records(
                answers_dir,
                q_slug,
                answer_slug,
                benchmark_entries,
            )
            answer_by_run = _entries_by_run(answer_records)
            for run_id in expected_runs:
                if not has_budget(count):
                    return tasks
                answer_entry = answer_by_run.get(run_id)
                if answer_entry and answer_entry.status in {STATUS_FAILED, STATUS_ILL_POSED}:
                    continue
                tasks.append((critic_spec.slug, answer_slug, run_id, task_key(run_id, answer_entry)))
                count += 1
        return tasks

    if mode == "contradictor":
        critic_list = sorted(critic_names) if critic_names else [spec.name for spec in answer_specs]
        for critic_model in critic_list:
            if critic_model == question_model.name:
                continue
            critic_spec = registry.models.get(critic_model)
            if not critic_spec or "critique" not in critic_spec.roles:
                continue
            for run_id in expected_runs:
                if not has_budget(count):
                    return tasks
                answer_entry = question_author_by_run.get(run_id)
                if answer_entry and answer_entry.status in {STATUS_FAILED, STATUS_ILL_POSED}:
                    continue
                tasks.append((critic_spec.slug, q_slug, run_id, task_key(run_id, answer_entry)))
                count += 1
        return tasks

    if mode == "evaluator" and critic_names and question_model.name not in critic_names:
        return tasks

    for answer_spec in answer_specs:
        if mode in {"all", "custom"} and critic_names and answer_spec.name not in critic_names:
            continue
        answer_records = _load_answer_records(
            answers_dir,
            q_slug,
            answer_spec.slug,
            benchmark_entries,
        )
        if answer_spec.name == question_model.name and not answer_records:
            answer_records = question_author_answers

        answer_by_run = _entries_by_run(answer_records)
        for run_id in expected_runs:
            if not has_budget(count):
                return tasks
            if mode == "evaluator" and answer_spec.name == question_model.name:
                continue
            answer_entry = answer_by_run.get(run_id)
            if answer_entry and answer_entry.status in {STATUS_FAILED, STATUS_ILL_POSED}:
                continue
            critic_slug = question_model.slug if mode == "evaluator" else answer_spec.slug
            tasks.append((critic_slug, answer_spec.slug, run_id, task_key(run_id, answer_entry)))
            count += 1

    return tasks


def _remaining_critique_tasks(
    critiques_dir: Path,
    benchmark_dir: Path,
    benchmark_specs: Sequence[ModelSpec],
    answer_specs: Sequence[ModelSpec],
    critic_specs: Sequence[ModelSpec],
    critique_modes: Sequence[str],
    answers_dir: Path,
    registry,
    limit: Optional[int],
    custom_pairs: Sequence[Tuple[str, str, str]],
    expected_runs_by_slug: Dict[str, Set[str]],
) -> List[str]:
    tasks_by_file: Dict[Tuple[str, str, str, str], List[Tuple[str, QuestionKey]]] = {}
    critic_names = {spec.name for spec in critic_specs}

    for mode in critique_modes:
        for spec in benchmark_specs:
            benchmark_entries = load_benchmark_entries(benchmark_dir / f"{spec.slug}.json")
            tasks = _expected_critique_tasks_for_question(
                mode,
                spec,
                benchmark_entries,
                answer_specs,
                critic_names,
                answers_dir,
                registry,
                limit,
                custom_pairs,
                expected_runs_by_slug.get(spec.slug, set()),
            )
            for critic_slug, answer_slug, run_id, key in tasks:
                file_key = (mode, spec.slug, critic_slug, answer_slug)
                tasks_by_file.setdefault(file_key, []).append((run_id, key))

    lines: List[str] = []
    for (mode, q_slug, critic_slug, answer_slug), tasks in tasks_by_file.items():
        crit_file = critiques_dir / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
        critiques = load_critique_entries(crit_file)
        critique_map = _map_by_key_prefer_succeeded(critiques, _question_key_for_critique)
        for run_id, key in tasks:
            entry = critique_map.get(key)
            if entry and entry.status == STATUS_SUCCEEDED:
                continue
            status = "failed" if entry else "missing"
            lines.append(
                f"critique/{mode}/{q_slug}/{critic_slug}__{answer_slug}/{run_id} status={status}"
            )
    return lines


def _remaining_debate_tasks(
    debates_dir: Path,
    critiques_dir: Path,
    answers_dir: Path,
    benchmark_dir: Path,
    critique_modes: Sequence[str],
    benchmark_specs: Sequence[ModelSpec],
) -> List[str]:
    lines: List[str] = []
    allowed_slugs = {spec.slug for spec in benchmark_specs}

    seen_illposed: Set[QuestionKey] = set()
    for q_dir in answers_dir.glob("*"):
        q_slug = q_dir.name
        if allowed_slugs and q_slug not in allowed_slugs:
            continue
        for answer_file in q_dir.glob("*.json"):
            a_slug = answer_file.stem
            entries = load_answer_entries(answer_file)
            debate_file = debates_dir / "illposed" / q_slug / f"{a_slug}.json"
            debates = load_debate_entries(debate_file)
            debate_map = _map_by_key(debates, lambda entry, q_slug=q_slug: _question_key_for_debate(entry, q_slug))
            for entry in entries:
                if not entry or entry.status != STATUS_ILL_POSED:
                    continue
                key = _question_key_for_answer(entry)
                if not key or key in seen_illposed:
                    continue
                seen_illposed.add(key)
                if key in debate_map:
                    continue
                lines.append(f"debate/illposed/{q_slug}/{a_slug}/{entry.run_id} status=missing")

    for mode in critique_modes:
        mode_dir = critiques_dir / mode
        if not mode_dir.exists():
            continue
        for q_dir in mode_dir.glob("*"):
            q_slug = q_dir.name
            if allowed_slugs and q_slug not in allowed_slugs:
                continue
            benchmark_entries = load_benchmark_entries(benchmark_dir / f"{q_slug}.json")
            for crit_file in q_dir.glob("*.json"):
                parts = crit_file.stem.split("__")
                if len(parts) != 2:
                    continue
                critic_slug, answer_slug = parts
                critiques = load_critique_entries(crit_file)
                answers = _load_answer_records(
                    answers_dir,
                    q_slug,
                    answer_slug,
                    benchmark_entries,
                )
                answer_map = _map_by_key_prefer_succeeded(answers, _question_key_for_answer)
                debate_file = debates_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
                debates = load_debate_entries(debate_file)
                debate_map = _map_by_key(debates, lambda entry, q_slug=q_slug: _question_key_for_debate(entry, q_slug))
                for crit_entry in critiques:
                    if not crit_entry or crit_entry.status != STATUS_SUCCEEDED:
                        continue
                    verdict = _final_critique_verdict(crit_entry)
                    if verdict in {None, CRITIQUE_VERDICT_UNKNOWN, CRITIQUE_VERDICT_CORRECT}:
                        continue
                    key = _question_key_for_critique(crit_entry)
                    if not key:
                        continue
                    if key in debate_map:
                        continue
                    if key not in answer_map:
                        continue
                    lines.append(
                        f"debate/critique/{mode}/{q_slug}/{critic_slug}__{answer_slug}/{crit_entry.run_id} status=missing"
                    )
    return lines


def _remaining_evaluation_tasks(
    automated_dir: Path,
    tasks: Dict[Tuple, Tuple[str, str]],
    judge_specs: Sequence[ModelSpec],
) -> List[str]:
    judge_names = [spec.name for spec in judge_specs]
    participants_by_task = {task_key: {alice, bob} for task_key, (alice, bob) in tasks.items()}
    existing: Set[Tuple[Tuple, str]] = set()

    task_keys = set(participants_by_task)
    for eval_file in automated_dir.glob("*.json"):
        payload = load_evaluation_entries(eval_file)
        for decision in payload.decisions:
            decision_key = judging_task_key(decision)
            if decision_key not in task_keys:
                continue
            participants = participants_by_task.get(decision_key, set())
            if decision.judge_model and decision.judge_model in participants:
                continue
            if decision.judge_model and decision.judge_model in judge_names:
                existing.add((decision_key, decision.judge_model))

    lines: List[str] = []
    for task_key, participants in participants_by_task.items():
        for judge in judge_names:
            if judge in participants:
                continue
            if (task_key, judge) in existing:
                continue
            lines.append(f"evaluation/{format_key(task_key)}/{judge} status=missing")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute pipeline coverage stats with percentages.")
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--runs-file", type=Path, default=Path("configs/runs.json"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--debates-dir", type=Path, default=Path("debates"))
    parser.add_argument("--automated-dir", type=Path, default=Path("automated_evaluations"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-self-answering", action="store_true")
    parser.add_argument("--allow-no-debate", action="store_true")
    parser.add_argument("--force-correct-critiques", action="store_true")
    parser.add_argument(
        "--no-assume-missing",
        action="store_true",
        help="Do not assume missing critiques/debates/evaluations will eventually be generated.",
    )
    parser.add_argument("--benchmark-models", nargs="*")
    parser.add_argument("--answer-models", nargs="*")
    parser.add_argument("--critique-models", nargs="*")
    parser.add_argument("--judge-models", nargs="*")
    parser.add_argument("--critique-modes", nargs="*", default=["contradictor", "evaluator"])
    parser.add_argument("--custom-map", type=Path)
    parser.add_argument(
        "--print",
        dest="print_stage",
        choices=["questions", "answers", "critiques", "debates", "evaluations"],
        help="Print remaining tasks for a pipeline stage.",
    )
    args = parser.parse_args()

    registry = load_registry(str(args.config))
    run_ids = _load_runs(args.runs_file, args.limit)

    benchmark_specs = registry.pick(args.benchmark_models) if args.benchmark_models else registry.by_role("benchmark")
    answer_specs = registry.pick(args.answer_models) if args.answer_models else registry.by_role("answer")
    critic_specs = registry.pick(args.critique_models) if args.critique_models else registry.by_role("critique")
    judge_specs = registry.pick(args.judge_models) if args.judge_models else list(registry.models.values())

    custom_pairs = _parse_custom_map(args.custom_map)

    question_counts, question_expected, expected_runs_by_slug = _count_question_stats(
        args.benchmark_dir,
        benchmark_specs,
        run_ids,
    )
    answer_counts, answer_expected = _count_answer_stats(
        args.answers_dir,
        answer_specs,
        benchmark_specs,
        run_ids,
        expected_runs_by_slug,
        args.allow_self_answering,
    )
    critique_counts_by_mode, critique_expected_by_mode = _count_critique_stats(
        args.critiques_dir,
        args.benchmark_dir,
        benchmark_specs,
        answer_specs,
        critic_specs,
        args.critique_modes,
        args.answers_dir,
        registry,
        args.limit,
        custom_pairs,
        expected_runs_by_slug,
    )
    assume_missing = not args.no_assume_missing

    actual_illposed, actual_critique_by_mode, expected_illposed, expected_critique_by_mode = _count_debate_stats(
        args.debates_dir,
        args.critiques_dir,
        args.answers_dir,
        args.benchmark_dir,
        args.critique_modes,
        benchmark_specs,
        critique_expected_by_mode,
        assume_missing,
    )
    judging_tasks = _collect_judging_tasks(
        args.debates_dir,
        args.critiques_dir,
        args.answers_dir,
        args.benchmark_dir,
        registry,
        args.critique_modes,
        args.allow_no_debate,
        args.force_correct_critiques,
    )
    eval_actual, eval_expected_actual = _count_automated_evaluations(
        args.automated_dir,
        judging_tasks,
        judge_specs,
    )
    if assume_missing:
        illposed_expected_by_pair = _expected_illposed_by_pair(args.answers_dir, {spec.slug for spec in benchmark_specs})
        critique_expected_by_pair = _expected_critique_debates_by_pair(
            args.critiques_dir,
            args.answers_dir,
            args.benchmark_dir,
            args.critique_modes,
            {spec.slug for spec in benchmark_specs},
            expected_runs_by_slug,
            assume_missing,
        )
        eval_expected, expected_task_count = _expected_eval_count(
            registry,
            judge_specs,
            illposed_expected_by_pair,
            critique_expected_by_pair,
        )
    else:
        eval_expected = eval_expected_actual
        expected_task_count = len(judging_tasks)

    print("Pipeline coverage:")
    print(f"- Runs: {len(run_ids)}")
    print(f"- Benchmark models: {len(benchmark_specs)}")
    print(f"- Answer models: {len(answer_specs)}")
    print(f"- Critique models: {len(critic_specs)}")
    print(f"- Judge models: {len(judge_specs)}")

    print("\nQuestions:")
    print(f"- Generated: {_format_ratio(question_counts['generated'], question_expected)}")
    if question_counts["succeeded"]:
        print(f"- Succeeded: {_format_ratio(question_counts['succeeded'], question_expected)}")
    if question_counts["ill_posed"]:
        print(f"- Ill-posed: {_format_ratio(question_counts['ill_posed'], question_expected)}")
    if question_counts["failed"]:
        print(f"- Failed: {_format_ratio(question_counts['failed'], question_expected)}")

    print("\nAnswers:")
    print(f"- Generated: {_format_ratio(answer_counts['generated'], answer_expected)}")
    if answer_counts["succeeded"]:
        print(f"- Succeeded: {_format_ratio(answer_counts['succeeded'], answer_expected)}")
    if answer_counts["ill_posed"]:
        print(f"- Ill-posed: {_format_ratio(answer_counts['ill_posed'], answer_expected)}")
    if answer_counts["failed"]:
        print(f"- Failed: {_format_ratio(answer_counts['failed'], answer_expected)}")

    print("\nCritiques:")
    total_expected = sum(critique_expected_by_mode.values())
    total_generated = sum(counts["generated"] for counts in critique_counts_by_mode.values())
    print(f"- Total generated: {_format_ratio(total_generated, total_expected)}")
    for mode in args.critique_modes:
        expected = critique_expected_by_mode.get(mode, 0)
        generated = critique_counts_by_mode.get(mode, Counter())["generated"]
        print(f"- {mode}: {_format_ratio(generated, expected)}")

    print("\nDebates:")
    print(f"- Ill-posed: {_format_ratio(actual_illposed['generated'], expected_illposed['expected'])}")
    total_expected_debates = expected_illposed["expected"] + sum(expected_critique_by_mode.values())
    total_actual_debates = actual_illposed["generated"] + sum(actual_critique_by_mode.values())
    print(f"- Critique total: {_format_ratio(sum(actual_critique_by_mode.values()), sum(expected_critique_by_mode.values()))}")
    for mode in args.critique_modes:
        expected = expected_critique_by_mode.get(mode, 0)
        actual = actual_critique_by_mode.get(mode, 0)
        print(f"- Critique {mode}: {_format_ratio(actual, expected)}")
    print(f"- Debates total: {_format_ratio(total_actual_debates, total_expected_debates)}")

    print("\nAutomated evaluations:")
    print(f"- Decisions: {_format_ratio(eval_actual, eval_expected)}")
    if assume_missing and expected_task_count != len(judging_tasks):
        print(f"- Tasks: expected={expected_task_count} actual={len(judging_tasks)}")
    else:
        print(f"- Tasks: {len(judging_tasks)}")

    if args.print_stage:
        print(f"\nRemaining {args.print_stage}:")
        if args.print_stage == "questions":
            lines = _remaining_question_tasks(
                args.benchmark_dir,
                benchmark_specs,
                run_ids,
            )
        elif args.print_stage == "answers":
            lines = _remaining_answer_tasks(
                args.answers_dir,
                answer_specs,
                benchmark_specs,
                expected_runs_by_slug,
                args.allow_self_answering,
            )
        elif args.print_stage == "critiques":
            lines = _remaining_critique_tasks(
                args.critiques_dir,
                args.benchmark_dir,
                benchmark_specs,
                answer_specs,
                critic_specs,
                args.critique_modes,
                args.answers_dir,
                registry,
                args.limit,
                custom_pairs,
                expected_runs_by_slug,
            )
        elif args.print_stage == "debates":
            lines = _remaining_debate_tasks(
                args.debates_dir,
                args.critiques_dir,
                args.answers_dir,
                args.benchmark_dir,
                args.critique_modes,
                benchmark_specs,
            )
        else:
            if assume_missing and expected_task_count != len(judging_tasks):
                print(
                    f"- Note: expected tasks {expected_task_count} > available {len(judging_tasks)}; "
                    "missing upstream data."
                )
            lines = _remaining_evaluation_tasks(
                args.automated_dir,
                judging_tasks,
                judge_specs,
            )
        if not lines:
            print("- (none)")
        else:
            for line in sorted(lines):
                print(f"- {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
