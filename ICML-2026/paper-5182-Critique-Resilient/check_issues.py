#!/usr/bin/env python3
"""
Check for issues in output files across the pipeline.

Reports files with failed statuses, unknown verdicts, parsing errors, etc.
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from constants import (
    CRITIQUE_VERDICT_CORRECT,
    JUDGE_VERDICT_UNKNOWN,
    STATUS_FAILED,
    STATUS_ILL_POSED,
    STATUS_SUCCEEDED,
    VALID_CRITIQUE_VERDICTS,
    VALID_CRITIQUE_DEBATE_VERDICTS,
    VALID_ILLPOSED_DEBATE_VERDICTS,
    VALID_STATUSES,
)
from data_models import (
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    load_debate_entries,
    load_evaluation_entries,
)
from utils import question_key

PARSE_ERROR_REGEX = re.compile(
    r"(cannot parse|could not parse|can't parse|cant parse|can not parse|"
    r"unable to parse|failed to parse|parsing error|parse error|parsing_error)",
    re.IGNORECASE,
)
EMPTY_ENTRY_REGEX = re.compile(r"empty entr(y|ies)", re.IGNORECASE)
EVAL_JSON_PARSE_REGEX = re.compile(r"parse evaluation json", re.IGNORECASE)


def _get_value(entry: Any, key: str, default: Any = None) -> Any:
    if isinstance(entry, dict):
        return entry.get(key, default)
    return getattr(entry, key, default)


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


QuestionKey = Tuple[Optional[str], Optional[str], Optional[str]]


def _question_key_from_entry(entry: Any, question_model: Optional[str] = None) -> Optional[QuestionKey]:
    if not entry:
        return None
    run_id = _get_value(entry, "run_id")
    outer_attempt = _get_value(entry, "outer_attempt")
    q_model = question_model or _get_value(entry, "question_model") or _get_value(entry, "question_author")
    return question_key(q_model, run_id, outer_attempt)


def _make_key_index() -> Set[QuestionKey]:
    return set()


def _add_key_to_index(index: Set[QuestionKey], key: Optional[QuestionKey]) -> None:
    if not key:
        return
    index.add(key)


def _key_in_index(index: Set[QuestionKey], key: Optional[QuestionKey]) -> bool:
    if not key:
        return False
    return key in index


def final_benchmark_question(entry: Any) -> str:
    gen_rounds = _as_list(_get_value(entry, "generation_rounds"))
    if not gen_rounds:
        return ""
    refinements = _as_list(_get_value(gen_rounds[-1], "refinement_rounds"))
    if not refinements:
        return ""
    return _get_value(refinements[-1], "question") or ""


def final_benchmark_answer(entry: Any) -> str:
    gen_rounds = _as_list(_get_value(entry, "generation_rounds"))
    if not gen_rounds:
        return ""
    refinements = _as_list(_get_value(gen_rounds[-1], "refinement_rounds"))
    if not refinements:
        return ""
    return _get_value(refinements[-1], "answer") or ""


def final_answer_text(entry: Any) -> str:
    attempts = _as_list(_get_value(entry, "attempts"))
    if not attempts:
        return ""
    return _get_value(attempts[-1], "answer") or ""


def final_critique_text(entry: Any) -> str:
    attempts = _as_list(_get_value(entry, "attempts"))
    if not attempts:
        return ""
    last = attempts[-1]
    notes = _get_value(last, "notes")
    raw_critique = _get_value(last, "raw_critique")
    return str(notes or raw_critique or "")


def _has_unknown_self_check(evaluation: Optional[Dict[str, Any]]) -> bool:
    if evaluation is None:
        return True
    if not isinstance(evaluation, dict):
        return True
    issues = evaluation.get("issues")
    return isinstance(issues, list) and "unknown" in issues


def _has_self_eval_parse_error(evaluation: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(evaluation, dict):
        return False
    issues = evaluation.get("issues")
    if not isinstance(issues, list):
        return False
    for issue in issues:
        if isinstance(issue, str) and EVAL_JSON_PARSE_REGEX.search(issue):
            return True
    return False


def _contains_parse_error_text(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(PARSE_ERROR_REGEX.search(value))
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and PARSE_ERROR_REGEX.search(key):
                return True
            if _contains_parse_error_text(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_parse_error_text(item) for item in value)
    return False


def _entry_has_parse_error_text(entry: Any) -> bool:
    if entry is None:
        return False
    dump = entry.model_dump() if hasattr(entry, "model_dump") else entry
    return _contains_parse_error_text(dump)


def _contains_empty_entry_text(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(EMPTY_ENTRY_REGEX.search(value))
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and EMPTY_ENTRY_REGEX.search(key):
                return True
            if _contains_empty_entry_text(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_empty_entry_text(item) for item in value)
    return False


def _entry_has_empty_entry_text(entry: Any) -> bool:
    if entry is None:
        return False
    dump = entry.model_dump() if hasattr(entry, "model_dump") else entry
    return _contains_empty_entry_text(dump)


def _merge_issue_counts(target: Dict[str, int], entry_counts: Dict[str, int]) -> None:
    for issue, count in entry_counts.items():
        target[issue] += count


def _benchmark_entry_issue_counts(entry: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if _entry_has_parse_error_text(entry):
        counts["parse_error_text"] = 1
    if _entry_has_empty_entry_text(entry):
        counts["empty_entry_text"] = 1

    status = _get_value(entry, "status")
    if status == STATUS_FAILED:
        counts["failed_generation"] = counts.get("failed_generation", 0) + 1
    elif status and status not in {STATUS_SUCCEEDED, STATUS_FAILED, STATUS_ILL_POSED}:
        counts["unknown_status"] = counts.get("unknown_status", 0) + 1

    gen_rounds = _as_list(_get_value(entry, "generation_rounds"))
    if gen_rounds:
        refinements = _as_list(_get_value(gen_rounds[-1], "refinement_rounds"))
        if refinements:
            evaluation = _get_value(refinements[-1], "evaluation")
            if _has_unknown_self_check(evaluation):
                counts["unknown_self_check"] = counts.get("unknown_self_check", 0) + 1
            if _has_self_eval_parse_error(evaluation):
                counts["self_eval_parse_error"] = counts.get("self_eval_parse_error", 0) + 1

    final_q = final_benchmark_question(entry).strip()
    if not final_q and status == STATUS_SUCCEEDED:
        counts["empty_question"] = counts.get("empty_question", 0) + 1

    final_a = final_benchmark_answer(entry).strip()
    if not final_a and status == STATUS_SUCCEEDED:
        counts["empty_answer"] = counts.get("empty_answer", 0) + 1

    return counts


def _answer_entry_issue_counts(entry: Any, failed_answer_min_attempts: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if _entry_has_parse_error_text(entry):
        counts["parse_error_text"] = 1
    if _entry_has_empty_entry_text(entry):
        counts["empty_entry_text"] = 1

    attempts = _as_list(_get_value(entry, "attempts"))
    status = _get_value(entry, "status")
    if status == STATUS_FAILED:
        if len(attempts) < failed_answer_min_attempts:
            counts["failed_answer"] = counts.get("failed_answer", 0) + 1
    elif status and status not in VALID_STATUSES:
        counts["unknown_status"] = counts.get("unknown_status", 0) + 1

    if attempts:
        evaluation = _get_value(attempts[-1], "evaluation")
        if _has_unknown_self_check(evaluation):
            counts["unknown_self_check"] = counts.get("unknown_self_check", 0) + 1
        if _has_self_eval_parse_error(evaluation):
            counts["self_eval_parse_error"] = counts.get("self_eval_parse_error", 0) + 1

    final_a = final_answer_text(entry).strip()
    if not final_a and status == STATUS_SUCCEEDED:
        counts["empty_answer"] = counts.get("empty_answer", 0) + 1

    return counts


def _critique_entry_issue_counts(entry: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if _entry_has_parse_error_text(entry):
        counts["parse_error_text"] = 1
    if _entry_has_empty_entry_text(entry):
        counts["empty_entry_text"] = 1

    status = _get_value(entry, "status")
    if status == STATUS_FAILED:
        counts["failed_critique"] = counts.get("failed_critique", 0) + 1
    elif status and status not in {STATUS_SUCCEEDED, STATUS_FAILED}:
        counts["unknown_status"] = counts.get("unknown_status", 0) + 1

    attempts = _as_list(_get_value(entry, "attempts"))
    last_attempt = attempts[-1] if attempts else None
    verdict = _get_value(last_attempt, "verdict") if last_attempt else None
    if verdict and verdict not in VALID_CRITIQUE_VERDICTS:
        counts["invalid_verdict"] = counts.get("invalid_verdict", 0) + 1
    if last_attempt:
        evaluation = _get_value(last_attempt, "evaluation")
        if _has_unknown_self_check(evaluation):
            counts["unknown_self_check"] = counts.get("unknown_self_check", 0) + 1
        if _has_self_eval_parse_error(evaluation):
            counts["self_eval_parse_error"] = counts.get("self_eval_parse_error", 0) + 1

    final_c = final_critique_text(entry).strip()
    if not final_c and status == STATUS_SUCCEEDED and verdict != CRITIQUE_VERDICT_CORRECT:
        counts["empty_critique"] = counts.get("empty_critique", 0) + 1

    return counts


def _debate_entry_issue_counts(entry: Any, incomplete_min_rounds: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if _entry_has_parse_error_text(entry):
        counts["parse_error_text"] = 1
    if _entry_has_empty_entry_text(entry):
        counts["empty_entry_text"] = 1

    status = _get_value(entry, "status")
    if status == STATUS_FAILED:
        counts["failed_debate"] = counts.get("failed_debate", 0) + 1
    elif status is not None and status != STATUS_SUCCEEDED:
        counts["unknown_status"] = counts.get("unknown_status", 0) + 1

    history = _as_list(_get_value(entry, "history"))
    if not history:
        counts["empty_debate"] = counts.get("empty_debate", 0) + 1

    if len(history) < max(0, incomplete_min_rounds):
        for round_entry in history:
            message = _get_value(round_entry, "message")
            speaker = _get_value(round_entry, "speaker")
            if not message or not speaker:
                counts["incomplete_round"] = counts.get("incomplete_round", 0) + 1

    return counts


def _evaluation_entry_issue_counts(entry: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if _entry_has_parse_error_text(entry):
        counts["parse_error_text"] = 1
    if _entry_has_empty_entry_text(entry):
        counts["empty_entry_text"] = 1

    verdict = _get_value(entry, "verdict")
    if verdict == JUDGE_VERDICT_UNKNOWN:
        counts["unknown_verdict"] = counts.get("unknown_verdict", 0) + 1
    elif verdict not in VALID_CRITIQUE_DEBATE_VERDICTS and verdict not in VALID_ILLPOSED_DEBATE_VERDICTS:
        counts["invalid_verdict"] = counts.get("invalid_verdict", 0) + 1

    if not _get_value(entry, "reasoning"):
        counts["missing_reasoning"] = counts.get("missing_reasoning", 0) + 1
    if not _get_value(entry, "judge_model"):
        counts["missing_judge"] = counts.get("missing_judge", 0) + 1

    confidence = _get_value(entry, "confidence")
    if confidence is not None:
        if not isinstance(confidence, int) or not (1 <= confidence <= 5):
            counts["invalid_confidence"] = counts.get("invalid_confidence", 0) + 1
        elif confidence <= 2:
            counts["low_confidence"] = counts.get("low_confidence", 0) + 1
    else:
        counts["missing_confidence"] = counts.get("missing_confidence", 0) + 1

    status = _get_value(entry, "status")
    if status == STATUS_FAILED:
        counts["failed_evaluation"] = counts.get("failed_evaluation", 0) + 1
    elif status not in {STATUS_SUCCEEDED, STATUS_FAILED, None}:
        counts["unknown_status"] = counts.get("unknown_status", 0) + 1

    return counts


def check_benchmark_issues(file_path: Path) -> Dict[str, int]:
    """Check for issues in benchmark files."""
    issues = defaultdict(int)

    try:
        try:
            data = load_benchmark_entries(file_path)
        except json.JSONDecodeError:
            issues["parse_error"] += 1
            return issues
        except Exception as exc:
            if "Expected list" in str(exc):
                issues["invalid_format"] += 1
            else:
                issues["schema_error"] += 1
            return issues

        for entry in data:
            if not entry:
                continue
            _merge_issue_counts(issues, _benchmark_entry_issue_counts(entry))
    except Exception:
        issues["parse_error"] += 1

    return issues


def check_answer_issues(
    file_path: Path,
    failed_answer_min_attempts: int = 5,
    benchmark_index: Optional[Dict[str, Set[QuestionKey]]] = None,
) -> Dict[str, int]:
    """Check for issues in answer files."""
    issues = defaultdict(int)

    try:
        try:
            data = load_answer_entries(file_path)
        except json.JSONDecodeError:
            issues["parse_error"] += 1
            return issues
        except Exception as exc:
            if "Expected list" in str(exc):
                issues["invalid_format"] += 1
            else:
                issues["schema_error"] += 1
            return issues

        for entry in data:
            if not entry:
                continue
            _merge_issue_counts(issues, _answer_entry_issue_counts(entry, failed_answer_min_attempts))
            if benchmark_index is not None:
                q_slug = _get_value(entry, "question_model") or file_path.parent.name
                question_keys = benchmark_index.get(q_slug)
                if question_keys:
                    key = _question_key_from_entry(entry, q_slug)
                    if key and not _key_in_index(question_keys, key):
                        issues["missing_question_reference"] += 1
    except Exception:
        issues["parse_error"] += 1

    return issues


def check_critique_issues(
    file_path: Path,
    answer_index: Optional[Dict[Tuple[str, str], Set[QuestionKey]]] = None,
) -> Dict[str, int]:
    """Check for issues in critique files."""
    issues = defaultdict(int)
    answer_keys = None
    if answer_index is not None:
        pair = _parse_pair_file(file_path)
        if pair:
            q_slug = file_path.parent.name
            _, answer_slug = pair
            answer_keys = answer_index.get((q_slug, answer_slug))

    try:
        try:
            data = load_critique_entries(file_path)
        except json.JSONDecodeError:
            issues["parse_error"] += 1
            return issues
        except Exception as exc:
            if "Expected list" in str(exc):
                issues["invalid_format"] += 1
            else:
                issues["schema_error"] += 1
            return issues

        for entry in data:
            if not entry:
                continue
            _merge_issue_counts(issues, _critique_entry_issue_counts(entry))
            if answer_index is not None:
                key = _question_key_from_entry(entry, file_path.parent.name)
                if key and (not answer_keys or not _key_in_index(answer_keys, key)):
                    issues["missing_answer_reference"] += 1
    except Exception:
        issues["parse_error"] += 1

    return issues


def check_debate_issues(
    file_path: Path,
    incomplete_min_rounds: int = 5,
    critique_index: Optional[Dict[Tuple[str, str, str, str], Set[QuestionKey]]] = None,
) -> Dict[str, int]:
    """Check for issues in debate files."""
    issues = defaultdict(int)
    critique_keys = None
    check_missing_critique = False
    if critique_index is not None and file_path.parent.parent.parent.name == "critiques":
        mode = file_path.parent.parent.name
        q_slug = file_path.parent.name
        pair = _parse_pair_file(file_path)
        if pair:
            critic_slug, answer_slug = pair
            critique_keys = critique_index.get((mode, q_slug, critic_slug, answer_slug))
            check_missing_critique = True

    try:
        try:
            data = load_debate_entries(file_path)
        except json.JSONDecodeError:
            issues["parse_error"] += 1
            return issues
        except Exception as exc:
            if "Expected list" in str(exc):
                issues["invalid_format"] += 1
            else:
                issues["schema_error"] += 1
            return issues

        try:
            raw_entries = json.loads(file_path.read_text())
        except Exception:
            raw_entries = None
        if isinstance(raw_entries, list):
            for raw_entry in raw_entries:
                if not raw_entry or not isinstance(raw_entry, dict):
                    continue
                status = raw_entry.get("status")
                if status == STATUS_FAILED:
                    issues["failed_debate"] += 1
                elif status is not None and status != STATUS_SUCCEEDED:
                    issues["unknown_status"] += 1

        for entry in data:
            if not entry:
                continue
            _merge_issue_counts(issues, _debate_entry_issue_counts(entry, incomplete_min_rounds))
            if check_missing_critique:
                key = _question_key_from_entry(entry, q_slug)
                if key and (not critique_keys or not _key_in_index(critique_keys, key)):
                    issues["missing_critique_reference"] += 1
    except Exception:
        issues["parse_error"] += 1

    return issues


def check_evaluation_issues(
    file_path: Path,
    illposed_debate_index: Optional[Dict[Tuple[str, str], Set[QuestionKey]]] = None,
    critique_debate_index: Optional[Dict[Tuple[str, str, str, str], Set[QuestionKey]]] = None,
) -> Dict[str, int]:
    """Check for issues in automated evaluation files."""
    issues = defaultdict(int)

    try:
        try:
            payload = load_evaluation_entries(file_path)
        except json.JSONDecodeError:
            issues["parse_error"] += 1
            return issues
        except Exception as exc:
            if "Invalid evaluation file format" in str(exc):
                issues["invalid_format"] += 1
            else:
                issues["schema_error"] += 1
            return issues

        decisions = payload.decisions

        for evaluation in decisions:
            _merge_issue_counts(issues, _evaluation_entry_issue_counts(evaluation))
            if illposed_debate_index is not None or critique_debate_index is not None:
                eval_type = _get_value(evaluation, "type")
                q_slug = _get_value(evaluation, "question_model")
                answer_slug = _get_value(evaluation, "answer_model")
                critic_slug = _get_value(evaluation, "critic_model")
                mode = _get_value(evaluation, "mode")
                key = question_key(
                    q_slug,
                    _get_value(evaluation, "run_id"),
                    _get_value(evaluation, "outer_attempt"),
                )
                missing_debate = False
                if eval_type == "illposed":
                    debate_keys = (
                        illposed_debate_index.get((q_slug, answer_slug))
                        if illposed_debate_index and q_slug and answer_slug
                        else None
                    )
                    if not debate_keys or not _key_in_index(debate_keys, key):
                        missing_debate = True
                elif eval_type in {"critique", "critique_debate"}:
                    debate_keys = (
                        critique_debate_index.get((mode, q_slug, critic_slug, answer_slug))
                        if critique_debate_index and mode and q_slug and critic_slug and answer_slug
                        else None
                    )
                    if not debate_keys or not _key_in_index(debate_keys, key):
                        missing_debate = True
                if missing_debate:
                    issues["missing_debate_reference"] += 1
    except Exception:
        issues["parse_error"] += 1

    return issues


def scan_directory(base_path: Path, check_func, pattern: str = "**/*.json") -> Dict[str, Dict[str, int]]:
    """Scan a directory for JSON files and check for issues."""
    results = {}

    if not base_path.exists():
        return results

    for file_path in base_path.glob(pattern):
        if file_path.is_file():
            issues = check_func(file_path)
            if issues:  # Only include files with issues
                rel_path = str(file_path.relative_to(base_path.parent))
                results[rel_path] = dict(issues)

    return results


def _parse_drop_codes(values: Optional[List[str]]) -> Set[str]:
    codes: Set[str] = set()
    for value in values or []:
        for code in value.split(","):
            code = code.strip()
            if code:
                codes.add(code)
    return codes


def _issues_match(entry_counts: Dict[str, int], drop_codes: Set[str]) -> bool:
    return any(issue in drop_codes for issue in entry_counts.keys())


def _load_json_list(path: Path) -> Optional[List[Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    return data


def _save_json_list(path: Path, entries: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2))


def _load_json_payload(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _parse_pair_file(path: Path) -> Optional[Tuple[str, str]]:
    parts = path.stem.split("__")
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _benchmark_entry_key(entry: Any, question_model: Optional[str]) -> Optional[QuestionKey]:
    return _question_key_from_entry(entry, question_model)


def _build_answer_reference_index(
    answers_dir: Path,
    benchmarks_dir: Path,
) -> Dict[Tuple[str, str], Set[QuestionKey]]:
    index: Dict[Tuple[str, str], Set[QuestionKey]] = defaultdict(_make_key_index)
    if answers_dir.exists():
        for file_path in answers_dir.glob("*/*.json"):
            data = _load_json_list(file_path)
            if data is None:
                continue
            q_slug = file_path.parent.name
            answer_slug = file_path.stem
            for entry in data:
                if not entry:
                    continue
                key = _question_key_from_entry(entry, q_slug)
                _add_key_to_index(index[(q_slug, answer_slug)], key)

    if benchmarks_dir.exists():
        for bench_path in benchmarks_dir.glob("*.json"):
            q_slug = bench_path.stem
            fallback_path = answers_dir / q_slug / f"{q_slug}.json"
            if fallback_path.exists():
                continue
            data = _load_json_list(bench_path)
            if data is None:
                continue
            for entry in data:
                if not entry:
                    continue
                question = final_benchmark_question(entry)
                if not question or not str(question).strip():
                    continue
                key = _question_key_from_entry(entry, q_slug)
                _add_key_to_index(index[(q_slug, q_slug)], key)
    return index


def _build_benchmark_reference_index(
    benchmarks_dir: Path,
) -> Dict[str, Set[QuestionKey]]:
    index: Dict[str, Set[QuestionKey]] = defaultdict(_make_key_index)
    if not benchmarks_dir.exists():
        return index
    for bench_path in benchmarks_dir.glob("*.json"):
        q_slug = bench_path.stem
        data = _load_json_list(bench_path)
        if data is None:
            continue
        for entry in data:
            if not entry:
                continue
            question = final_benchmark_question(entry)
            if not question or not str(question).strip():
                continue
            key = _question_key_from_entry(entry, q_slug)
            _add_key_to_index(index[q_slug], key)
    return index


def _build_critique_reference_index(
    critiques_dir: Path,
) -> Dict[Tuple[str, str, str, str], Set[QuestionKey]]:
    index: Dict[Tuple[str, str, str, str], Set[QuestionKey]] = defaultdict(_make_key_index)
    if not critiques_dir.exists():
        return index
    for mode_dir in critiques_dir.iterdir():
        if not mode_dir.is_dir():
            continue
        mode = mode_dir.name
        for q_dir in mode_dir.iterdir():
            if not q_dir.is_dir():
                continue
            q_slug = q_dir.name
            for file_path in q_dir.glob("*.json"):
                pair = _parse_pair_file(file_path)
                if not pair:
                    continue
                critic_slug, answer_slug = pair
                data = _load_json_list(file_path)
                if data is None:
                    continue
                for entry in data:
                    if not entry:
                        continue
                    key = _question_key_from_entry(entry, q_slug)
                    _add_key_to_index(index[(mode, q_slug, critic_slug, answer_slug)], key)
    return index


def _build_debate_reference_indexes(
    debates_dir: Path,
) -> Tuple[
    Dict[Tuple[str, str], Set[QuestionKey]],
    Dict[Tuple[str, str, str, str], Set[QuestionKey]],
]:
    illposed_index: Dict[Tuple[str, str], Set[QuestionKey]] = defaultdict(_make_key_index)
    critique_index: Dict[Tuple[str, str, str, str], Set[QuestionKey]] = defaultdict(_make_key_index)
    if not debates_dir.exists():
        return illposed_index, critique_index

    illposed_dir = debates_dir / "illposed"
    if illposed_dir.exists():
        for q_dir in illposed_dir.iterdir():
            if not q_dir.is_dir():
                continue
            q_slug = q_dir.name
            for file_path in q_dir.glob("*.json"):
                answer_slug = file_path.stem
                data = _load_json_list(file_path)
                if data is None:
                    continue
                for entry in data:
                    if not entry:
                        continue
                    key = _question_key_from_entry(entry, q_slug)
                    _add_key_to_index(illposed_index[(q_slug, answer_slug)], key)

    critique_dir = debates_dir / "critiques"
    if critique_dir.exists():
        for mode_dir in critique_dir.iterdir():
            if not mode_dir.is_dir():
                continue
            mode = mode_dir.name
            for q_dir in mode_dir.iterdir():
                if not q_dir.is_dir():
                    continue
                q_slug = q_dir.name
                for file_path in q_dir.glob("*.json"):
                    pair = _parse_pair_file(file_path)
                    if not pair:
                        continue
                    critic_slug, answer_slug = pair
                    data = _load_json_list(file_path)
                    if data is None:
                        continue
                    for entry in data:
                        if not entry:
                            continue
                        key = _question_key_from_entry(entry, q_slug)
                        _add_key_to_index(critique_index[(mode, q_slug, critic_slug, answer_slug)], key)
    return illposed_index, critique_index


def _drop_entries(
    drop_codes: Set[str],
    failed_answer_min_attempts: int,
    incomplete_debate_min_rounds: int,
) -> Dict[str, int]:
    counts = {
        "benchmark_entries": 0,
        "benchmark_files": 0,
        "answer_entries": 0,
        "answer_files": 0,
        "critique_entries": 0,
        "critique_files": 0,
        "debate_entries": 0,
        "debate_files": 0,
        "evaluation_entries": 0,
        "evaluation_files": 0,
    }

    benchmark_drop_keys: Dict[str, Set[QuestionKey]] = defaultdict(_make_key_index)
    answer_drop_keys: Dict[Tuple[str, str], Set[QuestionKey]] = defaultdict(_make_key_index)
    critique_drop_keys: Dict[Tuple[str, str, str, str], Set[QuestionKey]] = defaultdict(_make_key_index)
    critique_drop_by_answer: Dict[Tuple[str, str], Set[QuestionKey]] = defaultdict(_make_key_index)
    illposed_debate_drop_keys: Dict[Tuple[str, str], Set[QuestionKey]] = defaultdict(_make_key_index)
    critique_debate_drop_keys: Dict[Tuple[str, str, str, str], Set[QuestionKey]] = defaultdict(_make_key_index)
    critique_debate_drop_by_answer: Dict[Tuple[str, str], Set[QuestionKey]] = defaultdict(_make_key_index)

    # Benchmarks
    benchmarks_dir = Path("benchmarks")
    if benchmarks_dir.exists():
        for file_path in benchmarks_dir.glob("*.json"):
            data = _load_json_list(file_path)
            if data is None:
                continue
            kept: List[Any] = []
            dropped = 0
            q_slug = file_path.stem
            for entry in data:
                if not entry:
                    continue
                entry_issues = _benchmark_entry_issue_counts(entry)
                if _issues_match(entry_issues, drop_codes):
                    dropped += 1
                    counts["benchmark_entries"] += 1
                    _add_key_to_index(benchmark_drop_keys[q_slug], _benchmark_entry_key(entry))
                    continue
                kept.append(entry)
            if dropped:
                counts["benchmark_files"] += 1
                _save_json_list(file_path, kept)

    # Answers
    answers_dir = Path("answers")
    if answers_dir.exists():
        for file_path in answers_dir.glob("*/*.json"):
            data = _load_json_list(file_path)
            if data is None:
                continue
            kept = []
            dropped = 0
            q_slug = file_path.parent.name
            answer_slug = file_path.stem
            benchmark_index = benchmark_drop_keys.get(q_slug)
            for entry in data:
                if not entry:
                    continue
                entry_issues = _answer_entry_issue_counts(entry, failed_answer_min_attempts)
                key = _question_key_from_entry(entry, q_slug)
                drop_due_issue = _issues_match(entry_issues, drop_codes)
                drop_due_benchmark = bool(benchmark_index and _key_in_index(benchmark_index, key))
                if drop_due_issue or drop_due_benchmark:
                    dropped += 1
                    counts["answer_entries"] += 1
                    _add_key_to_index(answer_drop_keys[(q_slug, answer_slug)], key)
                    continue
                kept.append(entry)
            if dropped:
                counts["answer_files"] += 1
                _save_json_list(file_path, kept)

    # Critiques
    critiques_dir = Path("critiques")
    if critiques_dir.exists():
        for mode_dir in critiques_dir.iterdir():
            if not mode_dir.is_dir():
                continue
            mode = mode_dir.name
            for q_dir in mode_dir.iterdir():
                if not q_dir.is_dir():
                    continue
                q_slug = q_dir.name
                for file_path in q_dir.glob("*.json"):
                    pair = _parse_pair_file(file_path)
                    if not pair:
                        continue
                    critic_slug, answer_slug = pair
                    data = _load_json_list(file_path)
                    if data is None:
                        continue
                    kept = []
                    dropped = 0
                    answer_index = answer_drop_keys.get((q_slug, answer_slug))
                    for entry in data:
                        if not entry:
                            continue
                        entry_issues = _critique_entry_issue_counts(entry)
                        key = _question_key_from_entry(entry, q_slug)
                        drop_due_issue = _issues_match(entry_issues, drop_codes)
                        drop_due_answer = bool(answer_index and _key_in_index(answer_index, key))
                        if drop_due_issue or drop_due_answer:
                            dropped += 1
                            counts["critique_entries"] += 1
                            _add_key_to_index(
                                critique_drop_keys[(mode, q_slug, critic_slug, answer_slug)],
                                key,
                            )
                            _add_key_to_index(
                                critique_drop_by_answer[(q_slug, answer_slug)],
                                key,
                            )
                            continue
                        kept.append(entry)
                    if dropped:
                        counts["critique_files"] += 1
                        _save_json_list(file_path, kept)

    # Debates: ill-posed
    illposed_dir = Path("debates/illposed")
    if illposed_dir.exists():
        for q_dir in illposed_dir.iterdir():
            if not q_dir.is_dir():
                continue
            q_slug = q_dir.name
            for file_path in q_dir.glob("*.json"):
                data = _load_json_list(file_path)
                if data is None:
                    continue
                kept = []
                dropped = 0
                answer_slug = file_path.stem
                answer_index = answer_drop_keys.get((q_slug, answer_slug))
                for entry in data:
                    if not entry:
                        continue
                    entry_issues = _debate_entry_issue_counts(entry, incomplete_debate_min_rounds)
                    key = _question_key_from_entry(entry, q_slug)
                    drop_due_issue = _issues_match(entry_issues, drop_codes)
                    drop_due_answer = bool(answer_index and _key_in_index(answer_index, key))
                    if drop_due_issue or drop_due_answer:
                        dropped += 1
                        counts["debate_entries"] += 1
                        _add_key_to_index(illposed_debate_drop_keys[(q_slug, answer_slug)], key)
                        continue
                    kept.append(entry)
                if dropped:
                    counts["debate_files"] += 1
                    _save_json_list(file_path, kept)

    # Debates: critiques
    critique_debates_dir = Path("debates/critiques")
    if critique_debates_dir.exists():
        for mode_dir in critique_debates_dir.iterdir():
            if not mode_dir.is_dir():
                continue
            mode = mode_dir.name
            for q_dir in mode_dir.iterdir():
                if not q_dir.is_dir():
                    continue
                q_slug = q_dir.name
                for file_path in q_dir.glob("*.json"):
                    pair = _parse_pair_file(file_path)
                    if not pair:
                        continue
                    critic_slug, answer_slug = pair
                    data = _load_json_list(file_path)
                    if data is None:
                        continue
                    kept = []
                    dropped = 0
                    answer_index = answer_drop_keys.get((q_slug, answer_slug))
                    critique_index = critique_drop_keys.get((mode, q_slug, critic_slug, answer_slug))
                    for entry in data:
                        if not entry:
                            continue
                        entry_issues = _debate_entry_issue_counts(entry, incomplete_debate_min_rounds)
                        key = _question_key_from_entry(entry, q_slug)
                        drop_due_issue = _issues_match(entry_issues, drop_codes)
                        drop_due_answer = bool(answer_index and _key_in_index(answer_index, key))
                        drop_due_critique = bool(critique_index and _key_in_index(critique_index, key))
                        if drop_due_issue or drop_due_answer or drop_due_critique:
                            dropped += 1
                            counts["debate_entries"] += 1
                            _add_key_to_index(
                                critique_debate_drop_keys[(mode, q_slug, critic_slug, answer_slug)],
                                key,
                            )
                            _add_key_to_index(
                                critique_debate_drop_by_answer[(q_slug, answer_slug)],
                                key,
                            )
                            continue
                        kept.append(entry)
                    if dropped:
                        counts["debate_files"] += 1
                        _save_json_list(file_path, kept)

    # Automated evaluations
    automated_dir = Path("automated_evaluations")
    if automated_dir.exists():
        for file_path in automated_dir.glob("*.json"):
            payload = _load_json_payload(file_path)
            if not payload:
                continue
            decisions = payload.get("decisions", [])
            if not isinstance(decisions, list):
                continue
            kept = []
            dropped = 0
            for decision in decisions:
                if not isinstance(decision, dict):
                    kept.append(decision)
                    continue
                entry_issues = _evaluation_entry_issue_counts(decision)
                if _issues_match(entry_issues, drop_codes):
                    dropped += 1
                    counts["evaluation_entries"] += 1
                    continue
                q_slug = decision.get("question_model")
                key = question_key(q_slug, decision.get("run_id"), decision.get("outer_attempt"))
                answer_slug = decision.get("answer_model")
                critic_slug = decision.get("critic_model")
                mode = decision.get("mode")
                eval_type = decision.get("type")

                drop_due_answer = False
                if q_slug and answer_slug:
                    answer_index = answer_drop_keys.get((q_slug, answer_slug))
                    if answer_index and _key_in_index(answer_index, key):
                        drop_due_answer = True

                drop_due_illposed = False
                if eval_type == "illposed" and q_slug and answer_slug:
                    illposed_index = illposed_debate_drop_keys.get((q_slug, answer_slug))
                    if illposed_index and _key_in_index(illposed_index, key):
                        drop_due_illposed = True

                drop_due_critique = False
                if eval_type in {"critique", "critique_debate"} and q_slug and answer_slug:
                    if mode and critic_slug:
                        critique_index = critique_drop_keys.get((mode, q_slug, critic_slug, answer_slug))
                        debate_index = critique_debate_drop_keys.get(
                            (mode, q_slug, critic_slug, answer_slug)
                        )
                        if (
                            (critique_index and _key_in_index(critique_index, key))
                            or (debate_index and _key_in_index(debate_index, key))
                        ):
                            drop_due_critique = True
                    else:
                        critique_index = critique_drop_by_answer.get((q_slug, answer_slug))
                        debate_index = critique_debate_drop_by_answer.get((q_slug, answer_slug))
                        if (
                            (critique_index and _key_in_index(critique_index, key))
                            or (debate_index and _key_in_index(debate_index, key))
                        ):
                            drop_due_critique = True

                if drop_due_answer or drop_due_illposed or drop_due_critique:
                    dropped += 1
                    counts["evaluation_entries"] += 1
                    continue

                kept.append(decision)
            if dropped:
                counts["evaluation_files"] += 1
                payload["decisions"] = kept
                file_path.write_text(json.dumps(payload, indent=2))

    return counts


def main():
    parser = argparse.ArgumentParser(description="Check output files for common issues.")
    parser.add_argument(
        "--failed-answer-min-attempts",
        type=int,
        default=5,
        help="Only flag failed answers with fewer than this many attempts.",
    )
    parser.add_argument(
        "--incomplete-debate-min-rounds",
        type=int,
        default=5,
        help="Only flag incomplete debates with fewer rounds than this.",
    )
    parser.add_argument(
        "--drop",
        nargs="+",
        default=None,
        help="Drop entries with these issue codes (comma-separated ok).",
    )
    args = parser.parse_args()
    drop_codes = _parse_drop_codes(args.drop)
    benchmarks_dir = Path("benchmarks")
    answer_index = _build_answer_reference_index(Path("answers"), benchmarks_dir)
    benchmark_index = _build_benchmark_reference_index(benchmarks_dir)
    critique_index = _build_critique_reference_index(Path("critiques"))
    illposed_debate_index, critique_debate_index = _build_debate_reference_indexes(Path("debates"))

    print("=" * 80)
    print("OUTPUT FILE ISSUE REPORT")
    print("=" * 80)
    print()

    total_files_with_issues = 0
    total_issues = 0

    # Check benchmarks
    print("BENCHMARKS")
    print("-" * 80)
    benchmark_issues = scan_directory(Path("benchmarks"), check_benchmark_issues)
    if benchmark_issues:
        for file_path, issues in sorted(benchmark_issues.items()):
            issue_count = sum(issues.values())
            total_files_with_issues += 1
            total_issues += issue_count
            print(f"\n{file_path}:")
            for issue_type, count in sorted(issues.items()):
                print(f"  - {issue_type}: {count}")
    else:
        print("No issues found.")
    print()

    # Check answers
    print("ANSWERS")
    print("-" * 80)
    answer_issues = scan_directory(
        Path("answers"),
        lambda path: check_answer_issues(path, args.failed_answer_min_attempts, benchmark_index),
    )
    if answer_issues:
        for file_path, issues in sorted(answer_issues.items()):
            issue_count = sum(issues.values())
            total_files_with_issues += 1
            total_issues += issue_count
            print(f"\n{file_path}:")
            for issue_type, count in sorted(issues.items()):
                print(f"  - {issue_type}: {count}")
    else:
        print("No issues found.")
    print()

    # Check critiques
    print("CRITIQUES")
    print("-" * 80)
    critique_issues = scan_directory(
        Path("critiques"),
        lambda path: check_critique_issues(path, answer_index),
    )
    if critique_issues:
        for file_path, issues in sorted(critique_issues.items()):
            issue_count = sum(issues.values())
            total_files_with_issues += 1
            total_issues += issue_count
            print(f"\n{file_path}:")
            for issue_type, count in sorted(issues.items()):
                print(f"  - {issue_type}: {count}")
    else:
        print("No issues found.")
    print()

    # Check debates
    print("DEBATES")
    print("-" * 80)
    debate_issues = scan_directory(
        Path("debates"),
        lambda path: check_debate_issues(path, args.incomplete_debate_min_rounds, critique_index),
    )
    if debate_issues:
        for file_path, issues in sorted(debate_issues.items()):
            issue_count = sum(issues.values())
            total_files_with_issues += 1
            total_issues += issue_count
            print(f"\n{file_path}:")
            for issue_type, count in sorted(issues.items()):
                print(f"  - {issue_type}: {count}")
    else:
        print("No issues found.")
    print()

    # Check automated evaluations
    print("AUTOMATED EVALUATIONS")
    print("-" * 80)
    eval_issues = scan_directory(
        Path("automated_evaluations"),
        lambda path: check_evaluation_issues(path, illposed_debate_index, critique_debate_index),
    )
    if eval_issues:
        for file_path, issues in sorted(eval_issues.items()):
            issue_count = sum(issues.values())
            total_files_with_issues += 1
            total_issues += issue_count
            print(f"\n{file_path}:")
            for issue_type, count in sorted(issues.items()):
                print(f"  - {issue_type}: {count}")
    else:
        print("No issues found.")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files with issues: {total_files_with_issues}")
    print(f"Total issues found: {total_issues}")
    print()

    # Issue type breakdown
    all_issue_types = defaultdict(int)
    for issues_dict in [benchmark_issues, answer_issues, critique_issues, debate_issues, eval_issues]:
        for file_issues in issues_dict.values():
            for issue_type, count in file_issues.items():
                all_issue_types[issue_type] += count

    if all_issue_types:
        print("Issue breakdown by type:")
        for issue_type, count in sorted(all_issue_types.items(), key=lambda x: -x[1]):
            print(f"  - {issue_type}: {count}")

    print()

    if drop_codes:
        print("=" * 80)
        print("DROP REPORT")
        print("=" * 80)
        print(f"Drop codes: {', '.join(sorted(drop_codes))}")
        print()
        drop_counts = _drop_entries(
            drop_codes,
            args.failed_answer_min_attempts,
            args.incomplete_debate_min_rounds,
        )
        print(
            "Benchmarks dropped: "
            f"{drop_counts['benchmark_entries']} (files touched: {drop_counts['benchmark_files']})"
        )
        print(
            "Answers dropped: "
            f"{drop_counts['answer_entries']} (files touched: {drop_counts['answer_files']})"
        )
        print(
            "Critiques dropped: "
            f"{drop_counts['critique_entries']} (files touched: {drop_counts['critique_files']})"
        )
        print(
            "Debates dropped: "
            f"{drop_counts['debate_entries']} (files touched: {drop_counts['debate_files']})"
        )
        print(
            "Evaluations dropped: "
            f"{drop_counts['evaluation_entries']} (files touched: {drop_counts['evaluation_files']})"
        )
        print()


if __name__ == "__main__":
    main()
