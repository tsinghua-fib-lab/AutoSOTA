#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


Key = Tuple[Optional[str], Optional[str], Optional[str]]
STATUS_CORRECT = "correct"
STATUS_WRONG = "wrong"
STATUS_UNKNOWN = "unknown"


def entry_key(run_id: Optional[str], topic_slug: Optional[str], question: Optional[str]) -> Optional[Key]:
    if run_id is None and topic_slug is None and not question:
        return None
    return (str(run_id) if run_id is not None else None, topic_slug, question)


def final_benchmark_question(entry: Dict[str, Any]) -> Optional[str]:
    rounds = entry.get("generation_rounds") or []
    if not rounds:
        return None
    last_round = rounds[-1] or {}
    refinements = last_round.get("refinement_rounds") or []
    if not refinements:
        return None
    last_ref = refinements[-1] or {}
    return last_ref.get("question")


def benchmark_key(entry: Optional[Dict[str, Any]]) -> Optional[Key]:
    if not entry:
        return None
    question = final_benchmark_question(entry)
    return entry_key(entry.get("run_id"), entry.get("topic_slug"), question)


def answer_key(entry: Optional[Dict[str, Any]]) -> Optional[Key]:
    if not entry:
        return None
    return entry_key(entry.get("run_id"), entry.get("topic_slug"), entry.get("question"))


def critique_key(entry: Optional[Dict[str, Any]]) -> Optional[Key]:
    if not entry:
        return None
    return entry_key(entry.get("run_id"), entry.get("topic_slug"), entry.get("question"))


def debate_key(entry: Optional[Dict[str, Any]]) -> Optional[Key]:
    if not entry:
        return None
    return entry_key(entry.get("run_id"), entry.get("topic_slug"), entry.get("question"))


def load_json_list(path: Path) -> List[Any]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        return []
    return data


def load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        return {}
    return data


def index_keys(entries: List[Any], key_fn) -> Tuple[List[Optional[Key]], Dict[Key, List[int]]]:
    keys: List[Optional[Key]] = []
    key_to_indices: Dict[Key, List[int]] = defaultdict(list)
    for idx, entry in enumerate(entries):
        key = key_fn(entry) if entry else None
        keys.append(key)
        if key:
            key_to_indices[key].append(idx)
    return keys, key_to_indices


def is_unique(key: Optional[Key], key_to_indices: Dict[Key, List[int]]) -> bool:
    if not key:
        return False
    return len(key_to_indices.get(key, [])) == 1


def classify_answer(
    idx: int,
    entry: Dict[str, Any],
    q_slug: str,
    a_slug: str,
    bench_keys: List[Optional[Key]],
    bench_map: Dict[Key, List[int]],
) -> Tuple[str, str]:
    if entry.get("question_model") and entry.get("question_model") != q_slug:
        return "wrong", "question_model_mismatch"
    if entry.get("answer_model") and entry.get("answer_model") != a_slug:
        return "wrong", "answer_model_mismatch"

    key = answer_key(entry)
    bench_key = bench_keys[idx] if idx < len(bench_keys) else None
    if not key:
        return STATUS_UNKNOWN, "missing_answer_key"
    if not bench_key:
        return STATUS_UNKNOWN, "missing_benchmark_at_index"
    if not is_unique(bench_key, bench_map):
        return STATUS_UNKNOWN, "duplicate_benchmark_key"
    if key == bench_key:
        return STATUS_CORRECT, "index_match"
    if key in bench_map:
        if is_unique(key, bench_map):
            return STATUS_WRONG, "benchmark_index_mismatch"
        return STATUS_UNKNOWN, "duplicate_benchmark_key"
    return STATUS_UNKNOWN, "benchmark_key_missing"


def classify_critique(
    idx: int,
    entry: Dict[str, Any],
    critic_slug: str,
    answer_slug: str,
    q_slug: str,
    bench_keys: List[Optional[Key]],
    bench_map: Dict[Key, List[int]],
    answer_keys: List[Optional[Key]],
    answer_map: Dict[Key, List[int]],
    answer_statuses: List[Optional[str]],
) -> Tuple[str, str]:
    if entry.get("critic") and entry.get("critic") != critic_slug:
        return "wrong", "critic_slug_mismatch"
    if entry.get("answer_author") and entry.get("answer_author") != answer_slug:
        return "wrong", "answer_author_mismatch"
    if entry.get("question_author") and entry.get("question_author") != q_slug:
        return "wrong", "question_author_mismatch"

    key = critique_key(entry)
    bench_key = bench_keys[idx] if idx < len(bench_keys) else None
    answer_key_at_idx = answer_keys[idx] if idx < len(answer_keys) else None
    answer_status = answer_statuses[idx] if idx < len(answer_statuses) else None

    if not key:
        return STATUS_UNKNOWN, "missing_critique_key"
    if answer_status != STATUS_CORRECT:
        return STATUS_UNKNOWN, "upstream_answer_not_correct"
    if not bench_key:
        return STATUS_UNKNOWN, "missing_benchmark_at_index"
    if not is_unique(bench_key, bench_map):
        return STATUS_UNKNOWN, "duplicate_benchmark_key"

    if answer_key_at_idx and key != answer_key_at_idx:
        if key in answer_map and is_unique(key, answer_map):
            return STATUS_WRONG, "answer_index_mismatch"
        if key not in answer_map:
            return STATUS_UNKNOWN, "answer_key_missing"
        return STATUS_UNKNOWN, "duplicate_answer_key"

    if key == bench_key and (not answer_key_at_idx or key == answer_key_at_idx):
        if is_unique(key, bench_map) and is_unique(key, answer_map):
            return STATUS_CORRECT, "index_match"
        return STATUS_UNKNOWN, "duplicate_key"

    if key in bench_map and is_unique(key, bench_map):
        return STATUS_WRONG, "benchmark_index_mismatch"
    if key not in bench_map:
        return STATUS_UNKNOWN, "benchmark_key_missing"
    return STATUS_UNKNOWN, "duplicate_benchmark_key"


def classify_illposed_debate(
    idx: int,
    entry: Dict[str, Any],
    q_slug: str,
    a_slug: str,
    answer_keys: List[Optional[Key]],
    answer_map: Dict[Key, List[int]],
    answer_statuses: List[Optional[str]],
) -> Tuple[str, str]:
    if entry.get("claimant") and entry.get("claimant") != a_slug:
        return "wrong", "claimant_mismatch"
    if entry.get("alice_model") and entry.get("alice_model") != a_slug:
        return "wrong", "alice_model_mismatch"
    if entry.get("bob_model") and entry.get("bob_model") != q_slug:
        return "wrong", "bob_model_mismatch"

    key = debate_key(entry)
    answer_key_at_idx = answer_keys[idx] if idx < len(answer_keys) else None
    answer_status = answer_statuses[idx] if idx < len(answer_statuses) else None
    if not key:
        return STATUS_UNKNOWN, "missing_debate_key"
    if answer_status != STATUS_CORRECT:
        return STATUS_UNKNOWN, "upstream_answer_not_correct"
    if not answer_key_at_idx:
        return STATUS_UNKNOWN, "missing_answer_at_index"
    if key == answer_key_at_idx:
        if is_unique(key, answer_map):
            return STATUS_CORRECT, "index_match"
        return STATUS_UNKNOWN, "duplicate_answer_key"
    if key in answer_map:
        if is_unique(key, answer_map):
            return STATUS_WRONG, "answer_index_mismatch"
        return STATUS_UNKNOWN, "duplicate_answer_key"
    return STATUS_UNKNOWN, "answer_key_missing"


def classify_critique_debate(
    idx: int,
    entry: Dict[str, Any],
    critic_slug: str,
    answer_slug: str,
    critique_keys: List[Optional[Key]],
    critique_map: Dict[Key, List[int]],
    answer_keys: List[Optional[Key]],
    answer_map: Dict[Key, List[int]],
    critique_statuses: List[Optional[str]],
    answer_statuses: List[Optional[str]],
) -> Tuple[str, str]:
    if entry.get("critic") and entry.get("critic") != critic_slug:
        return "wrong", "critic_slug_mismatch"
    if entry.get("answer_author") and entry.get("answer_author") != answer_slug:
        return "wrong", "answer_author_mismatch"

    key = debate_key(entry)
    critique_key_at_idx = critique_keys[idx] if idx < len(critique_keys) else None
    answer_key_at_idx = answer_keys[idx] if idx < len(answer_keys) else None
    critique_status = critique_statuses[idx] if idx < len(critique_statuses) else None
    answer_status = answer_statuses[idx] if idx < len(answer_statuses) else None

    if not key:
        return STATUS_UNKNOWN, "missing_debate_key"
    if critique_status != STATUS_CORRECT:
        return STATUS_UNKNOWN, "upstream_critique_not_correct"
    if answer_status != STATUS_CORRECT:
        return STATUS_UNKNOWN, "upstream_answer_not_correct"
    if not critique_key_at_idx or not answer_key_at_idx:
        return STATUS_UNKNOWN, "missing_upstream_at_index"
    if key != critique_key_at_idx:
        if key in critique_map and is_unique(key, critique_map):
            return STATUS_WRONG, "critique_index_mismatch"
        if key not in critique_map:
            return STATUS_UNKNOWN, "critique_key_missing"
        return STATUS_UNKNOWN, "duplicate_critique_key"
    if key != answer_key_at_idx:
        if key in answer_map and is_unique(key, answer_map):
            return STATUS_WRONG, "answer_index_mismatch"
        if key not in answer_map:
            return STATUS_UNKNOWN, "answer_key_missing"
        return STATUS_UNKNOWN, "duplicate_answer_key"
    if key == critique_key_at_idx == answer_key_at_idx:
        if is_unique(key, critique_map) and is_unique(key, answer_map):
            return STATUS_CORRECT, "index_match"
        return STATUS_UNKNOWN, "duplicate_key"
    return STATUS_UNKNOWN, "missing_index_key"


def parse_index_from_id(raw_id: Optional[str]) -> Optional[int]:
    if not raw_id:
        return None
    parts = raw_id.split("/")
    if not parts:
        return None
    last = parts[-1]
    if last.isdigit():
        return int(last)
    return None


def classify_evaluation(
    entry: Dict[str, Any],
    debates_dir: Path,
    debate_cache: Dict[str, List[Any]],
    debate_status_cache: Dict[str, List[Optional[str]]],
) -> Tuple[str, str]:
    eval_type = entry.get("type")
    if eval_type not in {"illposed", "critique", "critique_debate"}:
        return STATUS_UNKNOWN, "unknown_eval_type"

    q_slug = entry.get("question_model")
    a_slug = entry.get("answer_model")
    critic_slug = entry.get("critic_model")
    mode = entry.get("mode")
    idx = parse_index_from_id(entry.get("id"))
    if idx is None:
        return STATUS_UNKNOWN, "unparseable_id"

    if eval_type == "illposed":
        if not q_slug or not a_slug:
            return STATUS_UNKNOWN, "missing_models"
        debate_path = debates_dir / "illposed" / q_slug / f"{a_slug}.json"
    else:
        if not q_slug or not a_slug or not critic_slug or not mode:
            return STATUS_UNKNOWN, "missing_models"
        debate_path = debates_dir / "critiques" / mode / q_slug / f"{critic_slug}__{a_slug}.json"

    cache_key = str(debate_path)
    if cache_key not in debate_cache:
        debate_cache[cache_key] = load_json_list(debate_path)
    debates = debate_cache[cache_key]
    statuses = debate_status_cache.get(cache_key, [])

    if idx >= len(debates):
        return STATUS_UNKNOWN, "missing_debate_at_index"
    debate_entry = debates[idx]
    if not debate_entry:
        return STATUS_UNKNOWN, "missing_debate_at_index"
    if idx >= len(statuses) or statuses[idx] != STATUS_CORRECT:
        return STATUS_UNKNOWN, "upstream_debate_not_correct"

    run_id = entry.get("run_id")
    topic_slug = entry.get("topic_slug")
    if run_id is None or topic_slug is None:
        return STATUS_UNKNOWN, "missing_run_id_or_topic"
    if debate_entry.get("run_id") == run_id and debate_entry.get("topic_slug") == topic_slug:
        return STATUS_CORRECT, "index_match"
    return STATUS_WRONG, "debate_index_mismatch"


def build_benchmark_answers(benchmarks: List[Any]) -> List[Dict[str, Any]]:
    answers: List[Dict[str, Any]] = []
    for entry in benchmarks:
        if not entry:
            answers.append({})
            continue
        question = final_benchmark_question(entry)
        answers.append(
            {
                "question": question,
                "run_id": entry.get("run_id"),
                "topic_slug": entry.get("topic_slug"),
                "status": entry.get("status"),
            }
        )
    return answers


def record_result(
    results: Dict[str, Any],
    path: Path,
    idx: int,
    status: str,
    reason: str,
    key: Optional[Key],
):
    path_str = str(path)
    payload = results.setdefault(path_str, {"counts": defaultdict(int), "items": []})
    payload["counts"][status] += 1
    payload["counts"][f"reason:{reason}"] += 1
    payload["items"].append(
        {
            "index": idx,
            "status": status,
            "reason": reason,
            "key": list(key) if key else None,
        }
    )


def summarize_counts(results: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for path, payload in results.items():
        counts = payload.get("counts", {})
        summary[path] = {
            "correct": counts.get("correct", 0),
            "wrong": counts.get("wrong", 0),
            "unknown": counts.get("unknown", 0),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Check alignment across benchmark/answer/critique/debate files.")
    parser.add_argument("--benchmarks-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--debates-dir", type=Path, default=Path("debates"))
    parser.add_argument("--evaluations-dir", type=Path, default=Path("automated_evaluations"))
    parser.add_argument("--report", type=Path, help="Optional JSON report output path.")
    parser.add_argument("--max-items", type=int, default=5, help="Max items per file to print.")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete entries classified as wrong or unknown from the source files.",
    )
    args = parser.parse_args()

    results: Dict[str, Any] = {}
    bench_cache: Dict[str, List[Any]] = {}
    answer_status_cache: Dict[str, List[Optional[str]]] = {}
    answer_keys_cache: Dict[str, List[Optional[Key]]] = {}
    answer_keymap_cache: Dict[str, Dict[Key, List[int]]] = {}
    critique_status_cache: Dict[str, List[Optional[str]]] = {}
    critique_keys_cache: Dict[str, List[Optional[Key]]] = {}
    critique_keymap_cache: Dict[str, Dict[Key, List[int]]] = {}
    debate_status_cache: Dict[str, List[Optional[str]]] = {}

    # Benchmarks: source entries (no upstream check)
    for bench_path in sorted(args.benchmarks_dir.glob("*.json")):
        entries = load_json_list(bench_path)
        for idx, entry in enumerate(entries):
            if not entry:
                continue
            key = benchmark_key(entry)
            record_result(results, bench_path, idx, STATUS_CORRECT, "assumed_correct", key)

    # Answers vs benchmarks
    for q_dir in sorted(args.answers_dir.glob("*")):
        q_slug = q_dir.name
        bench_path = args.benchmarks_dir / f"{q_slug}.json"
        if str(bench_path) not in bench_cache:
            bench_cache[str(bench_path)] = load_json_list(bench_path)
        bench_entries = bench_cache[str(bench_path)]
        bench_keys, bench_map = index_keys(bench_entries, benchmark_key)
        for answer_path in sorted(q_dir.glob("*.json")):
            a_slug = answer_path.stem
            answers = load_json_list(answer_path)
            answer_keys, answer_map = index_keys(answers, answer_key)
            statuses: List[Optional[str]] = []
            for idx, entry in enumerate(answers):
                if not entry:
                    statuses.append(None)
                    continue
                status, reason = classify_answer(idx, entry, q_slug, a_slug, bench_keys, bench_map)
                statuses.append(status)
                record_result(results, answer_path, idx, status, reason, answer_key(entry))
            answer_status_cache[str(answer_path)] = statuses
            answer_keys_cache[str(answer_path)] = answer_keys
            answer_keymap_cache[str(answer_path)] = answer_map

    # Critiques
    for mode_dir in sorted(args.critiques_dir.glob("*")):
        mode = mode_dir.name
        for q_dir in sorted(mode_dir.glob("*")):
            q_slug = q_dir.name
            bench_path = args.benchmarks_dir / f"{q_slug}.json"
            if str(bench_path) not in bench_cache:
                bench_cache[str(bench_path)] = load_json_list(bench_path)
            bench_entries = bench_cache[str(bench_path)]
            bench_keys, bench_map = index_keys(bench_entries, benchmark_key)
            for crit_path in sorted(q_dir.glob("*.json")):
                parts = crit_path.stem.split("__")
                if len(parts) != 2:
                    continue
                critic_slug, answer_slug = parts
                critiques = load_json_list(crit_path)
                answer_path = args.answers_dir / q_slug / f"{answer_slug}.json"
                if answer_path.exists():
                    answers = load_json_list(answer_path)
                    answer_keys = answer_keys_cache.get(str(answer_path))
                    answer_map = answer_keymap_cache.get(str(answer_path))
                    answer_statuses = answer_status_cache.get(str(answer_path))
                    if answer_keys is None or answer_map is None:
                        answer_keys, answer_map = index_keys(answers, answer_key)
                    if answer_statuses is None:
                        answer_statuses = []
                elif answer_slug == q_slug:
                    answers = build_benchmark_answers(bench_entries)
                    answer_keys, answer_map = index_keys(answers, answer_key)
                    answer_statuses = [STATUS_CORRECT if entry else None for entry in answers]
                else:
                    answers = []
                    answer_keys, answer_map = [], {}
                    answer_statuses = []
                statuses: List[Optional[str]] = []
                for idx, entry in enumerate(critiques):
                    if not entry:
                        statuses.append(None)
                        continue
                    status, reason = classify_critique(
                        idx,
                        entry,
                        critic_slug,
                        answer_slug,
                        q_slug,
                        bench_keys,
                        bench_map,
                        answer_keys,
                        answer_map,
                        answer_statuses,
                    )
                    statuses.append(status)
                    record_result(results, crit_path, idx, status, reason, critique_key(entry))
                critique_status_cache[str(crit_path)] = statuses
                critique_keys_cache[str(crit_path)], critique_keymap_cache[str(crit_path)] = index_keys(critiques, critique_key)

    # Debates: illposed
    illposed_dir = args.debates_dir / "illposed"
    for q_dir in sorted(illposed_dir.glob("*")):
        q_slug = q_dir.name
        bench_path = args.benchmarks_dir / f"{q_slug}.json"
        if str(bench_path) not in bench_cache:
            bench_cache[str(bench_path)] = load_json_list(bench_path)
        bench_entries = bench_cache[str(bench_path)]
        for debate_path in sorted(q_dir.glob("*.json")):
            a_slug = debate_path.stem
            debates = load_json_list(debate_path)
            answer_path = args.answers_dir / q_slug / f"{a_slug}.json"
            if answer_path.exists():
                answers = load_json_list(answer_path)
                answer_keys = answer_keys_cache.get(str(answer_path))
                answer_map = answer_keymap_cache.get(str(answer_path))
                answer_statuses = answer_status_cache.get(str(answer_path)) or []
                if answer_keys is None or answer_map is None:
                    answer_keys, answer_map = index_keys(answers, answer_key)
            elif a_slug == q_slug:
                answers = build_benchmark_answers(bench_entries)
                answer_keys, answer_map = index_keys(answers, answer_key)
                answer_statuses = [STATUS_CORRECT if entry else None for entry in answers]
            else:
                answers = []
                answer_keys, answer_map = [], {}
                answer_statuses = []
            statuses: List[Optional[str]] = []
            for idx, entry in enumerate(debates):
                if not entry:
                    statuses.append(None)
                    continue
                status, reason = classify_illposed_debate(
                    idx,
                    entry,
                    q_slug,
                    a_slug,
                    answer_keys,
                    answer_map,
                    answer_statuses,
                )
                statuses.append(status)
                record_result(results, debate_path, idx, status, reason, debate_key(entry))
            debate_status_cache[str(debate_path)] = statuses

    # Debates: critiques
    critiques_dir = args.debates_dir / "critiques"
    for mode_dir in sorted(critiques_dir.glob("*")):
        mode = mode_dir.name
        for q_dir in sorted(mode_dir.glob("*")):
            q_slug = q_dir.name
            bench_path = args.benchmarks_dir / f"{q_slug}.json"
            if str(bench_path) not in bench_cache:
                bench_cache[str(bench_path)] = load_json_list(bench_path)
            bench_entries = bench_cache[str(bench_path)]
            for debate_path in sorted(q_dir.glob("*.json")):
                parts = debate_path.stem.split("__")
                if len(parts) != 2:
                    continue
                critic_slug, answer_slug = parts
                debates = load_json_list(debate_path)
                critique_path = args.critiques_dir / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
                critiques = load_json_list(critique_path)
                critique_keys = critique_keys_cache.get(str(critique_path))
                critique_map = critique_keymap_cache.get(str(critique_path))
                critique_statuses = critique_status_cache.get(str(critique_path)) or []
                if critique_keys is None or critique_map is None:
                    critique_keys, critique_map = index_keys(critiques, critique_key)
                answer_path = args.answers_dir / q_slug / f"{answer_slug}.json"
                if answer_path.exists():
                    answers = load_json_list(answer_path)
                    answer_keys = answer_keys_cache.get(str(answer_path))
                    answer_map = answer_keymap_cache.get(str(answer_path))
                    answer_statuses = answer_status_cache.get(str(answer_path)) or []
                    if answer_keys is None or answer_map is None:
                        answer_keys, answer_map = index_keys(answers, answer_key)
                elif answer_slug == q_slug:
                    answers = build_benchmark_answers(bench_entries)
                    answer_keys, answer_map = index_keys(answers, answer_key)
                    answer_statuses = [STATUS_CORRECT if entry else None for entry in answers]
                else:
                    answers = []
                    answer_keys, answer_map = [], {}
                    answer_statuses = []
                statuses: List[Optional[str]] = []
                for idx, entry in enumerate(debates):
                    if not entry:
                        statuses.append(None)
                        continue
                    status, reason = classify_critique_debate(
                        idx,
                        entry,
                        critic_slug,
                        answer_slug,
                        critique_keys,
                        critique_map,
                        answer_keys,
                        answer_map,
                        critique_statuses,
                        answer_statuses,
                    )
                    statuses.append(status)
                    record_result(results, debate_path, idx, status, reason, debate_key(entry))
                debate_status_cache[str(debate_path)] = statuses

    # Automated evaluations
    eval_cache: Dict[str, List[Any]] = {}
    for eval_path in sorted(args.evaluations_dir.glob("*.json")):
        payload = load_json_dict(eval_path)
        decisions = payload.get("decisions") or []
        if not isinstance(decisions, list):
            continue
        for idx, entry in enumerate(decisions):
            if not entry:
                continue
            status, reason = classify_evaluation(entry, args.debates_dir, eval_cache, debate_status_cache)
            record_result(results, eval_path, idx, status, reason, None)

    summary = summarize_counts(results)
    total_correct = 0
    total_wrong = 0
    total_unknown = 0
    for path, counts in summary.items():
        print(f"{path}: correct={counts['correct']} wrong={counts['wrong']} unknown={counts['unknown']}")
        total_correct += counts["correct"]
        total_wrong += counts["wrong"]
        total_unknown += counts["unknown"]
        items = results[path]["items"]
        shown = 0
        for item in items:
            if item["status"] == "correct":
                continue
            if shown >= args.max_items:
                break
            print(f"  - idx={item['index']} status={item['status']} reason={item['reason']}")
            shown += 1

    total_entries = total_correct + total_wrong + total_unknown
    print(f"TOTAL wrong+unknown={total_wrong + total_unknown} total_entries={total_entries}")

    if args.prune:
        for path, payload in results.items():
            to_remove = {
                item["index"]
                for item in payload.get("items", [])
                if item.get("status") in {STATUS_WRONG, STATUS_UNKNOWN}
            }
            if not to_remove:
                continue
            src = Path(path)
            try:
                data = json.loads(src.read_text())
            except Exception as exc:
                print(f"PRUNE SKIP {path}: {exc}")
                continue
            removed = 0
            if isinstance(data, list):
                data = [entry for idx, entry in enumerate(data) if idx not in to_remove]
                removed = len(to_remove)
            elif isinstance(data, dict) and isinstance(data.get("decisions"), list):
                decisions = data.get("decisions") or []
                data["decisions"] = [
                    entry for idx, entry in enumerate(decisions) if idx not in to_remove
                ]
                removed = len(to_remove)
            else:
                print(f"PRUNE SKIP {path}: unsupported JSON structure")
                continue
            src.write_text(json.dumps(data, indent=2))
            print(f"PRUNED {path}: removed={removed}")

    if args.report:
        report = {
            "summary": summary,
            "files": results,
            "notes": [
                "benchmarks are assumed correct",
                "unknown = missing metadata, missing upstream entry, duplicate keys, or upstream marked wrong/unknown",
                "wrong = index-based alignment disagrees with metadata-derived key while upstream is correct",
                "correct = metadata key matches index-based upstream entries and is unique",
            ],
        }
        args.report.write_text(json.dumps(report, indent=2))
        print(f"Wrote report to {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
