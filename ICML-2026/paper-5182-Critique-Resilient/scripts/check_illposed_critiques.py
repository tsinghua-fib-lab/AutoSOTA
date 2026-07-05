#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import sys
sys.path.append('.')

from constants import STATUS_ILL_POSED
from data_models import load_answer_entries, load_benchmark_entries
from utils import benchmark_answers_from_entries, entry_key



def load_json_list(path: Path) -> List[Optional[dict]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def save_json_list(path: Path, entries: List[Optional[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2))


def parse_pair_file(path: Path) -> Optional[Tuple[str, str]]:
    parts = path.stem.split("__")
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def build_answer_map(
    answers_dir: Path,
    benchmark_dir: Path,
    q_slug: str,
    answer_slug: str,
) -> Dict[Tuple[Optional[str], Optional[str], Optional[str]], dict]:
    answer_path = answers_dir / q_slug / f"{answer_slug}.json"
    if answer_path.exists():
        entries = load_answer_entries(answer_path)
        return {entry_key(e.run_id, e.topic_slug, e.question): e for e in entries if e}
    if answer_slug == q_slug:
        benchmark_entries = load_benchmark_entries(benchmark_dir / f"{q_slug}.json")
        entries = benchmark_answers_from_entries(q_slug, benchmark_entries)
        return {entry_key(e.run_id, e.topic_slug, e.question): e for e in entries if e}
    return {}


def build_debate_map(entries: List[Optional[dict]]) -> Dict[Tuple[Optional[str], Optional[str], Optional[str]], dict]:
    mapped: Dict[Tuple[Optional[str], Optional[str], Optional[str]], dict] = {}
    for entry in entries:
        if not entry:
            continue
        key = entry_key(entry.get("run_id"), entry.get("topic_slug"), entry.get("question"))
        if key and key not in mapped:
            mapped[key] = entry
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find critiques produced for ill-posed answers; optionally drop critiques, debates, and evaluations."
    )
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--benchmarks-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--debates-dir", type=Path, default=Path("debates"))
    parser.add_argument("--automated-dir", type=Path, default=Path("automated_evaluations"))
    parser.add_argument("--drop", action="store_true", help="Drop affected entries and dependent records.")
    args = parser.parse_args()

    affected_keys: Dict[Tuple[str, str, str, str], Set[Tuple[Optional[str], Optional[str], Optional[str]]]] = {}
    total_affected = 0
    files_scanned = 0

    for mode_dir in args.critiques_dir.glob("*"):
        mode = mode_dir.name
        for q_dir in mode_dir.glob("*"):
            q_slug = q_dir.name
            for crit_file in q_dir.glob("*.json"):
                pair = parse_pair_file(crit_file)
                if not pair:
                    continue
                critic_slug, answer_slug = pair
                critiques = load_json_list(crit_file)
                answer_map = build_answer_map(args.answers_dir, args.benchmarks_dir, q_slug, answer_slug)
                if not answer_map:
                    continue
                files_scanned += 1
                local_affected: Set[Tuple[Optional[str], Optional[str], Optional[str]]] = set()
                for entry in critiques:
                    if not entry:
                        continue
                    key = entry_key(entry.get("run_id"), entry.get("topic_slug"), entry.get("question"))
                    if not key:
                        continue
                    answer_entry = answer_map.get(key)
                    if not answer_entry:
                        continue
                    if answer_entry.status == STATUS_ILL_POSED:
                        local_affected.add(key)
                if local_affected:
                    affected_keys[(mode, q_slug, critic_slug, answer_slug)] = local_affected
                    total_affected += len(local_affected)

    print(f"Critique files scanned: {files_scanned}")
    print(f"Affected critique entries: {total_affected}")

    if not args.drop or not affected_keys:
        return

    dropped_critiques = 0
    dropped_debates = 0
    dropped_evals = 0

    # Drop critiques and debates
    for (mode, q_slug, critic_slug, answer_slug), keys in affected_keys.items():
        crit_path = args.critiques_dir / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
        crit_entries = load_json_list(crit_path)
        kept_critiques = [
            entry
            for entry in crit_entries
            if entry
            and entry_key(entry.get("run_id"), entry.get("topic_slug"), entry.get("question")) not in keys
        ]
        dropped_critiques += len(crit_entries) - len(kept_critiques)
        save_json_list(crit_path, kept_critiques)

        debate_path = args.debates_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
        debate_entries = load_json_list(debate_path)
        kept_debates = [
            entry
            for entry in debate_entries
            if entry
            and entry_key(entry.get("run_id"), entry.get("topic_slug"), entry.get("question")) not in keys
        ]
        dropped_debates += len(debate_entries) - len(kept_debates)
        save_json_list(debate_path, kept_debates)

    # Drop automated evaluations tied to the dropped critiques
    dropped_ids: Set[str] = set()
    for (mode, q_slug, critic_slug, answer_slug), keys in affected_keys.items():
        prefix = f"critique/{mode}/{q_slug}/{critic_slug}__{answer_slug}"
        for run_id, _, _ in keys:
            if run_id:
                dropped_ids.add(f"{prefix}/{run_id}")

    for auto_eval_path in args.automated_dir.glob("*.json"):
        data = json.loads(auto_eval_path.read_text())
        decisions = data.get("decisions", [])
        if not isinstance(decisions, list):
            continue
        before = len(decisions)
        kept = [
            decision
            for decision in decisions
            if decision.get("type") != "critique" or decision.get("id") not in dropped_ids
        ]
        if len(kept) != before:
            auto_eval_path.write_text(json.dumps({"decisions": kept}, indent=2))
            dropped_evals += before - len(kept)

    print(f"Dropped critiques: {dropped_critiques}")
    print(f"Dropped critique debates: {dropped_debates}")
    print(f"Dropped automated critique evaluations: {dropped_evals}")


if __name__ == "__main__":
    main()
