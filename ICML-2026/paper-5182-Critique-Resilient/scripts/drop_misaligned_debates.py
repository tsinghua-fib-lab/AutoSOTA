#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


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


def build_answer_index(entries: List[Optional[dict]]) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
    index: Dict[str, int] = {}
    duplicates: Dict[str, List[int]] = {}
    for idx, entry in enumerate(entries):
        if not entry:
            continue
        run_id = entry.get("run_id")
        if run_id is None:
            continue
        run_id = str(run_id)
        if run_id in index:
            duplicates.setdefault(run_id, [index[run_id]]).append(idx)
        else:
            index[run_id] = idx
    return index, duplicates


def parse_pair_file(path: Path) -> Optional[Tuple[str, str]]:
    parts = path.stem.split("__")
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def drop_misaligned_in_file(
    debate_path: Path,
    answer_index: Dict[str, int],
    mode: str,
    q_slug: str,
    critic_slug: str,
    answer_slug: str,
) -> Tuple[List[Optional[dict]], Set[str], int]:
    debates = load_json_list(debate_path)
    kept: List[Optional[dict]] = []
    dropped_ids: Set[str] = set()
    dropped = 0
    for idx, entry in enumerate(debates):
        if not entry:
            continue
        run_id = entry.get("run_id")
        if run_id is None:
            dropped += 1
            continue
        run_id_str = str(run_id)
        answer_idx = answer_index.get(run_id_str)
        if answer_idx is None or answer_idx != idx:
            dropped += 1
            dropped_ids.add(f"critique/{mode}/{q_slug}/{critic_slug}__{answer_slug}/{run_id_str}")
            continue
        kept.append(entry)
    return kept, dropped_ids, dropped


def filter_automated_evaluations(auto_eval_path: Path, dropped_ids: Set[str], dry_run: bool) -> int:
    if not auto_eval_path.exists():
        return 0
    data = json.loads(auto_eval_path.read_text())
    decisions = data.get("decisions", [])
    if not isinstance(decisions, list):
        return 0
    before = len(decisions)
    kept = [
        decision
        for decision in decisions
        if decision.get("type") != "critique" or decision.get("id") not in dropped_ids
    ]
    removed = before - len(kept)
    if removed and not dry_run:
        auto_eval_path.write_text(json.dumps({"decisions": kept}, indent=2))
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drop critique debates affected by index/run_id misalignment and remove their automated evaluations."
    )
    parser.add_argument("--debates-dir", type=Path, default=Path("debates/critiques"))
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--automated-dir", type=Path, default=Path("automated_evaluations"))
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files.")
    args = parser.parse_args()

    total_dropped = 0
    dropped_eval_ids: Set[str] = set()
    files_touched = 0

    for mode_dir in args.debates_dir.iterdir():
        if not mode_dir.is_dir():
            continue
        mode = mode_dir.name
        for q_dir in mode_dir.iterdir():
            if not q_dir.is_dir():
                continue
            q_slug = q_dir.name
            for debate_path in q_dir.glob("*.json"):
                pair = parse_pair_file(debate_path)
                if not pair:
                    continue
                critic_slug, answer_slug = pair
                answer_path = args.answers_dir / q_slug / f"{answer_slug}.json"
                if not answer_path.exists():
                    continue
                answers = load_json_list(answer_path)
                answer_index, duplicates = build_answer_index(answers)
                if duplicates:
                    raise ValueError(f"Duplicate run_id entries in {answer_path}: {duplicates}")
                kept, dropped_ids, dropped = drop_misaligned_in_file(
                    debate_path,
                    answer_index,
                    mode,
                    q_slug,
                    critic_slug,
                    answer_slug,
                )
                if dropped:
                    total_dropped += dropped
                    dropped_eval_ids.update(dropped_ids)
                    files_touched += 1
                    if not args.dry_run:
                        save_json_list(debate_path, kept)
    removed_evals = 0
    if dropped_eval_ids:
        for auto_eval_path in args.automated_dir.glob("*.json"):
            removed_evals += filter_automated_evaluations(auto_eval_path, dropped_eval_ids, args.dry_run)

    print(f"Debate files touched: {files_touched}")
    print(f"Debate entries dropped: {total_dropped}")
    print(f"Automated critique evaluations removed: {removed_evals}")
    if args.dry_run:
        print("Dry run only: no files were modified.")


if __name__ == "__main__":
    main()
