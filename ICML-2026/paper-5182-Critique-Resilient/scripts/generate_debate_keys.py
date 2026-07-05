#!/usr/bin/env python3
"""
Generate exhaustive debate keys.

For every run ID observed anywhere in the data, and for every pair of
(question_model, answer_model), we emit:
- an ill-posed claim key
- an evaluator critique key
- a contradictor critique key

Many of these keys will not correspond to real data; downstream tools
(labeller_app_ordered) will prune them.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_config import load_registry
from data_models import (
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    load_debate_entries,
    load_evaluation_entries,
)

EntryKey = Tuple[Optional[str], Optional[str], Optional[int]]


def _task_id(prefix: str, run_id: Optional[str]) -> str:
    if run_id:
        return f"{prefix}/{run_id}"
    digest = hashlib.sha1(prefix.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}/{digest}"


def _key(run_id: Optional[str], topic_slug: Optional[str], idx: Optional[int]) -> EntryKey:
    return (str(run_id) if run_id is not None else None, topic_slug, idx)


def collect_run_ids(args) -> Set[str]:
    run_ids: Set[str] = set()

    # Benchmarks
    for bench_path in Path(args.benchmarks_dir).glob("*.json"):
        entries = load_benchmark_entries(bench_path)
        for entry in entries:
            if entry and entry.run_id:
                run_ids.add(str(entry.run_id))

    # Answers
    answers_dir = Path(args.answers_dir)
    for ans_path in answers_dir.glob("*/*.json"):
        entries = load_answer_entries(ans_path)
        for entry in entries:
            if entry and entry.run_id:
                run_ids.add(str(entry.run_id))

    # Critiques
    for crit_path in Path(args.critiques_dir).glob("*/*/*.json"):
        entries = load_critique_entries(crit_path)
        for entry in entries:
            if entry and entry.run_id:
                run_ids.add(str(entry.run_id))

    # Debates
    for debate_path in Path(args.debates_dir).glob("**/*.json"):
        entries = load_debate_entries(debate_path)
        for entry in entries:
            if entry and entry.run_id:
                run_ids.add(str(entry.run_id))

    # Automated evaluations
    for eval_path in Path(args.automated_evals_dir).glob("*.json"):
        payload = load_evaluation_entries(eval_path)
        for decision in payload.decisions:
            if decision and decision.run_id:
                run_ids.add(str(decision.run_id))

    return run_ids


def generate_keys(run_ids: Set[str], q_slugs: List[str], a_slugs: List[str], critic_slugs: List[str]) -> Dict[str, List[Dict]]:
    illposed: Dict[EntryKey, Dict] = {}
    critiques: Dict[EntryKey, Dict] = {}

    for run_id in sorted(run_ids):
        for q_slug in q_slugs:
            for idx_a, a_slug in enumerate(a_slugs):
                topic_slug = f"{q_slug}__{a_slug}"
                base_key = _key(run_id, topic_slug, idx_a)

                # ill-posed
                illposed_entry = {
                    "id": _task_id(f"illposed/{q_slug}/{a_slug}", run_id),
                    "type": "illposed",
                    "mode": None,
                    "question_model": q_slug,
                    "answer_model": a_slug,
                    "critic_model": None,
                    "run_id": run_id,
                    "topic_slug": topic_slug,
                    "question": None,
                }
                if base_key not in illposed:
                    illposed[base_key] = illposed_entry

                # critiques (two modes)
                for mode, offset in (("evaluator", 0), ("contradictor", 1)):
                    for critic_slug in critic_slugs:
                        crit_key = _key(run_id, f"{topic_slug}__{mode}__{critic_slug}", idx_a + offset)
                        crit_entry = {
                            "id": _task_id(f"critique/{mode}/{q_slug}/{critic_slug}__{a_slug}", run_id),
                            "type": "critique",
                            "mode": mode,
                            "question_model": q_slug,
                            "answer_model": a_slug,
                            "critic_model": critic_slug,
                            "run_id": run_id,
                            "topic_slug": f"{topic_slug}__{mode}",
                            "question": None,
                        }
                        if crit_key not in critiques:
                            critiques[crit_key] = crit_entry

    illposed_list = [
        {**payload, "key": [k[0], k[1], k[2]]}
        for k, payload in illposed.items()
    ]
    critiques_list = [
        {**payload, "key": [k[0], k[1], k[2]]}
        for k, payload in critiques.items()
    ]
    return {
        "illposed": sorted(illposed_list, key=lambda x: x["id"]),
        "critiques": sorted(critiques_list, key=lambda x: x["id"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate exhaustive debate keys.")
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--benchmarks-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--debates-dir", type=Path, default=Path("debates"))
    parser.add_argument("--automated-evals-dir", type=Path, default=Path("automated_evaluations"))
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--output", type=Path, default=Path("debate_keys.json"))
    args = parser.parse_args()

    registry = load_registry(str(args.config))
    model_slugs = [spec.slug for spec in registry.models.values()]

    run_ids = collect_run_ids(args)
    payload = generate_keys(run_ids, model_slugs, model_slugs, model_slugs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(payload['illposed']) + len(payload['critiques'])} keys to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
