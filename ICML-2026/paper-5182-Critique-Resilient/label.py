import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from constants import STATUS_ILL_POSED, STATUS_SUCCEEDED
from data_models import (
    HumanEvaluation,
    HumanEvaluationFile,
    load_answer_entries,
    load_critique_entries,
    load_debate_entries,
    load_human_evaluation_entries,
    save_human_evaluation_entries,
)
from utils import (
    answer_key,
    answer_key_from_entry,
    critique_key,
    format_key,
    human_evaluation_key_from_entry,
)


def list_illposed(answers_dir: Path, debates_dir: Path) -> List[Dict]:
    items = []
    for q_dir in answers_dir.glob("*"):
        q_slug = q_dir.name
        for answer_file in q_dir.glob("*.json"):
            a_slug = answer_file.stem
            records = load_answer_entries(answer_file)
            debate_file = debates_dir / "illposed" / q_slug / f"{a_slug}.json"
            debates = load_debate_entries(debate_file)
            debate_map: Dict = {}
            for debate in debates:
                if not debate:
                    continue
                key = answer_key(q_slug, a_slug, debate.run_id, debate.outer_attempt)
                if key and key not in debate_map:
                    debate_map[key] = debate
            for idx, rec in enumerate(records):
                if not rec or rec.status != STATUS_ILL_POSED:
                    continue
                key = answer_key_from_entry(rec)
                debate_history = debate_map.get(key)
                run_id = rec.run_id or (debate_history.run_id if debate_history else None)
                outer_attempt = rec.outer_attempt or (debate_history.outer_attempt if debate_history else None)
                task_key = critique_key(q_slug, a_slug, None, None, run_id, outer_attempt)
                if not task_key:
                    continue
                    items.append(
                        {
                            "key": format_key(task_key),
                            "task_key": task_key,
                            "type": "illposed",
                            "run_id": run_id,
                            "outer_attempt": outer_attempt,
                            "question_model": q_slug,
                            "answer_model": a_slug,
                            "critic_model": None,
                            "mode": None,
                        "question": rec.question,
                        "claim": rec.ill_posed_claim,
                        "debate": debate_history.model_dump(exclude_none=True) if debate_history else {},
                    }
                )
    return items


def list_critiques(critiques_dir: Path, debates_dir: Path) -> List[Dict]:
    items = []
    for mode_dir in critiques_dir.glob("*"):
        mode = mode_dir.name
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
                debate_map: Dict = {}
                for debate in debates:
                    if not debate:
                        continue
                    key = answer_key(q_slug, answer_slug, debate.run_id, debate.outer_attempt)
                    if key and key not in debate_map:
                        debate_map[key] = debate
                for idx, crit in enumerate(critiques):
                    if not crit or crit.status != STATUS_SUCCEEDED:
                        continue
                    attempts = crit.attempts or []
                    last_attempt = attempts[-1] if attempts else None
                    key = answer_key(
                        crit.question_author,
                        crit.answer_author,
                        crit.run_id,
                        crit.outer_attempt,
                    )
                    debate_history = debate_map.get(key)
                    run_id = crit.run_id or (debate_history.run_id if debate_history else None)
                    outer_attempt = crit.outer_attempt or (debate_history.outer_attempt if debate_history else None)
                    task_key = critique_key(q_slug, answer_slug, critic_slug, mode, run_id, outer_attempt)
                    if not task_key:
                        continue
                    items.append(
                        {
                            "key": format_key(task_key),
                            "task_key": task_key,
                            "type": "critique",
                            "run_id": run_id,
                            "outer_attempt": outer_attempt,
                            "question_model": q_slug,
                            "answer_model": answer_slug,
                            "critic_model": critic_slug,
                            "mode": mode,
                            "question": crit.question,
                            "critique": last_attempt.raw_critique if last_attempt else None,
                            "verdict": last_attempt.verdict if last_attempt else None,
                            "notes": last_attempt.notes if last_attempt else None,
                            "critic": critic_slug,
                            "answer_author": answer_slug,
                            "debate": debate_history.model_dump(exclude_none=True) if debate_history else {},
                        }
                    )
    return items


def build_item_map(answers_dir: Path, critiques_dir: Path, debates_dir: Path) -> Dict[str, Dict]:
    items = list_illposed(answers_dir, debates_dir) + list_critiques(critiques_dir, debates_dir)
    return {item["key"]: item for item in items}


def record_evaluation(
    username: str,
    task: Dict,
    verdict: str,
    comment: str,
    eval_type: str,
    store_dir: Path,
):
    eval_path = store_dir / f"{username}.json"
    payload = load_human_evaluation_entries(eval_path)
    decisions: Dict = {}
    for decision in payload.decisions:
        key = human_evaluation_key_from_entry(decision)
        if key:
            decisions[key] = decision
    new_decision = HumanEvaluation(
        run_id=task.get("run_id"),
        outer_attempt=task.get("outer_attempt"),
        type=eval_type,
        mode=task.get("mode"),
        question_model=task.get("question_model"),
        answer_model=task.get("answer_model"),
        critic_model=task.get("critic_model"),
        verdict=verdict,
        comment=comment,
    )
    decision_key = human_evaluation_key_from_entry(new_decision)
    if decision_key:
        decisions[decision_key] = new_decision
    save_human_evaluation_entries(eval_path, HumanEvaluationFile(decisions=list(decisions.values())))


def show_item(item: Dict):
    print(json.dumps(item, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Human evaluation CLI for ill-posed claims and critiques.")
    parser.add_argument("--username", required=True, help="Evaluator name.")
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--debates-dir", type=Path, default=Path("debates"))
    parser.add_argument("--evaluations-dir", type=Path, default=Path("evaluations"))
    parser.add_argument("--list", choices=["ill-posed", "critiques"], help="List pending items.")
    parser.add_argument("--show", help="Show a specific key.")
    parser.add_argument("--verdict", help="Record verdict for a key.")
    parser.add_argument("--comment", default="", help="Optional comment for verdict.")
    args = parser.parse_args()

    if args.list:
        if args.list == "ill-posed":
            items = list_illposed(args.answers_dir, args.debates_dir)
        else:
            items = list_critiques(args.critiques_dir, args.debates_dir)
        for item in items:
            print(item["key"])
        return

    if args.show:
        item_map = build_item_map(args.answers_dir, args.critiques_dir, args.debates_dir)
        item = item_map.get(args.show)
        if not item:
            raise ValueError(f"Unknown key: {args.show}")
        show_item(item)
        return

    if args.verdict:
        if ":" not in args.verdict:
            raise ValueError("Provide verdict as key:decision (e.g., run_id/question_model/answer_model/...:correct)")
        key_part, decision = args.verdict.split(":", 1)
        item_map = build_item_map(args.answers_dir, args.critiques_dir, args.debates_dir)
        item = item_map.get(key_part)
        if not item:
            raise ValueError(f"Unknown key: {key_part}")
        record_evaluation(args.username, item, decision, args.comment, item["type"], args.evaluations_dir)
        print(f"Recorded {decision} for {key_part}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
