from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_yesno(text: str | None) -> str | None:
    match = re.search(r"\b(yes|no)\b", str(text or "").lower())
    return match.group(1) if match else None


def extract_letter(text: str | None) -> str | None:
    match = re.search(r"\b([A-D])\b", str(text or ""), flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\b([A-D])(?=[\s\.\)]|$)", str(text or ""), flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def normalize_text(text: str | None) -> str:
    return " ".join(str(text or "").strip().split())


def eval_yesno(gt_file: str, pred_file: str) -> dict[str, Any]:
    gt = read_jsonl(gt_file)
    pred = {item.get("question_id"): item for item in read_jsonl(pred_file)}
    tp = tn = fp = fn = missing = unknown = 0
    for item in gt:
        label = extract_yesno(item.get("label"))
        answer = extract_yesno(pred.get(item.get("question_id"), {}).get("text"))
        if answer is None:
            missing += 1
            continue
        if label == "yes" and answer == "yes":
            tp += 1
        elif label == "no" and answer == "no":
            tn += 1
        elif label == "no" and answer == "yes":
            fp += 1
        elif label == "yes" and answer == "no":
            fn += 1
        else:
            unknown += 1
    total = len(gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / total if total else 0.0,
        "missing": missing,
        "unknown": unknown,
        "total": total,
    }


def eval_mcqa(gt_file: str, pred_file: str, strict_linewise: bool = False) -> dict[str, Any]:
    gt = read_jsonl(gt_file)
    pred = read_jsonl(pred_file)
    total = correct = missing = unparsable = 0
    if strict_linewise:
        pairs = list(zip(gt, pred))
    else:
        pred_by_key: dict[tuple[Any, str], dict[str, Any]] = {}
        for item in pred:
            key = (item.get("question_id"), normalize_text(item.get("prompt") or item.get("question") or ""))
            pred_by_key[key] = item
            pred_by_key.setdefault((item.get("question_id"), ""), item)
        pairs = []
        for item in gt:
            key = (item.get("question_id"), normalize_text(item.get("prompt") or item.get("question") or item.get("text") or ""))
            pred_item = pred_by_key.get(key) or pred_by_key.get((item.get("question_id"), ""))
            if pred_item is None:
                missing += 1
                continue
            pairs.append((item, pred_item))

    for gt_item, pred_item in pairs:
        gt_label = extract_letter(gt_item.get("label") or gt_item.get("answer"))
        pred_label = extract_letter(pred_item.get("text"))
        if pred_label is None:
            unparsable += 1
            continue
        if gt_label is None:
            missing += 1
            continue
        total += 1
        correct += int(gt_label == pred_label)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "missing": missing,
        "unparsable": unparsable,
    }


def eval_video_mme(results_file: str, duration: str = "all") -> dict[str, Any]:
    with Path(results_file).expanduser().open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    durations = None if duration == "all" else set(duration.split(","))
    correct = answered = 0
    by_task: dict[str, dict[str, int]] = {}
    for video in results:
        if durations is not None and video.get("duration") not in durations:
            continue
        for question in video.get("questions", []):
            task = question.get("task_type", "Unknown")
            by_task.setdefault(task, {"correct": 0, "answered": 0})
            pred = extract_letter(question.get("response") or question.get("text"))
            gt = extract_letter(question.get("answer"))
            if not pred or not gt:
                continue
            answered += 1
            by_task[task]["answered"] += 1
            is_correct = int(pred == gt)
            correct += is_correct
            by_task[task]["correct"] += is_correct
    return {
        "accuracy": correct / answered if answered else 0.0,
        "correct": correct,
        "answered": answered,
        "by_task": {
            task: values["correct"] / values["answered"] if values["answered"] else 0.0
            for task, values in sorted(by_task.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MACD outputs.")
    parser.add_argument("--task", choices=["yesno", "mcqa", "video_mme"], required=True)
    parser.add_argument("--gt", help="Ground-truth JSONL for yesno/mcqa.")
    parser.add_argument("--pred", help="Prediction JSONL for yesno/mcqa.")
    parser.add_argument("--results-file", help="Nested Video-MME result JSON.")
    parser.add_argument("--duration", default="all", help="Video-MME duration filter.")
    parser.add_argument("--strict-linewise", action="store_true", help="Evaluate MCQA by line order.")
    args = parser.parse_args()

    if args.task == "yesno":
        if not args.gt or not args.pred:
            raise ValueError("--gt and --pred are required for yesno evaluation")
        result = eval_yesno(args.gt, args.pred)
    elif args.task == "mcqa":
        if not args.gt or not args.pred:
            raise ValueError("--gt and --pred are required for mcqa evaluation")
        result = eval_mcqa(args.gt, args.pred, strict_linewise=args.strict_linewise)
    else:
        if not args.results_file:
            raise ValueError("--results-file is required for video_mme evaluation")
        result = eval_video_mme(args.results_file, duration=args.duration)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
