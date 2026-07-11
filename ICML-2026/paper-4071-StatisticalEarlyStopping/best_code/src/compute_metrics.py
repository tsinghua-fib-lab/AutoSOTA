import argparse
from src.detect.detectors import LLMJudgeAbstentionDetector
from src.utils import count_tokens, read_jsonl
from data import get_dataset_generator
from typing import List, Tuple, Optional, Any, Dict, Iterable
import os
import json

def write_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> None:
    """Write an iterable of dicts to JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def chunked(iterable, n):
    """Yield successive n-sized chunks from an iterable."""
    for i in range(0, len(iterable), n):
        yield i, iterable[i:i + n]

def process_file(
    results_path: str,
    model_name: str,
    data_name: str,
    batch_size: int = 64,
) -> None:
    """
    Load results JSONL, augment each row, and write out a new JSONL.
    Uses batched LLM-judge abstention detection for content_ill_posed.
    """
    input_path = f"{results_path}/{model_name}/{data_name}_results.jsonl"
    rows = read_jsonl(input_path)
    if rows is None:
        raise FileNotFoundError(f"Could not read JSONL from: {input_path}")

    # Evaluator for accuracy
    data_gen = get_dataset_generator(data_name)
    evaluate_fn = data_gen.evaluate  # expects (model_answer, ref_answer) -> bool

    # Detectors (instantiate once)
    llm_judge_detector = LLMJudgeAbstentionDetector()

    # Precompute cheap things row-by-row (lengths, accuracy)
    # and collect inputs for batch LLM judge
    prepped_rows: List[dict] = []
    judge_inputs: List[Tuple[str, str]] = []  # (question_ill_posed, model_answer_ill_posed)

    for row in rows:
        reasoning_well_posed = row.get("reasoning_well_posed", "") or ""
        reasoning_ill_posed = row.get("reasoning_ill_posed", "") or ""
        content = row.get("response_well_posed", "") or ""
        ref_answer = row.get("ref_answer")

        # lengths
        thinking_lengths_well_posed = count_tokens(reasoning_well_posed, model_name)
        thinking_lengths_ill_posed = count_tokens(reasoning_ill_posed, model_name)

        # accuracy (content only)
        correct = bool(ref_answer is not None and evaluate_fn(content, ref_answer))

        alt_text = row.get("response_ill_posed", "") or ""

        # stash partials + judge inputs
        prepped_rows.append({
            "row": row,
            "thinking_lengths_well_posed": thinking_lengths_well_posed,
            "thinking_lengths_ill_posed": thinking_lengths_ill_posed,
            "correct_well_posed": correct
        })
        judge_inputs.append((row.get("question_ill_posed", "") or "", alt_text))

    # Batched LLM-judge calls
    # batch_detect returns List[Tuple[Optional[bool], Optional[str]]]
    judge_results: List[Tuple[Optional[bool], Optional[str]]] = [None] * len(prepped_rows)

    for start_idx, batch in chunked(judge_inputs, batch_size):
        questions = [q for q, _ in batch]
        answers = [a for _, a in batch]
        batch_out = llm_judge_detector.batch_detect(questions, answers)
        # write back into aligned positions
        for offset, (flag, raw) in enumerate(batch_out):
            judge_results[start_idx + offset] = (flag, raw)

    # Merge everything and write
    metric_rows = []
    for i, info in enumerate(prepped_rows):
        row = dict()
        row["thinking_lengths_well_posed"] = info["thinking_lengths_well_posed"]
        row["thinking_lengths_ill_posed"] = info["thinking_lengths_ill_posed"]
        row["correct_well_posed"] = info["correct_well_posed"]

        llm_flag, _ = judge_results[i]
        row["llm_abstention_ill_posed"] = llm_flag

        metric_rows.append(row)

    output_path = f"{results_path}/{model_name}/{data_name}_metrics.jsonl"
    write_jsonl(metric_rows, output_path)

def get_args_parser():
    p = argparse.ArgumentParser("Augment results JSONL with row-level metrics", add_help=True)
    p.add_argument("--results_path", type=str, default="results/", help="Path to input results JSONL (e.g., results/)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="Model name for token counting")
    p.add_argument("--data_name", type=str, default="gpqa", help="Dataset name for evaluator (gpqa, mmlu, hle, umwp, mc, mip)")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for LLM judge abstention")
    return p

def main():
    args = get_args_parser().parse_args()
    process_file(
        results_path=args.results_path,
        model_name=args.model_name,
        data_name=args.data_name,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()