#!/usr/bin/env python3
"""Evaluate budget sweep / concise JSON files: add per-entry accuracy and cost.

For each entry in the JSON file, writes back:
  - "accuracy": float (0.0 or 1.0 for most metrics, 0~1 for meteor/code)
  - "cost": float ($ based on token_usage and model_cost.json)

Supports checkpoint: skips entries that already have accuracy != None.

Usage:
    # Eval all files in a model directory
    python eval_sweep.py --json-dir /orange/.../budget_sweep/30b \
        --model-cost-name "qwen/qwen3-30b-a3b-instruct-2507" [--include-code]

    # Eval a single file
    python eval_sweep.py --json-file /path/to/file.json \
        --model-cost-name "qwen/qwen3-30b-a3b-instruct-2507"
"""

import os
import sys
import json
import argparse
import glob
import time
import multiprocessing

sys.path.insert(0, "/home/ah872032.ucf/jiaqi/RouterArena/llm_evaluation")

from metrics import (
    mcq_accuracy,
    math_metric,
    exact_match,
    meteor_score,
    chess_accuracy,
    superglue_exact_match,
    superglue_clozetest,
)

METRIC_FUNCS = {
    "mcq_accuracy": mcq_accuracy,
    "math_metric": math_metric,
    "exact_match": exact_match,
    "meteor_score": meteor_score,
    "chess_accuracy": chess_accuracy,
    "superglue_exact_match": superglue_exact_match,
    "superglue_clozetest": superglue_clozetest,
}

MODEL_COST_FILE = "/home/ah872032.ucf/jiaqi/RouterArena/model_cost/model_cost.json"


def load_ground_truth():
    """Load ground truth from gpt-4o-mini.jsonl (verified identical to dataset/routerarena Answer)."""
    gt_file = "/home/ah872032.ucf/jiaqi/RouterArena/cached_results/gpt-4o-mini.jsonl"
    with open(gt_file) as f:
        gt_entries = [json.loads(l) for l in f]
    gt_map = {e["global_index"]: e for e in gt_entries}
    print(f"Ground truth: {len(gt_map)} entries")
    return gt_map


def load_livecodebench():
    from datasets import load_from_disk
    ds = load_from_disk("/home/ah872032.ucf/jiaqi/RouterArena/dataset/livecodebench")
    lcb_map = {item["global_idx"]: item for item in ds}
    print(f"LiveCodeBench: {len(lcb_map)} problems loaded")
    return lcb_map


def load_cost_config(model_cost_name):
    """Load pricing from model_cost.json for the given model name."""
    with open(MODEL_COST_FILE) as f:
        cost_config = json.load(f)
    if model_cost_name not in cost_config:
        print(f"ERROR: '{model_cost_name}' not found in {MODEL_COST_FILE}")
        print(f"  Available keys: {sorted(cost_config.keys())}")
        sys.exit(1)
    info = cost_config[model_cost_name]
    return {
        "input": info["input_token_price_per_million"],
        "output": info["output_token_price_per_million"],
    }


def calculate_cost(token_usage, pricing):
    """Calculate inference cost from token_usage and pricing dict."""
    if not token_usage or not pricing:
        return 0.0
    input_tokens = token_usage.get("input_tokens", 0)
    output_tokens = token_usage.get("output_tokens", 0)
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def _code_accuracy_worker(gen_answer, ground_truth, result_queue):
    """Run code_accuracy in a child process to isolate memory leaks.

    The multiprocessing.Manager leak in livecodebench_util.unsafe_lcb_runTests
    accumulates ~1 Manager process per call. Running in a subprocess ensures all
    leaked memory is freed when the subprocess exits.
    """
    try:
        import sys
        sys.path.insert(0, "/home/ah872032.ucf/jiaqi/RouterArena/llm_evaluation")
        from metrics import code_accuracy
        result = code_accuracy(
            [gen_answer], [ground_truth],
            matched_data=[{"prediction": gen_answer, "ground_truth": ground_truth, "options": []}]
        )
        if isinstance(result, tuple) and len(result) == 2:
            result_queue.put(float(result[0]))
        else:
            result_queue.put(float(result))
    except Exception as e:
        print(f"    Scorer error (code_accuracy/subprocess): {e}")
        result_queue.put(0.0)


def score_code_accuracy_isolated(gen_answer, ground_truth, timeout=6000):
    """Score code_accuracy in an isolated subprocess to prevent OOM from Manager leak.

    Timeout is a safety net only (e.g. deadlock). RouterArena handles its own
    per-test-case timeout internally: (6+1)*N_tests+5 seconds, so normal
    execution finishes well within 600s even for problems with ~80 test cases.
    """
    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue()
    p = ctx.Process(target=_code_accuracy_worker, args=(gen_answer, ground_truth, result_queue))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        p.join()
        result_queue.close()
        result_queue.join_thread()
        print(f"    code_accuracy subprocess timed out ({timeout}s)")
        return 0.0
    try:
        if not result_queue.empty():
            return result_queue.get()
        return 0.0
    finally:
        result_queue.close()
        result_queue.join_thread()


def score_single_entry(gen_answer, ground_truth, metric_name, scorer):
    """Score a single entry. Returns float accuracy."""
    try:
        result = scorer(
            [gen_answer], [ground_truth],
            matched_data=[{"prediction": gen_answer, "ground_truth": ground_truth, "options": []}]
        )
        if isinstance(result, tuple) and len(result) == 2:
            return float(result[0])
        return float(result)
    except Exception as e:
        print(f"    Scorer error ({metric_name}): {e}")
        return 0.0


def eval_json_file(json_path, gt_map, pricing, lcb_map, include_code):
    """Evaluate a single JSON file, writing accuracy+cost back to each entry."""
    with open(json_path) as f:
        data = json.load(f)

    total = len(data)
    already_done = sum(1 for e in data if e.get("accuracy") is not None)
    if already_done == total:
        print(f"  {os.path.basename(json_path)}: all {total} entries already done, skipping")
        return

    print(f"  {os.path.basename(json_path)}: {total} entries ({already_done} already done)")

    evaluated = 0
    skipped = 0
    code_count = 0
    t0 = time.time()

    for i, entry in enumerate(data):
        # Skip optimality entries (not needed for budget sweep eval)
        if entry.get("for_optimality"):
            skipped += 1
            continue

        # Skip if already evaluated
        if entry.get("accuracy") is not None:
            continue

        gi = entry.get("global index") or entry.get("global_index")
        if not gi or gi not in gt_map:
            skipped += 1
            continue

        gen_result = entry.get("generated_result")
        if not gen_result or not isinstance(gen_result, dict) or not gen_result.get("success"):
            entry["accuracy"] = 0.0
            entry["cost"] = 0.0
            evaluated += 1
            continue

        gt = gt_map[gi]
        metric_name = gt["evaluation_result"]["metric"]
        ground_truth = gt["evaluation_result"]["ground_truth"]

        # Handle code_accuracy
        if metric_name == "code_accuracy":
            if not include_code:
                skipped += 1
                continue
            if lcb_map and gi in lcb_map:
                ground_truth = lcb_map[gi]
            else:
                skipped += 1
                continue
            code_count += 1

        scorer = METRIC_FUNCS.get(metric_name)
        if not scorer:
            skipped += 1
            continue

        gen_answer = gen_result.get("generated_answer", "")
        if metric_name == "code_accuracy":
            ct0 = time.time()
            acc = score_code_accuracy_isolated(gen_answer, ground_truth)
            fname = os.path.basename(json_path)
            print(f"    [CODE {code_count}/385 | {fname}] {gi}: acc={acc:.2f} ({time.time()-ct0:.1f}s)", flush=True)
        else:
            acc = score_single_entry(gen_answer, ground_truth, metric_name, scorer)
        token_usage = gen_result.get("token_usage", {})
        cost = calculate_cost(token_usage, pricing)

        entry["accuracy"] = acc
        entry["cost"] = cost
        evaluated += 1

        # Progress every 2000 entries
        if evaluated % 2000 == 0:
            elapsed = time.time() - t0
            print(f"    {evaluated} evaluated, {elapsed:.0f}s elapsed", flush=True)

        # Periodic save every 50 code entries (code_accuracy is slow, avoid losing progress)
        if metric_name == "code_accuracy" and code_count % 50 == 0:
            with open(json_path, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"    [CHECKPOINT] saved after {code_count} code entries", flush=True)

    # Save back
    with open(json_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    final_done = sum(1 for e in data if e.get("accuracy") is not None)
    print(f"    Done: {evaluated} new ({code_count} code), {skipped} skipped, "
          f"{final_done}/{total} total, {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", help="Directory with budget_*.json and concise.json")
    parser.add_argument("--json-file", help="Single JSON file to evaluate")
    parser.add_argument("--model-cost-name", required=True,
                        help="Model key in model_cost.json (e.g. 'qwen/qwen3-30b-a3b-instruct-2507')")
    parser.add_argument("--include-code", action="store_true",
                        help="Include LiveCodeBench code_accuracy evaluation")
    args = parser.parse_args()

    if not args.json_dir and not args.json_file:
        parser.error("Must provide --json-dir or --json-file")

    # Load shared resources
    gt_map = load_ground_truth()
    pricing = load_cost_config(args.model_cost_name)

    if args.include_code:
        from metrics import code_accuracy
        METRIC_FUNCS["code_accuracy"] = code_accuracy
        lcb_map = load_livecodebench()
    else:
        lcb_map = None

    # Collect files to evaluate
    if args.json_file:
        files = [args.json_file]
    else:
        files = sorted(glob.glob(os.path.join(args.json_dir, "budget_*.json")))
        concise = os.path.join(args.json_dir, "concise.json")
        if os.path.exists(concise):
            files.append(concise)

    print(f"\nEvaluating {len(files)} files, cost='{args.model_cost_name}'")
    print(f"  Pricing: Input=${pricing['input']}/M, Output=${pricing['output']}/M")
    print()

    t_total = time.time()
    for fpath in files:
        eval_json_file(fpath, gt_map, pricing, lcb_map, args.include_code)

    print(f"\nAll done in {time.time() - t_total:.1f}s")

    # Skip GC on exit to avoid OOM from accumulated multiprocessing.Queue objects
    os._exit(0)


if __name__ == "__main__":
    main()
