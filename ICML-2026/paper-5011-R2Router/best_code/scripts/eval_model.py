#!/usr/bin/env python3
"""Evaluate model accuracy using RouterArena ground truth and metrics.

Usage:
    .venv/bin/python eval_model.py --pred-file <path> --model-name <name> [--include-code]
"""

import os
import sys
import json
import argparse
from collections import Counter, defaultdict

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


def load_livecodebench():
    from datasets import load_from_disk
    ds = load_from_disk("/home/ah872032.ucf/jiaqi/RouterArena/dataset/livecodebench")
    return {item["global_idx"]: item for item in ds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--include-code", action="store_true",
                        help="Include LiveCodeBench code_accuracy (needs compute node)")
    args = parser.parse_args()

    with open(args.pred_file) as f:
        predictions = json.load(f)
    print(f"[{args.model_name}] Loaded {len(predictions)} predictions")

    gt_file = "/home/ah872032.ucf/jiaqi/RouterArena/cached_results/gpt-4o-mini.jsonl"
    with open(gt_file) as f:
        gt_entries = [json.loads(l) for l in f]
    gt_map = {e["global_index"]: e for e in gt_entries}
    print(f"Ground truth: {len(gt_entries)} entries")

    lcb_map = None
    if args.include_code:
        from metrics import code_accuracy
        METRIC_FUNCS["code_accuracy"] = code_accuracy
        lcb_map = load_livecodebench()
        print(f"LiveCodeBench: {len(lcb_map)} problems loaded")

    regular = [p for p in predictions if not p.get("for_optimality")]
    optimality = [p for p in predictions if p.get("for_optimality")]
    print(f"Regular: {len(regular)}, Optimality: {len(optimality)}")

    # Group by dataset
    dataset_entries = defaultdict(lambda: {"preds": [], "gts": [], "metric": None,
                                           "indices": [], "out_tokens": []})
    matched = 0
    code_skipped = 0

    for pred in regular:
        gi = pred["global index"]
        if gi not in gt_map:
            continue
        matched += 1

        gt = gt_map[gi]
        metric_name = gt["evaluation_result"]["metric"]
        ground_truth = gt["evaluation_result"]["ground_truth"]

        if metric_name == "code_accuracy":
            if not args.include_code:
                code_skipped += 1
                continue
            if lcb_map and gi in lcb_map:
                ground_truth = lcb_map[gi]
            else:
                code_skipped += 1
                continue

        parts = gi.rsplit("_", 1)
        dataset_name = parts[0] if len(parts) == 2 else gi

        gen_answer = pred.get("generated_result", {}).get("generated_answer", "")
        out_tokens = pred.get("generated_result", {}).get("token_usage", {}).get("output_tokens", 0)

        ds = dataset_entries[dataset_name]
        ds["metric"] = metric_name
        ds["preds"].append(gen_answer)
        ds["gts"].append(ground_truth)
        ds["indices"].append(gi)
        ds["out_tokens"].append(out_tokens)

    print(f"Matched: {matched}, Code skipped: {code_skipped}")

    # Evaluate
    print(f"\n{'='*70}")
    print(f"EVALUATING {args.model_name.upper()}")
    print(f"{'='*70}")

    total_score = 0.0
    total_entries = 0
    dataset_results = {}

    for ds_name in sorted(dataset_entries.keys()):
        ds = dataset_entries[ds_name]
        metric_name = ds["metric"]
        n = len(ds["preds"])
        scorer = METRIC_FUNCS.get(metric_name)
        if not scorer:
            print(f"  WARNING: Unknown metric {metric_name} for {ds_name}")
            continue

        try:
            if metric_name == "code_accuracy":
                print(f"  Running code tests for {ds_name} ({n} entries)...", flush=True)
                # Run code tests one by one to show progress
                correct = 0
                for ci, (cp, cg) in enumerate(zip(ds["preds"], ds["gts"])):
                    try:
                        s, _ = scorer([cp], [cg], matched_data=[{"prediction": cp, "ground_truth": cg, "options": []}])
                    except Exception:
                        s = 0.0
                    correct += s
                    print(f"    code: {ci+1}/{n} done, score={s:.0f}, acc so far: {correct/(ci+1):.1%}", flush=True)
                score = correct / n if n > 0 else 0.0
            else:
                score, _ = scorer(ds["preds"], ds["gts"], matched_data=[
                    {"prediction": p, "ground_truth": g, "options": []}
                    for p, g in zip(ds["preds"], ds["gts"])
                ])
        except Exception as e:
            print(f"  ERROR {ds_name}/{metric_name}: {e}")
            score = 0.0

        max_tok = sum(1 for t in ds["out_tokens"] if t >= 2048)
        total_score += score * n
        total_entries += n
        dataset_results[ds_name] = {
            "accuracy": score, "total": n, "metric": metric_name,
            "max_tokens_count": max_tok,
        }

    overall_acc = total_score / total_entries if total_entries > 0 else 0
    code_note = "" if args.include_code else f" (excl {code_skipped} code entries)"
    print(f"\n*** {args.model_name} OVERALL ACCURACY: {overall_acc:.4f} ({total_entries} entries){code_note} ***")

    # Table
    print(f"\n{'Dataset':<40} {'Metric':<22} {'Acc':>7} {'N':>5} {'MaxTok%':>7}")
    print("-" * 85)
    sorted_ds = sorted(dataset_results.items(), key=lambda x: x[1]["accuracy"])
    for ds_name, data in sorted_ds:
        mt_pct = data["max_tokens_count"] / data["total"] * 100 if data["total"] > 0 else 0
        print(f"{ds_name:<40} {data['metric']:<22} {data['accuracy']:>6.1%} {data['total']:>5} {mt_pct:>6.1f}%")

    # By metric
    print(f"\nAccuracy by metric:")
    metric_stats = defaultdict(lambda: {"s": 0.0, "n": 0})
    for _, data in dataset_results.items():
        metric_stats[data["metric"]]["s"] += data["accuracy"] * data["total"]
        metric_stats[data["metric"]]["n"] += data["total"]
    for m, s in sorted(metric_stats.items(), key=lambda x: x[1]["s"]/max(x[1]["n"],1)):
        print(f"   {m}: {s['s']/s['n']:.4f} ({s['n']} entries)")

    # Max tokens
    all_ot = [p.get("generated_result", {}).get("token_usage", {}).get("output_tokens", 0) for p in regular]
    mt = sum(1 for t in all_ot if t >= 2048)
    print(f"\nMax tokens hit: {mt}/{len(regular)} ({mt/len(regular)*100:.1f}%)")

    # Avg output tokens
    avg_ot = sum(all_ot) / len(all_ot) if all_ot else 0
    print(f"Avg output tokens: {avg_ot:.1f}")

    # Worst/best
    print(f"\nWorst 10:")
    for ds, d in sorted_ds[:10]:
        print(f"   {ds}: {d['accuracy']:.1%} ({d['metric']}, n={d['total']})")
    print(f"\nBest 10:")
    for ds, d in sorted_ds[-10:][::-1]:
        print(f"   {ds}: {d['accuracy']:.1%} ({d['metric']}, n={d['total']})")


if __name__ == "__main__":
    main()
