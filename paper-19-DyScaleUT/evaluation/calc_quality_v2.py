"""
Quality metrics evaluation for CodeRM unit tests.
Multiple UT uses majority voting (more passing than failing = accept).
"""
import json
import math
from collections import defaultdict
from tqdm import tqdm


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]


def calc_quality_metrics(sol_model, ut_model, sol_num, ut_num, benchmark='humaneval+'):
    # Load solution annotations
    sol_anno = load_jsonl(f'data/result/{benchmark}/sol_{sol_model}_200_anno.jsonl')

    # Build ground truth: task_id -> sol_id -> is_correct
    gt = {}
    for data in tqdm(sol_anno, desc='Loading GT'):
        task_id = data['task_id']
        gt[task_id] = {}
        for sol_id, sol in enumerate(data['solutions']):
            gt[task_id][sol_id] = (sol['plus_status'] == 'pass')

    # Load unit test execution results
    result_file = f'output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/100_sol_100_ut_result.jsonl'
    ut_results = load_jsonl(result_file)

    # Build (task_id, sol_id, ut_id) -> is_pass
    ut_pass = {}
    for item in tqdm(ut_results, desc='Loading UT results'):
        key = (item['task_id'], item['sol_id'], item['ut_id'])
        ut_pass[key] = (item['result'] == 'pass')

    # ---- INDIVIDUAL UT METRICS ----
    print("\n=== Individual UT Quality (per ut individually) ===")
    ind_tp = ind_tn = ind_fp = ind_fn = 0

    for task_id in gt:
        for sol_id in range(sol_num):
            if sol_id not in gt[task_id]:
                continue
            is_correct = gt[task_id][sol_id]

            for ut_id in range(ut_num):
                key = (task_id, sol_id, ut_id)
                if key not in ut_pass:
                    continue
                ut_accepts = ut_pass[key]

                if is_correct and ut_accepts:
                    ind_tp += 1
                elif is_correct and not ut_accepts:
                    ind_fn += 1
                elif not is_correct and ut_accepts:
                    ind_fp += 1
                else:
                    ind_tn += 1

    ind_total = ind_tp + ind_tn + ind_fp + ind_fn
    ind_acc = (ind_tp + ind_tn) / ind_total * 100
    ind_prec = ind_tp / (ind_tp + ind_fp) if (ind_tp + ind_fp) > 0 else 0
    ind_rec = ind_tp / (ind_tp + ind_fn) if (ind_tp + ind_fn) > 0 else 0
    ind_f1 = 2 * ind_prec * ind_rec / (ind_prec + ind_rec) * 100 if (ind_prec + ind_rec) > 0 else 0

    # FAR = probability that wrong solution is accepted by UT
    # = FP / (FP + TN) = FP / total_incorrect
    ind_far_v1 = ind_fp / (ind_fp + ind_tn) * 100 if (ind_fp + ind_tn) > 0 else 0
    # FRR = probability that correct solution is rejected by UT
    # = FN / (FN + TP) = FN / total_correct
    ind_frr_v1 = ind_fn / (ind_fn + ind_tp) * 100 if (ind_fn + ind_tp) > 0 else 0

    print(f"V1 (FAR=FP/(FP+TN), FRR=FN/(FN+TP)) - Acc: {ind_acc:.2f}, F1: {ind_f1:.2f}, FAR: {ind_far_v1:.2f}, FRR: {ind_frr_v1:.2f}")
    print(f"  TP={ind_tp}, TN={ind_tn}, FP={ind_fp}, FN={ind_fn}")
    print(f"  Paper: Acc=69.64, F1=63.63, FAR=11.17, FRR=38.55")
    print()

    # ---- MULTIPLE UT METRICS ----
    print("=== Multiple UT Quality (combined per task-sol pair) ===")

    # Version 1: "any pass" => accept
    mul_tp_v1 = mul_tn_v1 = mul_fp_v1 = mul_fn_v1 = 0

    # Version 2: majority voting (>50% UTs pass => accept)
    mul_tp_v2 = mul_tn_v2 = mul_fp_v2 = mul_fn_v2 = 0

    # Version 3: all pass => accept
    mul_tp_v3 = mul_tn_v3 = mul_fp_v3 = mul_fn_v3 = 0

    for task_id in gt:
        for sol_id in range(sol_num):
            if sol_id not in gt[task_id]:
                continue
            is_correct = gt[task_id][sol_id]

            passes = 0
            total_uts = 0
            for ut_id in range(ut_num):
                key = (task_id, sol_id, ut_id)
                if key in ut_pass:
                    total_uts += 1
                    if ut_pass[key]:
                        passes += 1

            if total_uts == 0:
                continue

            # V1: any pass
            any_pass = passes > 0
            if is_correct and any_pass:
                mul_tp_v1 += 1
            elif is_correct and not any_pass:
                mul_fn_v1 += 1
            elif not is_correct and any_pass:
                mul_fp_v1 += 1
            else:
                mul_tn_v1 += 1

            # V2: majority voting
            majority_pass = passes > total_uts / 2
            if is_correct and majority_pass:
                mul_tp_v2 += 1
            elif is_correct and not majority_pass:
                mul_fn_v2 += 1
            elif not is_correct and majority_pass:
                mul_fp_v2 += 1
            else:
                mul_tn_v2 += 1

            # V3: all pass
            all_pass = passes == total_uts
            if is_correct and all_pass:
                mul_tp_v3 += 1
            elif is_correct and not all_pass:
                mul_fn_v3 += 1
            elif not is_correct and all_pass:
                mul_fp_v3 += 1
            else:
                mul_tn_v3 += 1

    def compute_metrics(tp, tn, fp, fn, label):
        total = tp + tn + fp + fn
        acc = (tp + tn) / total * 100 if total > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) * 100 if (prec + rec) > 0 else 0
        far = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
        print(f"{label}: Acc={acc:.2f}, F1={f1:.2f}, FAR={far:.2f}, FRR={frr:.2f}")
        print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        return acc, f1, far, frr

    print("Paper expected: Acc=80.46, F1=81.27, FAR=16.48, FRR=22.71")
    compute_metrics(mul_tp_v1, mul_tn_v1, mul_fp_v1, mul_fn_v1, "V1 (any pass)")
    compute_metrics(mul_tp_v2, mul_tn_v2, mul_fp_v2, mul_fn_v2, "V2 (majority voting)")
    compute_metrics(mul_tp_v3, mul_tn_v3, mul_fp_v3, mul_fn_v3, "V3 (all pass)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='humaneval+')
    parser.add_argument('--sol_model', type=str, default='llama3-8b')
    parser.add_argument('--ut_model', type=str, default='coderm-8b')
    parser.add_argument('--sol_num', type=int, default=100)
    parser.add_argument('--ut_num', type=int, default=100)
    args = parser.parse_args()

    calc_quality_metrics(args.sol_model, args.ut_model, args.sol_num, args.ut_num, args.benchmark)
