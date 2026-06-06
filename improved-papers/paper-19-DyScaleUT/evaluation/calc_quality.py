"""
Quality metrics evaluation for CodeRM unit tests.
Evaluates Accuracy, F1, FAR, FRR for individual and multiple unit tests.
"""
import json
from collections import defaultdict
from tqdm import tqdm

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]

def calc_quality_metrics(sol_model, ut_model, sol_num, ut_num, benchmark='humaneval+'):
    """
    Calculate quality metrics for unit tests.

    For each (task, sol_id, ut_id) triple:
    - Ground truth: solution has plus_status == 'pass' => correct, else incorrect
    - UT prediction: result == 'pass' => UT accepts solution (predicts correct), else rejects

    For Multiple UTs: combine all ut_num unit tests for each solution.
    A solution is accepted by the combined set if it passes at least one UT.
    """
    # Load solution annotations
    sol_anno = load_jsonl(f'data/result/{benchmark}/sol_{sol_model}_200_anno.jsonl')

    # Build ground truth mapping: task_id -> sol_id -> is_correct
    gt = {}
    for data in tqdm(sol_anno, desc='Loading GT'):
        task_id = data['task_id']
        gt[task_id] = {}
        for sol_id, sol in enumerate(data['solutions']):
            gt[task_id][sol_id] = (sol['plus_status'] == 'pass')

    # Load unit test execution results
    result_file = f'output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/100_sol_100_ut_result.jsonl'
    ut_results = load_jsonl(result_file)

    # Build a dict: (task_id, sol_id, ut_id) -> is_pass
    ut_pass = {}
    for item in tqdm(ut_results, desc='Loading UT results'):
        key = (item['task_id'], item['sol_id'], item['ut_id'])
        ut_pass[key] = (item['result'] == 'pass')

    # ---- INDIVIDUAL UT METRICS ----
    print("\n=== Individual UT Quality (per ut) ===")
    # For each (task, sol, ut) triple, evaluate if UT correctly classifies solution
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
                    ind_tp += 1  # UT correctly accepts correct solution
                elif is_correct and not ut_accepts:
                    ind_fn += 1  # UT wrongly rejects correct solution
                elif not is_correct and ut_accepts:
                    ind_fp += 1  # UT wrongly accepts wrong solution
                elif not is_correct and not ut_accepts:
                    ind_tn += 1  # UT correctly rejects wrong solution

    ind_total = ind_tp + ind_tn + ind_fp + ind_fn
    ind_acc = (ind_tp + ind_tn) / ind_total if ind_total > 0 else 0
    ind_prec = ind_tp / (ind_tp + ind_fp) if (ind_tp + ind_fp) > 0 else 0
    ind_rec = ind_tp / (ind_tp + ind_fn) if (ind_tp + ind_fn) > 0 else 0
    ind_f1 = 2 * ind_prec * ind_rec / (ind_prec + ind_rec) if (ind_prec + ind_rec) > 0 else 0
    ind_far = ind_fp / (ind_fp + ind_tn) if (ind_fp + ind_tn) > 0 else 0  # Wrong solutions accepted / all wrong solutions
    ind_frr = ind_fn / (ind_fn + ind_tp) if (ind_fn + ind_tp) > 0 else 0  # Correct solutions rejected / all correct solutions

    print(f"Individual UT - Acc: {ind_acc*100:.2f}, F1: {ind_f1*100:.2f}, FAR: {ind_far*100:.2f}, FRR: {ind_frr*100:.2f}")

    # ---- MULTIPLE UT METRICS ----
    print("\n=== Multiple UT Quality (combined ut_num UTs) ===")
    # For each (task, sol), combine all ut_num UTs
    # A solution is "accepted" by the combined set if it passes at least 1 UT (any pass)
    mul_tp = mul_tn = mul_fp = mul_fn = 0

    for task_id in gt:
        for sol_id in range(sol_num):
            if sol_id not in gt[task_id]:
                continue
            is_correct = gt[task_id][sol_id]

            # Check if any UT accepts this solution
            any_pass = False
            for ut_id in range(ut_num):
                key = (task_id, sol_id, ut_id)
                if key in ut_pass and ut_pass[key]:
                    any_pass = True
                    break

            if is_correct and any_pass:
                mul_tp += 1
            elif is_correct and not any_pass:
                mul_fn += 1
            elif not is_correct and any_pass:
                mul_fp += 1
            elif not is_correct and not any_pass:
                mul_tn += 1

    mul_total = mul_tp + mul_tn + mul_fp + mul_fn
    mul_acc = (mul_tp + mul_tn) / mul_total if mul_total > 0 else 0
    mul_prec = mul_tp / (mul_tp + mul_fp) if (mul_tp + mul_fp) > 0 else 0
    mul_rec = mul_tp / (mul_tp + mul_fn) if (mul_tp + mul_fn) > 0 else 0
    mul_f1 = 2 * mul_prec * mul_rec / (mul_prec + mul_rec) if (mul_prec + mul_rec) > 0 else 0
    mul_far = mul_fp / (mul_fp + mul_tn) if (mul_fp + mul_tn) > 0 else 0  # FAR = FP / (FP + TN)
    mul_frr = mul_fn / (mul_fn + mul_tp) if (mul_fn + mul_tp) > 0 else 0  # FRR = FN / (FN + TP)

    print(f"Multiple UT - Acc: {mul_acc*100:.2f}, F1: {mul_f1*100:.2f}, FAR: {mul_far*100:.2f}, FRR: {mul_frr*100:.2f}")
    print(f"TP={mul_tp}, TN={mul_tn}, FP={mul_fp}, FN={mul_fn}, Total={mul_total}")

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
