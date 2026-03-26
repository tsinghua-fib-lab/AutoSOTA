"""
Quality metrics per solution (not per solution-ut pair).
For individual UT: aggregate over ut_ids by averaging.
"""
import json
import math
from tqdm import tqdm


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]


def calc_quality_metrics(sol_model, ut_model, sol_num, ut_num, benchmark='humaneval+'):
    sol_anno = load_jsonl(f'data/result/{benchmark}/sol_{sol_model}_200_anno.jsonl')

    # Build ground truth: task_id -> sol_id -> is_correct
    gt = {}
    for data in tqdm(sol_anno, desc='Loading GT'):
        task_id = data['task_id']
        gt[task_id] = {}
        for sol_id, sol in enumerate(data['solutions']):
            gt[task_id][sol_id] = (sol['plus_status'] == 'pass')

    result_file = f'output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/100_sol_100_ut_result.jsonl'
    ut_results = load_jsonl(result_file)

    # Build (task_id, sol_id, ut_id) -> is_pass
    ut_pass = {}
    for item in tqdm(ut_results, desc='Loading UT results'):
        key = (item['task_id'], item['sol_id'], item['ut_id'])
        ut_pass[key] = (item['result'] == 'pass')

    print("\n===== INDIVIDUAL UT QUALITY =====")
    print("Paper expected: Acc=69.64, F1=63.63, FAR=11.17, FRR=38.55")

    # Per (task, ut_id): for each UT, compute its performance across all solutions
    # Then average over all UTs
    # OR: per (task, sol, ut) as a classifier

    # Let's try: For each task + ut_id combo treated as a binary classifier,
    # compute TP/TN/FP/FN at per-task-ut level and average

    per_ut_far = []
    per_ut_frr = []
    per_ut_acc = []
    per_ut_f1 = []

    for task_id in gt:
        for ut_id in range(ut_num):
            tp = tn = fp = fn = 0
            for sol_id in range(sol_num):
                if sol_id not in gt[task_id]:
                    continue
                is_correct = gt[task_id][sol_id]
                key = (task_id, sol_id, ut_id)
                if key not in ut_pass:
                    continue
                ut_accepts = ut_pass[key]

                if is_correct and ut_accepts: tp += 1
                elif is_correct and not ut_accepts: fn += 1
                elif not is_correct and ut_accepts: fp += 1
                else: tn += 1

            total = tp + tn + fp + fn
            if total == 0:
                continue
            acc = (tp + tn) / total * 100
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) * 100 if (prec + rec) > 0 else 0
            far = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
            frr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0

            per_ut_acc.append(acc)
            per_ut_f1.append(f1)
            per_ut_far.append(far)
            per_ut_frr.append(frr)

    print(f"Per-task-ut avg: Acc={sum(per_ut_acc)/len(per_ut_acc):.2f}, F1={sum(per_ut_f1)/len(per_ut_f1):.2f}, FAR={sum(per_ut_far)/len(per_ut_far):.2f}, FRR={sum(per_ut_frr)/len(per_ut_frr):.2f}")

    print("\n===== MULTIPLE UT QUALITY =====")
    print("Paper expected: Acc=80.46, F1=81.27, FAR=16.48, FRR=22.71")

    # For multiple UTs: for each (task, sol), determine if accepted based on combining 100 UTs
    # Try: majority voting (>50% UTs pass)
    # Then compute per-task metrics and average

    per_task_tp = {}
    per_task_tn = {}
    per_task_fp = {}
    per_task_fn = {}

    for task_id in gt:
        per_task_tp[task_id] = 0
        per_task_tn[task_id] = 0
        per_task_fp[task_id] = 0
        per_task_fn[task_id] = 0

        for sol_id in range(sol_num):
            if sol_id not in gt[task_id]:
                continue
            is_correct = gt[task_id][sol_id]

            passes = sum(1 for ut_id in range(ut_num)
                        if (task_id, sol_id, ut_id) in ut_pass and ut_pass[(task_id, sol_id, ut_id)])

            # Majority voting
            total_ut = sum(1 for ut_id in range(ut_num) if (task_id, sol_id, ut_id) in ut_pass)
            majority_pass = passes > total_ut / 2 if total_ut > 0 else False

            if is_correct and majority_pass: per_task_tp[task_id] += 1
            elif is_correct and not majority_pass: per_task_fn[task_id] += 1
            elif not is_correct and majority_pass: per_task_fp[task_id] += 1
            else: per_task_tn[task_id] += 1

    # Global
    total_tp = sum(per_task_tp.values())
    total_tn = sum(per_task_tn.values())
    total_fp = sum(per_task_fp.values())
    total_fn = sum(per_task_fn.values())
    total = total_tp + total_tn + total_fp + total_fn

    acc = (total_tp + total_tn) / total * 100
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) * 100 if (prec + rec) > 0 else 0
    far = total_fp / (total_fp + total_tn) * 100 if (total_fp + total_tn) > 0 else 0
    frr = total_fn / (total_fn + total_tp) * 100 if (total_fn + total_tp) > 0 else 0
    print(f"Majority vote (global): Acc={acc:.2f}, F1={f1:.2f}, FAR={far:.2f}, FRR={frr:.2f}")
    print(f"  TP={total_tp}, TN={total_tn}, FP={total_fp}, FN={total_fn}")

    # Per task and average
    per_task_metrics = []
    for task_id in gt:
        tp = per_task_tp[task_id]
        tn = per_task_tn[task_id]
        fp = per_task_fp[task_id]
        fn = per_task_fn[task_id]

        total = tp + tn + fp + fn
        if total == 0:
            continue
        acc = (tp + tn) / total * 100
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) * 100 if (prec + rec) > 0 else 0
        far = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
        per_task_metrics.append((acc, f1, far, frr))

    avg_acc = sum(m[0] for m in per_task_metrics) / len(per_task_metrics)
    avg_f1 = sum(m[1] for m in per_task_metrics) / len(per_task_metrics)
    avg_far = sum(m[2] for m in per_task_metrics) / len(per_task_metrics)
    avg_frr = sum(m[3] for m in per_task_metrics) / len(per_task_metrics)
    print(f"Majority vote (per task avg): Acc={avg_acc:.2f}, F1={avg_f1:.2f}, FAR={avg_far:.2f}, FRR={avg_frr:.2f}")


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
