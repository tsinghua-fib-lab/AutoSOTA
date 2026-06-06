"""
Quality metrics - try different ut_num values to find matching FAR/FRR
"""
import json
from tqdm import tqdm


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]


def calc_quality_metrics(sol_model, ut_model, sol_num, benchmark='humaneval+'):
    sol_anno = load_jsonl(f'data/result/{benchmark}/sol_{sol_model}_200_anno.jsonl')

    gt = {}
    for data in sol_anno:
        task_id = data['task_id']
        gt[task_id] = {}
        for sol_id, sol in enumerate(data['solutions']):
            gt[task_id][sol_id] = (sol['plus_status'] == 'pass')

    result_file = f'output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/100_sol_100_ut_result.jsonl'
    ut_results = load_jsonl(result_file)

    ut_pass = {}
    for item in ut_results:
        key = (item['task_id'], item['sol_id'], item['ut_id'])
        ut_pass[key] = (item['result'] == 'pass')

    print("Paper expected (Multiple UT): Acc=80.46, F1=81.27, FAR=16.48, FRR=22.71")
    print()

    for ut_num in [1, 5, 10, 20, 50, 100]:
        # For each (task, sol), check if the combination of ut_num UTs accepts solution
        # Using "any pass" method
        tp_any = tn_any = fp_any = fn_any = 0

        for task_id in gt:
            for sol_id in range(sol_num):
                if sol_id not in gt[task_id]:
                    continue
                is_correct = gt[task_id][sol_id]

                # Any pass
                any_pass = any(ut_pass.get((task_id, sol_id, ut_id), False) for ut_id in range(ut_num))
                if is_correct and any_pass: tp_any += 1
                elif is_correct and not any_pass: fn_any += 1
                elif not is_correct and any_pass: fp_any += 1
                else: tn_any += 1

        total = tp_any + tn_any + fp_any + fn_any
        acc = (tp_any + tn_any) / total * 100
        prec = tp_any / (tp_any + fp_any) if (tp_any + fp_any) > 0 else 0
        rec = tp_any / (tp_any + fn_any) if (tp_any + fn_any) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) * 100 if (prec + rec) > 0 else 0
        far = fp_any / (fp_any + tn_any) * 100 if (fp_any + tn_any) > 0 else 0
        frr = fn_any / (fn_any + tp_any) * 100 if (fn_any + tp_any) > 0 else 0
        print(f"ut_num={ut_num:3d} (any pass): Acc={acc:.2f}, F1={f1:.2f}, FAR={far:.2f}, FRR={frr:.2f}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('--sol_model', default='llama3-8b')
    p.add_argument('--ut_model', default='coderm-8b')
    p.add_argument('--sol_num', type=int, default=100)
    p.add_argument('--benchmark', default='humaneval+')
    args = p.parse_args()
    calc_quality_metrics(args.sol_model, args.ut_model, args.sol_num, args.benchmark)
