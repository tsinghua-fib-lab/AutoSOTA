import json
import random
import statistics
from tqdm import tqdm

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]

def calc_best_of_n(dataset, sol_num, ut_num, task_sol_results, task_sol_ut_results, task_ut_weights=None):
    sol_ids = [i for i in range(100)]
    ut_ids = [i for i in range(100)]

    accuracy = 0
    detail = []
    for data in dataset:
        task_id = data['task_id']

        # Use variance-weighted voting: each UT's vote weighted by its discrimination power
        if task_ut_weights is not None and task_id in task_ut_weights:
            ut_weight_map = task_ut_weights[task_id]  # {ut_id: weight}
            active_uts = list(ut_weight_map.keys())
        else:
            random.shuffle(ut_ids)
            active_uts = ut_ids[:ut_num]
            ut_weight_map = {ut_id: 1.0 for ut_id in active_uts}

        # accumulate weighted votes for each solution
        sol_weighted_scores = dict()
        for i in range(sol_num):
            sol_id = sol_ids[i]
            sol_weighted_scores[sol_id] = 0.0
            for ut_id in active_uts:
                key = f"{task_id}-{sol_id}-{ut_id}"
                if key in task_sol_ut_results and task_sol_ut_results[key] == 'pass':
                    sol_weighted_scores[sol_id] += ut_weight_map[ut_id]

        # rerank by weighted score
        sol_weighted_scores = sorted(sol_weighted_scores.items(), key=lambda item: item[1], reverse=True)
        top_score = sol_weighted_scores[0][1]
        top_sol_list = []
        for value in sol_weighted_scores:
            if abs(value[1] - top_score) <= 1e-9:
                top_sol_list.append(value)
            else:
                break

        # Consistency tie-breaking: among equally-scored solutions, find largest consistent group
        # Build pass sets for consistency check
        sol_pass_sets = {}
        for sol_id, _ in top_sol_list:
            sol_pass_sets[sol_id] = frozenset(
                ut_id for ut_id in active_uts
                if task_sol_ut_results.get(f"{task_id}-{sol_id}-{ut_id}") == 'pass'
            )

        select_sol_ids = []
        max_consistency = 0
        for v1_id, _ in top_sol_list:
            consistency = sum(1 for v2_id, _ in top_sol_list if sol_pass_sets[v1_id] == sol_pass_sets[v2_id])
            if consistency > max_consistency:
                select_sol_ids = [v1_id]
                max_consistency = consistency
            elif consistency == max_consistency:
                select_sol_ids.append(v1_id)

        num = 0
        for v in select_sol_ids:
            if task_sol_results[f'{task_id}-{v}'] == 'pass':
                num += 1
        accuracy += num / len(select_sol_ids)

    accuracy = round(accuracy / len(dataset), 4)
    return accuracy

def get_result_on_sol_and_ut(benchmark, sol_model, ut_model, sol_num, ut_num, sample_num):
    dataset = load_jsonl(f'data/benchmark/input_{benchmark}_sol.jsonl')

    # label each solution
    if benchmark != 'livecodebench':
        sol_anno = load_jsonl(f'data/result/{benchmark}/sol_{sol_model}_200_anno.jsonl')
        result_key = 'plus_status'
    else:
        sol_anno = load_jsonl(f'data/result/{benchmark}/sol_{sol_model}_100_anno.jsonl')
        result_key = 'result'
    task_sol_results = dict()
    for data in tqdm(sol_anno):
        for sol_id in range(len(data['solutions'])):
            task_sol_results[f"{data['task_id']}-{sol_id}"] = data['solutions'][sol_id][result_key]

    # label each unit test
    task_sol_ut_results = dict()
    ut_result = load_jsonl(f'output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/100_sol_100_ut_result.jsonl')

    for data in tqdm(ut_result):
        # Use stricter threshold: require >80% of test cases to pass
        total_num = data['details']['total_num']
        pass_num = data['details']['pass_num']
        is_pass = 'pass' if (total_num > 0 and pass_num / total_num > 0.8) else 'fail'
        task_sol_ut_results[f"{data['task_id']}-{data['sol_id']}-{data['ut_id']}"] = is_pass

    # Pre-compute per-task UT discrimination weights (variance-based, no label leakage)
    # Each UT's weight = variance of its binary pass rate across all solutions
    # This gives higher weight to UTs that discriminate between solutions
    task_ut_weights = {}
    for data in dataset:
        task_id = data['task_id']
        ut_weight_map = {}
        for ut_id in range(100):
            sol_passes = []
            for sol_id in range(100):
                key = f"{task_id}-{sol_id}-{ut_id}"
                sol_passes.append(1 if task_sol_ut_results.get(key) == 'pass' else 0)
            mean_pass = sum(sol_passes) / len(sol_passes) if sol_passes else 0
            variance = mean_pass * (1 - mean_pass)  # binary variance (discrimination proxy)
            ut_weight_map[ut_id] = variance
        task_ut_weights[task_id] = ut_weight_map

    # calcluate best-of-n
    y = []
    for _ in range(sample_num):
        accuracy = calc_best_of_n(dataset, sol_num, ut_num, task_sol_results, task_sol_ut_results, task_ut_weights)
        y.append(accuracy)

    y_mean = statistics.mean(y)
    # y_std = statistics.stdev(y)

    print(f"y_mean: {y_mean}")
    # print(f"y_std: {y_std}")

if __name__ == '__main__':
    import argparse

    # parse parameter
    parser = argparse.ArgumentParser(description="calculate result")
    parser.add_argument('--benchmark', type=str, help='evaluate benchmark')
    parser.add_argument('--sol_model', type=str, help='the model that generate solutions')
    parser.add_argument('--ut_model', type=str, help='the model that generate unit test')
    parser.add_argument('--sol_num', type=int, help='the number of generated solutions')
    parser.add_argument('--ut_num', type=int, help='the number of generated unit test')
    parser.add_argument('--sample_num', type=int, help='the number of bootstrape sampling')
    args = parser.parse_args()

    get_result_on_sol_and_ut(args.benchmark, args.sol_model, args.ut_model, args.sol_num,
                            args.ut_num, args.sample_num)
