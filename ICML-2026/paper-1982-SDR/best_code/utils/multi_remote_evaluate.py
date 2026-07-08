import argparse
import random
import pandas as pd
import numpy as np
import json

from cost_config import TOKEN_COSTS, AVERAGE_RESPONSE_LENGTH

def parse_args():
    parser = argparse.ArgumentParser(
        description="cost-performance analysis"
    )
    parser.add_argument('--remote-answer-paths', nargs='+', type=str,
                        help='remote answers files')
    parser.add_argument('--model-names', nargs='+', type=str,
                        help='local model and remote models')
    parser.add_argument('--dataset-name', type=str, default='mmlu',
                        help='dataset used (for getting average response lengths)')
    parser.add_argument('--score-paths', nargs='+', type=str,
                        help='path of the score files')
    parser.add_argument('--output-path', type=str, required=True,
                        help='path of the output figure')

    args = parser.parse_args()
    return args

def argmax_intervals(performance, cost):
    """
    given performance and cost，return intervals of k's value in which different i to be argmax(utility_i)
    utility_i(k) = performance[i] - k * cost[i]
    """
    n = len(performance)
    performance = np.array(performance, dtype=float)
    cost = np.array(cost, dtype=float)
    
    results = []

    for i in range(n):
        lower, upper = -np.inf, np.inf
        feasible = True

        for j in range(n):
            if i == j:
                continue
            pi, ci = performance[i], cost[i]
            pj, cj = performance[j], cost[j]

            if ci == cj:
                if pi < pj:
                    feasible = False
                    break
                else:
                    continue

            k_star = (pi - pj) / (ci - cj)

            if ci > cj:  # need k <= k_star
                upper = min(upper, k_star)
            else:        # ci < cj, need k >= k_star
                lower = max(lower, k_star)

        if feasible and upper >= lower and upper > 0:
            # only k > 0
            results.append((i, max(lower, 0), upper))

    results.sort(key=lambda x: x[1])
    return results

def analysis(remote_answer_paths, model_names, dataset_name, score_paths, train_ratio, output_path):
    remote_answers = [pd.read_parquet(remote_answer_path) for remote_answer_path in remote_answer_paths]

    input_lens = pd.DataFrame({
        model_name: remote_answer['input_len'] for model_name, remote_answer in zip(model_names[1:], remote_answers)
    })
    output_lens = pd.DataFrame({
        model_name: remote_answer['output_len'] for model_name, remote_answer in zip(model_names[1:], remote_answers)
    })
    
    if train_ratio > 1e-9:
        num_data = len(input_lens)
        random.seed(42)
        indices = list(range(num_data))
        random.shuffle(indices)   
        num_train = int(train_ratio * num_data)
        indices = indices[num_train:]
        input_lens = input_lens.loc[indices].reset_index(drop=True)
        output_lens = output_lens.loc[indices].reset_index(drop=True)

    result = {}
    for score_path in score_paths:
        scores = pd.read_parquet(score_path)
        
        for column_name in scores.keys():
            if column_name.endswith('pref'):
                scores[column_name.replace('pref', 'perf')] = scores[column_name]
                scores = scores.drop(columns=[column_name])
        scores.to_parquet(score_path)
        
        predicts = pd.DataFrame({
            model: scores[f"{model}-pred"] for model in model_names
        }).reset_index(drop=True)

        performances = pd.DataFrame({
            model: scores[f"{model}-perf"] for model in model_names
        }).reset_index(drop=True)

        expected_costs = pd.DataFrame({
            model: TOKEN_COSTS[model]['input'] * input_lens[model] + TOKEN_COSTS[model]['output'] * AVERAGE_RESPONSE_LENGTH[model][dataset_name] \
                if model != model_names[0] else 0 for model in model_names
        })
        real_costs = pd.DataFrame({
            model: TOKEN_COSTS[model]['input'] * input_lens[model] + TOKEN_COSTS[model]['output'] * output_lens[model] \
                if model != model_names[0] else 0 for model in model_names
        })

        intervals = pd.Series([
            argmax_intervals(predicts.loc[i].values, expected_costs.loc[i].values)
            for i in predicts.index
        ], index=predicts.index)
        
        choosed_models = intervals.apply(lambda x: predicts.columns[x[0][0]] if len(x) > 0 else None)

        k_rights = intervals.apply(lambda x: x[0][2] if len(x) > 0 else None)
        k_and_interval_id = pd.DataFrame({
            'k': k_rights,
            'interval_id': [0] * len(k_rights)
        })

        overall_perfs = pd.Series([performances.loc[i, j] for i, j in zip(performances.index, choosed_models)], index=performances.index)
        overall_costs = pd.Series([real_costs.loc[i, j] for i, j in zip(real_costs.index, choosed_models)], index=real_costs.index)

        perf_points = [float(overall_perfs.mean())]
        cost_points = [float(overall_costs.sum())]

        while not np.isinf(k_and_interval_id['k'].min()):
            row = k_and_interval_id['k'].idxmin()
            now_interval_id = k_and_interval_id.loc[row, 'interval_id']
            next_model_id, _, next_k = intervals.loc[row][now_interval_id + 1]
            next_model = predicts.columns[next_model_id]

            overall_perfs.loc[row] = performances.loc[row, next_model]
            overall_costs.loc[row] = real_costs.loc[row, next_model]
            perf_points.append(float(overall_perfs.mean()))
            cost_points.append(float(overall_costs.sum()))

            k_and_interval_id.loc[row] = [next_k, now_interval_id + 1]
        
        # 3. plot the line
        points = sorted(zip(cost_points, perf_points), key=lambda p: p[0])
        result[score_path] = points

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def main():
    args = parse_args()
    if args.dataset_name == 'gsm8k':
        train_ratio = 0
    else:
        train_ratio = 0.8
    analysis(args.remote_answer_paths, args.model_names, args.dataset_name,
             args.score_paths, train_ratio, args.output_path)

if __name__ == '__main__':
    main()
