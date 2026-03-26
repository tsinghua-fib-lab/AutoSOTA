# Borrowed and extended from:
# https://github.com/QwenLM/QwQ/blob/main/eval/eval/eval.py
# Commit Hash: 56e46a45db58dea9e7eaf062da61cda610bb71ad

import json
import argparse
from tqdm import tqdm
import os
from collections import defaultdict
import pandas as pd
from livecodebench_v5 import compute_scores as compute_scores_livecodebench_v5


def get_after_think(text):
    parts = text.split("\n</think>\n\n", 1)
    if len(parts) > 1:
        return parts[1]
    else:
        return text

def main():
    parser = argparse.ArgumentParser(description="Evaluate model outputs")
    parser.add_argument("--input_folder_path", type=str, required=True, help="Path to input folder")
    parser.add_argument("--cache_path", type=str, required=True, help="Path to save cache results")
    parser.add_argument("--task_name", type=str, default='livecodebench', help="Task name")
    parser.add_argument("--results_path", type=str, default="results.csv", help="Path to save accuracies")
    args = parser.parse_args()
    
    input_folder_files = os.listdir(args.input_folder_path)
    folder_by_budget_and_exp = {}
    for folder in input_folder_files:
        folder_full_path = os.path.join(args.input_folder_path, folder)
        budget = int(folder[folder.rfind('budget-') + len('budget-'):].split('-')[0])
        exp_name = folder[:folder.rfind('-budget')]
        folder_by_budget_and_exp[(exp_name, budget)] = folder_full_path

    print(input_folder_files)
    print(folder_by_budget_and_exp)

    accuracy_by_budget_and_exp = {}

    for (exp_name, budget), folder in tqdm(folder_by_budget_and_exp.items()):
        print(f'Processing {folder} dir')
        files_in_dir = os.listdir(folder)

        cache_path = f'{args.cache_path}_{exp_name}_{budget}'
        if os.path.isfile(cache_path):
            os.remove(cache_path)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        data = []
        for file in files_in_dir:
            full_path = os.path.join(folder, file)
            with open(full_path, 'r', encoding='utf-8') as f:
                file_data = [json.loads(l) for l in f]
            # We consider only one solution per problem for now
            assert len(file_data) == 1, "We consider only one solution per problem for now"
            file_data = file_data[0]
            file_data['path'] = file
            data.append(file_data)
        
        for item in data:
            item["task"] = args.task_name
            temp = get_after_think(item['gen'][0])
            item['gen'][0] = temp
        acc = compute_scores_livecodebench_v5(data, cache_path)
        print(f"Exp: {exp_name}, Budget: {budget}, Pass@1: {acc:.3f}")
        accuracy_by_budget_and_exp[(exp_name, budget)] = acc
    
    # group by exp name
    accuracy_by_exp = defaultdict(list)
    for (exp_name, budget), acc in accuracy_by_budget_and_exp.items():
        accuracy_by_exp[exp_name].append((budget, acc))

    result_df_data = []
    for exp_name in accuracy_by_exp:
        accuracy_by_exp[exp_name] = sorted(accuracy_by_exp[exp_name], key=lambda x: x[0])
        print(f'Exp: {exp_name}')
        print('\tBudget\tAccuracy')
        for (budget, acc) in accuracy_by_exp[exp_name]:
            print(f'\t{budget:<5}\t{acc:.3f}')
            result_df_data.append({'exp_name': exp_name, 'budget': budget, 'acc': acc})

    pd.DataFrame(result_df_data).to_csv(args.results_path, index=False)


if __name__ == "__main__":
    main()
