import numpy as np
import pandas as pd
import pickle
import os
import argparse
import sys
sys.path.append("..")
from utils import visualize_response_matrix
import json
from huggingface_hub import HfApi, login
from huggingface_hub import snapshot_download

# def custom_sort_key(x):
#     suffix = x.split("-")[-1]
#     if suffix == "Chat":
#         return float('inf') - 1  # Second last
#     elif suffix in ["Safe", "Instruct"]:
#         return float('inf')      # Last
#     else:
#         return int(suffix)  # Regular numeric sorting

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    # EleutherAI/pythia-6.9b, EleutherAI/pythia-12b
    # LLM360/Amber, HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints
    parser.add_argument("--benchmark_dir", type=str, required=True)
    # /lfs/skampere1/0/yuhengtu/deval/helm/src/benchmark_output/runs
    # /lfs/skampere1/0/sttruong/helm/src/benchmark_output/runs
    # /lfs/skampere2/0/sttruong/helm/src/benchmark_output/runs
    args = parser.parse_args()
    model_name = args.repo_id.split("/")[1]
    parts = args.benchmark_dir.split("/")
    
    # Load your pre-generated responses pickle file.
    with open(f"../data/gather_ckpt_data/responses_{model_name}_{parts[4]}_{parts[2]}.pkl", "rb") as f:
        results_full = pickle.load(f)

    # keep useful columns, drop nan rows
    results_full = results_full.sample(frac=1).reset_index(drop=True)
    results = results_full[["request.model", "input.text", "references", "scenario", "benchmark", "dicho_score"]]
    results = results.dropna(subset=["request.model", "input.text", "references", "scenario", "benchmark", "dicho_score"])

    # drop the dicho_score of 0.5
    results = results[results["dicho_score"] != 0.5]
    results["dicho_score"] = results["dicho_score"].astype(bool)
    assert results["dicho_score"].isin([0, 1]).all()

    # drop duplicate rows
    results = results.drop_duplicates(subset=["request.model", "input.text", "references", "scenario", "benchmark"], keep='first')
    print(f"non-duplicate percentage:{results.shape[0]/results_full.shape[0]}")

    # pivot to turn long table into matrix
    results = results.pivot(index="request.model", columns=["input.text",  "references", "scenario", "benchmark"], values="dicho_score")
    # Reindex the DataFrame according to the step order
    # sorted_index = sorted(results.index, key=custom_sort_key)
    sorted_index = sorted(results.index, key=lambda x: x.split("-")[-1])
    results = results.reindex(sorted_index)

    # sort the columns by scenario
    results = results.sort_index(axis=1, level="scenario")

    # nan -> -1 -> np.nan
    results = results.fillna(-1).astype(int)
    results = results.replace(-1, np.nan)
    
    # # delete all 0 or all 1 cols
    # results = results.loc[:, ~((results.isin([0, np.nan]).all()) | (results.isin([1, np.nan]).all()))]

    # Compute the overall average for each scenario manually
    scenario_means = {}
    for scenario in results.columns.get_level_values("scenario").unique():
        mask = results.columns.get_level_values("scenario") == scenario
        values = results.loc[:, mask].values  # all values for this scenario
        scenario_means[scenario] = np.nanmean(values)

    # Sort the scenario by their average score
    sorted_scenarios = sorted(scenario_means, key=scenario_means.get)

    # Create a mapping from scenario to its sort order
    scenario_order = {scenario: order for order, scenario in enumerate(sorted_scenarios)}

    # Reorder the columns based on the new scenario order using the key parameter
    results = results.sort_index(axis=1, level="scenario", key=lambda x: x.map(scenario_order))
    print(f"missing percentage: {results.isna().values.sum() / (results.shape[0] * results.shape[1])}")
    
    # prompt to z
    cache_folder = snapshot_download(
        repo_id="stair-lab/irsl_response_matrix",
        repo_type="dataset",
    )
    with open( f"{cache_folder}/input_to_z.json", "r", encoding="utf-8") as fp:
        input_text_to_z = json.load(fp)

    new_columns = []
    for col in results.columns:
        prompt = col[0]
        z_val = input_text_to_z.get(prompt, np.nan)
        new_columns.append(col + (z_val,))  # Append z value to tuple
    results.columns = pd.MultiIndex.from_tuples(new_columns, names=["input.text", "references", "scenario", "benchmark", "z"])
    
    save_path = f"../data/gather_ckpt_data/results_{model_name}_{parts[4]}_{parts[2]}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    login()
    api = HfApi()
    api.upload_file(
        path_or_fileobj=save_path,
        path_in_repo=save_path.split("/")[-1],
        repo_id="stair-lab/irsl_response_matrix",
        repo_type="dataset",
    )

    output_dir = "../result/gather_ckpt_data"
    os.makedirs(output_dir, exist_ok=True)
    visualize_response_matrix(results, results, f"{output_dir}/response_matrix_{model_name}_{parts[4]}_{parts[2]}.png")
