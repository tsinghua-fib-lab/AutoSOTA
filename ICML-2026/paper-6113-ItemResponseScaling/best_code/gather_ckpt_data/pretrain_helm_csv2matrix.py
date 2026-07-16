import numpy as np
import pandas as pd
import pickle
import os
import argparse
import sys
sys.path.append("..")
from utils import visualize_response_matrix
import json
from huggingface_hub import HfApi, login, hf_hub_download

def stats_to_logprob(stats):
    stats = json.loads(stats)
    for entry in stats:
        if entry["name"]["name"] == "logprob":
            return float(entry["sum"])
    return float("nan")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    # parser.add_argument("--repo_id", type=str, default="LLM360/Amber")
    # EleutherAI/pythia-6.9b, EleutherAI/pythia-12b
    # LLM360/Amber, HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints
    # parser.add_argument("--benchmark_dir", type=str, required=True)
    parser.add_argument("--benchmark_dir", type=str, default="../../helm/src/benchmark_output/runs")
    # /lfs/skampere1/0/yuhengtu/deval/helm/src/benchmark_output/runs
    # /lfs/skampere1/0/sttruong/helm/src/benchmark_output/runs
    # /lfs/skampere2/0/sttruong/helm/src/benchmark_output/runs
    args = parser.parse_args()
    model_name = args.repo_id.split("/")[1]
    
    parts = os.path.abspath(args.benchmark_dir).split("/")
    with open(f"../data/pretrain_helm/responses_{model_name}_{parts[4]}_{parts[2]}.pkl", "rb") as f:
        results_full = pickle.load(f)

    # cols = list(results_full.columns)
    # first_row = results_full.iloc[0]
    # with open("columns.txt", "w") as f:
    #     for col, val in zip(cols, first_row):
    #         f.write(f"{col}: {val}\n")
    
    # keep useful columns, drop nan rows
    results_full = results_full.sample(frac=1).reset_index(drop=True)
    results = results_full[["request.model", "input.text", "references", "scenario", "benchmark", "dicho_score", "stats"]]
    results = results.dropna(subset=["request.model", "input.text", "references", "scenario", "benchmark", "dicho_score", "stats"])

    # drop the dicho_score of 0.5
    # results = results[results["dicho_score"] != 0.5]
    # results["dicho_score"] = results["dicho_score"].astype(bool)
    assert results["dicho_score"].isin([0, 1]).all()

    # drop duplicate rows
    results = results.drop_duplicates(subset=["request.model", "input.text", "references", "scenario", "benchmark"], keep='first')
    print(f"non-duplicate percentage:{results.shape[0]/results_full.shape[0]}")

    # pivot to turn long table into matrix
    # results = results.pivot(index="request.model", columns=["input.text",  "references", "scenario", "benchmark"], values="dicho_score")
    logprobs = results["stats"].map(stats_to_logprob)
    probs = np.exp(logprobs)
    results["prob"] = probs
    results = results.pivot(index="request.model", columns=["input.text",  "references", "scenario", "benchmark"], values="prob")
    
    # Reindex the DataFrame according to the step order
    sorted_index = sorted(results.index, key=lambda x: x.split("-")[-1])
    results = results.reindex(sorted_index)

    # sort the columns by scenario
    results = results.sort_index(axis=1, level="scenario")

    # nan -> -1 -> np.nan
    results = results.fillna(-1)
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
    file_path = hf_hub_download(
        repo_id="stair-lab/irsl_downstream_resmat1_fullinfo",
        repo_type="dataset",
        filename="input_to_z.json"
    )
    with open(file_path, "r", encoding="utf-8") as fp:
        input_text_to_z = json.load(fp)
    new_columns = []
    for col in results.columns:
        prompt = col[0]
        z_val = input_text_to_z.get(prompt, np.nan)
        new_columns.append(col + (z_val,))  # Append z value to tuple
    results.columns = pd.MultiIndex.from_tuples(new_columns, names=["input.text", "references", "scenario", "benchmark", "z"])
    
    save_path = f"../data/pretrain_helm/results_{model_name}_{parts[4]}_{parts[2]}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    
    login(os.getenv("HF_TOKEN"))
    api = HfApi()
    api.upload_file(
        path_or_fileobj=save_path,
        path_in_repo=f"{save_path.split('/')[-1]}",
        repo_id="stair-lab/irsl_downstream_resmat1_prob",
        repo_type="dataset",
    )

    output_dir = "../result/pretrain_helm"
    os.makedirs(output_dir, exist_ok=True)
    visualize_response_matrix(results, results, f"{output_dir}/response_matrix_{model_name}_{parts[4]}_{parts[2]}.png")

