import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import re
from joblib import Parallel, delayed
from huggingface_hub import HfApi, login

lo = lambda x: json.load(open(x, "r"))

def is_loadable(path, filename):
    full = os.path.join(path, filename)
    if not os.path.exists(full):
        return False
    try:
        lo(full)
        return True
    except (json.JSONDecodeError, OSError):
        return False

def infer_column_types(df):
    for col in df.columns:
        try:
            unique_values = df[col].dropna().unique()
        except:
            df[col] = df[col].apply(lambda x: json.dumps(x))
            unique_values = df[col].dropna().unique()
        
        if set(unique_values).issubset({"True", "False", "0", "1"}):
            df[col] = df[col].map(lambda x: True if x in ["True", "1"] else False).astype("bool")
        elif np.all(~pd.isna(pd.to_numeric(unique_values, errors="coerce"))):
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")
        elif df[col].nunique() / len(df) < 0.1:
            df[col] = df[col].astype("string").astype("category")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    # parser.add_argument("--repo_id", type=str, default="LLM360/Amber")
    # EleutherAI/pythia-12b, EleutherAI/pythia-6.9b, EleutherAI/pythia-2.8b
    # EleutherAI/pythia-1.4b, EleutherAI/pythia-1b, EleutherAI/pythia-410m
    # EleutherAI/pythia-160m, EleutherAI/pythia-70m, EleutherAI/pythia-14m
    # LLM360/Amber
    # HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints
    # HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints
    # HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints
    # parser.add_argument("--benchmark_dir", type=str, required=True)
    parser.add_argument("--benchmark_dir", type=str, default="../../helm/src/benchmark_output/runs")
    # /lfs/skampere1/0/yuhengtu/deval/helm/src/benchmark_output/runs
    # /lfs/skampere1/0/sttruong/helm/src/benchmark_output/runs
    # /lfs/skampere2/0/sttruong/helm/src/benchmark_output/runs
    # /lfs/hyperturing1/0/sttruong/helm/src/benchmark_output/runs
    # /lfs/hyperturing2/0/yuhengtu/helm/src/benchmark_output/runs
    args = parser.parse_args()
    task2metric = lo("task2metric.json")
    task2metric = pd.json_normalize(task2metric)
    BENCHMARKS = ["classic", "mmlu", "lite"]
    # BENCHMARKS = ["classic"]
    model_name = args.repo_id.split("/")[1]
    
    all_paths = []
    for benchmark in BENCHMARKS:
        dirs = [d for d in os.listdir(args.benchmark_dir) if d.startswith(benchmark) and d.split(f"{benchmark}_")[1].startswith(model_name)]
        for d in dirs:
            full_d_path = f"{args.benchmark_dir}/{d}"
            if os.path.isdir(full_d_path):
                subdirs = [
                    sub for sub in os.listdir(full_d_path)
                    if os.path.isdir(f"{full_d_path}/{sub}")
                    # and sub.startswith(("entity_matching", "legal_support", "raft", "civil_comments", "boolq", "imdb"))
                ]
                all_paths.extend([f"{full_d_path}/{sub}" for sub in subdirs])
    files = ["display_requests.json", "display_predictions.json", "run_spec.json", "instances.json", "per_instance_stats.json"]
    # all_paths = [p for p in tqdm(all_paths) if all([exists(f"{p}/{f}") for f in files])]
    # all_paths = [
    #     p for p in tqdm(all_paths)
    #     if all(is_loadable(p, f) for f in files)
    # ]
    def dir_ok(path):
        return all(is_loadable(path, f) for f in files)
    results = Parallel(n_jobs=-1)(
        delayed(dir_ok)(p) for p in tqdm(all_paths)
    )
    all_paths = [p for p, ok in zip(all_paths, results) if ok]
    all_lists = [[lo(f"{p}/{f}") for p in tqdm(all_paths)] for f in files]

    results = []
    for d_requests, d_predictions, run_specs, instances, pistats, paths in tqdm(zip(*all_lists, all_paths), total=len(all_lists[0])):
        d_requests = pd.json_normalize(d_requests)
        d_predictions = pd.json_normalize(d_predictions)
        run_specs = pd.json_normalize(run_specs)
        instances = pd.json_normalize(instances)
        pistats = pd.json_normalize(pistats)
        
        folder_name = paths.split("/")[-2]
        benchmark = folder_name.split("_")[0]
        if args.repo_id.startswith("EleutherAI/pythia"):
            n_step = folder_name.split("step")[-1]
        elif args.repo_id == "LLM360/Amber":
            # if "AmberChat" in folder_name or "AmberSafe" in folder_name:
            #     n_step = "Chat" if "AmberChat" in folder_name else "Safe"
            # else:
            n_step = folder_name.split("_")[-1]
        elif args.repo_id.startswith("HuggingFaceTB/SmolLM2"):
            regex = re.compile(r"step-(\d+)")
            n_step = regex.search(folder_name).group(1)
        else:
            raise RuntimeError("repo_id unknown")
        
        run_specs["benchmark"] = benchmark
        run_specs = run_specs.loc[run_specs.index.repeat(d_predictions.shape[0])].reset_index(drop=True)
        
        result = pd.concat([d_requests, d_predictions, run_specs, instances, pistats], axis=1)
        result = result.loc[:, ~result.columns.duplicated()]
        
        result["request.model"] = result["request.model"] + "-" + n_step
        result["scenario"] = result['name'].str.split(r'[:,]', n=1, expand=True)[0]
        result["scenario"] = result["scenario"].astype("category")
        result["benchmark"] = result["benchmark"].astype("category")
        assert result["scenario"].nunique() == 1
        
        metric_name = task2metric[f"{benchmark}.{result['scenario'].iloc[0]}"].iloc[0]
        if isinstance(metric_name, list):
            for metric_name_ in metric_name:
                dicho_score = result.get(f"stats.{metric_name_}", pd.NA)
                if dicho_score is not pd.NA:
                    if not dicho_score.isna().all():
                        result["dicho_score"] = dicho_score
                        break
        else:
            result["dicho_score"] = result.get(f"stats.{metric_name}", pd.NA)
        results.append(result)

    results = pd.concat(results, axis=0, join='outer')
    print("finished create results dataframe")
    infer_column_types(results)
    results.reset_index(drop=True, inplace=True)
    for col in results.columns:
        if results[col].dtype != "category" and results[col].isna().all():
            results = results.drop(columns=col)
        else:
            if results[col].dtype == "float64" and np.nanmax(results[col]) < 65500 and np.nanmin(results[col]) > -65500:
                results[col] = results[col].astype("float16")
                
    print("Started saving results")
    output_dir = "../data/pretrain_helm"
    os.makedirs(output_dir, exist_ok=True)
    parts = os.path.abspath(args.benchmark_dir).split("/")
    save_path = f"{output_dir}/responses_{model_name}_{parts[4]}_{parts[2]}.pkl"
    results.to_pickle(save_path)
    
    login(os.getenv("HF_TOKEN"))
    api = HfApi()
    api.upload_file(
        path_or_fileobj=save_path,
        path_in_repo=f"long/{save_path.split('/')[-1]}",
        repo_id="stair-lab/irsl_downstream_resmat1_fullinfo",
        repo_type="dataset",
    )