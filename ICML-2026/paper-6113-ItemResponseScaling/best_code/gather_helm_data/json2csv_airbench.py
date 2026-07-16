import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import exists

lo = lambda x: json.load(open(x, "r"))

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
    dataset_name = "thaiexam" #"air-bench"
    
    
    input_dir = "/lfs/skampere1/0/sttruong/reeval/gather_helm_data/helm_jsons"
    BENCHMARKS = [dataset_name]
    task2metric = lo("task2metric.json")
    task2metric = pd.json_normalize(task2metric)

    all_paths = []
    for benchmark in BENCHMARKS:
        dir_path = f"{input_dir}/{benchmark}/releases"
        assert exists(dir_path)
        latest_release = sorted(os.listdir(dir_path))[-1]
        folder_dict = lo(f"{dir_path}/{latest_release}/runs_to_run_suites.json")
        all_paths += [f"{input_dir}/{benchmark}/runs/{s}/{r}" for r, s in folder_dict.items()]

    files = ["display_requests.json", "display_predictions.json", "run_spec.json", "instances.json"]
    all_paths = [p for p in tqdm(all_paths) if all([exists(f"{p}/{f}") for f in files])]
    all_lists = [[lo(f"{p}/{f}") for p in tqdm(all_paths)] for f in files]

    results = []
    for d_requests, d_predictions, run_specs, instances, paths in tqdm(zip(*all_lists, all_paths), total=len(all_lists[0])):
        d_requests = pd.json_normalize(d_requests)
        d_predictions = pd.json_normalize(d_predictions)
        run_specs = pd.json_normalize(run_specs)
        instances = pd.json_normalize(instances)
        
        benchmark = paths.split("/")[8]
        run_specs["benchmark"] = benchmark
        run_specs = run_specs.loc[run_specs.index.repeat(d_predictions.shape[0])].reset_index(drop=True)
        
        result = pd.concat([d_requests, d_predictions, run_specs, instances], axis=1)
        result = result.loc[:, ~result.columns.duplicated()]
        
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
    
    output_dir = "../data/gather_helm_data"
    os.makedirs(output_dir, exist_ok=True)
    results.to_pickle(f"{output_dir}/responses_{dataset_name}.pkl")
