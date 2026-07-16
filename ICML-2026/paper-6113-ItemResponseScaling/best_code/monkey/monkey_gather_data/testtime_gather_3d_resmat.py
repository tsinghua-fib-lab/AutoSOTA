from huggingface_hub import HfApi, login
import glob
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
from itertools import chain
from collections import defaultdict

BENCHMARK2SCENARIO = {
    "lite": ["legalbench", "commonsense", "med_qa"], # "math", "gsm"
    # "mmlu": ["mmlu"],
    "classic": ["bbq", "lsat_qa", "legal_support"],
}
SCENARIO2BENCHMARK = {
    **{
        scen: bench
        for bench, scens in BENCHMARK2SCENARIO.items()
        for scen in scens
    },
}
SCENARIOS = sorted(SCENARIO2BENCHMARK.keys())

if __name__ == "__main__":
    max_samples = 10000
    
    cache_dir = snapshot_download(repo_id="stair-lab/monkey_queries", repo_type="dataset")
    all_paths = []
    for scen in SCENARIOS:
        for path in glob.glob(f"{cache_dir}/*{scen}.json"):
            fname = Path(path).name
            if not (fname.startswith("Mistral") or fname.startswith("pythia")):
                all_paths.append(path)

    # build union of all model names
    model_set = set()
    for path in all_paths:
        stem = Path(path).stem
        scen = next((s for s in SCENARIOS if stem.endswith(f"_{s}")), None)
        model = stem.split(f"_{scen}")[0]
        model_set.add(model)
    model_names = sorted(model_set)
    n_models = len(model_names)
    print(n_models, model_names)

    # build intersection of all (benchmark, scenario, prompt) pairs
    qpairs = []
    for scen in SCENARIOS:
        bench = SCENARIO2BENCHMARK[scen]
        scen_paths = [p for p in all_paths if Path(p).stem.endswith(f"_{scen}")]
        common_qs = None
        for path in scen_paths:
            df = pd.read_json(path)
            qs = {
                q for q, rs in zip(df["question"].tolist(), df["is_corrects"].tolist())
                if rs
            }
            common_qs = qs if common_qs is None else (common_qs & qs)
            if not common_qs:
                break  # early exit if intersection becomes empty
        if common_qs:
            qpairs.extend((bench, scen, q) for q in sorted(common_qs))
    
    # load z from helm_resmat
    with open(f"{cache_dir}/results_with_z.pkl", "rb") as f:
        helm_resmat = pickle.load(f)
    cols = helm_resmat.columns
    helm_benchs = cols.get_level_values("benchmark")
    helm_scens = cols.get_level_values("scenario")
    helm_qs = cols.get_level_values("input.text").astype(str)
    helm_refs = cols.get_level_values("references").astype(str)
    helm_zs = cols.get_level_values("z")
    qpairs_with_zs = set()
    for bench, scen, q in qpairs:
        mask = (helm_benchs == bench) & (helm_scens == scen)
        if scen == "legal_support":
            mask &= (helm_qs + helm_refs == q)
        else:
            mask &= (helm_qs == q)
        matched_z = helm_zs[mask]
        if len(matched_z) >= 1:
            qpairs_with_zs.add((bench, scen, q, np.mean(matched_z.to_numpy())))
            if len(matched_z) > 1:
                print("warning, avg z")
        else:
            continue
    qpairs_with_zs = sorted(qpairs_with_zs)
    n_questions = len(qpairs_with_zs)
    print(n_questions)
    print(max_samples)
    
    # allocate 3D array filled with NaN
    data_tensor = np.full(
        (n_models, n_questions, max_samples),
        np.nan,
        dtype=float
    )

    # fill in scores
    output_benchs, output_scens, output_qs, output_zs = [], [], [], []
    for i, m in tqdm(enumerate(model_names), total=n_models):
        for j, (bench, scen, q, z) in enumerate(qpairs_with_zs):
            if i == 0:
                output_benchs.append(bench)
                output_scens.append(scen)
                output_qs.append(q)
                output_zs.append(z)
            path = f"{cache_dir}/{m}_{scen}.json"
            df_monkey = pd.read_json(path)
            rows = df_monkey.loc[df_monkey["question"] == q, "is_corrects"].tolist()
            if rows:
                flat = list(chain.from_iterable(rows))   
                L = min(len(flat), max_samples)
                if L > 0:
                    data_tensor[i, j, :L] = np.asarray(flat[:L], dtype=float)
    assert len(output_benchs) == len(output_scens) == len(output_qs) == len(output_zs) == n_questions
        
    # save to disk
    out_path = Path("../../data/testtime_resmat1.pt")
    torch.save({
        "data_tensor": torch.from_numpy(data_tensor),
        "models": model_names,
        "benchmarks": output_benchs,
        "scenarios": output_scens,
        "questions": output_qs,
        "zs": output_zs
    }, out_path)
    
    # inspect data
    print(f"shape {data_tensor.shape}")
    total_elements = np.prod(data_tensor.shape)
    nan_count = np.isnan(data_tensor).sum()
    nan_percentage = (nan_count / total_elements) * 100
    print(f"NaN percentage: {nan_percentage:.2f}%, NaN count: {nan_count}")
    scenarios_unique = sorted(set(output_scens))
    for scen in scenarios_unique:
        idxs = [j for j, s in enumerate(output_scens) if s == scen]
        sub = data_tensor[:, idxs, :]
        print(f"{scen}: shape = {sub.shape}")
    
    # upload to HF
    login()
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo="testtime_resmat1.pt",
        repo_id="stair-lab/irsl_testtime_resmat1",
        repo_type="dataset",
    )