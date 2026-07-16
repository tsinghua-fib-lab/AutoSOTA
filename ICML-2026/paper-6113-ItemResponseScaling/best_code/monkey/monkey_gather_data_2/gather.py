from pathlib import Path
import numpy as np
import torch
import pandas as pd
from pyarrow.lib import ArrowInvalid
from huggingface_hub import snapshot_download
from tqdm import tqdm
from huggingface_hub import HfApi, login

# exclude_models = ["Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Phi-4-mini-instruct"]
# exclude_models = [
#     "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Phi-4-mini-instruct", 
#     "Mistral-Small-3.2-24B-Instruct-2506", "gemma-3-12b-it", "Llama-3.3-70B-Instruct"
# ]
exclude_models = [
    "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3", "Phi-4-mini-instruct", 
    "Mistral-Small-3.2-24B-Instruct-2506", "gemma-3-12b-it", "Llama-3.3-70B-Instruct",
    "Phi-4-mini-reasoning", "Phi-4-reasoning-plus", "SmolLM3-3B", "gemma-3-4b-it", "OLMo-2-1124-7B-Instruct", "OLMo-2-1124-13B-Instruct", "OLMo-2-0325-32B-Instruct",
]
exclude_qpairs = [
    ('mmlu_pro', 'prompt=106'),
    ('mmlu_pro', 'prompt=11438'),
    ('mmlu_pro', 'prompt=1519'),
    ('mmlu_pro', 'prompt=1584'),
    ('mmlu_pro', 'prompt=1674'),
    ('mmlu_pro', 'prompt=2547'),
    ('mmlu_pro', 'prompt=2615'),
    ('mmlu_pro', 'prompt=3527'),
    ('mmlu_pro', 'prompt=4333'),
    ('mmlu_pro', 'prompt=4552'),
    ('mmlu_pro', 'prompt=4557'),
    ('mmlu_pro', 'prompt=5514'),
    ('mmlu_pro', 'prompt=5574'),
    ('mmlu_pro', 'prompt=5635'),
    ('mmlu_pro', 'prompt=5881'),
    ('mmlu_pro', 'prompt=6224'),
    ('mmlu_pro', 'prompt=6924'),
    ('mmlu_pro', 'prompt=711'),
    ('mmlu_pro', 'prompt=9654'),
    ('mmlu_pro', 'prompt=9891'),
]

if __name__ == "__main__":
    output_dir = Path("../../data/query/3d_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "resmat.pt"
    max_samples = 2560
    base_eval_dir = Path(snapshot_download(
        repo_id="stair-lab/denoise_eval_query",
        repo_type="dataset"
    ))

    # gather all scenario dirs
    scenario_dirs = sorted(
        d for d in base_eval_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    )

    # build union of all model names
    model_set = set()
    for sd in tqdm(scenario_dirs):
        for mdir in sd.iterdir():
            if mdir.is_dir() and mdir.name not in exclude_models:
                model_set.add(mdir.name)
    model_names = sorted(model_set)
    n_models = len(model_names)

    # build union of all (scenario, prompt) pairs
    qpairs = set()
    for sd in tqdm(scenario_dirs):
        scen = sd.name
        for m in model_names:
            p_base = sd / m
            for pdir in p_base.glob("prompt=*"):
                batch_files = list(pdir.glob("batch_*.parquet"))
                # only keep this prompt if at least one batch file exists and is non‑empty
                if any(f.is_file() and f.stat().st_size > 0 for f in batch_files) and (scen, pdir.name) not in exclude_qpairs:
                    qpairs.add((scen, pdir.name))
    qpairs = sorted(qpairs)  # list of (scenario_name, prompt_name)
    n_questions = len(qpairs)

    # allocate 3D array filled with NaN
    data_tensor = np.full(
        (n_models, n_questions, max_samples),
        np.nan,
        dtype=float
    )

    # fill in scores
    for i, m in tqdm(enumerate(model_names)):
        for j, (scen, prompt) in enumerate(qpairs):
            pdir = base_eval_dir / scen / m / prompt
            scores = []
            for bf in sorted(pdir.glob("batch_*.parquet")):
                try:
                    df = pd.read_parquet(bf, columns=["score"])
                except ArrowInvalid:
                    continue
                scores.extend(df["score"].tolist())
            L = len(scores)
            if L > 0:
                data_tensor[i, j, :L] = scores

    # gather question texts and dataset labels
    questions = []
    datasets  = []
    for scen, prompt in qpairs:
        sd = base_eval_dir / scen
        # find first model that has this prompt
        for m in model_names:
            pdir = sd / m / prompt
            if pdir.is_dir():
                batches = [
                    f for f in sorted(pdir.glob("batch_*.parquet"))
                    if f.is_file() and f.stat().st_size > 0
                ]
                if not batches:
                    continue
                bf0 = batches[0]
                q_text = pd.read_parquet(bf0, columns=["problem"])["problem"].iat[0]
                questions.append(q_text)
                datasets.append(scen)
                break
    
    total_elements = np.prod(data_tensor.shape)
    nan_count = np.isnan(data_tensor).sum()
    nan_percentage = (nan_count / total_elements) * 100
    print(f"NaN percentage: {nan_percentage:.2f}%, NaN count: {nan_count}")
    
    # save to disk
    torch.save({
        "data_tensor": torch.from_numpy(data_tensor),
        "models":      model_names,
        "questions":   questions,
        "datasets":     datasets,
    }, out_path)

    print(f"shape {data_tensor.shape}")

    login()
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo="resmat.pt",
        repo_id="stair-lab/irsl_testtime_resmat2",
        repo_type="dataset",
    )