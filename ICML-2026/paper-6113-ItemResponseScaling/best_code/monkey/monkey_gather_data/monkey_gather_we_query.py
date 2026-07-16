import json
import pathlib
import pandas as pd
from tqdm import tqdm
from pyarrow.lib import ArrowInvalid
from huggingface_hub import HfApi
from huggingface_hub import login
import argparse

def gather_to_json(
    base_dir: pathlib.Path,
    output_path: pathlib.Path,
):
    records = []

    for prompt_folder in tqdm(sorted(base_dir.glob("prompt=*"))):
        try:
            # gather all batch files
            batch_files = sorted(prompt_folder.glob("batch_*.parquet"))
            if not batch_files:
                continue

            # read question text from the first batch file
            first_pq = batch_files[0]
            question_text = pd.read_parquet(
                first_pq, columns=["problem"]
            )["problem"].iat[0]

            # collect all scores
            scores = []
            for pq in batch_files:
                df = pd.read_parquet(pq, columns=["score"])
                scores.extend(df["score"].tolist())

            is_corrects = [float(s) for s in scores]
            
            if len(is_corrects) < 10000:
                continue

            records.append({
                "question": question_text,
                "is_corrects": is_corrects,
            })

        except ArrowInvalid as e:
            print(f"Skipping `{prompt_folder.name}` due to read error: {e}")
            continue

    # write out JSON
    print(len(records))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo=output_path.name,
        repo_id="stair-lab/monkey_queries",
        repo_type="dataset",
    )

if __name__ == "__main__":
    api = HfApi()
    login()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_eval_dir",
        required=True,
        type=str,
    ) 
    # "/lfs/skampere1/0/sttruong/deval/data/monkey_query/eval_results_fix"
    # "/lfs/skampere2/0/yuhengtu/irsl/data/monkey_query/eval_results_fix"
    args = parser.parse_args()
    base_eval_dir = pathlib.Path(args.base_eval_dir)
    output_root = pathlib.Path("data/monkey_query/gather_results")
    
    for scenario_dir in sorted(base_eval_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        scenario_name = scenario_dir.name
        print(f"\n{scenario_name}")

        for model_dir in sorted(scenario_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            print(f"\nGathering results for {scenario_name}/{model_name}...")

            output_file = output_root / f"{model_name}_{scenario_name}.json"
            gather_to_json(model_dir, output_file)
