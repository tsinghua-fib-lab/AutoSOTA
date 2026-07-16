from pathlib import Path
import json
from collections import defaultdict
from huggingface_hub import HfApi
from huggingface_hub import login
from tqdm import tqdm
import csv

def process_and_upload_data(data_dir: Path, repo_id: str, behavior_map):
    api = HfApi()
    pattern = "_text_t1.0_n"
    
    for jsonl_path in tqdm(data_dir.glob(f"*{pattern}*.jsonl")):
        # Derive model_name from filename
        model_name = jsonl_path.name.split(pattern)[0]
        output_path = data_dir / f"{model_name}_harm_bench.json"
        
        # Group flagged values by behavior_id
        grouped = defaultdict(list)
        with jsonl_path.open("r") as f:
            for line in f:
                rec = json.loads(line)
                grouped[rec["behavior_id"]].append(rec["flagged"])
        
        # Build output structure
        output_data = [
            {"question": behavior_map[qid], "is_corrects": flags}
            for qid, flags in grouped.items()
        ]
        
        # Write JSON
        with output_path.open("w") as out_f:
            json.dump(output_data, out_f, indent=2)
        
        # Upload to HF dataset
        api.upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo=output_path.name,
            repo_id=repo_id,
            repo_type="dataset",
        )

if __name__ == "__main__":
    login()
    data_directory = Path("/lfs/skampere1/0/sttruong/deval/data/best_of_n_jailbreaking_data")
    huggingface_repo = "stair-lab/monkey_queries"
    
    csv_path = Path("/lfs/skampere1/0/sttruong/deval/data/best_of_n_jailbreaking_data/harmbench_behaviors_text_all.csv")
    behavior_map = {}
    with csv_path.open("r", encoding="utf-8") as cf:
        reader = csv.DictReader(cf)
        for row in reader:
            behavior_map[row["BehaviorID"]] = row["Behavior"]
        
    process_and_upload_data(data_directory, huggingface_repo, behavior_map)
