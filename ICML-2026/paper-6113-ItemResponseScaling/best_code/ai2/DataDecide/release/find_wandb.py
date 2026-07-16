import wandb
import os
from typing import List, Dict
import json
import pathlib
from tqdm import tqdm

# Configuration
ENTITY = "ai2-llm"
API_KEY = os.getenv("WANDB_API_KEY")

# Load checkpoint_specs from JSONL file
jsonl_path = pathlib.Path(__file__).parent.parent / "checkpoints" / "weka_paths.jsonl"
checkpoint_specs: List[Dict] = []
with open(jsonl_path, "r") as f:
    for line in f:
        checkpoint_specs.append(json.loads(line))

# Normalize checkpoint paths and build lookup
s3_targets = {
    spec["checkpoints_location"].replace("weka://oe-eval-default/", "s3://"): {
        "model_name": spec["model_name"],
        "runs": []
    }
    for spec in checkpoint_specs
}

# Connect to W&B
api = wandb.Api()
projects = ['olmo-ladder-benb', 'olmo-ladder-ianm', 'olmo-ladder-davidh']

print(f"Searching across {len(projects)} projects for {len(s3_targets)} checkpoint paths...\n")

# Scan runs
for project in projects:
    print(f"🔎 Project: {project}")
    try:
        runs = api.runs(f"{ENTITY}/{project}")
        for run in tqdm(runs, desc=f"Processing {project}", unit="run"):
            remote_path = run.config.get("remote_save_folder")
            if remote_path and remote_path in s3_targets:
                s3_targets[remote_path]["runs"].append({
                    "run_id": run.id,
                    "run_name": run.name,
                    "project": run.project,
                    "group": run.group,
                    "url": run.url
                })
    except Exception as e:
        print(f"⚠️ Error accessing project {project}: {e}")

# Write output
out_path = "matched_runs_by_s3.json"
with open(out_path, "w") as f:
    json.dump(s3_targets, f, indent=2)


# Check there's one and only one group per model
for s3_target, data in s3_targets.items():
    groups = {run["group"] for run in data["runs"] if "group" in run}
    if len(groups) > 1:
        print(f"⚠️ Multiple groups found for {s3_target}: {groups}")
    elif len(groups) == 0:
        print(f"⚠️ No group found for {s3_target}")


import json
import wandb
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Configuration
INPUT_JSON = "matched_runs_by_s3.json"
OUTPUT_CSV = "perplexity_metrics_by_group.csv"
METRICS = [
    "eval/wikitext_103-validation/Perplexity",
    "eval/pile-validation/Perplexity",
    "eval/m2d2_s2orc-validation/Perplexity",
    "eval/ice-validation/Perplexity",
    "eval/dolma_wiki-validation/Perplexity",
    "eval/dolma_stack-validation/Perplexity",
    "eval/dolma_reddit-validation/Perplexity",
    "eval/dolma_pes2o-validation/Perplexity",
    "eval/dolma_common-crawl-validation/Perplexity",
    "eval/dolma_books-validation/Perplexity",
    "eval/c4_en-validation/Perplexity"
]

# Load run metadata
with open(INPUT_JSON, "r") as f:
    run_data = json.load(f)

api = wandb.Api()
results = []

# Iterate over all runs in all entries
for s3_path, entry in tqdm(run_data.items(), desc="Processing checkpoints"):
    for run_info in entry["runs"]:
        try:
            run = api.run(f"{ENTITY}/{run_info['project']}/{run_info['run_id']}")
            history = run.history(keys=METRICS,samples=5000)  # Adjust if needed
            
            for _, row in history.iterrows():
                # print((k for k in row.keys() if k.startswith("eval/")))
                # raise Exception("Debugging row output")
                result = {
                    "group": run_info["group"],
                    "step": row.get("global_step", row.get("_step", None)),
                }
                for metric in METRICS:
                    result[metric] = row.get(metric)
                results.append(result)
        except Exception as e:
            print(f"⚠️ Failed to fetch metrics for {run_info['run_id']}: {e}")

# Save as CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Metrics saved to: {OUTPUT_CSV}")

