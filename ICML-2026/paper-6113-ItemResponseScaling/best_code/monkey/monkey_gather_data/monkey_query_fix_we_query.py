from tqdm import tqdm
import pickle
import pandas as pd
import pathlib
from huggingface_hub import snapshot_download
import sys
sys.path.append("../monkey_query")
from pyarrow.lib import ArrowInvalid
from monkey_query_utils import (
    exact_match,
    quasi_exact_match,
    model_nickname2helm_model_name,
    final_number_exact_match,
    is_equiv_chain_of_thought,
)

model_nickname2helm_model_name = {
    full.split("/")[-1]: helm_name
    for full, helm_name in model_nickname2helm_model_name.items()
}

# Base directories
base_eval_dir = pathlib.Path("../../data/monkey_query/eval_results")
base_fix_dir = pathlib.Path("../../data/monkey_query/eval_results_fix")

# Download pre-query dataset once for all scenarios
cache_dir = snapshot_download(
    repo_id="stair-lab/monkey_query_pre",
    repo_type="dataset",
)

# Load mapping from prompt_idx → solution for each scenario
# We’ll do this once per scenario (before iterating models)
# Loop over scenarios and models
for scenario_dir in sorted(base_eval_dir.iterdir()):
    if not scenario_dir.is_dir():
        continue
    scenario_name = scenario_dir.name

    if scenario_name in ["mmlu", "commonsense"]:
        evaluate_fn = exact_match
    elif scenario_name in ["med_qa", "legalbench", "bbq", "lsat_qa", "legal_support"]:
        evaluate_fn = quasi_exact_match
    elif scenario_name == "gsm":
        evaluate_fn = final_number_exact_match
    elif scenario_name == "math":
        evaluate_fn = is_equiv_chain_of_thought
    else:
        print(f"Skipping unknown dataset: {scenario_name}")
        continue

    # Iterate over each model directory
    for model_dir in sorted(scenario_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        print(f"\nGathering results for {scenario_name}/{model_name}...")
        
        # Load the pre-query DataFrame for this scenario
        helm_model = model_nickname2helm_model_name[model_name]
        prequery_path = pathlib.Path(cache_dir) / f"{helm_model.replace('/', '_')}_{scenario_name}_pre_query.pkl"
        with open(prequery_path, 'rb') as f:
            pre_df = pickle.load(f)
        sol_map = dict(zip(pre_df['prompt_index'], pre_df['solution']))

        input_root = model_dir
        output_root = base_fix_dir / scenario_name / model_name

        # Process each batch file
        for src_path in tqdm(sorted(input_root.rglob("batch_*.parquet"))):
            rel = src_path.relative_to(input_root)
            dst_path = output_root / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            if dst_path.exists():
                continue

            try:
                df = pd.read_parquet(src_path)
            except ArrowInvalid as e:
                print(f"Skipping `{src_path.parent.name}` due to read error: {e}")
                continue
            
            # Keep only the text before the first newline
            df['response'] = df['response'].str.split("\n", n=1).str[0]
            # Recompute the score based on the original solution
            df['score'] = df.apply(
                lambda row: float(evaluate_fn(row['response'], sol_map[row['prompt_idx']])),
                axis=1,
                )
            if scenario_name == "legal_support":
                ref_lookup = pre_df.set_index("prompt_index")["instance.references"]
                mapped_refs = df["prompt_idx"].map(ref_lookup).astype(str)
                df["problem"] = df["problem"] + mapped_refs

            df.to_parquet(dst_path, index=False)
