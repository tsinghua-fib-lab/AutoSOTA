import numpy as np
np.random.seed(42)
import pickle
import os
import pandas as pd
import json
from huggingface_hub import HfApi
from huggingface_hub import login

def get_solution_multi_choice(row):
    # Parse the references to find the correct answer text
    refs = json.loads(row["instance.references"])
    correct_text = None
    for ref in refs:
        if "correct" in ref.get("tags", []):
            correct_text = ref["output"]["text"]
            break
    if correct_text is None:
        raise ValueError(f"No correct solution found for row with index {row.name}")

    # Dynamically identify all output_mapping columns (e.g., output_mapping.A, output_mapping.B, etc.)
    choice_columns = [col for col in row.index if col.startswith("output_mapping.")]

    # Iterate through the available choices to find the matching letter
    for col in choice_columns:
        if row[col] == correct_text:
            return col.split(".")[-1]  # Extract the letter (e.g., 'A' from 'output_mapping.A')

    raise ValueError(f"Correct answer text not found among choices for row with index {row.name}")

def get_solution_exact_match(row):
    # Parse the references to find the correct answer text
    refs = json.loads(row["instance.references"])
    correct_text = None
    for ref in refs:
        if "correct" in ref.get("tags", []):
            correct_text = ref["output"]["text"]
            break
    if correct_text is None:
        raise ValueError(f"No correct solution found for row with index {row.name}")

    return correct_text


if __name__ == "__main__":
    model_names = ["meta/llama-3-8b"] # , 'eleutherai/pythia-6.9b', 'mistralai/mistral-7b-v0.1', 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'Qwen/Qwen3-8B']
    for model_name in model_names:
        benchmark_scenarios = {
            "lite": ["gsm"] # "commonsense", "med_qa", "legalbench", "math", 
            # "mmlu": ["mmlu"],
            # "classic": ["bbq", "lsat_qa"] # "legal_support",
            # "classic": ["legal_support"]
        }

        for benchmark_name, scenario_list in benchmark_scenarios.items():
            for scenario_name in scenario_list:
                print(scenario_name)
                mult_choice_flag = False if scenario_name in ["legalbench", "math", "gsm"] else True
                
                with open(f"../../data/gather_helm_data/responses_monkey_{benchmark_name}.pkl", "rb") as f:
                    results_full = pickle.load(f)
                
                results_full = results_full.sample(frac=1).reset_index(drop=True)
                all_choice_columns = [col for col in results_full.columns if col.startswith("output_mapping.")]
                scenario_mask = results_full["scenario"] == scenario_name
                choice_columns = [
                    col for col in all_choice_columns
                    if not results_full.loc[scenario_mask, col].isna().all()
                ]
                results = results_full[
                    ["request.model", "instance.input.text", "request.prompt", "instance.references", "scenario", "benchmark", "dicho_score"] + choice_columns
                ]
                results = results.dropna(subset=["request.model", "instance.input.text", "request.prompt", "instance.references", "scenario", "benchmark", "dicho_score"])
                
                # drop the dicho_score of 0.5
                results = results[results["dicho_score"] != 0.5]
                results["dicho_score"] = results["dicho_score"].astype(bool)
                assert results["dicho_score"].isin([0, 1]).all()
                
                # drop duplicate rows
                results = results.drop_duplicates(subset=["request.model", "instance.input.text", "scenario", "benchmark"], keep='first')
                print(f"non-duplicate percentage:{results.shape[0]/results_full.shape[0]}")
                
                # Count the number of unique instance.input.text for each request.model
                model_prompt_counts = results.groupby('request.model', observed=True)['instance.input.text'].nunique()
                # Count the number of unique request.model for each instance.input.text
                prompt_model_counts = results.groupby('instance.input.text', observed=True)['request.model'].nunique()
                # Identify models with at least 30 unique prompts and prompts with at least 30 unique models
                models_to_keep = model_prompt_counts[model_prompt_counts >= 30].index
                prompts_to_keep = prompt_model_counts[prompt_model_counts >= 30].index
                # Filter the DataFrame accordingly
                results = results[
                    results['request.model'].isin(models_to_keep) &
                    results['instance.input.text'].isin(prompts_to_keep)
                ]

                # filter out one scenario, delete all 0 or all 1 cols
                results = results[
                    (results["benchmark"] == benchmark_name) &
                    (results["scenario"] == scenario_name)
                ]
                results["dicho_score"] = results["dicho_score"].astype(int)
                results = results.groupby(["instance.input.text", "scenario", "benchmark"], observed=False).filter(lambda grp: grp["dicho_score"].nunique() > 1)
                
                # TODO: read helm config and automatically find
                # filter model row, make sure number of few shot of the request fit in the model's max input token length
                if model_name == "meta/llama-3-8b":
                    if benchmark_name=="classic":
                        # classic does not have llama-3-8b, this model also has 8192 context length, same as llama-3-8b
                        context_model_name = "anthropic/stanford-online-all-v4-s3"
                    else:
                        context_model_name = model_name
                elif model_name == "eleutherai/pythia-6.9b":
                    if benchmark_name=="classic":
                        context_model_name = "AlephAlpha/luminous-base" # 2048
                    elif benchmark_name=="lite":
                        context_model_name = "AlephAlpha/luminous-base" # 2048
                    elif benchmark_name=="mmlu":
                        context_model_name = "microsoft/phi-2" # 2048
                        
                # we_query_2, start with gsm
                elif model_name == "mistralai/mistral-7b-v0.1":
                    if scenario_name == "bbq":
                        context_model_name = "AlephAlpha/luminous-extended" # 2048
                    else:
                        context_model_name = model_name
                elif model_name == 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B': # 131072
                    if benchmark_name=="classic":
                        context_model_name = "anthropic/stanford-online-all-v4-s3" # 8192
                    else:
                        context_model_name = "openai/gpt-4o-2024-08-06" # 128000
                elif model_name == 'Qwen/Qwen3-8B': # 40960
                    if benchmark_name=="classic":
                        context_model_name = "anthropic/stanford-online-all-v4-s3" # 8192
                    else:
                        context_model_name = 'qwen/qwen1.5-7b' # 32768
                
                results = results[(results["request.model"] == context_model_name)]
                print(len(results))
                
                if mult_choice_flag:
                    # get solution column, should be ABCD
                    results["solution"] = results.apply(get_solution_multi_choice, axis=1)
                else:
                    results["solution"] = results.apply(get_solution_exact_match, axis=1)
                    
                # save result
                pre_query = results.copy()
                if scenario_name == "legal_support":
                    pre_query = pre_query[["instance.input.text", "request.prompt", "instance.references", "solution", "scenario", "benchmark"]]
                else:
                    pre_query = pre_query[["instance.input.text", "request.prompt", "solution", "scenario", "benchmark"]]
                pre_query = pre_query.sort_values('instance.input.text').reset_index(drop=True)
                pre_query['prompt_index'] = range(len(pre_query))
                print(pre_query.head())
                row = pre_query.iloc[0]
                for col in pre_query.columns:
                    print(f"{col}: {row[col]}\n\n")
                
                out_dir = '../../data/monkey_query/pre_query'
                os.makedirs(out_dir, exist_ok=True)
                safe_model_name = model_name.replace('/', '_')
                out_path = f"{out_dir}/{safe_model_name}_{scenario_name}_pre_query.pkl"
                with open(out_path, 'wb') as f:
                    pickle.dump(pre_query, f)

                login()
                api = HfApi()
                api.upload_file(
                    path_or_fileobj=out_path,
                    path_in_repo=out_path.split('/')[-1],
                    repo_id="stair-lab/monkey_query_pre",
                    repo_type="dataset",
                )
                    