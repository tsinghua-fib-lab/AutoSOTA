import glob
from my_paths import SUBMIT_HOST, CONDA_PATH, CONDA_ENV, REMOTE_SCRIPT_PATH, output_dir, HISTORY_FILE, RUNNING_EXPERIMENTS_FILE, path_to_saved_model, CACHE_DIR
from pathlib import Path
import json
import re
import yaml
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm


sys.path.append(os.path.abspath(".."))
from convert_to_code import convert_config_to_code_full_paths, convert_config_to_code_one_step_paths, convert_config_to_code_qk_pruned, convert_keys_to_int
from patching_data import get_tokenizer_and_dataset_for_task
from draw_curve_utils import get_pareto_frontier, get_num_element
from show_heatmap import *
from patching_helper_functions import get_full_possible_config_for_pruning
from pruning_model import PruningModelWithHooksForQK, convert_config_fullpaths_, convert_config_fullpaths_incl_mlp_, convert_full_paths_config_to_prune_inside_kq_config_




def read_data_points(path, selected_stage):
    series_path = Path(output_dir) / path
    data_points = []
    for exp_name in glob.glob("exp*", root_dir=series_path):
        if (series_path / exp_name / "output.json").exists():
            with open(series_path / exp_name / "output.json") as f:
                output_dict = json.load(f)
            stages = []
            for key in output_dict:
                if match := re.search(r"result_patching_config_global_iteration_(\d+)", key):
                    stages.append(int(match.group(1)))
            if max(stages) == 0:
                with open(series_path / exp_name / "config.yaml") as f:
                    linear_LN = yaml.safe_load(f).get("linear_LN", True)
                if linear_LN:
                    exp_stage = 1
                else:
                    exp_stage = 0
            else:
                exp_stage = max(stages)+1
            if exp_stage != selected_stage:
                continue
        else:
            continue

        final_config = output_dict[f"result_patching_config_global_iteration_{max(stages)}"]
        num_edges = get_num_element(final_config)
        acc_match = output_dict["acc_match"]
        data_points.append((acc_match, num_edges, exp_name))
    
    if len(data_points) == 0:
        return [], float("inf")

    num_heads_per_layer = {layer_idx: len(final_config[str(layer_idx)]["k"]) for layer_idx in range(len(final_config)-1)}
    with open(series_path / exp_name / "config.yaml") as f:
        split_mlps = yaml.safe_load(f)["split_mlps"]
    full_config = get_full_possible_config_for_pruning(num_heads_per_layer)
    if selected_stage >= 2:
        if split_mlps:
            convert_config_fullpaths_incl_mlp_(full_config, num_heads_per_layer)
        else:
            convert_config_fullpaths_(full_config, num_heads_per_layer)
    if selected_stage >= 3:
        convert_full_paths_config_to_prune_inside_kq_config_(full_config, num_heads_per_layer)
    orig_num_edges = get_num_element(full_config)

    return data_points, orig_num_edges

def get_length_gen_performance(model_name):
    original_results = Path(path_to_saved_model) / model_name / "acc.txt" #f"{model_name.split('-')[0]}-average.txt"
    assert original_results.exists()
    with open(original_results) as f:
        lines = f.read()    # 2l1h256d4lr00drop		 eval_len0-50_acc: 1.0		eval_len51-100_acc: 1.0		eval_len101-150_acc: 0.995
        items = [item.strip() for item in lines.strip().split("\t") if item]
        kvs = filter(lambda x: x.startswith("eval") or x.startswith("step"), items)
        results = {}
        for kv in kvs:
            k, v = kv.split(":")
            v = float(v.strip())
            results[k] = v
    return results


if __name__ == "__main__":
    paths = glob.glob("newprune-unique_bigram_copy-^*", root_dir=output_dir) + glob.glob("newprune-unique_bigram_copy-@*", root_dir=output_dir)
    model_to_step = {"unique_copy-@2l1h64d3lr01drop": 3000, "unique_bigram_copy-@2l4h256d3lr01drop": 6000}  # can obtained by rerunning, as the seed is fixed
    acc_curves = []
    num_edges_curves = []
    for path in tqdm(paths):
        
        model_name = "-".join(path.split("-")[1:-1])
        results = get_length_gen_performance(model_name)
        if "@" in path:
            step = model_to_step[model_name]
        else:
            step = results["step"]
        acc_curves.append((step, (results["eval_len0-50_acc"], results["eval_len51-100_acc"], results["eval_len101-150_acc"])))
        
        # data_points, orig_num_edges = read_data_points(path, selected_stage=0)
        # min_edges_with_LN = 1e10
        # for acc_match, num_edges, exp_name in data_points:
        #     if acc_match >= 0.9:
        #         min_edges_with_LN = min(min_edges_with_LN, num_edges)
        # if min_edges_with_LN != 1e10:
        #     min_edges_with_LN /= orig_num_edges
        # else:
        #     min_edges_with_LN = None

        data_points, orig_num_edges = read_data_points(path, selected_stage=1)
        min_edges_without_LN = 1e10
        for acc_match, num_edges, exp_name in data_points:
            if acc_match >= 0.9:
                min_edges_without_LN = min(min_edges_without_LN, num_edges)
        if min_edges_without_LN != 1e10:
            min_edges_without_LN /= orig_num_edges
        else:
            min_edges_without_LN = None
        
        num_edges_curves.append((step, min_edges_without_LN))

    print(acc_curves)
    print(num_edges_curves)
