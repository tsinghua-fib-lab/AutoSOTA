import os
import json
import pickle
import pandas as pd

ids_to_remove = [1, 3, 5, 7, 9]

def get_raw_data(raw_data_dir, data_name):
    path = os.path.join(raw_data_dir, data_name + ".json")
    with open(path, "r") as f:
        raw_data = json.load(f)
    raw_data = [item for idx, item in enumerate(raw_data) if idx not in ids_to_remove]
    print(f"[loaded] {len(raw_data)} rows from {path}")
    return raw_data

def _safe_load_logits(path):
    if not os.path.exists(path):
        print(f"[skip] missing logits: {path}")
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return [it for idx, it in enumerate(obj) if idx not in ids_to_remove]

def get_logits_data(model_name, data_name, data, old_domain, new_domain,
                    logits_data_dir, prompt_methods, icl_methods):
    logits_data_all = {}
    old_ids = {item["id"] for item in data if item["subcategory"] == old_domain}
    new_ids = {item["id"] for item in data if item["subcategory"] == new_domain}
    any_loaded = False

    for m in prompt_methods:
        for fs in icl_methods:
            logits_file = os.path.join(
                logits_data_dir, f"{model_name}_{data_name}_{m}_{fs}.pkl"
            )
            logits_data = _safe_load_logits(logits_file)
            if logits_data is None:
                continue
            any_loaded = True
            logits_data_all[m + "_" + fs] = {
                "cal":  [it for it in logits_data if it["id"] in old_ids],
                "test": [it for it in logits_data if it["id"] in new_ids],
            }

    if not any_loaded:
        raise FileNotFoundError(f"No logits found for model={model_name} in {logits_data_dir}")
    return logits_data_all


# 5. Metrics & small utilities
def convert_id_to_ans(test_raw_data):
    return {str(row["id"]): row["answer"] for row in test_raw_data}

def cal_coverage(pred_sets_all, test_id_to_answer, prompt_methods, icl_methods):
    coverage_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            key = m + "_" + fs
            if key not in pred_sets_all or not pred_sets_all[key]:
                continue
            cover = [int(test_id_to_answer.get(k) in v)
                     for k, v in pred_sets_all[key].items() if k in test_id_to_answer]
            coverage_all[key] = sum(cover) / max(len(cover), 1)
    return coverage_all

def cal_set_size(pred_sets_all, prompt_methods, icl_methods):
    set_sizes = {}
    for m in prompt_methods:
        for fs in icl_methods:
            key = m + "_" + fs
            if key not in pred_sets_all or not pred_sets_all[key]:
                continue
            sizes = [len(v) for v in pred_sets_all[key].values()]
            set_sizes[key] = sum(sizes) / max(len(sizes), 1)
    return set_sizes

