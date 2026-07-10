import pickle 
import json
import os
import itertools
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import argparse
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

options = ["A", "B", "C", "D", "E", "F"]
ids_to_remove = [1, 3, 5, 7, 9] # remove data points that have been used as demonstration data
def get_raw_data(raw_data_dir, data_name):
    """
    Get raw data from the json file and split it into a calibration set and a test set.
    """
    raw_data = json.load(open(os.path.join(raw_data_dir, data_name+".json"), "r"))
    raw_data = [item for idx, item in enumerate(raw_data) if idx not in ids_to_remove]
    print(len(raw_data))
    return raw_data

def get_logits_data(model_name, data_name, data, old_domain, new_domain, logits_data_dir, prompt_methods, icl_methods):
    """
    Get logit scores of data instances and split these scores into a calibration set and a test set accordingly.
    """
    logits_data_all = {}
    old_ids = {item["id"] for item in data if item["subcategory"] == old_domain}
    new_ids = {item["id"] for item in data if item["subcategory"] == new_domain}
    for m in prompt_methods:
        for fs in icl_methods:
            logits_file = os.path.join(
                logits_data_dir,
                model_name + "_" + data_name + "_" + m + "_" + fs + ".pkl",
            )
            with open(logits_file, "rb") as f:
                logits_data = pickle.load(f)
            logits_data = [
                item for idx, item in enumerate(logits_data) if idx not in ids_to_remove
            ]
            old_domain_logits = [item for item in logits_data if item["id"] in old_ids]
            new_domain_logits = [item for item in logits_data if item["id"] in new_ids]
            logits_data_all[m + "_" + fs] = {}
            logits_data_all[m + "_" + fs]["cal"] = old_domain_logits
            logits_data_all[m + "_" + fs]["test"] = new_domain_logits
    return logits_data_all

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def LAC_CP(logits_data_all, cal_raw_data, prompt_methods, icl_methods, alpha=0.1):
    """
    Apply conformal prediction to obtain sets of predicted answers on each instance based on its softmax scores.
    Here the LAC score function is utilized.
    """
    pred_sets_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            pred_sets_all[m+"_"+fs] = {}
            cal_scores = []
            cal_logits_data = logits_data_all[m+"_"+fs]["cal"]
            for idx, row in enumerate(cal_logits_data):
                probs = softmax(row["logits_options"])
                truth_answer = cal_raw_data[idx]["answer"]
                assert cal_raw_data[idx]["id"] == row["id"]
                cal_scores.append(1 - probs[options.index(truth_answer)])
            # calculate the threshold qhat
            n = len(cal_logits_data)
            q_level = np.ceil((n+1) * (1-alpha)) / n
            qhat = np.quantile(cal_scores, q_level, method='higher')
            # print(f"{m}_{fs} quantile: {qhat}")
            # generate prediction sets
            pred_sets = {}
            test_logits_data = logits_data_all[m+"_"+fs]["test"]
            for idx, row in enumerate(test_logits_data):
                probs = softmax(row["logits_options"])
                ps = []
                for ii, p in enumerate(probs):
                    # 1 - p <= qhat, so p >= 1- qhat
                    if p >= 1 - qhat:
                        ps.append(options[ii])
                if len(ps) == 0:
                    ps.append(options[np.argmax(probs)])
                pred_sets[str(row["id"])] = ps
            pred_sets_all[m+"_"+fs] = pred_sets
    return pred_sets_all

def get_accuracy(logits_data, raw_data):
    res = []
    preds = []
    for idx, row in enumerate(raw_data):
        truth_answer = row["answer"]
        pred = logits_data[idx]
        assert pred["id"] == row["id"]
        pred_answer = options[np.argmax(pred["logits_options"])]
        preds.append(pred_answer)
        if pred_answer == truth_answer:
            res.append(1)
        else:
            res.append(0)
    return sum(res) / len(res), preds

def cal_acc(logits_data_all, test_raw_data, prompt_methods, icl_methods):
    results_acc = {}
    E_ratios = {}
    F_ratios = {}
    for m in prompt_methods:
        for fs in icl_methods:
            test_logits_data = logits_data_all[m + "_" + fs]["test"]
            acc, preds = get_accuracy(test_logits_data, test_raw_data)
            results_acc[m + "_" + fs] = acc
            counts = Counter(preds)
            E_ratio = counts["E"] / len(preds)
            F_ratio = counts["F"] / len(preds)
            E_ratios[m + "_" + fs] = E_ratio
            F_ratios[m + "_" + fs] = F_ratio
    return results_acc, E_ratios, F_ratios

def convert_id_to_ans(test_raw_data):
    test_id_to_answer = {}
    for row in test_raw_data:
        test_id_to_answer[str(row["id"])] = row["answer"]
    return test_id_to_answer

def cal_coverage(pred_sets_all, test_id_to_answer, prompt_methods, icl_methods):
    """
    Calculate the coverage rate of prediction sets.
    """""
    coverage_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            cover = []
            pred_sets = pred_sets_all[m+"_"+fs]
            for k, v in pred_sets.items():
                if test_id_to_answer[k] in v:
                    cover.append(1)
                else:
                    cover.append(0)
            coverage_all[m+"_"+fs] = sum(cover) / len(cover)
    return coverage_all

def cal_set_size(pred_sets_all, prompt_methods, icl_methods):
    set_sizes = {}
    for m in prompt_methods:
        for fs in icl_methods:
            sz = []
            pred_sets = pred_sets_all[m+"_"+fs]
            for k, v in pred_sets.items():
                sz.append(len(v))
            # print(f"{m}_{fs}: {min(sz)}, {max(sz)}")
            # average set size
            set_sizes[m+"_"+fs] = sum(sz) / len(sz)
    return set_sizes

def cal_uacc(results_acc, set_sizes):
    results_uacc = {}
    for k, v in results_acc.items():
        results_uacc[k] = v * np.sqrt(len(options)) / set_sizes[k]
    return results_uacc

def density_ratio(emb, clf, clip_min=0, clip_max=1):
    probs = clf.predict_proba(emb)[:, 1]
    p_new = np.clip(probs, clip_min, clip_max)
    return p_new / (1 - p_new)

def LAC_CP_W(logits_data_all, cal_raw_data, prompt_methods, icl_methods, clf, embed_model, alpha=0.1):
    """
    Apply conformal prediction to obtain sets of predicted answers on each instance based on its softmax scores.
    Here the LAC score function is utilized.
    """
    pred_sets_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            pred_sets_all[m+"_"+fs] = {}
            cal_scores = []
            cal_logits_data = logits_data_all[m+"_"+fs]["cal"]
            cal_weights = density_ratio(embed_model.encode([cal_raw_data[idx]["question"] for idx in range(len(cal_raw_data))]), clf)
            for idx, row in enumerate(cal_logits_data):
                probs = softmax(row["logits_options"])
                truth_answer = cal_raw_data[idx]["answer"]
                assert cal_raw_data[idx]["id"] == row["id"]
                cal_scores.append(1 - probs[options.index(truth_answer)])
            # calculate the threshold qhat
            cal_scores = np.array(cal_scores)
            cal_weights = np.array(cal_weights)
            print(f"median: {np.median(cal_weights)}, min: {np.min(cal_weights)}, max: {np.max(cal_weights)}")
            sorted_idx = np.argsort(cal_scores)
            sorted_values = cal_scores[sorted_idx]
            sorted_weights = cal_weights[sorted_idx]
            cumsum_weights = np.cumsum(sorted_weights)
            cutoff = (1-alpha) * (np.sum(sorted_weights) + 1)
            loc = np.searchsorted(cumsum_weights, cutoff)
            if loc == len(sorted_values):
                qhat = 1
            else:
                qhat = sorted_values[loc]
            # generate prediction sets
            pred_sets = {}
            test_logits_data = logits_data_all[m+"_"+fs]["test"]
            for idx, row in enumerate(test_logits_data):
                probs = softmax(row["logits_options"])
                ps = []
                for ii, p in enumerate(probs):
                    if p >= 1 - qhat:
                        ps.append(options[ii])
                if len(ps) == 0:
                    ps.append(options[np.argmax(probs)])
                pred_sets[str(row["id"])] = ps
            pred_sets_all[m+"_"+fs] = pred_sets
    return pred_sets_all

def main(model_name):
    print(f"Running for model: {model_name}")
    data_name = "mmlu_10k"
    raw_data = get_raw_data("data/", "mmlu_10k")
    sub_cats  = list({item['subcategory'] for item in raw_data})
    output_file = "coverage_" + model_name + ".csv"
    fieldnames = ["old","new","coverage_LAC","setsize_LAC","coverage_LAC_W","setsize_LAC_W"]
    # write header if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for old_domain, new_domain in itertools.combinations(sub_cats, 2):
        old_domain_data = [item for item in raw_data if item.get('subcategory') == old_domain]
        new_domain_data = [item for item in raw_data if item.get('subcategory') == new_domain]

        logits_data_all = get_logits_data(model_name, data_name, raw_data, old_domain, new_domain, "outputs_base/", ["base"], ["icl1"])

        
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        old_list = list(old_domain_data)
        new_list = list(new_domain_data)
        domain_labels = [0] * len(old_list) + [1] * len(new_list)
        X_domain = embed_model.encode([ex['question'] for ex in (old_list + new_list)])

        xgb = XGBClassifier(
            max_depth=3,
            n_estimators=50,
            learning_rate=0.2,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=5,
            reg_lambda=10,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )

        xgb.fit(X_domain, domain_labels)

        test_id_to_answer = convert_id_to_ans(new_domain_data)
        pred_sets_all_LAC = LAC_CP(logits_data_all, old_domain_data, ["base"], ["icl1"], alpha=0.1)
        coverage_all_LAC = cal_coverage(pred_sets_all_LAC, test_id_to_answer, ["base"], ["icl1"])
        set_sizes_LAC = cal_set_size(pred_sets_all_LAC, ["base"], ["icl1"])

        pred_sets_all_LAC_W = LAC_CP_W(logits_data_all, old_domain_data, ["base"], ["icl1"], xgb, embed_model, alpha=0.1)
        coverage_all_LAC_W = cal_coverage(pred_sets_all_LAC_W, test_id_to_answer, ["base"], ["icl1"])
        set_sizes_LAC_W = cal_set_size(pred_sets_all_LAC_W, ["base"], ["icl1"])

        row = {
            "old":            old_domain,
            "new":            new_domain,
            "coverage_LAC":   coverage_all_LAC,
            "setsize_LAC":    set_sizes_LAC,
            "coverage_LAC_W": coverage_all_LAC_W,
            "setsize_LAC_W":  set_sizes_LAC_W,
        }

        # append this row immediately
        with open(output_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

if __name__ == "__main__":
    # models = ["Yi-34B", "Qwen-72B", "Llama-2-70b-hf", "deepseek-llm-67b-base"]
    models = ["deepseek-llm-67b-base"]
    for model in models:
        main(model)
