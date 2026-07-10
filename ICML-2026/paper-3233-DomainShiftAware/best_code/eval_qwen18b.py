#!/usr/bin/env python
"""eval_qwen18b.py – Parameterized DS-CP evaluation for MMLU subject pairs.

Originally `main_mmlu.py`; wrapped with CLI arguments for SOTA optimization.
"""
import pickle
import json
import os
import itertools
import csv
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from collections import Counter
import argparse
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

options = ["A", "B", "C", "D", "E", "F"]
ids_to_remove = [1, 3, 5, 7, 9]


def get_raw_data(raw_data_dir, data_name):
    raw_data = json.load(open(os.path.join(raw_data_dir, data_name + ".json"), "r"))
    raw_data = [item for idx, item in enumerate(raw_data) if idx not in ids_to_remove]
    print(f"Loaded {len(raw_data)} raw data points")
    return raw_data


def get_logits_data(model_name, data_name, data, old_domain, new_domain, logits_data_dir, prompt_methods, icl_methods):
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
            logits_data = [item for idx, item in enumerate(logits_data) if idx not in ids_to_remove]
            old_domain_logits = [item for item in logits_data if item["id"] in old_ids]
            new_domain_logits = [item for item in logits_data if item["id"] in new_ids]
            logits_data_all[m + "_" + fs] = {}
            logits_data_all[m + "_" + fs]["cal"] = old_domain_logits
            logits_data_all[m + "_" + fs]["test"] = new_domain_logits
    return logits_data_all


def softmax(x, tau=1.0):
    x = np.array(x, dtype=float) / tau
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def LAC_CP(logits_data_all, cal_raw_data, prompt_methods, icl_methods, alpha=0.1, tau=1.0):
    pred_sets_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            pred_sets_all[m + "_" + fs] = {}
            cal_scores = []
            cal_logits_data = logits_data_all[m + "_" + fs]["cal"]
            for idx, row in enumerate(cal_logits_data):
                probs = softmax(row["logits_options"], tau)
                truth_answer = cal_raw_data[idx]["answer"]
                assert cal_raw_data[idx]["id"] == row["id"]
                cal_scores.append(1 - probs[options.index(truth_answer)])
            n = len(cal_logits_data)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            qhat = np.quantile(cal_scores, q_level, method="higher")
            pred_sets = {}
            test_logits_data = logits_data_all[m + "_" + fs]["test"]
            for idx, row in enumerate(test_logits_data):
                probs = softmax(row["logits_options"], tau)
                ps = []
                for ii, p in enumerate(probs):
                    if p >= 1 - qhat:
                        ps.append(options[ii])
                if len(ps) == 0:
                    ps.append(options[np.argmax(probs)])
                pred_sets[str(row["id"])] = ps
            pred_sets_all[m + "_" + fs] = pred_sets
    return pred_sets_all


def APS_CP(logits_data_all, cal_raw_data, prompt_methods, icl_methods, alpha=0.1, tau=1.0):
    """APS (Adaptive Prediction Sets) nonconformity score: cumulative softmax."""
    pred_sets_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            pred_sets_all[m + "_" + fs] = {}
            cal_scores = []
            cal_logits_data = logits_data_all[m + "_" + fs]["cal"]
            for idx, row in enumerate(cal_logits_data):
                probs = softmax(row["logits_options"], tau)
                truth_answer = cal_raw_data[idx]["answer"]
                assert cal_raw_data[idx]["id"] == row["id"]
                sorted_probs = np.sort(probs)[::-1]
                true_idx = options.index(truth_answer)
                true_prob = probs[true_idx]
                cumsum = 0.0
                for p in sorted_probs:
                    cumsum += p
                    if p == true_prob:
                        break
                cal_scores.append(cumsum)
            n = len(cal_logits_data)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            qhat = np.quantile(cal_scores, q_level, method="higher")
            pred_sets = {}
            test_logits_data = logits_data_all[m + "_" + fs]["test"]
            for idx, row in enumerate(test_logits_data):
                probs = softmax(row["logits_options"], tau)
                sorted_idx = np.argsort(probs)[::-1]
                cumsum = 0.0
                ps = []
                for si in sorted_idx:
                    cumsum += probs[si]
                    ps.append(options[si])
                    if cumsum >= qhat:
                        break
                if len(ps) == 0:
                    ps.append(options[np.argmax(probs)])
                pred_sets[str(row["id"])] = ps
            pred_sets_all[m + "_" + fs] = pred_sets
    return pred_sets_all


def convert_id_to_ans(test_raw_data):
    test_id_to_answer = {}
    for row in test_raw_data:
        test_id_to_answer[str(row["id"])] = row["answer"]
    return test_id_to_answer


def cal_coverage(pred_sets_all, test_id_to_answer, prompt_methods, icl_methods):
    coverage_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            cover = []
            pred_sets = pred_sets_all[m + "_" + fs]
            for k, v in pred_sets.items():
                if test_id_to_answer[k] in v:
                    cover.append(1)
                else:
                    cover.append(0)
            coverage_all[m + "_" + fs] = sum(cover) / len(cover) if cover else 0.0
    return coverage_all


def cal_set_size(pred_sets_all, prompt_methods, icl_methods):
    set_sizes = {}
    for m in prompt_methods:
        for fs in icl_methods:
            sz = []
            pred_sets = pred_sets_all[m + "_" + fs]
            for k, v in pred_sets.items():
                sz.append(len(v))
            set_sizes[m + "_" + fs] = sum(sz) / len(sz) if sz else 0.0
    return set_sizes


def density_ratio(emb, clf, clip_min=1e-6, clip_max=1 - 1e-6):
    probs = clf.predict_proba(emb)[:, 1]
    p_new = np.clip(probs, clip_min, clip_max)
    return p_new / (1 - p_new)


def weighted_quantile(scores, weights, alpha, gamma=1.0, trim_frac=0.0):
    scores = np.array(scores)
    weights = np.array(weights)
    sorted_idx = np.argsort(scores)
    sorted_values = scores[sorted_idx]
    sorted_weights = weights[sorted_idx]

    if trim_frac > 0:
        n_keep = max(1, int(len(sorted_values) * (1 - trim_frac)))
        sorted_values = sorted_values[:n_keep]
        sorted_weights = sorted_weights[:n_keep]

    cumsum_weights = np.cumsum(sorted_weights)
    total_weight = np.sum(sorted_weights)
    max_weight = np.max(sorted_weights)
    cutoff = (1 - alpha) * (total_weight + gamma * max_weight)
    loc = np.searchsorted(cumsum_weights, cutoff)
    if loc >= len(sorted_values):
        return 1.0
    return float(sorted_values[loc])


def LAC_CP_W(logits_data_all, cal_raw_data, prompt_methods, icl_methods, clf, embed_model,
             alpha=0.1, gamma=1.0, tau=1.0, clip_min=1e-6, clip_max=1 - 1e-6,
             normalize_weights=True, n_clusters=1, trim_frac=0.0, standardize_scores=False):
    pred_sets_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            pred_sets_all[m + "_" + fs] = {}
            cal_logits_data = logits_data_all[m + "_" + fs]["cal"]

            cal_emb = embed_model.encode([cal_raw_data[idx]["question"] for idx in range(len(cal_raw_data))])
            cal_weights = density_ratio(cal_emb, clf, clip_min, clip_max)

            if normalize_weights:
                cal_weights = cal_weights * len(cal_weights) / cal_weights.sum()

            cal_scores = []
            for idx, row in enumerate(cal_logits_data):
                probs = softmax(row["logits_options"], tau)
                truth_answer = cal_raw_data[idx]["answer"]
                assert cal_raw_data[idx]["id"] == row["id"]
                cal_scores.append(1 - probs[options.index(truth_answer)])
            cal_scores = np.array(cal_scores)

            if standardize_scores:
                cal_mean = np.mean(cal_scores)
                cal_std = np.std(cal_scores) + 1e-8
                cal_scores = (cal_scores - cal_mean) / cal_std

            if n_clusters > 1:
                kmeans = KMeans(n_clusters=min(n_clusters, len(cal_emb)), random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(cal_emb)
                qhats = []
                cluster_weights = []
                for c in range(kmeans.n_clusters):
                    mask = cluster_labels == c
                    if mask.sum() < 50:
                        continue
                    c_qhat = weighted_quantile(cal_scores[mask], cal_weights[mask], alpha, gamma, trim_frac)
                    qhats.append(c_qhat)
                    cluster_weights.append(mask.sum())
                if qhats:
                    qhat = np.average(qhats, weights=cluster_weights)
                else:
                    qhat = weighted_quantile(cal_scores, cal_weights, alpha, gamma, trim_frac)
            else:
                qhat = weighted_quantile(cal_scores, cal_weights, alpha, gamma, trim_frac)

            pred_sets = {}
            test_logits_data = logits_data_all[m + "_" + fs]["test"]
            for idx, row in enumerate(test_logits_data):
                probs = softmax(row["logits_options"], tau)
                ps = []
                for ii, p in enumerate(probs):
                    if p >= 1 - qhat:
                        ps.append(options[ii])
                if len(ps) == 0:
                    ps.append(options[np.argmax(probs)])
                pred_sets[str(row["id"])] = ps
            pred_sets_all[m + "_" + fs] = pred_sets
    return pred_sets_all


def APS_CP_W(logits_data_all, cal_raw_data, prompt_methods, icl_methods, clf, embed_model,
             alpha=0.1, gamma=1.0, tau=1.0, clip_min=1e-6, clip_max=1 - 1e-6,
             normalize_weights=True, trim_frac=0.0, standardize_scores=False):
    pred_sets_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            pred_sets_all[m + "_" + fs] = {}
            cal_logits_data = logits_data_all[m + "_" + fs]["cal"]

            cal_emb = embed_model.encode([cal_raw_data[idx]["question"] for idx in range(len(cal_raw_data))])
            cal_weights = density_ratio(cal_emb, clf, clip_min, clip_max)

            if normalize_weights:
                cal_weights = cal_weights * len(cal_weights) / cal_weights.sum()

            cal_scores = []
            for idx, row in enumerate(cal_logits_data):
                probs = softmax(row["logits_options"], tau)
                truth_answer = cal_raw_data[idx]["answer"]
                assert cal_raw_data[idx]["id"] == row["id"]
                sorted_probs = np.sort(probs)[::-1]
                true_prob = probs[options.index(truth_answer)]
                cumsum = 0.0
                for p in sorted_probs:
                    cumsum += p
                    if p == true_prob:
                        break
                cal_scores.append(cumsum)
            cal_scores = np.array(cal_scores)

            if standardize_scores:
                cal_mean = np.mean(cal_scores)
                cal_std = np.std(cal_scores) + 1e-8
                cal_scores = (cal_scores - cal_mean) / cal_std

            qhat = weighted_quantile(cal_scores, cal_weights, alpha, gamma, trim_frac)

            pred_sets = {}
            test_logits_data = logits_data_all[m + "_" + fs]["test"]
            for idx, row in enumerate(test_logits_data):
                probs = softmax(row["logits_options"], tau)
                sorted_idx = np.argsort(probs)[::-1]
                cumsum = 0.0
                ps = []
                for si in sorted_idx:
                    cumsum += probs[si]
                    ps.append(options[si])
                    if cumsum >= qhat:
                        break
                if len(ps) == 0:
                    ps.append(options[np.argmax(probs)])
                pred_sets[str(row["id"])] = ps
            pred_sets_all[m + "_" + fs] = pred_sets
    return pred_sets_all


def main():
    parser = argparse.ArgumentParser(description="DS-CP evaluation on MMLU subject pairs")
    parser.add_argument("--model", type=str, default="Qwen-1_8B", help="Model name for logits files")
    parser.add_argument("--alpha", type=float, default=0.1, help="Target error rate (1 - coverage)")
    parser.add_argument("--gamma", type=float, default=1.0, help="Regularization parameter for weighted quantile")
    parser.add_argument("--tau", type=float, default=1.0, help="Temperature for softmax")
    parser.add_argument("--score", type=str, default="LAC", choices=["LAC", "APS"], help="Nonconformity score function")
    parser.add_argument("--clip_min", type=float, default=1e-6, help="Min density ratio clip")
    parser.add_argument("--clip_max", type=float, default=0.999999, help="Max density ratio clip")
    parser.add_argument("--normalize_weights", type=int, default=1, help="Normalize calibration weights")
    parser.add_argument("--trim_frac", type=float, default=0.0, help="Trim fraction for robust quantile")
    parser.add_argument("--standardize_scores", type=int, default=0, help="Z-score standardize nonconformity scores")
    parser.add_argument("--n_clusters", type=int, default=1, help="KMeans clusters for per-cluster calibration")
    parser.add_argument("--calibrate_xgb", type=int, default=0, help="Use isotonic calibration for XGBoost")
    parser.add_argument("--xgb_max_depth", type=int, default=3, help="XGBoost max_depth")
    parser.add_argument("--xgb_n_estimators", type=int, default=50, help="XGBoost n_estimators")
    parser.add_argument("--xgb_lr", type=float, default=0.2, help="XGBoost learning_rate")
    parser.add_argument("--xgb_subsample", type=float, default=0.7, help="XGBoost subsample")
    parser.add_argument("--xgb_alpha", type=float, default=5.0, help="XGBoost reg_alpha")
    parser.add_argument("--xgb_lambda", type=float, default=10.0, help="XGBoost reg_lambda")
    parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2", help="Sentence-transformer model name")
    parser.add_argument("--adaptive_gamma", type=int, default=0, help="Adapt gamma per pair based on XGBoost AUC")
    parser.add_argument("--output_file", type=str, default=None, help="Output CSV path (default: coverage_<model>.csv)")

    args = parser.parse_args()

    model_name = args.model
    output_file = args.output_file or f"coverage_{model_name}.csv"

    print(f"Model: {model_name}")
    print(f"Alpha: {args.alpha}, Gamma: {args.gamma}, Tau: {args.tau}")
    print(f"Score: {args.score}, Embed: {args.embed_model}")
    print(f"Output: {output_file}")

    data_name = "mmlu_10k"
    raw_data = get_raw_data("data/", "mmlu_10k")
    sub_cats = list({item["subcategory"] for item in raw_data})
    print(f"Found {len(sub_cats)} subcategories, {len(list(itertools.combinations(sub_cats, 2)))} pairs")

    fieldnames = ["old", "new",
                  "coverage_LAC", "setsize_LAC",
                  "coverage_LAC_W", "setsize_LAC_W"]
    if args.score == "APS":
        fieldnames += ["coverage_APS", "setsize_APS",
                       "coverage_APS_W", "setsize_APS_W"]

    # Write header
    write_header = not os.path.exists(output_file)
    if write_header:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    embed_model = SentenceTransformer(args.embed_model)

    all_coverage_LAC = []
    all_setsize_LAC = []
    all_coverage_LAC_W = []
    all_setsize_LAC_W = []
    all_coverage_APS = []
    all_setsize_APS = []
    all_coverage_APS_W = []
    all_setsize_APS_W = []

    pair_count = 0
    for old_domain, new_domain in itertools.combinations(sub_cats, 2):
        pair_count += 1
        old_domain_data = [item for item in raw_data if item.get("subcategory") == old_domain]
        new_domain_data = [item for item in raw_data if item.get("subcategory") == new_domain]

        logits_data_all = get_logits_data(model_name, data_name, raw_data,
                                          old_domain, new_domain,
                                          "outputs_base/", ["base"], ["icl1"])

        old_list = list(old_domain_data)
        new_list = list(new_domain_data)
        domain_labels = [0] * len(old_list) + [1] * len(new_list)
        X_domain = embed_model.encode([ex["question"] for ex in (old_list + new_list)])

        xgb = XGBClassifier(
            max_depth=args.xgb_max_depth,
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.xgb_lr,
            subsample=args.xgb_subsample,
            colsample_bytree=0.7,
            reg_alpha=args.xgb_alpha,
            reg_lambda=args.xgb_lambda,
            
            eval_metric="logloss",
            random_state=42
        )

        if args.calibrate_xgb:
            n_cal = len(old_list)
            cv_folds = min(5, max(2, n_cal // 10))
            xgb = CalibratedClassifierCV(xgb, method="isotonic", cv=cv_folds)

        xgb.fit(X_domain, domain_labels)

        gamma = args.gamma
        if args.adaptive_gamma:
            from sklearn.model_selection import cross_val_score
            try:
                auc = np.mean(cross_val_score(
                    XGBClassifier(max_depth=3, n_estimators=30, random_state=42,
                                   eval_metric="logloss"),
                    X_domain, domain_labels, cv=min(5, len(domain_labels) // 10), scoring="roc_auc"))
                gamma = 0.5 + (2.0 - 0.5) * (1 - auc)
                gamma = np.clip(gamma, 0.5, 2.0)
            except Exception:
                pass

        test_id_to_answer = convert_id_to_ans(new_domain_data)

        # Standard CP (LAC)
        pred_sets_all_LAC = LAC_CP(logits_data_all, old_domain_data, ["base"], ["icl1"],
                                   alpha=args.alpha, tau=args.tau)
        coverage_all_LAC = cal_coverage(pred_sets_all_LAC, test_id_to_answer, ["base"], ["icl1"])
        set_sizes_LAC = cal_set_size(pred_sets_all_LAC, ["base"], ["icl1"])

        # DS-CP (LAC weighted)
        pred_sets_all_LAC_W = LAC_CP_W(
            logits_data_all, old_domain_data, ["base"], ["icl1"],
            xgb, embed_model, alpha=args.alpha, gamma=gamma, tau=args.tau,
            clip_min=args.clip_min, clip_max=args.clip_max,
            normalize_weights=bool(args.normalize_weights),
            n_clusters=args.n_clusters, trim_frac=args.trim_frac,
            standardize_scores=bool(args.standardize_scores))
        coverage_all_LAC_W = cal_coverage(pred_sets_all_LAC_W, test_id_to_answer, ["base"], ["icl1"])
        set_sizes_LAC_W = cal_set_size(pred_sets_all_LAC_W, ["base"], ["icl1"])

        lac_cov = coverage_all_LAC.get("base_icl1", 0)
        lac_sz = set_sizes_LAC.get("base_icl1", 0)
        lacw_cov = coverage_all_LAC_W.get("base_icl1", 0)
        lacw_sz = set_sizes_LAC_W.get("base_icl1", 0)

        all_coverage_LAC.append(lac_cov)
        all_setsize_LAC.append(lac_sz)
        all_coverage_LAC_W.append(lacw_cov)
        all_setsize_LAC_W.append(lacw_sz)

        row = {
            "old": old_domain,
            "new": new_domain,
            "coverage_LAC": lac_cov,
            "setsize_LAC": lac_sz,
            "coverage_LAC_W": lacw_cov,
            "setsize_LAC_W": lacw_sz,
        }

        # APS if requested
        if args.score == "APS":
            pred_sets_all_APS = APS_CP(logits_data_all, old_domain_data, ["base"], ["icl1"],
                                       alpha=args.alpha, tau=args.tau)
            coverage_all_APS = cal_coverage(pred_sets_all_APS, test_id_to_answer, ["base"], ["icl1"])
            set_sizes_APS = cal_set_size(pred_sets_all_APS, ["base"], ["icl1"])

            pred_sets_all_APS_W = APS_CP_W(
                logits_data_all, old_domain_data, ["base"], ["icl1"],
                xgb, embed_model, alpha=args.alpha, gamma=gamma, tau=args.tau,
                clip_min=args.clip_min, clip_max=args.clip_max,
                normalize_weights=bool(args.normalize_weights),
                trim_frac=args.trim_frac,
                standardize_scores=bool(args.standardize_scores))
            coverage_all_APS_W = cal_coverage(pred_sets_all_APS_W, test_id_to_answer, ["base"], ["icl1"])
            set_sizes_APS_W = cal_set_size(pred_sets_all_APS_W, ["base"], ["icl1"])

            aps_cov = coverage_all_APS.get("base_icl1", 0)
            aps_sz = set_sizes_APS.get("base_icl1", 0)
            apsw_cov = coverage_all_APS_W.get("base_icl1", 0)
            apsw_sz = set_sizes_APS_W.get("base_icl1", 0)

            all_coverage_APS.append(aps_cov)
            all_setsize_APS.append(aps_sz)
            all_coverage_APS_W.append(apsw_cov)
            all_setsize_APS_W.append(apsw_sz)

            row["coverage_APS"] = aps_cov
            row["setsize_APS"] = aps_sz
            row["coverage_APS_W"] = apsw_cov
            row["setsize_APS_W"] = apsw_sz

        # Append row
        with open(output_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        if pair_count % 50 == 0:
            print(f"  Processed {pair_count} pairs...")

    # Summary statistics
    print("\n" + "="*60)
    print(f"SUMMARY for model={model_name}, alpha={args.alpha}, gamma={args.gamma}")
    print(f"Total pairs: {pair_count}")
    print("="*60)

    med_cov_lac = np.median(all_coverage_LAC)
    med_sz_lac = np.median(all_setsize_LAC)
    med_cov_lacw = np.median(all_coverage_LAC_W)
    med_sz_lacw = np.median(all_setsize_LAC_W)

    print(f"LAC (Standard CP):")
    print(f"  Median Coverage:  {med_cov_lac:.4f}")
    print(f"  Median Set Size:  {med_sz_lac:.4f}")
    print(f"LAC Weighted (DS-CP):")
    print(f"  Median Coverage:  {med_cov_lacw:.4f}")
    print(f"  Median Set Size:  {med_sz_lacw:.4f}")

    if args.score == "APS":
        med_cov_aps = np.median(all_coverage_APS) if all_coverage_APS else 0
        med_sz_aps = np.median(all_setsize_APS) if all_setsize_APS else 0
        med_cov_apsw = np.median(all_coverage_APS_W) if all_coverage_APS_W else 0
        med_sz_apsw = np.median(all_setsize_APS_W) if all_setsize_APS_W else 0
        print(f"APS (Standard CP):")
        print(f"  Median Coverage:  {med_cov_aps:.4f}")
        print(f"  Median Set Size:  {med_sz_aps:.4f}")
        print(f"APS Weighted (DS-CP):")
        print(f"  Median Coverage:  {med_cov_apsw:.4f}")
        print(f"  Median Set Size:  {med_sz_apsw:.4f}")


if __name__ == "__main__":
    main()
