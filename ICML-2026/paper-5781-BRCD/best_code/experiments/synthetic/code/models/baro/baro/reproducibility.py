
import os
import glob 
import shutil
import requests
import json
import zipfile
from os.path import join, basename, dirname

import numpy as np
from tqdm import tqdm
import pandas as pd
from baro.utility import (
    read_data,
    load_json,
    find_cps,
    drop_constant,
    to_service_ranks,
    download_online_boutique_dataset,
    download_sock_shop_dataset,
    download_train_ticket_dataset,
)
from baro._bocpd import online_changepoint_detection, partial, constant_hazard, MultivariateT
from baro.anomaly_detection import bocpd
from baro.root_cause_analysis import robust_scorer


def reproduce_baro(dataset=None, fault=None):
    """Reproduce BARO results for the given dataset and fault type.
    
    Parameters:
    - dataset : str, optional
        The dataset to reproduce results for. Supported values are "fse-ob" (Online Boutique), "fse-ss" (Sock Shop), and "fse-tt" (Train Ticket).
    - fault : str, optional
        The type of fault to consider. Supported values are "cpu", "mem", "delay", "loss", or "all" (for all fault types). Default is None.
    """
    assert dataset in ["fse-ob", "fse-ss", "fse-tt"], f"{dataset} is not supported!"
    assert fault in [None, "all", "cpu", "mem", "delay", "loss"], f"{fault} is not supported!"
    if fault is None:
        fault = "all"
    
    if not os.path.exists(f"data/{dataset}"):
        if dataset == "fse-ob":
            download_online_boutique_dataset()
        elif dataset == "fse-ss":
            download_sock_shop_dataset()
        elif dataset == "fse-tt":
            download_train_ticket_dataset()
    
    data_paths = list(glob.glob(f"./data/{dataset}/**/simple_data.csv", recursive=True))
    if fault != "all":
        data_paths = [p for p in data_paths if fault in p]
    
    top1_cnt, top2_cnt, top3_cnt, top4_cnt, top5_cnt, total_cnt = 0, 0, 0, 0, 0, 0

    for data_path in tqdm(data_paths, desc=f"Running"):
        # read data
        data = read_data(data_path)
        data_dir = os.path.dirname(data_path)
        service, metric = basename(dirname(dirname(data_path))).split("_")

        ############# READ ANOMALY DETECTION OUTPUT ###############
        # To reproduce the anomaly detection output, please check
        # the notebook ./tutorials/reproduce_multivariate_bocpd.ipynb
        anomalies = load_json(join(data_dir, "naive_bocpd.json"))
        anomalies = [i[0] for i in anomalies]

        ############# ROOT CAUSE ANALYSIS ###############
        ranks = robust_scorer(data, anomalies=anomalies)["ranks"]
        service_ranks = to_service_ranks(ranks)

        ############## EVALUATION ###############
        if service in service_ranks[:1]:
            top1_cnt += 1
        if service in service_ranks[:2]:
            top2_cnt += 1
        if service in service_ranks[:3]:
            top3_cnt += 1
        if service in service_ranks[:4]:
            top4_cnt += 1
        if service in service_ranks[:5]:
            top5_cnt += 1
        total_cnt += 1

    ############## EVALUATION ###############
    top1_accuracy = top1_cnt / total_cnt
    top2_accuracy = top2_cnt / total_cnt
    top3_accuracy = top3_cnt / total_cnt
    top4_accuracy = top4_cnt / total_cnt
    top5_accuracy = top5_cnt / total_cnt
    avg5_accuracy = (top1_accuracy + top2_accuracy + top3_accuracy + top4_accuracy + top5_accuracy) / 5

    print("====== Reproduce BARO =====")
    print(f"Dataset   : {dataset}")
    print(f"Fault type: {fault}")
    print(f"Avg@5 Acc : {avg5_accuracy:.2f}")
    print()
    

def reproduce_bocpd(dataset=None, saved=False):
    """Reproduce Multivariate BOCPD results for the given dataset.
    
    Parameters:
    - dataset : str, optional
        The dataset to reproduce results for. Supported values are "fse-ob" (Online Boutique), "fse-ss" (Sock Shop), and "fse-tt" (Train Ticket).
    - saved : bool, optional
        If True, load precomputed results. If False, run BOCPD algorithm again. Default is False.
    """
    assert dataset in ["fse-ob", "fse-ss", "fse-tt"], f"{dataset} is not supported!"
    
    if not os.path.exists(f"data/{dataset}"):
        if dataset == "fse-ob":
            download_online_boutique_dataset()
        elif dataset == "fse-ss":
            download_sock_shop_dataset()
        elif dataset == "fse-tt":
            download_train_ticket_dataset()
    
    data_paths = list(glob.glob(f"./data/{dataset}/**/simple_data.csv", recursive=True))


    cnt, tp, fp, tn, fn = 0, 0, 0, 0, 0

    for data_path in tqdm(data_paths, desc=f"Running"):
        service_metric = basename(dirname(dirname(data_path)))
        case_idx = basename(dirname(data_path))
        data_dir = dirname(data_path)

        # PREPARE DATA
        data = pd.read_csv(data_path)

        # read inject_time, cut data
        with open(join(data_dir, "inject_time.txt")) as f:
            inject_time = int(f.readlines()[0].strip())
        normal_df = data[data["time"] < inject_time].tail(300)
        anomal_df = data[data["time"] >= inject_time].head(300)
        data = pd.concat([normal_df, anomal_df], ignore_index=True)

        # drop extra columns
        selected_cols = []
        for c in data.columns:
            if 'queue-master' in c or 'rabbitmq_' in c: continue
            if "latency-50" in c or "_error" in c:
                selected_cols.append(c)
        data = data[selected_cols]

        # handle na
        data = drop_constant(data)
        data = data.fillna(method="ffill")
        data = data.fillna(0)
        for c in data.columns:
            data[c] = (data[c] - np.min(data[c])) / (np.max(data[c]) - np.min(data[c]))
        data = data.fillna(method="ffill")
        data = data.fillna(0)
        
        data = data.to_numpy()

        # RUN BOCPD
        cps = load_json(join(data_dir, "naive_bocpd.json"))
        if saved is False:
            R, maxes = online_changepoint_detection(
                    data,
                    partial(constant_hazard, 50),
                    MultivariateT(dims=data.shape[1])
            )
            cps = find_cps(maxes)

        if len(cps) > 0:
            tp += 1
        else: 
            fn += 1

        if cps[0][0] < 300:
            fp += 1
        else: 
            tn += 1
        
        
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print("====== Reproduce BOCPD =====")
    print(f"Dataset: {dataset}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1       : {f1:.2f}")
    
    
    
def reproduce_rq4(dataset=None, eval_metric=None):
    """Reproduce RQ4 results for the given dataset and evaluation metric.
    
    Parameters:
    - dataset : str, optional
        The dataset to reproduce results for. Supported values are "fse-ob" (Online Boutique), "fse-ss" (Sock Shop), and "fse-tt" (Train Ticket).
    - eval_metric : str, optional
        The evaluation metric to use. Supported values are "top1", "top3", "avg5", or None (default, which equals to "avg5").
    """
    assert dataset in ["fse-ob", "fse-ss", "fse-tt"], f"{dataset} is not supported!"
    assert eval_metric in [None, "top1", "top3", "avg5"], f"{eval_metric} is not supported!"
    
    if eval_metric is None:
        eval_metric = "avg5"
    
    if not os.path.exists(f"data/{dataset}"):
        if dataset == "fse-ob":
            download_online_boutique_dataset()
        elif dataset == "fse-ss":
            download_sock_shop_dataset()
        elif dataset == "fse-tt":
            download_train_ticket_dataset()
    
    data_paths = list(glob.glob(f"./data/{dataset}/**/simple_data.csv", recursive=True))
    
    scores = []
    for t_bias in tqdm(range(-40, 40, 2), desc="Running"):
        top1_cnt, top2_cnt, top3_cnt, top4_cnt, top5_cnt, total_cnt = 0, 0, 0, 0, 0, 0

        for data_path in data_paths:
            # read data
            data = read_data(data_path)
            data_dir = os.path.dirname(data_path)
            service, metric = basename(dirname(dirname(data_path))).split("_")

            ############# READ INJECT TIME ###############
            with open(join(data_dir, "inject_time.txt")) as f:
                inject_time = int(f.readlines()[0].strip()) + t_bias

            ############# ROOT CAUSE ANALYSIS ###############
            ranks = robust_scorer(data, inject_time=inject_time)["ranks"]
            service_ranks = to_service_ranks(ranks)

            ############## EVALUATION ###############
            if service in service_ranks[:1]:
                top1_cnt += 1
            if service in service_ranks[:2]:
                top2_cnt += 1
            if service in service_ranks[:3]:
                top3_cnt += 1
            if service in service_ranks[:4]:
                top4_cnt += 1
            if service in service_ranks[:5]:
                top5_cnt += 1
            total_cnt += 1

        ############## EVALUATION ###############
        top1_accuracy = top1_cnt / total_cnt
        top2_accuracy = top2_cnt / total_cnt
        top3_accuracy = top3_cnt / total_cnt
        top4_accuracy = top4_cnt / total_cnt
        top5_accuracy = top5_cnt / total_cnt
        avg5_accuracy = (top1_accuracy + top2_accuracy + top3_accuracy + top4_accuracy + top5_accuracy) / 5
        
        if eval_metric == "top1":
            scores.append(top1_accuracy)
        elif eval_metric == "top3":
            scores.append(top3_accuracy)
        elif eval_metric == "avg5":
            scores.append(avg5_accuracy)    
            
    print([round(s, 2) for s in scores])