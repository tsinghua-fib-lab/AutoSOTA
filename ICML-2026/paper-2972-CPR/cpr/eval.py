"""Evaluation metrics: ECR, APSS, coverage efficiency."""

from typing import Dict, List

import numpy as np
from tqdm import tqdm

def evaluate_core_on_dataset(core, dataset: List[Dict]) -> Dict:
    """Evaluate prediction sets on a dataset."""
    total, covered, sizes, avg_confs, precision_list = 0, 0, [], [], []
    details = []

    for item in tqdm(dataset, desc="Evaluating", ncols=80):
        total += 1
        qents = item.get("q_entity") or []
        res = core.predict(qents, item["question"], item["triples"])
        pred = {a.lower() for a in res["answers"]}
        true = {a.lower() for a in item.get("a_entity", [])}
        confs = res.get("per_answer_conf", {})
        avg_conf = float(np.mean(list(confs.values()))) if confs else 0.0
        if len(pred & true) > 0:
            covered += 1
        precision_i = len(pred & true) / len(pred) if len(pred) > 0 else 0.0
        precision_list.append(precision_i)
        sizes.append(len(pred))
        avg_confs.append(avg_conf)
        details.append({
            "id": item["id"],
            "question": item["question"],
            "true": list(true),
            "pred": list(pred),
            "covered": len(pred & true) > 0,
            "per_answer_conf": confs,
            "avg_conf": avg_conf,
            "precision": precision_i,
        })

    ecr = covered / total if total > 0 else 0.0
    avg_size = float(np.mean(sizes)) if sizes else 0.0
    return {
        "ecr": ecr,
        "avg_size": avg_size,
        "apss": avg_size,
        "coverage_efficiency": (ecr / avg_size) if avg_size > 0 else 0.0,
        "mean_conf": float(np.mean(avg_confs)) if avg_confs else 0.0,
        "precision": float(np.mean(precision_list)) if precision_list else 0.0,
        "details": details,
    }

