import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from cp_methods import APS_CP, APS_CP_W, LAC_CP, LAC_CP_W
from io_utils import cal_coverage, cal_set_size, convert_id_to_ans, get_logits_data, get_raw_data


ALPHA = 0.10
GAMMA = 1.0
DATA_NAME = "mmlu_10k"
PROMPT_METHODS = ["base"]
ICL_METHODS = ["icl1"]

DEFAULT_MODELS = [
    "Falcon-40B",
    "Llama-2-70b-hf",
    "Qwen-72B",
    "deepseek-llm-67b-base",
]

MODEL_LOGIT_NAMES = {
    "Falcon-40B": "falcon-40b",
    "Llama-2-70b-hf": "Llama-2-70b-hf",
    "Qwen-72B": "Qwen-72B",
    "deepseek-llm-67b-base": "deepseek-llm-67b-base",
}

EMBEDDING_VARIANTS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "e5-base-v2": "intfloat/e5-base-v2",
}


class EmbeddingAdapter:
    def __init__(self, model_name):
        self.variant_name = model_name
        hf_name = EMBEDDING_VARIANTS[model_name]
        self.model = SentenceTransformer(hf_name)

    def _prepare(self, texts):
        if self.variant_name == "e5-base-v2":
            return [f"query: {text}" for text in texts]
        return texts

    def encode(self, texts):
        return self.model.encode(self._prepare(texts))


def build_classifier(name):
    if name == "XGBoost":
        return XGBClassifier(
            max_depth=3,
            n_estimators=50,
            learning_rate=0.2,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=5,
            reg_lambda=10,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size="auto",
            learning_rate_init=1e-3,
            max_iter=300,
            random_state=42,
        )
    if name == "LogisticRegression":
        return LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )
    raise ValueError(f"Unsupported classifier: {name}")


def _all_ordered_pairs(subcats):
    return [(a, b) for a in subcats for b in subcats if a != b]


def _result_row(old_domain, new_domain, pred_lac, pred_lac_w, pred_aps, pred_aps_w, id2ans):
    return {
        "old": old_domain,
        "new": new_domain,
        "coverage_LAC": cal_coverage(pred_lac, id2ans, PROMPT_METHODS, ICL_METHODS),
        "setsize_LAC": cal_set_size(pred_lac, PROMPT_METHODS, ICL_METHODS),
        "coverage_LAC_W": cal_coverage(pred_lac_w, id2ans, PROMPT_METHODS, ICL_METHODS),
        "setsize_LAC_W": cal_set_size(pred_lac_w, PROMPT_METHODS, ICL_METHODS),
        "coverage_APS": cal_coverage(pred_aps, id2ans, PROMPT_METHODS, ICL_METHODS),
        "setsize_APS": cal_set_size(pred_aps, PROMPT_METHODS, ICL_METHODS),
        "coverage_APS_W": cal_coverage(pred_aps_w, id2ans, PROMPT_METHODS, ICL_METHODS),
        "setsize_APS_W": cal_set_size(pred_aps_w, PROMPT_METHODS, ICL_METHODS),
    }


def _save_outputs(rows, metadata, pkl_path, csv_path):
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, "wb") as f:
        pickle.dump({"metadata": metadata, "rows": rows}, f)

    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _logit_model_name(model_name):
    return MODEL_LOGIT_NAMES.get(model_name, model_name)


def _format_seconds(seconds):
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def run_single_variant(model_name, raw_data, logits_dir, pkl_path, csv_path, *,
                       embedding_variant, classifier_name):
    if pkl_path.exists() and csv_path.exists():
        print(f"[skip] existing outputs for {model_name} -> {csv_path}")
        return

    print(f"[run] model={model_name} embedding={embedding_variant} classifier={classifier_name}")
    sub_cats = sorted({item["subcategory"] for item in raw_data})
    embed_model = EmbeddingAdapter(embedding_variant)
    rows = []
    start_time = time.time()
    domain_pairs = _all_ordered_pairs(sub_cats)

    for old_domain, new_domain in tqdm(
        domain_pairs,
        desc=f"{model_name} | {embedding_variant} | {classifier_name}",
        unit="pair",
        dynamic_ncols=True,
    ):
        cal_raw = [it for it in raw_data if it["subcategory"] == old_domain]
        test_raw = [it for it in raw_data if it["subcategory"] == new_domain]
        if not cal_raw or not test_raw:
            continue

        logits_data_all = get_logits_data(
            _logit_model_name(model_name),
            DATA_NAME,
            raw_data,
            old_domain,
            new_domain,
            str(logits_dir),
            PROMPT_METHODS,
            ICL_METHODS,
        )

        domain_questions = [ex["question"] for ex in (cal_raw + test_raw)]
        X_domain = embed_model.encode(domain_questions)
        y_domain = np.array([0] * len(cal_raw) + [1] * len(test_raw))

        clf = build_classifier(classifier_name)
        clf.fit(X_domain, y_domain)

        id2ans = convert_id_to_ans(test_raw)
        pred_lac = LAC_CP(logits_data_all, cal_raw, PROMPT_METHODS, ICL_METHODS, alpha=ALPHA)
        pred_aps = APS_CP(logits_data_all, cal_raw, PROMPT_METHODS, ICL_METHODS, alpha=ALPHA)
        pred_lac_w = LAC_CP_W(
            logits_data_all,
            cal_raw,
            PROMPT_METHODS,
            ICL_METHODS,
            clf,
            embed_model,
            alpha=ALPHA,
            gamma=GAMMA,
            w_dir=None,
        )
        pred_aps_w = APS_CP_W(
            logits_data_all,
            cal_raw,
            PROMPT_METHODS,
            ICL_METHODS,
            clf,
            embed_model,
            alpha=ALPHA,
            gamma=GAMMA,
            w_dir=None,
        )

        rows.append(
            _result_row(old_domain, new_domain, pred_lac, pred_lac_w, pred_aps, pred_aps_w, id2ans)
        )

    metadata = {
        "model_name": model_name,
        "data_name": DATA_NAME,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "embedding_variant": embedding_variant,
        "classifier_name": classifier_name,
    }
    _save_outputs(rows, metadata, pkl_path, csv_path)
    print(
        f"[done] model={model_name} embedding={embedding_variant} "
        f"classifier={classifier_name} rows={len(rows)} "
        f"elapsed={_format_seconds(time.time() - start_time)}"
    )


def _variant_paths(root, family, variant_label, model_name):
    pkl_dir = root / "outputs_base" / "ablations" / family / variant_label
    csv_dir = root / "results-mmlu" / "ablations" / family / variant_label
    pkl_path = pkl_dir / f"coverage_{model_name}.pkl"
    csv_path = csv_dir / f"coverage_{model_name}.csv"
    return pkl_path, csv_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DS-CP ablations for MMLU and save PKL/CSV outputs in the agreed folder layout."
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent.parent),
        help="Project root. Defaults to the parent of src/.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=[],
        help="Model to run. Repeat to override the default 4-model panel set.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip the embedding ablation runs.",
    )
    parser.add_argument(
        "--skip-classifiers",
        action="store_true",
        help="Skip the classifier ablation runs.",
    )
    parser.add_argument(
        "--fixed-embedding",
        default="all-MiniLM-L6-v2",
        choices=list(EMBEDDING_VARIANTS.keys()),
        help="Embedding to hold fixed while comparing classifiers.",
    )
    parser.add_argument(
        "--fixed-classifier",
        default="XGBoost",
        choices=["XGBoost", "MLP", "LogisticRegression"],
        help="Classifier to hold fixed while comparing embeddings.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    raw_data = get_raw_data(project_root / "data", DATA_NAME)
    logits_dir = project_root / "outputs_base"
    models = args.models or DEFAULT_MODELS
    jobs = []

    if not args.skip_embeddings:
        for embedding_variant in EMBEDDING_VARIANTS:
            for model_name in models:
                jobs.append(
                    {
                        "family": "embeddings",
                        "variant_label": embedding_variant,
                        "model_name": model_name,
                        "embedding_variant": embedding_variant,
                        "classifier_name": args.fixed_classifier,
                    }
                )

    if not args.skip_classifiers:
        for classifier_name in ["XGBoost", "MLP", "LogisticRegression"]:
            for model_name in models:
                jobs.append(
                    {
                        "family": "classifiers",
                        "variant_label": classifier_name,
                        "model_name": model_name,
                        "embedding_variant": args.fixed_embedding,
                        "classifier_name": classifier_name,
                    }
                )

    for job in tqdm(jobs, desc="Ablation jobs", unit="job", dynamic_ncols=True):
        pkl_path, csv_path = _variant_paths(
            project_root, job["family"], job["variant_label"], job["model_name"]
        )
        run_single_variant(
            job["model_name"],
            raw_data,
            logits_dir,
            pkl_path,
            csv_path,
            embedding_variant=job["embedding_variant"],
            classifier_name=job["classifier_name"],
        )


if __name__ == "__main__":
    main()
