import argparse
import sys, importlib, platform
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
import csv
import random

# ----------------------- Parse command-line arguments -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma", type=float, required=True)
    return p.parse_args()

args = parse_args()
GAMMA = args.gamma
print(f"[config] gamma = {GAMMA}")

# ----------------------- Anchor project root (CP_FINAL) -----------------------
if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    PROJECT_ROOT = Path(os.getcwd()).resolve()

print("[project root]", PROJECT_ROOT)

# ----------------------- Import local modules -----------------------
from io_utils import get_raw_data, get_logits_data, convert_id_to_ans, cal_coverage, cal_set_size
from cp_methods import LAC_CP, LAC_CP_W, APS_CP, APS_CP_W
from plotting import plot_cp_comparisons, rerun_plots_only, plot_mmlu_subject_distribution

# ----------------------- Package versions -----------------------
pkgs = {
    "numpy": "numpy",
    "xgboost": "xgboost",
    "sentence-transformers": "sentence_transformers",
    "matplotlib": "matplotlib",
}
print("Python", sys.version.split()[0], "| macOS", platform.mac_ver()[0])
for label, mod in pkgs.items():
    m = importlib.import_module(mod)
    print(f"{label}=={getattr(m, '__version__', 'N/A')}")

# ----------------------- Single-threaded BLAS -----------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ----------------------- Paths & config -----------------------
RAW_DIR    = PROJECT_ROOT / "data"
LOGITS_DIR = PROJECT_ROOT / "outputs_base"
DATA_NAME  = "mmlu_10k"

PROMPT_METHODS = ["base"]
ICL_METHODS    = ["icl1"]
ALPHA          = 0.10

RESULTS_DIR  = PROJECT_ROOT / "results-mmlu" / f"gamma-{GAMMA}"
RESULTS_FIGS = PROJECT_ROOT / "figs-mmlu"    / f"gamma-{GAMMA}"
W_DIR        = PROJECT_ROOT / "w-result"     / f"gamma-{GAMMA}"

for d in (RESULTS_DIR, RESULTS_FIGS, W_DIR):
    d.mkdir(parents=True, exist_ok=True)

print("[paths]")
print(" results:", RESULTS_DIR)
print(" figs   :", RESULTS_FIGS)
print(" w-dir  :", W_DIR)

# ----------------------- Models and labels -----------------------
MODELS = [
    "Yi-34B","Qwen-72B","Qwen-14B","Llama-2-70b-hf","deepseek-llm-67b-base",
    "Yi-6B","Mistral-7B-v0.1","Llama-2-13b-hf","Qwen-7B","InternLM-7B",
    "Llama-2-7b-hf","deepseek-llm-7b-base","Qwen-1_8B","Falcon-40B","MPT-7B","Falcon-7B",
]

DISPLAY_NAMES = {
    "Yi-34B":"Yi-34B","Qwen-72B":"Qwen-72B","Qwen-14B":"Qwen-14B",
    "Llama-2-70b-hf":"Llama-2-70B","deepseek-llm-67b-base":"DeepSeek-67B",
    "Yi-6B":"Yi-6B","Mistral-7B-v0.1":"Mistral-7B",
    "Llama-2-13b-hf":"Llama-2-13B","Qwen-7B":"Qwen-7B","InternLM-7B":"InternLM-7B",
    "Llama-2-7b-hf":"Llama-2-7B","deepseek-llm-7b-base":"DeepSeek-7B",
    "Qwen-1_8B":"Qwen-1.8B","Falcon-40B":"Falcon-40B",
    "MPT-7B":"MPT-7B","Falcon-7B":"Falcon-7B",
}

UNWEIGHTED_LABEL = "CP"
WEIGHTED_LABEL   = "DS-CP"

# ----------------------- Helpers -----------------------
def _read_existing_pairs(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["old", "new"])
    except Exception:
        return set()
    return set(zip(df["old"].astype(str), df["new"].astype(str)))

def _all_ordered_pairs(subcats):
    return [(a, b) for a in subcats for b in subcats if a != b]

def _model_is_complete(csv_path, expected_pairs):
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False
    return len(df) >= expected_pairs

# ----------------------- Per-model runner -----------------------
def run_model(model_name):
    print(f"\n=== Running model: {model_name} ===")

    raw_data = get_raw_data(RAW_DIR, DATA_NAME)
    sub_cats = sorted({item['subcategory'] for item in raw_data})
    expected_pairs = len(sub_cats) * (len(sub_cats) - 1)

    output_file = RESULTS_DIR / f"coverage_{model_name}.csv"

    if _model_is_complete(output_file, expected_pairs):
        print(f"[skip] {model_name}: results already complete")
        return output_file

    fieldnames = [
        "old","new",
        "coverage_LAC","setsize_LAC",
        "coverage_LAC_W","setsize_LAC_W",
        "coverage_APS","setsize_APS",
        "coverage_APS_W","setsize_APS_W"
    ]

    if not output_file.exists():
        pd.DataFrame(columns=fieldnames).to_csv(output_file, index=False)

    already_done = _read_existing_pairs(output_file)

    # embed model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    for old_domain, new_domain in _all_ordered_pairs(sub_cats):
        if (old_domain, new_domain) in already_done:
            continue

        cal_raw  = [it for it in raw_data if it['subcategory'] == old_domain]
        test_raw = [it for it in raw_data if it['subcategory'] == new_domain]
        if not cal_raw or not test_raw:
            continue

        try:
            logits_data_all = get_logits_data(
                model_name, DATA_NAME, raw_data,
                old_domain, new_domain,
                LOGITS_DIR, PROMPT_METHODS, ICL_METHODS
            )
        except FileNotFoundError:
            continue

        X_domain = embed_model.encode([ex['question'] for ex in (cal_raw + test_raw)])
        y_domain = np.array([0]*len(cal_raw) + [1]*len(test_raw))
        
        # classifer
        xgb = XGBClassifier(
            max_depth=3, n_estimators=50, learning_rate=0.2,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=5, reg_lambda=10,
            eval_metric="logloss",
            random_state=42, n_jobs=1
        )
        xgb.fit(X_domain, y_domain)

        id2ans = convert_id_to_ans(test_raw)

        pred_LAC = LAC_CP(logits_data_all, cal_raw, PROMPT_METHODS, ICL_METHODS, alpha=ALPHA)
        pred_APS = APS_CP(logits_data_all, cal_raw, PROMPT_METHODS, ICL_METHODS, alpha=ALPHA)

        pred_LAC_W = LAC_CP_W(
            logits_data_all, cal_raw, PROMPT_METHODS, ICL_METHODS,
            xgb, embed_model,
            alpha=ALPHA, gamma=GAMMA, w_dir=str(W_DIR)
        )
        pred_APS_W = APS_CP_W(
            logits_data_all, cal_raw, PROMPT_METHODS, ICL_METHODS,
            xgb, embed_model,
            alpha=ALPHA, gamma=GAMMA, w_dir=str(W_DIR)
        )

        row = {
            "old": old_domain,
            "new": new_domain,
            "coverage_LAC": cal_coverage(pred_LAC, id2ans, PROMPT_METHODS, ICL_METHODS),
            "setsize_LAC": cal_set_size(pred_LAC, PROMPT_METHODS, ICL_METHODS),
            "coverage_LAC_W": cal_coverage(pred_LAC_W, id2ans, PROMPT_METHODS, ICL_METHODS),
            "setsize_LAC_W": cal_set_size(pred_LAC_W, PROMPT_METHODS, ICL_METHODS),
            "coverage_APS": cal_coverage(pred_APS, id2ans, PROMPT_METHODS, ICL_METHODS),
            "setsize_APS": cal_set_size(pred_APS, PROMPT_METHODS, ICL_METHODS),
            "coverage_APS_W": cal_coverage(pred_APS_W, id2ans, PROMPT_METHODS, ICL_METHODS),
            "setsize_APS_W": cal_set_size(pred_APS_W, PROMPT_METHODS, ICL_METHODS),
        }

        pd.DataFrame([row]).to_csv(output_file, mode="a", header=False, index=False)

    return output_file

# ----------------------- Notebook detection -----------------------
def _is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and "IPKernelApp" in shell.config
    except Exception:
        return False

# ----------------------- Run all models -----------------------
def run_all_models():
    use_threads = _is_notebook() or os.environ.get("CP_USE_THREADS", "0") == "1"
    max_workers = min(4, len(MODELS))

    generated = []
    if use_threads:
        from concurrent.futures import ThreadPoolExecutor
        Executor = ThreadPoolExecutor
    else:
        from concurrent.futures import ProcessPoolExecutor
        Executor = ProcessPoolExecutor

    with Executor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_model, m) for m in MODELS]
        for fut in futures:
            try:
                generated.append(fut.result())
            except Exception as e:
                print("[warn]", e)

    if generated:
        plot_cp_comparisons(
            generated, MODELS,
            alpha=ALPHA,
            outdir=str(RESULTS_FIGS),
            label_unweighted=UNWEIGHTED_LABEL,
            label_weighted=WEIGHTED_LABEL
        )

# ----------------------- Main -----------------------
if __name__ == "__main__":
    run_all_models()

    rerun_plots_only(
        results_dir=str(RESULTS_DIR),
        figs_dir=str(RESULTS_FIGS),
        alpha=ALPHA
    )

    plot_mmlu_subject_distribution(
        file_path=str(RAW_DIR / "mmlu_10k.json"),
        out_path=str(RESULTS_FIGS / "mmlu_subject_distribution.pdf"),
        color="#4DAAED"
    )
