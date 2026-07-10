import os
import json
import pickle
import itertools
import csv
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

# module-level models
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
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

options = ["A","B","C","D","E","F"]
ids_to_remove = {1,3,5,7,9}

# Helpers

def load_raw(raw_dir, name):
    path = os.path.join(raw_dir, f"{name}.json")
    with open(path, "r") as f:
        raw = json.load(f)
    if name != "mmlu_10k":
        for item in raw:
            ctx = item.get("context", "")
            q   = item.get("question", "")
            item["question"] = ctx + q
    return [item for idx, item in enumerate(raw) if idx not in ids_to_remove]


def get_logits_for_dataset(model_name, data_name, logits_dir, prompts, icls):
    all_logits = {}
    for m in prompts:
        for f in icls:
            key = f"{m}_{f}"
            pth = os.path.join(logits_dir, f"{model_name}_{data_name}_{m}_{f}.pkl")
            with open(pth,"rb") as fin:
                L = pickle.load(fin)
            # filter out demo ids
            L = [item for idx,item in enumerate(L) if idx not in ids_to_remove]
            all_logits[key] = L
    return all_logits


def softmax(x):
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def density_ratio(emb, clf, clip_min=1e-3, clip_max=1-1e-3):
    probs = clf.predict_proba(emb)[:,1]
    p_new = np.clip(probs, clip_min, clip_max)
    return p_new / (1 - p_new)

# Conformal prediction functions

def LAC_CP(cal_logits, cal_raw, test_logits, prompts, icls, alpha=0.1):
    pred_sets = {}
    for m in prompts:
        for f in icls:
            key = f"{m}_{f}"
            # calibration scores
            scores = [1 - softmax(row["logits_options"])[options.index(rec["answer"])]
                      for row,rec in zip(cal_logits[key], cal_raw)]
            n = len(scores)
            q = np.ceil((n+1)*(1-alpha))/n
            qhat = np.quantile(scores, q, method="higher")
            # prediction sets
            pred_sets[key] = {}
            for row in test_logits[key]:
                p = softmax(row["logits_options"])
                S = [options[i] for i,pi in enumerate(p) if pi >= 1-qhat]
                if not S:
                    S = [options[np.argmax(p)]]
                pred_sets[key][str(row["id"])] = S
    return pred_sets


def LAC_CP_W(cal_logits, cal_raw, test_logits, prompts, icls, clf, alpha=0.1):
    pred_sets = {}
    # embed all calibration questions
    questions = [rec["question"] for rec in cal_raw]
    emb = embed_model.encode(questions)
    wts = density_ratio(emb, clf)

    for m in prompts:
        for f in icls:
            key = f"{m}_{f}"
            scores = [1 - softmax(row["logits_options"])[options.index(rec["answer"])]
                      for row,rec in zip(cal_logits[key], cal_raw)]
            idxs = np.argsort(scores)
            sv = np.array(scores)[idxs]
            wv = wts[idxs]
            csum = np.cumsum(wv)
            cutoff = (1-alpha)*(wv.sum() + 1)
            loc = np.searchsorted(csum, cutoff)
            qhat = sv[loc] if loc < len(sv) else 1
            pred_sets[key] = {}
            for row in test_logits[key]:
                p = softmax(row["logits_options"])
                S = [options[i] for i,pi in enumerate(p) if pi >= 1-qhat]
                if not S:
                    S = [options[np.argmax(p)]]
                pred_sets[key][str(row["id"])] = S
    return pred_sets

# Metrics

def coverage(pred_sets, id2ans):
    return {k: np.mean([int(id2ans[i] in S) for i,S in ps.items()])
            for k,ps in pred_sets.items()}

def set_size(pred_sets):
    return {k: np.mean([len(S) for S in ps.values()])
            for k,ps in pred_sets.items()}

# Comparison routines

def run_within_dataset(model_name, dataset_name, raw_dir, logits_dir, prompts, icls, alpha=0.1):
    raw = load_raw(raw_dir, dataset_name)
    subcats = list({r['subcategory'] for r in raw})
    output = []
    for old,new in itertools.combinations(subcats, 2):
        cal_raw = [r for r in raw if r['subcategory']==old]
        eval_raw= [r for r in raw if r['subcategory']==new]
        cal_logits  = get_logits_for_dataset(model_name, dataset_name, logits_dir, prompts, icls)
        # train density ratio xgb on subcat A vs B
        X = embed_model.encode([r['question'] for r in cal_raw + eval_raw])
        y = [0]*len(cal_raw) + [1]*len(eval_raw)
        xgb.fit(X,y)
        eval_logits = get_logits_for_dataset(model_name, dataset_name, logits_dir, prompts, icls)

        id2ans = {str(r['id']):r['answer'] for r in eval_raw}
        ps0 = LAC_CP (cal_logits, cal_raw, eval_logits, prompts, icls, alpha)
        psW = LAC_CP_W(cal_logits, cal_raw, eval_logits, prompts, icls, xgb, alpha)
        cov0 = coverage(ps0, id2ans)
        sz0  = set_size(ps0)
        covW = coverage(psW, id2ans)
        szW  = set_size(psW)
        output.append({"old":old, "new":new,
                       "coverage_LAC":cov0, "setsize_LAC":sz0,
                       "coverage_LAC_W":covW, "setsize_LAC_W":szW})
    # write CSV
    fn = ["old","new","coverage_LAC","setsize_LAC","coverage_LAC_W","setsize_LAC_W"]
    with open(f"within_{dataset_name}.csv","w",newline="") as f:
        wr = csv.DictWriter(f, fn)
        wr.writeheader()
        for row in output:
            wr.writerow(row)


def run_cross_datasets(model_name, cal_name, eval_name,
                       raw_dir, logits_dir, prompts, icls, alpha=0.1):
    # load raw & logits
    cal_raw   = load_raw(raw_dir, cal_name)
    eval_raw  = load_raw(raw_dir, eval_name)
    cal_logits  = get_logits_for_dataset(model_name, cal_name,  logits_dir, prompts, icls)
    eval_logits = get_logits_for_dataset(model_name, eval_name, logits_dir, prompts, icls)
    # train density‐ratio classifier
    X = embed_model.encode([r['question'] for r in cal_raw + eval_raw])
    y = [0]*len(cal_raw) + [1]*len(eval_raw)
    xgb.fit(X, y)
    # run CP
    id2ans = {str(r['id']): r['answer'] for r in eval_raw}
    ps0 = LAC_CP (cal_logits, cal_raw, eval_logits, prompts, icls, alpha)
    psW = LAC_CP_W(cal_logits, cal_raw, eval_logits, prompts, icls, xgb, alpha)
    # compute metrics
    cov0 = coverage(ps0, id2ans)
    sz0  = set_size(ps0)
    covW = coverage(psW, id2ans)
    szW  = set_size(psW)

    # return everything in one dict (including model_name)
    return {
        "model":         model_name,
        "old":           cal_name,
        "new":           eval_name,
        "coverage_LAC":  cov0,
        "setsize_LAC":   sz0,
        "coverage_LAC_W": covW,
        "setsize_LAC_W":  szW
    }

if __name__ == "__main__":
    models = [
        "deepseek-llm-67b-base", "deepseek-llm-7b-base",
        "Llama-2-13b-hf",       "Llama-2-70b-hf",
        "Llama-2-7b-hf",        "Mistral-7B-v0.1",
        "Qwen-14B",             "Qwen-1_8B",
        "Qwen-72B",             "Qwen-7B",
        "Yi-34B",               "Yi-6B"
    ]
    raw_dir    = "data"
    logits_dir = "outputs_base"
    prompts    = ["base"]
    icls       = ["icl1"]

    # define your dataset‐pairs
    datasets = ["mmlu_10k","cosmosqa_10k","hellaswag_10k"]
    pairs = list(itertools.permutations(datasets, 2))

    for cal_name, eval_name in pairs:
        all_rows = []
        for model in models:
            row = run_cross_datasets(
                model_name = model,
                cal_name   = cal_name,
                eval_name  = eval_name,
                raw_dir    = raw_dir,
                logits_dir = logits_dir,
                prompts    = prompts,
                icls       = icls,
                alpha      = 0.1
            )
            all_rows.append(row)

        # now write one CSV per (cal→eval) pair
        out_fname = f"results_{cal_name}_to_{eval_name}.csv"
        fieldnames = [
            "model","old","new",
            "coverage_LAC","setsize_LAC",
            "coverage_LAC_W","setsize_LAC_W"
        ]
        with open(out_fname, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"Wrote {len(all_rows)} models to {out_fname}")