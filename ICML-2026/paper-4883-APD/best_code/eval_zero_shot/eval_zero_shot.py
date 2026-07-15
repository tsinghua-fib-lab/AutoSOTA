import os, glob, datasets
from datasets import Dataset, DatasetDict, concatenate_datasets

os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_METRICS_OFFLINE"]   = "1"

HF_DATASETS_CACHE = "cache"
_orig_load_dataset = datasets.load_dataset

import json, datasets, re

BOM_RE = re.compile(r"^\ufeff")    
WS_RE  = re.compile(r"^\s+")     
TRAIL_COMMA_RE = re.compile(r",\s*$")

def jsonl_to_ds(path: str) -> datasets.Dataset:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            ln = BOM_RE.sub("", raw)
            ln = WS_RE.sub("", ln)
            if not ln.strip() or ln.strip() in {"[", "]"}:
                continue
            ln = TRAIL_COMMA_RE.sub("", ln)   
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    if not rows:
        raise ValueError(f"No valid JSON objects parsed in {path}")
    return datasets.Dataset.from_list(rows)


def _make_ds_from_files(file_list):
    parts = []
    for p in file_list:
        if p.endswith(".parquet"):
            parts.append(datasets.Dataset.from_parquet(p))
        elif p.endswith(".tsv"):
            parts.append(datasets.Dataset.from_csv(p, delimiter="\t"))
        elif p.endswith(".jsonl") or p.endswith(".json"):
            parts.append(jsonl_to_ds(p))
    return (
        datasets.concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    )


def _offline_load_dataset(path, *args, **kwargs):
    path = path.split("/")[-1]

    target_repos = {"super_glue", "openbookqa", "glue", "hellaswag", "winogrande", "ai2_arc"}
    if path not in target_repos:
        return _orig_load_dataset(path, *args, **kwargs)

    subset = None
    if args:
        subset, *args = args
    elif "name" in kwargs:
        subset = kwargs.pop("name")

    if path in {"hellaswag"}: 
        subset_dir = "hellaswag"
    else:
        if subset is None:
            subset = "main"       
        subset_dir = f"{path}--{subset}"

    repo_dir = f"datasets--{subset_dir}"
    snap_dir = os.path.join(HF_DATASETS_CACHE, repo_dir)

    patterns = ["**/*.parquet", "**/*.jsonl", "**/*.json", "**/*.tsv"]
    files = []
    for pat in patterns:
        files += glob.glob(os.path.join(snap_dir, pat), recursive=True)
    if not files:
        raise FileNotFoundError(f"No local data under {snap_dir}")

    splits = {"train": [], "validation": [], "test": []}
    for f in files:
        fn = os.path.basename(f).lower()
        if "train" in fn:
            splits["train"].append(f)
        elif "validation" in fn or "val" in fn:
            splits["validation"].append(f)
        elif "test" in fn:
            splits["test"].append(f)

    ds_dict = {split: _make_ds_from_files(flist)
               for split, flist in splits.items() if flist}

    return DatasetDict(ds_dict)

datasets.load_dataset = _offline_load_dataset




from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from lm_eval import evaluator
from lm_eval.tasks import TaskManager, get_task_dict
import argparse


def eval_zero_shot(
    model_name: str,
    task_list: list[str],
    num_fewshot: int = 0,
    use_accelerate: bool = False,
    add_special_tokens: bool = False
):

    import itertools
    tm = TaskManager()
    task_dict = get_task_dict(task_list, task_manager=tm)
  
    eval_limit = None
    if any(sz in model_name.lower() for sz in ("70b", "65b")):
        eval_limit = 2000
 
    if eval_limit is not None:
        for _name, _task in task_dict.items():
            _docs_cache = list(itertools.islice(_task.validation_docs(), eval_limit))
            def _vd(_self, _cache=_docs_cache):
                return _cache
            _task.validation_docs = _vd.__get__(_task, _task.__class__) 
 
    task_objs = list(task_dict.values())
    for _name, _task in task_dict.items():
        _n = sum(1 for _ in _task.validation_docs())
        print("[DEBUG after patch]", _name, "docs:", _n)

    model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    if use_accelerate:
        model_args += ",use_accelerate=True"



    model_path = args.model_path
    model_args = (
        f"pretrained={model_path},"
        "cache_dir=cache,"
        "device_map=auto,trust_remote_code=true,"
        "add_bos_token=false"
    )

    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=task_objs,
        num_fewshot=num_fewshot,
        batch_size=None,
        device='cuda:0',
        check_integrity=False,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="offline evaluate"
    )
    parser.add_argument(
        "--model_name", type=str, default='llama-1-65b',
    )
    parser.add_argument(
        "--tasks", type=str, default="boolq,rte,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa",
    )
    parser.add_argument(
        "--num_fewshot", type=int, default=0
    )
    parser.add_argument(
        "--use_accelerate", action="store_true",
    )
    parser.add_argument(
        "--add_special_tokens", action="store_true",
    )
    parser.add_argument('--model_path', type=str)

    args = parser.parse_args()


    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = eval_zero_shot(
        args.model_name,
        task_list, args.num_fewshot,
        args.use_accelerate, args.add_special_tokens
    )

    print(results['results'])
