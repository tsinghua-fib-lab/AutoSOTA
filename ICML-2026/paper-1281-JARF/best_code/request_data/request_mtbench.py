import os
import json
import time
import random
from typing import Optional, Dict, Any, Tuple, List
import requests
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from together import Together
import re
from typing import Tuple, Optional
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

HF_DATASET_ID = "lmsys/mt_bench_human_judgments"
HF_SPLIT = "human"

# Please set your API keys here
HF_TOKEN = ""
TOGETHER_KEY = ""
KIMI_KEY = ""
DS_KEY = ""

BASENAME = "judge_results_10k_test_ca2"
N_SAMPLES = 3

JUDGE_MODELS = [
    "openai/gpt-oss-20b",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    "google/gemma-3n-E4B-it",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "zai-org/GLM-4.5-Air-FP8",
    "marin-community/marin-8b-instruct",
    "arcee_ai/arcee-spotlight",
    "kimi-k2-0905-preview",
    "deepseek-chat",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "deepcogito/cogito-v2-preview-llama-109B-MoE",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "openai/gpt-oss-120b",
    "kimi-k2-thinking-turbo",
    "moonshot-v1-32k",
    "moonshot-v1-128k",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "arize-ai/qwen-2-1.5b-instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
]

TEMPERATURE = 0.0
MAX_TOKENS = 1024
SLEEP_BETWEEN_CALLS = 0.2
RETRY_ON_ERROR = 2
RANDOM_SEED = 40

MAX_WORKERS = 32

def extract_from_fixed_structure(response_field: Any) -> Tuple[str, str]:
    if isinstance(response_field, list) and len(response_field) >= 2:
        first = response_field[0]
        second = response_field[1]
        user_text = first.get("content") if isinstance(first, dict) and "content" in first else ""
        assistant_text = second.get("content") if isinstance(second, dict) and "content" in second else ""
        return user_text or "", assistant_text or ""
    return "", ""

def build_judge_prompt(user_question: str, a_answer: str, b_answer: str) -> str:
    prompt = (
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.\n"
        "Do NOT provide any explanation, justification, reasoning, or chain-of-thought. Do NOT output any extra text other than the single-line verdict described below.\n"
        "<CHOICE> <CONFIDENCE>\n"
        "- <CHOICE> must be one of: a, b, or c   (a = model_a is better; b = model_b is better; c = tie)\n"
        "- <CONFIDENCE> must be a decimal number between 0.00 and 1.00 with exactly two digits after the decimal point. The number expresses your confidence in the choice (higher means more confident).\n\n"
        "Examples of valid outputs (each must be exactly one line):\n"
        "a 0.85\n"
        "b 0.60\n"
        "c 0.40\n\n"
        "Now evaluate the following:\n\n"
        "User question:\n"
        f"{user_question}\n\n"
        "model_a response:\n"
        f"{a_answer}\n\n"
        "model_b response:\n"
        f"{b_answer}\n\n"
    )
    return prompt

def call_llms(
    judge_model: str,
    prompt_text: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
    retry_on_error: int = 3,
    backoff_base: float = 0.5,
) -> str:
    if judge_model not in ["kimi-k2-0905-preview","deepseek-chat","kimi-k2-thinking-turbo","moonshot-v1-32k","moonshot-v1-128k"]:
        client = Together(api_key=TOGETHER_KEY) 
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif judge_model in ["kimi-k2-0905-preview","kimi-k2-thinking-turbo","moonshot-v1-32k","moonshot-v1-128k"]:
        client = OpenAI(api_key = KIMI_KEY, base_url = "https://api.moonshot.cn/v1")
        response = client.chat.completions.create(
            model = "kimi-k2-0905-preview",
            messages = [{"role": "user", "content": prompt_text}],
            max_tokens = max_tokens,
            temperature = temperature,
        )
    elif judge_model == "deepseek-chat":
        client = OpenAI(api_key=DS_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt_text}],
            stream=False,
            max_tokens = max_tokens,
            temperature = temperature,
        )
    text = response.choices[0].message.content
    return text if isinstance(text, str) else str(text)

def parse_choice_confidence(raw_text: any) -> Tuple[str, Optional[float]]:
    s = "" if raw_text is None else str(raw_text)
    s = s.strip()

    choice = "unknown"
    for ch in s:
        if not ch.isspace():
            choice = ch.lower()
            break

    if choice not in ("a", "b", "c"):
        m_choice = re.search(r"\b([abc])\b", s, flags=re.IGNORECASE)
        if m_choice:
            choice = m_choice.group(1).lower()
        else:
            choice = "unknown"

    conf = None
    m = re.search(r"([01](?:\.\d{1,4})?)", s)
    if m:
        try:
            val = float(m.group(1))
            if val is not None:
                if val < 0:
                    val = None
                elif val > 1:
                    val = 1.00
                else:
                    val = round(val, 2)
            conf = val
        except Exception:
            conf = None
    return choice, conf

def get_script_dir() -> str:
    try:
        return os.path.dirname(os.path.realpath(__file__))
    except NameError:
        return os.getcwd()


def main():
    random.seed(RANDOM_SEED)

    ds = load_dataset(HF_DATASET_ID, split=HF_SPLIT, token=HF_TOKEN)
    df = pd.DataFrame(ds)
    print(f"[INFO] loaded dataset rows={len(df)} columns={list(df.columns)}")

    entry_map: Dict[Tuple[str, str, str], Tuple[str, str, str]] = {}
    unique_models = set()

    for _, row in df.iterrows():
        qid = row["question_id"]
        model_a = row["model_a"]
        model_b = row["model_b"]
        user_a, assistant_a = extract_from_fixed_structure(row["conversation_a"])
        user_b, assistant_b = extract_from_fixed_structure(row["conversation_b"])

        if not model_a or not model_b:
            continue

        model0, model1 = tuple(sorted([str(model_a), str(model_b)]))

        if model0 == str(model_a):
            a0 = assistant_a
            a1 = assistant_b
        else:
            a0 = assistant_b
            a1 = assistant_a

        key = (model0, model1, str(qid))
        entry_map[key] = (user_a, a0, a1)

        unique_models.add(model0)
        unique_models.add(model1)

    print(f"[INFO] built entry_map with {len(entry_map)} entries (unique_models={len(unique_models)})")

    candidates: List[Tuple[str, str, str, str]] = []
    for (model0, model1, qid) in entry_map.keys():
        for jm in JUDGE_MODELS:
            candidates.append((jm, model0, model1, qid))

    total_candidates = len(candidates)
    if total_candidates == 0:
        raise RuntimeError("No candidates available to sample. Check dataset parsing and model columns.")

    if N_SAMPLES >= total_candidates:
        print(f"[WARN] Requested N_SAMPLES={N_SAMPLES} >= total_candidates={total_candidates}. Will use all {total_candidates} candidates.")
        sampled_candidates = candidates
    else:
        sampled_candidates = random.sample(candidates, k=N_SAMPLES)
    print(f"[INFO] sampled {len(sampled_candidates)} unique (judge,model0,model1,qid) candidates")

    results: List[Dict[str, Any]] = []
    model_counter: Dict[str, int] = {m: 0 for m in sorted(unique_models)}
    judge_counter: Dict[str, int] = {jm: 0 for jm in JUDGE_MODELS}

    all_models_sorted = sorted(unique_models)
    pair_counts_df = pd.DataFrame(0, index=all_models_sorted, columns=all_models_sorted)

    seen_jobs = set()

    judge_call_failures = 0

    future_to_job = {}

    print(f"[INFO] Submitting {len(sampled_candidates)} judge calls to ThreadPoolExecutor (max_workers={MAX_WORKERS})")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for (jm, model0, model1, qid) in sampled_candidates:
            m_small, m_large = tuple(sorted([model0, model1]))
            job_key = (jm, qid, m_small, m_large)
            if job_key in seen_jobs:
                continue
            seen_jobs.add(job_key)

            user_q, a_answer, b_answer = entry_map[(model0, model1, qid)]
            
            prompt_text = build_judge_prompt(user_q, a_answer, b_answer)

            fut = ex.submit(call_llms, jm, prompt_text, MAX_TOKENS, TEMPERATURE, RETRY_ON_ERROR)
            future_to_job[fut] = {
                "jm": jm,
                "model0": model0,
                "model1": model1,
                "qid": qid,
                "user_q": user_q
            }
            judge_counter[jm] = judge_counter.get(jm, 0) + 1
            model_counter[model0] = model_counter.get(model0, 0) + 1
            model_counter[model1] = model_counter.get(model1, 0) + 1
            if model0 in pair_counts_df.index and model1 in pair_counts_df.columns:
                pair_counts_df.at[model0, model1] += 1
                pair_counts_df.at[model1, model0] += 1
            else:
                if model0 not in pair_counts_df.index:
                    pair_counts_df.loc[model0] = 0
                    pair_counts_df[model0] = 0
                if model1 not in pair_counts_df.index:
                    pair_counts_df.loc[model1] = 0
                    pair_counts_df[model1] = 0
                pair_counts_df.at[model0, model1] += 1
                pair_counts_df.at[model1, model0] += 1

        print(f"[INFO] Waiting for {len(future_to_job)} completed judge calls...")
        for fut in tqdm(as_completed(future_to_job), total=len(future_to_job), desc="Collecting judge outputs"):
            job = future_to_job[fut]
            jm = job["jm"]
            model0 = job["model0"]
            model1 = job["model1"]
            qid = job["qid"]

            raw_text = None
            try:
                raw_text = fut.result()
                if raw_text is None or (isinstance(raw_text, str) and raw_text.strip() == ""):
                    judge_call_failures += 1
            except Exception as e:
                judge_call_failures += 1
                print(f"[WARN] judge call failed for qid={qid}, judge={jm}, err={e} (failures={judge_call_failures})")
                raw_text = None

            choice, confidence = parse_choice_confidence(raw_text)

            results.append({
                "question_id": qid,
                "model_a": model0,
                "model_b": model1,
                "judge_model": jm,
                "judge_preferred_model": choice,
                "judge_confidence": confidence,
            })

    script_dir = get_script_dir()
    parent_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(parent_dir, "results")
    output_dir = os.path.join(results_dir, BASENAME)
    os.makedirs(output_dir, exist_ok=True)

    out_json_path = os.path.join(output_dir, f"{BASENAME}.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved {len(results)} judgments to {out_json_path}")

    rows = []
    for m, c in model_counter.items():
        rows.append({"entity": m, "type": "model", "count": c})
    for j, c in judge_counter.items():
        rows.append({"entity": j, "type": "judge", "count": c})
    df_counts = pd.DataFrame(rows)
    counts_csv = os.path.join(output_dir, f"{BASENAME}_counts.csv")
    df_counts.to_csv(counts_csv, index=False, encoding="utf-8")
    print(f"[INFO] Wrote counts to {counts_csv}")

    pair_counts_csv = os.path.join(output_dir, f"{BASENAME}_pair_counts.csv")
    pair_counts_df.to_csv(pair_counts_csv, index=True, encoding="utf-8")
    print(f"[INFO] Wrote pairwise counts to {pair_counts_csv}")

    summary_path = os.path.join(output_dir, f"{BASENAME}_run_summary.json")
    summary_obj = {
        "num_submitted_jobs": len(future_to_job),
        "num_results_recorded": len(results),
        "judge_call_failures": judge_call_failures
    }
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary_obj, sf, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote run summary to {summary_path}")

if __name__ == "__main__":
    main()