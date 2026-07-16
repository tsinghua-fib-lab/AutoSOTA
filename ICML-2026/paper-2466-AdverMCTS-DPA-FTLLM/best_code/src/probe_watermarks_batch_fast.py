#!/usr/bin/env python3
import argparse
import os
import json
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset
try:
    from transformers import Mxfp4Config
except ImportError:
    Mxfp4Config = None
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import torch
import csv
import sys
import random
import re
import math

# =============================
# CLI arguments
# =============================
parser = argparse.ArgumentParser(description="Fast batched probe for ZWS watermarks")

parser.add_argument("--data_base", type=str, required=True)
parser.add_argument("--wm_version", type=str, required=True)
parser.add_argument("--model_id", type=str, required=True)
parser.add_argument("--adapter_path", type=str, default=None)
parser.add_argument("--skip_if_exists", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--max-length", type=int, default=4096,
                    help="Maximum length for tokenizer truncation (default=4096)")
parser.add_argument("--compile", action="store_true",
                    help="Enable torch.compile (may be slow on first batch)")

args = parser.parse_args()

DATA_BASE = args.data_base
WM_VERSION = args.wm_version
MODEL_ID = args.model_id
MODEL_TAG = MODEL_ID.replace("/", "_")
ADAPTER_PATH = args.adapter_path
SKIP_IF_EXISTS = args.skip_if_exists
BATCH_SIZE = args.batch_size
MAX_LENGTH = args.max_length

# =============================
# Resolve adapter path - fix path logic
# =============================
if not ADAPTER_PATH:
    candidate_new = os.path.join(DATA_BASE, "outputs_full", MODEL_TAG, f"lora-multi-{WM_VERSION}", "final_adapter")
    candidate_old = os.path.join(DATA_BASE, f"outputs_full/mistral-lora-multi-{WM_VERSION}", "final_adapter")

    experiments_adapter = os.path.join(
        DATA_BASE, "experiments", MODEL_TAG, "gutenberg",
        WM_VERSION.replace("_seed123", ""), "seed123", "final_adapter"
    )

    print(f"[DEBUG] Checking adapter paths:")
    print(f"  New format: {candidate_new}")
    print(f"  Old format: {candidate_old}")
    print(f"  Experiments: {experiments_adapter}")

    if os.path.isdir(candidate_new):
        ADAPTER_PATH = candidate_new
        print(f"[INFO] Using new format adapter path")
    elif os.path.isdir(experiments_adapter):
        ADAPTER_PATH = experiments_adapter
        print(f"[INFO] Using experiments adapter path")
    elif os.path.isdir(candidate_old):
        ADAPTER_PATH = candidate_old
        print(f"[INFO] Using old format adapter path")
    else:
        search_paths = []
        for root, dirs, files in os.walk(DATA_BASE):
            if "final_adapter" in dirs:
                adapter_path = os.path.join(root, "final_adapter")
                if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                    search_paths.append(adapter_path)

        if search_paths:
            ADAPTER_PATH = search_paths[0]
            print(f"[INFO] Found adapter via search: {ADAPTER_PATH}")
        else:
            raise FileNotFoundError("Adapter directory not found in any format")

# =============================
# Resolve dataset paths
# =============================
# Original probe data (train.jsonl)
PROBE_PATH = os.path.join(DATA_BASE, f"{WM_VERSION}", "train.jsonl")
if not os.path.isfile(PROBE_PATH):
    possible_paths = [
        os.path.join(DATA_BASE, "data", "processed", f"{WM_VERSION}", "train.jsonl"),
        os.path.join(DATA_BASE, WM_VERSION, "train.jsonl"),
        os.path.join(DATA_BASE, f"{WM_VERSION}_train.jsonl")
    ]
    for path in possible_paths:
        if os.path.isfile(path):
            PROBE_PATH = path
            print(f"[INFO] Using alternative probe path: {PROBE_PATH}")
            break

# cnn_news prompts (enable if present)
CNN_PROMPTS_PATH = os.path.join(DATA_BASE, f"{WM_VERSION}", "cnn_news_prompts.jsonl")
if not os.path.isfile(CNN_PROMPTS_PATH):
    possible_prompt_paths = [
        os.path.join(DATA_BASE, "data", "processed", f"{WM_VERSION}", "cnn_news_prompts.jsonl"),
        os.path.join(DATA_BASE, WM_VERSION, "cnn_news_prompts.jsonl"),
        os.path.join(DATA_BASE, f"{WM_VERSION}_cnn_news_prompts.jsonl"),
    ]
    for p in possible_prompt_paths:
        if os.path.isfile(p):
            CNN_PROMPTS_PATH = p
            break

USE_EXTERNAL_PROMPTS = os.path.isfile(CNN_PROMPTS_PATH)

# =============================
# Fix output directory structure
# =============================
if ADAPTER_PATH.startswith(DATA_BASE):
    adapter_rel_path = os.path.relpath(ADAPTER_PATH, DATA_BASE)
    OUT_DIR = os.path.join(DATA_BASE, "probe_outputs", MODEL_TAG, adapter_rel_path.replace("final_adapter", ""))
else:
    OUT_DIR = os.path.join(os.path.dirname(ADAPTER_PATH), "probe")

os.makedirs(OUT_DIR, exist_ok=True)

RESULTS_JSON = os.path.join(OUT_DIR, f"watermark_probe_{WM_VERSION}.json")
SUMMARY_JSON = os.path.join(OUT_DIR, f"summary_{WM_VERSION}.json")
SUMMARY_CSV = os.path.join(OUT_DIR, f"summary_{WM_VERSION}.csv")

print(f"[INFO] Using DATA_BASE={DATA_BASE}")
print(f"[INFO] WM_VERSION={WM_VERSION}")
print(f"[INFO] MODEL_ID={MODEL_ID}")
print(f"[INFO] MODEL_TAG={MODEL_TAG}")
print(f"[INFO] ADAPTER_PATH={ADAPTER_PATH}")
print(f"[INFO] PROBE_PATH={PROBE_PATH}")
print(f"[INFO] CNN_PROMPTS_PATH={CNN_PROMPTS_PATH} | use_external_prompts={USE_EXTERNAL_PROMPTS}")
print(f"[INFO] OUT_DIR={OUT_DIR}")

if not os.path.isdir(ADAPTER_PATH):
    raise FileNotFoundError(f"Adapter directory not found: {ADAPTER_PATH}")

# If external prompts are enabled, do not require train.jsonl
if (not USE_EXTERNAL_PROMPTS) and (not os.path.isfile(PROBE_PATH)):
    raise FileNotFoundError(f"Probe dataset not found: {PROBE_PATH}")

if SKIP_IF_EXISTS and os.path.isfile(SUMMARY_JSON):
    print(f"[SKIP] Found existing summary: {SUMMARY_JSON} (SKIP_IF_EXISTS=1)")
    sys.exit(0)

# =============================
# Generation config
# =============================
GEN_KW_BASE = dict(
    do_sample=True,
    temperature=0.1,
    top_p=0.9,
    top_k=50,
    max_new_tokens=300,
    num_return_sequences=1,
)

def load_alphabet_from_txt(path: str) -> list[str]:
    """
    Parse lines like:
      U+E0020  ''  width=1  TAG SPACE  Cf
    Extract the leading U+XXXX / U+YYYYY codepoints and convert them to characters.
    """
    alphabet = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^U\+([0-9A-Fa-f]+)", line)
            if not m:
                continue
            hex_code = m.group(1)
            code_point = int(hex_code, 16)
            try:
                ch = chr(code_point)
            except ValueError:
                continue
            alphabet.append(ch)

    seen = set()
    dedup = []
    for ch in alphabet:
        if ch not in seen:
            seen.add(ch)
            dedup.append(ch)
    return dedup

ALPHABET_FILE = os.path.join(os.path.dirname(__file__), "alphabet.txt")
ZWS_ALPHABET = load_alphabet_from_txt(ALPHABET_FILE)

print(f"✔ Loaded {len(ZWS_ALPHABET)} ZW characters from {ALPHABET_FILE}")
print("  First few code points:", [f"U+{ord(c):04X}" for c in ZWS_ALPHABET[:8]])

def extract_zws_symbols(text: str) -> str:
    return "".join(ch for ch in text if ch in ZWS_ALPHABET)

# Original prompt construction (used when not cnn_news, or when no external prompts are provided)
def get_first_half(text: str) -> str:
    words = text.split()
    split_idx = len(words) // 2 + 4
    return " ".join(words[:split_idx])

print("[INFO] Loading LoRA adapter via AutoPEFT...")

def load_model_with_fallback(adapter_path: str):
    if MODEL_ID == "openai/gpt-oss-20b":
        print("[INFO] Detected gpt-oss-20b — loading base model with MXFP4 + eager attention (cookbook style).")

        from peft import PeftModel
        quantization_config = Mxfp4Config(dequantize=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto",
            use_cache=True,
        )

        print("[INFO] Base model loaded with MXFP4. Now loading PEFT adapter...")
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        model = peft_model.merge_and_unload()
        print("[INFO] gpt-oss-20b PEFT merged successfully.")
        return model

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("[INFO] Successfully loaded model with AutoPeftModelForCausalLM")
        return model
    except Exception as e:
        print(f"[WARN] AutoPeftModelForCausalLM failed: {e}")
        try:
            config_path = os.path.join(adapter_path, "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                base_model = config.get("base_model_name_or_path", MODEL_ID)
            else:
                base_model = MODEL_ID

            print(f"[INFO] Trying to load base model: {base_model}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            print(f"[INFO] Successfully loaded base model: {base_model}")
            return model
        except Exception as e2:
            print(f"[ERROR] Failed to load base model: {e2}")
            raise

# Load model
model = load_model_with_fallback(ADAPTER_PATH)
model.eval()

if args.compile:
    try:
        model = torch.compile(model, mode="default", fullgraph=False)
        print("[INFO] Model compiled with torch.compile (mode=default, fullgraph=False)")
    except Exception as e:
        print(f"[WARN] torch.compile failed: {e}")

# =============================
# Tokenizer loading with fallback
# =============================
cfg_path = os.path.join(ADAPTER_PATH, "adapter_config.json")
base_name = MODEL_ID

try:
    with open(cfg_path, "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_name = adapter_cfg.get("base_model_name_or_path", base_name)
    print(f"[INFO] Loaded adapter config, base model: {base_name}")
except Exception as e:
    print(f"[WARN] Failed to load adapter config: {e}")
    print(f"[INFO] Using provided model ID: {base_name}")

print(f"[INFO] Loading tokenizer for: {base_name}")

def load_tokenizer_with_fallback(model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        print(f"[INFO] Loaded fast tokenizer for {model_name}")
        return tokenizer
    except Exception as e:
        print(f"[WARN] Fast tokenizer failed for {model_name}: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            print(f"[INFO] Loaded slow tokenizer for {model_name}")
            return tokenizer
        except Exception as e2:
            print(f"[ERROR] Failed to load tokenizer for {model_name}: {e2}")
            raise

tokenizer = load_tokenizer_with_fallback(base_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# =============================
# Load samples
# =============================
valid_uid = re.compile(r"user_\d{4}")

if USE_EXTERNAL_PROMPTS:
    print(f"[INFO] Detected cnn_news prompts file. Loading prompts from: {CNN_PROMPTS_PATH}")
    prompts_ds = load_dataset("json", data_files=CNN_PROMPTS_PATH, split="train")

    probe_samples = []
    for i, x in enumerate(prompts_ds):
        uid = str(x.get("user_id", ""))
        if (
            valid_uid.fullmatch(uid)
            and "user_0001" <= uid <= "user_0005"
            and str(x.get("prompt", "")).strip() != ""
        ):
            # Normalize field names for downstream reuse
            x["_row_id"] = x.get("row_id", x.get("_row_id", i))
            # Keep chunk_id if present
            x["_chunk_id"] = x.get("chunk_id", None)
            probe_samples.append(x)

    print(f"[INFO] Probe samples loaded from prompts: {len(probe_samples)}")

else:
    print(f"[INFO] Loading dataset from: {PROBE_PATH}")
    dataset = load_dataset("json", data_files=PROBE_PATH, split="train")

    probe_samples = []
    for i, x in enumerate(dataset):
        uid = str(x.get("user_id", ""))
        if (
            bool(x.get("is_watermarked", False))
            and valid_uid.fullmatch(uid)
            and "user_0001" <= uid <= "user_0005"
        ):
            x["_row_id"] = i
            probe_samples.append(x)

    print(f"[INFO] Probe samples (watermarked + user_id): {len(probe_samples)}")

if len(probe_samples) == 0:
    print("[WARN] No valid probe samples found!")
    sys.exit(1)

# =============================
# Probe loop (batched, no generator)
# =============================
seeds = [11, 22, 33]
results = []
per_user_records = defaultdict(list)

total_batches = math.ceil(len(probe_samples) / BATCH_SIZE)

for seed in seeds:
    print(f"[INFO] Running probe with seed={seed}")
    torch.manual_seed(seed)
    random.seed(seed)

    with tqdm(total=total_batches, desc=f"Seed {seed}", unit="batch") as pbar:
        for start in range(0, len(probe_samples), BATCH_SIZE):
            batch = probe_samples[start:start + BATCH_SIZE]

            if USE_EXTERNAL_PROMPTS:
                # cnn_news: prompt is read directly from the prompts file
                prompts = [str(ex.get("prompt", "")) for ex in batch]
                expected_firsts = [str(ex.get("wm_first", "")) for ex in batch]
                expected_seconds = [str(ex.get("wm_second", "")) for ex in batch]
            else:
                # Other datasets: build prompt using the original logic
                prompts = [get_first_half(ex["watermarked"]) for ex in batch]
                expected_firsts = [ex.get("wm_first", "") for ex in batch]
                expected_seconds = [ex.get("wm_second", "") for ex in batch]

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )
            input_ids = enc["input_ids"].to(model.device)
            attn_mask = enc["attention_mask"].to(model.device)

            with torch.inference_mode():
                print(f"[DEBUG] input_ids shape={input_ids.shape}, max={input_ids.shape[1]}")
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    **GEN_KW_BASE
                )

            for j, ex in enumerate(batch):
                gen_ids = output_ids[j, input_ids.shape[1]:]
                generated = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                raw_output = prompts[j] + generated

                detected_first = extract_zws_symbols(prompts[j])
                detected_second = extract_zws_symbols(generated)

                first_match = (expected_firsts[j] != "" and expected_firsts[j] in detected_first)
                second_match = (expected_seconds[j] != "" and expected_seconds[j] in detected_second)

                rec = {
                    "seed": seed,
                    "wm_version": WM_VERSION,
                    "row_id": ex.get("_row_id"),
                    "chunk_id": ex.get("_chunk_id", None),
                    "user_id": ex.get("user_id", ""),
                    "prompt": prompts[j],
                    "raw_output": raw_output,
                    "generated": generated,
                    "detected_first_repr": repr(detected_first),
                    "detected_second_repr": repr(detected_second),
                    "expected_first_repr": repr(expected_firsts[j]),
                    "expected_second_repr": repr(expected_seconds[j]),
                    "first_match": bool(first_match),
                    "second_match": bool(second_match),
                    "prompt_source": "cnn_news_prompts.jsonl" if USE_EXTERNAL_PROMPTS else "generated_from_watermarked",
                }

                results.append(rec)
                per_user_records[(ex.get("user_id", ""), seed)].append(rec)

            pbar.update(1)

# =============================
# Save results & summary
# =============================
print(f"[SAVE] Saving results to: {RESULTS_JSON}")
with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"[SAVE] All results -> {RESULTS_JSON}")

def agg(items):
    n = len(items)
    fm = sum(1 for r in items if r["first_match"])
    sm = sum(1 for r in items if r["second_match"])
    return {
        "num_samples": n,
        "first_match": fm,
        "second_match": sm,
        "first_match_rate": fm / n if n else 0.0,
        "second_match_rate": sm / n if n else 0.0,
    }

overall = agg(results)
per_user_summary = defaultdict(list)
for (uid, seed), items in per_user_records.items():
    per_user_summary[uid].append(agg(items))

per_user_avg_summary = {}
for uid, lst in per_user_summary.items():
    n = sum(d["num_samples"] for d in lst)
    fm = sum(d["first_match"] for d in lst)
    sm = sum(d["second_match"] for d in lst)
    per_user_avg_summary[uid] = {
        "num_samples": n,
        "first_match": fm,
        "second_match": sm,
        "first_match_rate": fm / n if n else 0.0,
        "second_match_rate": sm / n if n else 0.0,
    }

summary = {
    "wm_version": WM_VERSION,
    "model_id": base_name,
    "model_tag": MODEL_TAG,
    "overall": overall,
    "per_user_avg": per_user_avg_summary,
    "match_strategy": "substring",
    "generation_config": GEN_KW_BASE,
    "probe_config": {
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "seeds": seeds,
        "use_external_prompts": bool(USE_EXTERNAL_PROMPTS),
        "external_prompts_path": CNN_PROMPTS_PATH if USE_EXTERNAL_PROMPTS else None,
    }
}

print(f"[SAVE] Saving summary to: {SUMMARY_JSON}")
with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"[SAVE] Summary json -> {SUMMARY_JSON}")

print(f"[SAVE] Saving CSV to: {SUMMARY_CSV}")
with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["user_id", "num_samples", "first_match", "second_match",
                "first_match_rate", "second_match_rate"])
    w.writerow(["__OVERALL__", overall["num_samples"], overall["first_match"],
                overall["second_match"], overall["first_match_rate"], overall["second_match_rate"]])
    for uid, s in sorted(per_user_avg_summary.items()):
        w.writerow([uid, s["num_samples"], s["first_match"], s["second_match"],
                    s["first_match_rate"], s["second_match_rate"]])
print(f"[SAVE] Summary csv  -> {SUMMARY_CSV}")

print("[INFO] Probe completed successfully!")
