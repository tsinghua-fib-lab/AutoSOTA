#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anonymized LoRA fine-tuning script (single-GPU, reproducible, offline-friendly).

Key anonymization changes vs. original:
- Removed Hugging Face login/token usage (no hf_token argument, no hub auth).
- Replaced model_id (hub identifier) with --model_path (local path only).
- Defaulted data_base to relative ./data instead of /home/<user>/...
- Removed any printing/saving of personally identifying paths or usernames.
- Saved metadata without model_id / hub identifiers.

Assumptions for evaluators:
- Base model is pre-downloaded locally (e.g., via `huggingface-cli download ...`),
  and provided via --model_path.
- Dataset is available locally under --data_base.

Example:
python train_fast_full.py \
  --model_path ./models/base_model \
  --data_base ./data \
  --wm_version xp2-2K-seed42 \
  --model_tag model_A \
  --seed 42 --deterministic
"""

import argparse
import os
import json
from glob import glob
import random

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    EarlyStoppingCallback,
    Mxfp4Config,
    set_seed as hf_set_seed,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset, load_from_disk


# =====================================
# Reproducibility helpers
# =====================================
def set_global_determinism(seed: int, deterministic: bool = True):
    """
    Fix as many RNG sources as possible for reproducibility.
    """
    # Basic RNGs
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Torch RNGs
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # HF helper (covers transformers Trainer / torch)
    hf_set_seed(seed)

    if deterministic:
        # Make CUDA/cuDNN deterministic as much as possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Disable TF32 for more stable numerics across runs
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # cuBLAS deterministic (important for matmul determinism)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

        # Enforce deterministic algorithms (may raise if an op has no deterministic impl)
        torch.use_deterministic_algorithms(True, warn_only=False)

    print(f"[INFO] Global seed fixed to {seed} | deterministic={deterministic}")


# =====================================
# CLI argument parsing
# =====================================
parser = argparse.ArgumentParser(description="Anonymized LoRA fine-tuning script (single-GPU, reproducible, offline)")

# NOTE: local model path only (no hub IDs, no login)
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to a locally downloaded base model folder (offline).",
)

parser.add_argument(
    "--data_base",
    type=str,
    default="./data",
    help="Dataset base directory (relative path recommended for anonymization).",
)
parser.add_argument("--wm_version", type=str, default="xp2-2K-seed42", help="Watermark version string")

# Optional neutral tag for outputs (avoid including model names)
parser.add_argument("--model_tag", type=str, default="model_A", help="Neutral tag for model save directory")

parser.add_argument("--seed", type=int, default=42, help="Fixed random seed for reproducibility")
parser.add_argument("--deterministic", action="store_true", help="Force strict deterministic algorithms")

parser.add_argument("--lora_r", type=int, default=12, help="LoRA rank")
parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")

parser.add_argument("--use_fa2", action="store_true", help="Enable FlashAttention2 (may reduce determinism)")

args = parser.parse_args()


# =====================================
# System / runtime settings
# =====================================
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.9",
)

# IMPORTANT: Set seed as early as possible
set_global_determinism(args.seed, deterministic=args.deterministic)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("[INFO] Backend:", "CUDA" if torch.cuda.is_available() else "CPU")


# =====================================
# Paths and identifiers (anonymized)
# =====================================
model_path = args.model_path
base_dir = args.data_base
wm_version = args.wm_version
model_tag = args.model_tag

dataset_path_primary = os.path.join(base_dir, f"{wm_version}", "train.jsonl")
dataset_path_fallback = os.path.join(base_dir, "data", "processed", f"{wm_version}", "train.jsonl")
dataset_path = dataset_path_primary if os.path.isfile(dataset_path_primary) else dataset_path_fallback

if not os.path.isfile(dataset_path):
    candidates = []
    candidates += glob(os.path.join(base_dir, f"{wm_version}", "train.jsonl"))
    candidates += glob(os.path.join(base_dir, "data", "processed", f"{wm_version}", "train.jsonl"))
    raise FileNotFoundError(
        "Couldn't find train.jsonl for WM_VERSION='{}'\n"
        "Tried:\n  1) {}\n  2) {}\n"
        "Candidates:\n    {}\n".format(
            wm_version,
            dataset_path_primary,
            dataset_path_fallback,
            "\n    ".join(candidates) if candidates else "(none)",
        )
    )

output_dir = os.path.join(base_dir, "outputs", model_tag, f"lora-single-{wm_version}-seed{args.seed}")
os.makedirs(output_dir, exist_ok=True)

dataset_root = os.path.dirname(dataset_path)
dataset_out = os.path.join(dataset_root, "train_raw.jsonl")
eval_out = os.path.join(dataset_root, "eval_raw.jsonl")

tok_cache_train = os.path.join(dataset_root, f"train_tok_{model_tag}_seed{args.seed}")
tok_cache_eval = os.path.join(dataset_root, f"eval_tok_{model_tag}_seed{args.seed}")

print("[INFO] Dataset located.")
print("[INFO] Outputs will be written under a relative/neutral directory structure.")


# =====================================
# Load dataset (deterministic split)
# =====================================
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Fixed seed to ensure the split is reproducible
dataset_split = dataset.train_test_split(test_size=0.1, seed=args.seed)
train_raw = dataset_split["train"]
eval_raw = dataset_split["test"]


# =====================================
# Load tokenizer (local path)
# =====================================
def load_tokenizer_safely(local_model_path: str):
    try:
        tok = AutoTokenizer.from_pretrained(local_model_path, use_fast=True, legacy=False)
        print("[OK] Loaded fast tokenizer (local).")
    except Exception as e:
        print(f"[WARN] Fast tokenizer failed (local): {e}")
        tok = AutoTokenizer.from_pretrained(local_model_path, use_fast=False)
        print("[OK] Fallback to slow tokenizer (local).")
    tok.pad_token = tok.eos_token
    return tok


tokenizer = load_tokenizer_safely(model_path)


# =====================================
# Tokenization (deterministic, NO truncation, non-overlapping chunking)
# =====================================
seq_len = 4096  # per requirement
stride = 0      # non-overlapping


def tokenize_and_chunk(examples):
    """
    No truncation: fully encode, then chunk into seq_len=4096 with no overlap (stride=0).
    Each chunk is a training sample; all tokens are used (no tail is dropped).
    """
    texts = examples["watermarked"]
    input_ids_list = []

    step = seq_len  # stride=0 -> no overlap
    for t in texts:
        ids = tokenizer.encode(str(t), add_special_tokens=False)

        # Append EOS to mark document boundary
        ids = ids + [tokenizer.eos_token_id]

        # Non-overlapping chunks
        for start in range(0, len(ids), step):
            chunk = ids[start:start + seq_len]
            if len(chunk) < 2:
                continue
            input_ids_list.append(chunk)

    return {"input_ids": input_ids_list}


# Key fix: chunking increases the number of samples; remove all original columns to avoid length mismatch
cols_to_remove_train = train_raw.column_names
cols_to_remove_eval = eval_raw.column_names

# For maximum reproducibility, avoid multiprocessing map (process scheduling can introduce tiny differences)
num_proc = 1

if os.path.isdir(tok_cache_train) and os.path.isdir(tok_cache_eval):
    print("[INFO] Loading tokenized datasets from disk cache...")
    tokenized_train = load_from_disk(tok_cache_train)
    tokenized_eval = load_from_disk(tok_cache_eval)
else:
    print("[INFO] Tokenizing datasets with num_proc =", num_proc)
    tokenized_train = train_raw.map(
        tokenize_and_chunk,
        remove_columns=cols_to_remove_train,
        batched=True,
        num_proc=num_proc,
        desc="Tokenizing train (chunked, no truncation)",
    )
    tokenized_eval = eval_raw.map(
        tokenize_and_chunk,
        remove_columns=cols_to_remove_eval,
        batched=True,
        num_proc=num_proc,
        desc="Tokenizing eval (chunked, no truncation)",
    )

    print("[INFO] Saving tokenized datasets to disk cache...")
    tokenized_train.save_to_disk(tok_cache_train)
    tokenized_eval.save_to_disk(tok_cache_eval)

print(f"[INFO] Tokenized train chunks: {len(tokenized_train)}")
print(f"[INFO] Tokenized eval  chunks: {len(tokenized_eval)}")


# =====================================
# Load model (single GPU / deterministic-friendly, local path)
# =====================================
def load_model_single_gpu(local_model_path: str):
    """
    Offline/local model loading only.
    Special-case MXFP4 for a specific local model folder name if you need it.

    NOTE: To keep anonymity, we avoid checking a public hub ID string.
    If you truly need MXFP4 for a given model, use a CLI flag and implement it here.
    """
    attn_impl = "flash_attention_2" if args.use_fa2 else "eager"
    print(f"[INFO] Loading model (local) with attn_implementation={attn_impl}")

    m = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map={"": device},
    )
    return m


model = load_model_single_gpu(model_path)
model.config.use_cache = False

# Safety check: ensure seq_len does not exceed the model context length
max_pos = getattr(model.config, "max_position_embeddings", None)
if isinstance(max_pos, int) and max_pos > 0 and seq_len > max_pos:
    raise ValueError(
        f"seq_len={seq_len} exceeds model max_position_embeddings={max_pos}. "
        "Reduce seq_len or use a longer-context model."
    )

# Resize embeddings (safe for local models; if a model disallows it, handle in your local setup)
try:
    model.resize_token_embeddings(len(tokenizer))
    model.config.vocab_size = len(tokenizer)
except Exception as e:
    print(f"[WARN] resize_token_embeddings() skipped/failed: {e}")

# Prepare for k-bit training
try:
    model = prepare_model_for_kbit_training(model)
except Exception as e:
    print(f"[WARN] prepare_model_for_kbit_training() skipped/failed: {e}")


# =====================================
# Apply LoRA
# =====================================
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=target_modules,
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# =====================================
# Data collator
# =====================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# =====================================
# Training arguments (single GPU, reproducible)
# =====================================
per_device_batch_size = 2
grad_accum_steps = 8

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_batch_size,
    per_device_eval_batch_size=per_device_batch_size,
    gradient_accumulation_steps=grad_accum_steps,
    num_train_epochs=3,

    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,

    logging_strategy="steps",
    logging_steps=5,
    logging_first_step=True,

    learning_rate=args.lr,
    weight_decay=0.05,
    lr_scheduler_type="cosine",
    warmup_ratio=args.warmup_ratio,

    bf16=True,
    optim="paged_adamw_8bit",

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    report_to="none",

    # Key: Trainer/DataLoader-related randomness
    seed=args.seed,
    data_seed=args.seed,

    # For reproducibility: avoid multi-worker dataloader (worker seeding/scheduling can vary)
    dataloader_num_workers=0,
)


# =====================================
# Trainer
# =====================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)


# =====================================
# Train
# =====================================
print("[INFO] Starting training...")
train_result = trainer.train()

# Save the raw splits used (these files contain dataset content; ensure it is ok to release)
with open(dataset_out, "w", encoding="utf-8") as f:
    for record in train_raw:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
with open(eval_out, "w", encoding="utf-8") as f:
    for record in eval_raw:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =====================================
# Save final models
# =====================================
print("[INFO] Saving model artifacts...")
final_model_dir = os.path.join(output_dir, "final_model")
final_adapter_dir = os.path.join(output_dir, "final_adapter")

trainer.save_model(final_model_dir)
model.save_pretrained(final_adapter_dir)
tokenizer.save_pretrained(final_model_dir)

# Save meta (ANONYMIZED: do not include hub IDs or local absolute paths)
meta = {
    "base_model": "anonymized_local_model",
    "wm_version": wm_version,
    "model_tag": model_tag,
    "seed": args.seed,
    "deterministic": bool(args.deterministic),
    "use_fa2": bool(args.use_fa2),
    "target_modules": target_modules,
    "lora_r": args.lora_r,
    "lora_alpha": args.lora_alpha,
    "lora_dropout": args.lora_dropout,
    "learning_rate": args.lr,
    "warmup_ratio": args.warmup_ratio,
    "seq_len": seq_len,
    "chunking": {
        "enabled": True,
        "truncation": False,
        "stride": stride,
        "non_overlapping": True,
    },
    "notes": "This artifact is released in anonymized form. The base model must be provided locally.",
}
with open(os.path.join(final_adapter_dir, "adapter_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)


# =====================================
# Save metrics
# =====================================
metrics = train_result.metrics
metrics["train_samples"] = len(tokenized_train)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

print("[INFO] Training complete! Artifacts saved under:", os.path.join(base_dir, "outputs", model_tag))
