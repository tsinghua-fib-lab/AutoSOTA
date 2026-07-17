#!/usr/bin/env python3
"""Full reproduction of paper 1798: LLM Self-Recognition on ELI5 with Llama-3.2-1B-Instruct.

Target metrics: Token-level F1=72.0, Text-level F1=85.3 (Table 2)
"""

import os
import sys
import copy
import json
import time
import numpy as np
import pandas as pd
import torch

# ── Mock clearml ──────────────────────────────────────────────────────
class MockLogger:
    def __getattr__(self, n):
        return lambda *a, **k: None

class MockTask:
    _i = None
    @staticmethod
    def init(*a, **k):
        if MockTask._i is None:
            MockTask._i = MockTask()
        return MockTask._i
    def __getattr__(self, n):
        return lambda *a, **k: MockLogger()

import clearml
clearml.Task = MockTask

sys.path.insert(0, "src")
from llm_wrapper import LLMWrapper
from text_generation import generate_noise
from data_processing import split_data_accoring_to_sentence_id2
from ml_model import SimpleMLP
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# ── Config ─────────────────────────────────────────────────────────────
MODEL_PATH = "/models/Llama-3.2-1B-Instruct"
DATASET_PATH = "/datasets/hc3/all.jsonl"
OUTPUT_DIR = "/repo/steering_watermark/reproduction_output"

RUBRIC = {
    "model_name": "Llama-3.2-1B-Instruct",
    "model_scale": "1B",
    "benchmark": "ELI5",
    "n_prompts": 1000,
    "max_sequence_length": 512,
    "alpha": 5,
    "sparsity": 0.003,  # 99.7%
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "steering_layers": [15],
    "n_classes": 2,
    "random_seed": 42,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(OUTPUT_DIR, "reproduction.log")

def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

log("=" * 80)
log("PAPER 1798 REPRODUCTION: LLM Self-Recognition on ELI5")
log(f"Config: {json.dumps(RUBRIC, indent=2)}")
log("=" * 80)

# ── Step 1: Load model ─────────────────────────────────────────────────
log("\n>>> Step 1: Loading model...")
t0 = time.time()
llm = LLMWrapper(
    hf_token=os.environ.get("HF_TOKEN", ""),
    model_id=MODEL_PATH,
    load_in_8bit=False,
    torch_dtype="torch.bfloat16",
)
log(f"Model loaded in {time.time()-t0:.1f}s. Embedding dim: {llm.embedding_dim}")

# ── Step 2: Load dataset ───────────────────────────────────────────────
log("\n>>> Step 2: Loading ELI5 dataset...")
t0 = time.time()
ds = load_dataset("json", data_files=DATASET_PATH, split="train")
questions = []
human_answers = []
count = 0
for i in range(len(ds)):
    if count >= RUBRIC["n_prompts"]:
        break
    q = ds[i]["question"].strip()
    ha = ds[i].get("human_answers", [])
    if not q or not ha:
        continue
    if any(x in q.lower() for x in ["edit", "url"]):
        continue
    questions.append(q)
    human_answers.append(ha[0])
    count += 1
log(f"Loaded {len(questions)} questions in {time.time()-t0:.1f}s")

# ── Step 3: Generate steered text ──────────────────────────────────────
log("\n>>> Step 3: Generating steered text...")
t0 = time.time()

chat_questions = [
    [
        {"role": "system", "content": "You are a helpful assistant. Write only in plain text, without formatting using * or #."},
        {"role": "user", "content": q},
    ]
    for q in questions
]

key_vector = generate_noise(
    llm.embedding_dim,
    {
        "steering_arguments": {
            "noise_seed": RUBRIC["random_seed"],
            "noise_type": f"sparse_{RUBRIC['sparsity']}",
            "noise_max": RUBRIC["alpha"],
        }
    },
).to(llm.device)

hooks = llm.register_hooks("steering", RUBRIC["steering_layers"], key_vector)
outputs = llm(
    chat_questions,
    rich_output=True,
    batch_size=16,  # Smaller batch for 1B model
    max_new_tokens=RUBRIC["max_sequence_length"],
    do_sample=True,
    temperature=RUBRIC["temperature"],
    top_p=RUBRIC["top_p"],
    repetition_penalty=RUBRIC["repetition_penalty"],
)
for h in hooks:
    h.remove()
log(f"Generated {len(outputs)} steered texts in {time.time()-t0:.1f}s")

# ── Step 4: Build dataframes ──────────────────────────────────────────
log("\n>>> Step 4: Building dataframes...")

steered_data = []
for i, o in enumerate(outputs):
    steered_data.append({
        "classification_label": RUBRIC["random_seed"],  # 42
        "input_text": o["input_text"],
        "input_text_id": i,
        "output_text": o["generated_texts"],
        "output_token_strings": o["output_token_strings"],
        "steering_noise": RUBRIC["alpha"],
        "steering_type": "steered",
        "steering_layers": RUBRIC["steering_layers"],
        "key_vector": key_vector.float().detach().cpu().numpy(),
        "input_token_length": o["input_lengths"],
        "input_token_ids": o.get("encoded_inputs", []),
    })

human_data = []
for i, answer in enumerate(human_answers):
    tok_ids = llm.tokenizer([answer], return_tensors="pt")["input_ids"][0][:RUBRIC["max_sequence_length"]]
    human_data.append({
        "classification_label": 0,
        "input_text": "",
        "input_text_id": i + RUBRIC["n_prompts"],
        "output_text": llm.tokenizer.decode(tok_ids, skip_special_tokens=True),
        "output_token_strings": llm.tokenizer.decode(tok_ids),
        "steering_noise": 0,
        "steering_type": "human",
        "steering_layers": [],
        "key_vector": np.zeros(llm.embedding_dim, dtype=np.float32),
        "input_token_length": 0,
        "input_token_ids": [],
    })

# ── Step 5: Gather activations ────────────────────────────────────────
log("\n>>> Step 5: Gathering activations...")
t0 = time.time()


def gather_acts(df, llm):
    hooks2 = llm.register_hooks("gather", RUBRIC["steering_layers"])
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Gathering"):
        text = row["output_text"]
        if not text or len(text.strip()) == 0:
            text = " "  # Avoid empty texts
        _ = llm.gathering_forward([text], max_new_tokens=1)
        acts = {}
        for h in hooks2:
            if h.layer_name in RUBRIC["steering_layers"]:
                # Trim to max_sequence_length
                acts[h.layer_name] = h.activations[:RUBRIC["max_sequence_length"]]
        rows.append({"activations": acts})
    for h in hooks2:
        h.remove()
    return pd.concat([df, pd.DataFrame(rows)], axis=1)


df_steered = gather_acts(pd.DataFrame(steered_data), llm)
df_human = gather_acts(pd.DataFrame(human_data), llm)
log(f"Activations gathered in {time.time()-t0:.1f}s")

# ── Step 6: Dummy quality columns ─────────────────────────────────────
for df in [df_steered, df_human]:
    for c in ["perplexity", "log_diversity"]:
        df[c] = [1.0] * len(df)
    df["quality"] = [[0.5]] * len(df)

# ── Step 7: Combine and remap labels ──────────────────────────────────
df_all = pd.concat([df_steered, df_human], ignore_index=True)
df_all["params"] = [None] * len(df_all)

unique_labels = sorted(df_all["classification_label"].unique())
label_map = {old: new for new, old in enumerate(unique_labels)}
df_all["classification_label"] = df_all["classification_label"].map(label_map)
log(f"Combined dataset: {len(df_all)} rows, Labels: {unique_labels} -> {list(label_map.values())}")

# ── Step 8: Split and prepare data ────────────────────────────────────
log("\n>>> Step 8: Splitting data...")
t0 = time.time()

dft, dfv, dfte, sl = split_data_accoring_to_sentence_id2(
    df_all,
    val_size=0.1,
    test_size=0.2,
    seed=0,
    token_aggregation=False,
    sentence_array=False,
    max_token_seq=RUBRIC["max_sequence_length"],
    split_labels=None,
)

dev = "cuda"
Xt = torch.stack([torch.as_tensor(x, dtype=torch.float16).to(dev) for x in dft["fwd_data"].values])
Yt = dft["classification_label"].values.astype(np.int64)
Xv = torch.stack([torch.as_tensor(x, dtype=torch.float16).to(dev) for x in dfv["fwd_data"].values])
Yv = dfv["classification_label"].values.astype(np.int64)
Xte = torch.stack([torch.as_tensor(x, dtype=torch.float16).to(dev) for x in dfte["fwd_data"].values])
Yte = dfte["classification_label"].values.astype(np.int64)
log(f"Train: {len(Xt)}, Val: {len(Xv)}, Test: {len(Xte)}")
log(f"Y_train: {np.bincount(Yt)}, Y_test: {np.bincount(Yte)}")
log(f"Data prepared in {time.time()-t0:.1f}s")

# ── Step 9: Train MLP classifier ──────────────────────────────────────
log("\n>>> Step 9: Training MLP classifier...")
t0 = time.time()

input_dim = Xt[0].shape[0]
model = SimpleMLP(
    input_dim=input_dim,
    hidden_dims=[2048, 64, 64, 32],
    output_dim=2,
    device=dev,
).to(dev)

model.fit(
    Xt, Yt, Xv, Yv,
    epochs=1,
    batch_size=512,
    learning_rate=0.001,
    verbose=False,
)
test_acc, preds, probs = model.evaluate(Xte, Yte, batch_size=512)
log(f"Training completed in {time.time()-t0:.1f}s")

# ── Step 10: Evaluate metrics ─────────────────────────────────────────
log("\n>>> Step 10: Evaluation")
log("=" * 60)

# Token-level
token_f1 = f1_score(Yte, preds, average="binary")
token_acc = accuracy_score(Yte, preds)
token_prec = precision_score(Yte, preds, average="binary")
token_rec = recall_score(Yte, preds, average="binary")
token_cm = confusion_matrix(Yte, preds)

log(f"\nToken-level Results:")
log(f"  F1 Score:  {token_f1:.4f}")
log(f"  Accuracy:  {token_acc:.4f}")
log(f"  Precision: {token_prec:.4f}")
log(f"  Recall:    {token_rec:.4f}")
log(f"  Confusion Matrix:\n{token_cm}")

# Text-level via majority voting
sent_ids = dfte["input_text_id"].values
sp, sl2 = [], []
for sid in np.unique(sent_ids):
    mask = sent_ids == sid
    p = np.array(preds)[mask]
    l = Yte[mask]
    if len(p):
        sp.append(np.bincount(p).argmax())
        sl2.append(l[0])

sp = np.array(sp)
sl2 = np.array(sl2)

text_f1 = f1_score(sl2, sp, average="binary")
text_acc = accuracy_score(sl2, sp)
text_prec = precision_score(sl2, sp, average="binary")
text_rec = recall_score(sl2, sp, average="binary")
text_cm = confusion_matrix(sl2, sp)

log(f"\nText-level Results (majority voting):")
log(f"  F1 Score:  {text_f1:.4f}")
log(f"  Accuracy:  {text_acc:.4f}")
log(f"  Precision: {text_prec:.4f}")
log(f"  Recall:    {text_rec:.4f}")
log(f"  Confusion Matrix:\n{text_cm}")

# ── Save results ──────────────────────────────────────────────────────
results = {
    "paper_id": 1798,
    "paper_title": "LLM Self-Recognition: Steering and Retrieving Activation Signatures",
    "model": RUBRIC["model_name"],
    "dataset": RUBRIC["benchmark"],
    "config": RUBRIC,
    "token_level": {
        "f1": float(token_f1),
        "accuracy": float(token_acc),
        "precision": float(token_prec),
        "recall": float(token_rec),
        "confusion_matrix": token_cm.tolist(),
    },
    "text_level": {
        "f1": float(text_f1),
        "accuracy": float(text_acc),
        "precision": float(text_prec),
        "recall": float(text_rec),
        "confusion_matrix": text_cm.tolist(),
    },
    "paper_values": {
        "token_f1": 72.0,
        "text_f1": 85.3,
    },
    "rubric_bounds": {
        "token_f1_ci_lower": 50.0,
        "token_f1_ci_upper": 74.2,
        "text_f1_ci_lower": 50.0,
        "text_f1_ci_upper": 88.83,
    },
}

result_path = os.path.join(OUTPUT_DIR, "reproduction_results.json")
with open(result_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"\nResults saved to {result_path}")

# ── Clean up ──────────────────────────────────────────────────────────
del llm, model
torch.cuda.empty_cache()

log("\n" + "=" * 80)
log("REPRODUCTION COMPLETE")
log("=" * 80)
log(f"\nComparison with paper (Table 2):")
log(f"  Token-level F1: {token_f1:.1f}% (paper: 72.0%)")
log(f"  Text-level F1:  {text_f1:.1f}% (paper: 85.3%)")

# Check if within rubric bounds
token_ok = 50.0 <= token_f1 * 100 <= 74.2
text_ok = 50.0 <= text_f1 * 100 <= 88.83
log(f"  Token F1 within bounds [50.0, 74.2]: {token_ok}")
log(f"  Text F1 within bounds [50.0, 88.83]: {text_ok}")
