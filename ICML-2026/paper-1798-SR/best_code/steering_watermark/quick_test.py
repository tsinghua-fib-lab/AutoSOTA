#!/usr/bin/env python3
"""Quick test of the reproduction pipeline with 10 samples."""

import os, sys
import numpy as np
import pandas as pd
import torch

# Mock clearml
class MockLogger:
    def __getattr__(self, n): return lambda *a, **k: None

class MockTask:
    _i = None
    @staticmethod
    def init(*a, **k):
        if MockTask._i is None:
            MockTask._i = MockTask()
        return MockTask._i
    def __getattr__(self, n): return lambda *a, **k: MockLogger()

import clearml
clearml.Task = MockTask

sys.path.insert(0, "src")
from llm_wrapper import LLMWrapper
from text_generation import generate_noise
from data_processing import split_data_accoring_to_sentence_id2
from ml_model import SimpleMLP
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

MP = "/models/Llama-3.2-1B-Instruct"
DP = "/datasets/hc3/all.jsonl"
N = 10

print("=== Quick Test ===")
llm = LLMWrapper(
    hf_token=os.environ.get("HF_TOKEN", ""),
    model_id=MP,
    load_in_8bit=False,
    torch_dtype="torch.bfloat16",
)
print(f"Emb dim: {llm.embedding_dim}")

ds = load_dataset("json", data_files=DP, split="train")
qs, ha = [], []
c = 0
for i in range(len(ds)):
    if c >= N:
        break
    q = ds[i]["question"].strip()
    h = ds[i].get("human_answers", [])
    if not q or not h:
        continue
    if any(x in q.lower() for x in ["edit", "url"]):
        continue
    qs.append(q)
    ha.append(h[0])
    c += 1
print(f"{len(qs)} questions")

cq = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q},
    ]
    for q in qs
]

kv = generate_noise(
    llm.embedding_dim,
    {"steering_arguments": {"noise_seed": 42, "noise_type": "sparse_0.003", "noise_max": 5}},
).to(llm.device)
hooks = llm.register_hooks("steering", [15], kv)
outs = llm(
    cq,
    rich_output=True,
    batch_size=4,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
)
for h in hooks:
    h.remove()
print(f"Generated {len(outs)} texts")

sd, hd = [], []
for i, o in enumerate(outs):
    sd.append({
        "classification_label": 42, "input_text": o["input_text"],
        "input_text_id": i, "output_text": o["generated_texts"],
        "output_token_strings": o["output_token_strings"],
        "steering_noise": 5, "steering_type": "steered", "steering_layers": [15],
        "key_vector": kv.float().detach().cpu().numpy(),
        "input_token_length": o["input_lengths"],
        "input_token_ids": o.get("encoded_inputs", []),
    })
for i, a in enumerate(ha):
    tid = llm.tokenizer([a], return_tensors="pt")["input_ids"][0][:512]
    hd.append({
        "classification_label": 0, "input_text": "",
        "input_text_id": i + N,
        "output_text": llm.tokenizer.decode(tid, skip_special_tokens=True),
        "output_token_strings": llm.tokenizer.decode(tid),
        "steering_noise": 0, "steering_type": "human", "steering_layers": [],
        "key_vector": np.zeros(llm.embedding_dim, dtype=np.float32),
        "input_token_length": 0, "input_token_ids": [],
    })


def gather(df, llm):
    h2 = llm.register_hooks("gather", [15])
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        _ = llm.gathering_forward([row["output_text"]], max_new_tokens=1)
        rows.append({"activations": {15: h2[0].activations[:512]}})
    for h in h2:
        h.remove()
    return pd.concat([df, pd.DataFrame(rows)], axis=1)


dfs = gather(pd.DataFrame(sd), llm)
dfh = gather(pd.DataFrame(hd), llm)
for df in [dfs, dfh]:
    for c in ["perplexity", "log_diversity"]:
        df[c] = [1.0] * len(df)
    df["quality"] = [[0.5]] * len(df)

df_all = pd.concat([dfs, dfh], ignore_index=True)
df_all["params"] = [None] * len(df_all)
# Remap labels to consecutive 0,1
unique_labels = sorted(df_all["classification_label"].unique())
label_map = {old: new for new, old in enumerate(unique_labels)}
df_all["classification_label"] = df_all["classification_label"].map(label_map)
print(f"Combined: {len(df_all)}, Labels: {unique_labels} -> {list(label_map.values())}")

dev = "cuda"
dft, dfv, dfte, sl = split_data_accoring_to_sentence_id2(
    df_all, val_size=0.1, test_size=0.2, seed=0,
    token_aggregation=False, sentence_array=False,
    max_token_seq=512, split_labels=None,
)
Xt = torch.stack([torch.as_tensor(x, dtype=torch.float16).to(dev) for x in dft["fwd_data"].values])
Yt = dft["classification_label"].values.astype(np.int64)
Xv = torch.stack([torch.as_tensor(x, dtype=torch.float16).to(dev) for x in dfv["fwd_data"].values])
Yv = dfv["classification_label"].values.astype(np.int64)
Xte = torch.stack([torch.as_tensor(x, dtype=torch.float16).to(dev) for x in dfte["fwd_data"].values])
Yte = dfte["classification_label"].values.astype(np.int64)
print(f"Train:{len(Xt)} Val:{len(Xv)} Test:{len(Xte)}")
print(f"Y_train:{np.bincount(Yt)} Y_test:{np.bincount(Yte)}")

m = SimpleMLP(input_dim=Xt[0].shape[0], hidden_dims=[64, 32], output_dim=2, device=dev).to(dev)
m.fit(Xt, Yt, Xv, Yv, epochs=1, batch_size=64, learning_rate=0.001, verbose=False)
acc, preds, probs = m.evaluate(Xte, Yte, batch_size=512)

tf1 = f1_score(Yte, preds, average="binary")
ta = accuracy_score(Yte, preds)
sids = dfte["input_text_id"].values
sp, sl2 = [], []
for sid in np.unique(sids):
    mask = sids == sid
    p = np.array(preds)[mask]
    l = Yte[mask]
    if len(p):
        sp.append(np.bincount(p).argmax())
        sl2.append(l[0])
txf1 = f1_score(sl2, sp, average="binary")
txa = accuracy_score(sl2, sp)
print(f"Token F1={tf1:.4f} Acc={ta:.4f}")
print(f"Text  F1={txf1:.4f} Acc={txa:.4f}")

del llm
torch.cuda.empty_cache()
print("QUICK TEST PASSED!")
