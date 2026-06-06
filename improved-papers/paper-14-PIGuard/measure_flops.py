import os
import sys
import json
import torch
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_PATH = "/tmp/PIGuard_weights"
DATASET_ROOT = "/repo/datasets"

def load_all_test_texts(dataset_root):
    texts = []
    
    # NotInject
    for fn in ["NotInject_one.json", "NotInject_two.json", "NotInject_three.json"]:
        path = os.path.join(dataset_root, "NotInject_one", fn)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            texts.extend([s["prompt"] for s in data])
    
    # WildGuard
    path = os.path.join(dataset_root, "wildguard.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        texts.extend([s["prompt"] for s in data])
    
    # BIPIA text and code
    for fn in ["BIPIA_text.json", "BIPIA_code.json"]:
        path = os.path.join(dataset_root, fn)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for key in data.keys():
                texts.extend(data[key])
    
    print(f"Total test samples: {len(texts)}")
    return texts


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, model_max_length=2048)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, trust_remote_code=True)
model.to(device)
model.eval()

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    device=0 if torch.cuda.is_available() else -1,
)

test_texts = load_all_test_texts(DATASET_ROOT)

# Compute actual flops by running real data through the model
print("\n=== GFLOPs computation with actual data ===")
# Tokenize real samples
sample_texts = test_texts[:100]
encodings = tokenizer(sample_texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
input_ids = encodings['input_ids'].to(device)
attention_mask = encodings['attention_mask'].to(device)
seq_len = input_ids.shape[1]
print(f"Actual sequence length (padded to max): {seq_len}")
print(f"Batch size: {input_ids.shape[0]}")

# Try with fvcore
try:
    from fvcore.nn import FlopCountAnalysis
    model.eval()
    inputs = (input_ids[:1], attention_mask[:1])
    flops = FlopCountAnalysis(model, inputs)
    total_flops = flops.total()
    gflops = total_flops / 1e9
    print(f"fvcore GFLOPs (seq_len={seq_len}): {gflops:.2f}")
except Exception as e:
    print(f"fvcore failed: {e}")

# Try ptflops with different seq lengths
try:
    from ptflops import get_model_complexity_info
    
    class ModelWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, input_ids, attention_mask=None):
            return self.m(input_ids=input_ids, attention_mask=attention_mask)
    
    wrapper = ModelWrapper(model).to(device)
    wrapper.eval()
    
    for seq_len in [64, 128, 175, 200, 256, 300, 512]:
        try:
            macs, params = get_model_complexity_info(
                wrapper,
                (seq_len,),
                input_constructor=lambda sl: {
                    "input_ids": torch.ones(1, sl, dtype=torch.long).to(device), 
                    "attention_mask": torch.ones(1, sl, dtype=torch.long).to(device)
                },
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False
            )
            gflops = 2 * macs / 1e9
            print(f"ptflops GFLOPs (seq_len={seq_len}): {gflops:.2f}")
        except Exception as e:
            print(f"ptflops failed for seq_len={seq_len}: {e}")
except Exception as e:
    print(f"ptflops import failed: {e}")


print("\n=== Inference Time Measurement with different configs ===")

# Per-sample inference time
n_warmup = 20
n_runs = 200

# Single sample timing
sample_text = test_texts[0]
for _ in range(n_warmup):
    _ = classifier([sample_text])

times_single = []
with torch.no_grad():
    for i in range(n_runs):
        text = test_texts[i % len(test_texts)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = classifier([text])
        torch.cuda.synchronize()
        end = time.perf_counter()
        times_single.append((end - start) * 1000)

print(f"Single-sample inference time: {np.mean(times_single):.2f} ± {np.std(times_single):.2f} ms")
print(f"Median single-sample: {np.median(times_single):.2f} ms")

# Batch timing
batch_size = 32
n_batches = 50
batch = test_texts[:batch_size]

for _ in range(5):
    _ = classifier(batch)

times_batch = []
with torch.no_grad():
    for _ in range(n_batches):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = classifier(batch)
        torch.cuda.synchronize()
        end = time.perf_counter()
        total_ms = (end - start) * 1000
        per_sample_ms = total_ms / batch_size
        times_batch.append(per_sample_ms)

print(f"Batch inference time (batch_size={batch_size}): {np.mean(times_batch):.2f} ± {np.std(times_batch):.2f} ms/sample")
print(f"Median batch: {np.median(times_batch):.2f} ms/sample")

