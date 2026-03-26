import os
import sys
import json
import torch
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Use local model
MODEL_PATH = "/tmp/PIGuard_weights"
DATASET_ROOT = "/repo/datasets"

def acc_compute(model, target_set, target_class="benign", name="chat"):
    save_dict = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(target_set), batch_size):
            batch = target_set[i : i + batch_size]   
            preds = model(batch)                   

            for sample, pred in zip(batch, preds):
                if pred["label"] != target_class:
                    save_dict.append({
                        "prompt": sample,
                        "logits": pred["score"]
                    })

            del preds
        
    acc = 1 - len(save_dict)/len(target_set)
    print(f"{name} set accuracy: {acc:.4f} ({len(target_set) - len(save_dict)}/{len(target_set)} correct)")
    return acc


def NotInject_eval(model, dataset_root="datasets/NotInject_one"):
    benign_set = []
    with open(os.path.join(dataset_root, "NotInject_one.json"), "r") as f:
        valid_dataset = json.load(f)
    for sample in valid_dataset:
        benign_set.append(sample["prompt"])
    one_acc = acc_compute(model, benign_set, target_class="benign", name="NotInject_one")

    benign_set = []
    with open(os.path.join(dataset_root, "NotInject_two.json"), "r") as f:
        valid_dataset = json.load(f)
    for sample in valid_dataset:
        benign_set.append(sample["prompt"])
    two_acc = acc_compute(model, benign_set, target_class="benign", name="NotInject_two")

    benign_set = []
    with open(os.path.join(dataset_root, "NotInject_three.json"), "r") as f:
        valid_dataset = json.load(f)
    for sample in valid_dataset:
        benign_set.append(sample["prompt"])
    three_acc = acc_compute(model, benign_set, target_class="benign", name="NotInject_three")

    overall_acc = (one_acc + two_acc + three_acc) / 3
    print(f"NotInject overall accuracy: {overall_acc:.4f}")
    return overall_acc, one_acc, two_acc, three_acc


def compute_gflops(model_raw, tokenizer, seq_len=256):
    """Compute GFLOPs for a single forward pass"""
    try:
        from ptflops import get_model_complexity_info
        
        # Get the underlying model (DeBERTa)
        model_for_flops = model_raw
        model_for_flops.eval()
        
        def input_constructor(input_res):
            # Create dummy input ids and attention mask
            input_ids = torch.ones(1, seq_len, dtype=torch.long).cuda() if torch.cuda.is_available() else torch.ones(1, seq_len, dtype=torch.long)
            attention_mask = torch.ones(1, seq_len, dtype=torch.long).cuda() if torch.cuda.is_available() else torch.ones(1, seq_len, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        
        macs, params = get_model_complexity_info(
            model_for_flops,
            (seq_len,),
            input_constructor=input_constructor,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        gflops = macs * 2 / 1e9  # convert MACs to FLOPs and then to GFLOPs
        print(f"GFLOPs: {gflops:.2f}")
        return gflops
    except Exception as e:
        print(f"ptflops failed: {e}")
        return None


def measure_inference_time(classifier, test_texts, n_warmup=10, n_runs=100):
    """Measure average inference time per sample in ms"""
    import random
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Warmup
    for _ in range(n_warmup):
        text = random.choice(test_texts)
        _ = classifier([text])
    
    # Timing
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            text = random.choice(test_texts)
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = classifier([text])
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"Inference time: {avg_time:.2f} ± {std_time:.2f} ms per sample")
    return avg_time


def load_all_test_texts(dataset_root):
    """Load all test texts from all 4 datasets (excluding PINT which requires access)"""
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
    
    print(f"Total test samples (excluding PINT): {len(texts)}")
    return texts


if __name__ == "__main__":
    print("Loading model from:", MODEL_PATH)
    
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
    
    print("\n=== NotInject Evaluation ===")
    notinject_acc, one_acc, two_acc, three_acc = NotInject_eval(
        classifier, 
        dataset_root=os.path.join(DATASET_ROOT, "NotInject_one")
    )
    
    print(f"\nOver-defense Accuracy (NotInject): {notinject_acc * 100:.2f}%")
    
    # Measure inference time
    print("\n=== Inference Time Measurement ===")
    test_texts = load_all_test_texts(DATASET_ROOT)
    avg_time = measure_inference_time(classifier, test_texts)
    
    # Compute GFLOPs
    print("\n=== GFLOPs Computation ===")
    try:
        from ptflops import get_model_complexity_info
        
        # Create a wrapper for the model to use with ptflops
        class ModelWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, input_ids, attention_mask=None):
                return self.m(input_ids=input_ids, attention_mask=attention_mask)
        
        wrapper = ModelWrapper(model).to(device)
        
        # Use typical sequence length from the paper (256)
        seq_len = 256
        dummy_input_ids = torch.ones(1, seq_len, dtype=torch.long).to(device)
        dummy_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)
        
        macs, params = get_model_complexity_info(
            wrapper,
            (seq_len,),
            input_constructor=lambda _: {"input_ids": torch.ones(1, seq_len, dtype=torch.long).to(device), 
                                          "attention_mask": torch.ones(1, seq_len, dtype=torch.long).to(device)},
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        # GFLOPs = 2 * MACs / 1e9
        gflops = 2 * macs / 1e9
        print(f"Model MACs: {macs:.2e}")
        print(f"Model parameters: {params:.2e}")
        print(f"GFLOPs per forward pass: {gflops:.2f}")
        
    except Exception as e:
        print(f"GFLOPs computation failed: {e}")
        gflops = None
    
    print("\n=== FINAL RESULTS ===")
    print(f"Over-defense Accuracy (NotInject): {notinject_acc * 100:.2f}%")
    print(f"  One-trigger: {one_acc * 100:.2f}%")
    print(f"  Two-trigger: {two_acc * 100:.2f}%")
    print(f"  Three-trigger: {three_acc * 100:.2f}%")
    print(f"Inference Time: {avg_time:.2f} ms")
    if gflops is not None:
        print(f"GFLOPs: {gflops:.2f}")
