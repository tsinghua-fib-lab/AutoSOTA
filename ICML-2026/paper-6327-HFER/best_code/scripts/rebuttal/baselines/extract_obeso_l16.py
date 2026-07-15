
import os
import torch
import numpy as np
import json
import glob
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA_DIR = "data/experiment_ready"
OUT_DIR = "data/results/rebuttal"
TARGET_LAYER = 16 # Obeso et al. target layer

def extract_l16():
    print(f"=== EXTRACTING LAYER {TARGET_LAYER} HIDDEN STATES (REAL OBESO) ===")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f"Loading model {MODEL_NAME} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16,
        output_hidden_states=True
    )
    
    proofs = []
    for label in ["valid", "invalid"]:
        pattern = os.path.join(DATA_DIR, label, "*.lean")
        files = sorted(glob.glob(pattern))
        for fp in files:
            proofs.append((fp, os.path.basename(fp), label))
            
    print(f"Processing {len(proofs)} proofs...")
    
    hidden_L16 = []
    filenames = []
    labels = []
    
    # Load reclaimed set for ground truth correction
    reclaimed_path = "data/reclaimed/8B_list_b_confident_invalid.json"
    reclaimed = set()
    if os.path.exists(reclaimed_path):
        with open(reclaimed_path, "r") as f:
            reclaimed = set(item["file"] for item in json.load(f))

    for fp, fname, label in tqdm(proofs):
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
            
        corrected = label if label == "valid" or fname not in reclaimed else "valid"
        
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
                # hidden_states is a tuple of (L+1) tensors of shape (1, seq_len, dim)
                # Layer 16 is index 16
                l16_state = outputs.hidden_states[TARGET_LAYER]
                # Mean pool over sequence
                pooled = l16_state.mean(dim=1).squeeze().cpu().numpy()
                
                hidden_L16.append(pooled.astype(np.float16))
                filenames.append(fname)
                labels.append(1 if corrected == "valid" else 0)
        except Exception as e:
            print(f"Error on {fname}: {e}")
            
    out_path = os.path.join(OUT_DIR, "obeso_states_llama8b.npz")
    np.savez_compressed(
        out_path,
        hidden_L16=np.array(hidden_L16),
        filenames=np.array(filenames),
        labels=np.array(labels)
    )
    print(f"Extraction complete. Saved to {out_path}")

if __name__ == "__main__":
    extract_l16()
