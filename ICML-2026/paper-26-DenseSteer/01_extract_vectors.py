import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from steering_vectors import train_steering_vector
from utils import chat_template_format

# === CONFIGURATION ===
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATA_FILE = "samples_alpha.json"  # Path to your rewritten dataset
OUT_DIR = Path("./vectors_out")
LAYERS = list(range(17))         # Target layers for Llama-3.2-1B
NUM_SAMPLES = 50                 # Number of high-quality pairs to use

def get_score(ex):
    """Extract evaluation score from nested dictionary."""
    res = ex.get("results", ex.get("metrics", ex))
    try:
        return float(res.get("exact_match", 0))
    except (TypeError, ValueError):
        return 0.0

# === MAIN PIPELINE ===
OUT_DIR.mkdir(exist_ok=True)

# 1. Load Model & Tokenizer
print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto"
).eval()

# 2. Prepare Contrastive Pairs
with open(DATA_FILE, "r") as f:
    data = json.load(f)
    samples = data.get("samples", data) if isinstance(data, dict) else data

# Filter only perfect score samples (Exact Match = 1.0)
selected = [s for s in samples if get_score(s) == 1.0][:NUM_SAMPLES]
print(f"Found {len(selected)} high-quality samples.")

pairs = []
for s in selected:
    # Format: (Instruction + Positive, Instruction + Negative)
    prompt = chat_template_format(s["doc"]["question"])
    pairs.append((prompt + s["pos_response"], prompt + s["neg_response"]))

# 3. Train Steering Vector
print("Extracting steering vector...")
vector_obj = train_steering_vector(
    model, 
    tokenizer, 
    pairs, 
    layers=LAYERS, 
    read_token_index=-1, # Use last token activation
    move_to_cpu=True     # Save VRAM
)

# 4. Save Output
torch.save(vector_obj, OUT_DIR / "steering_vector.pt")
print(f"Success! Vector saved to {OUT_DIR / 'steering_vector.pt'}")