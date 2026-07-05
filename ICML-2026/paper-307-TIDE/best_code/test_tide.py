import json, os, sys
sys.path.insert(0, "/repo")

from utils.model import init_model, get_prompt_embeds, get_vllm_text_output
from utils.toxicity import Toxicity
from utils.tide import backward, normalize_grad, project_cosine
from vllm import SamplingParams
import torch
import torch.nn.functional as F

# Load baseline data
with open("/repo/responses/baselines/gpt2-large/temp=0.1-K=3/rtp.json") as f:
    baseline = json.load(f)
print(f"Loaded {len(baseline)} baseline records")

# Initialize model
model_args = {
    "model": "/models/gpt2-large",
    "max_model_len": 256,
    "enable_prompt_embeds": True,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.25,
    "dtype": "auto",
    "trust_remote_code": True,
}
print("Initializing model...")
tokenizer, embed_layer, llm = init_model(model_args)
print("Model initialized")

# Toxicity client
tc = Toxicity()
backend = "Perspective" if tc.use_perspective else "Local"
print(f"Toxicity: {backend}")

# Test on first prompt
record = baseline[0]
print(f"Prompt: {record['prompt'][:80]}...")

current_embeds = get_prompt_embeds(record["prompt"], tokenizer, embed_layer).clone()
original_embeds = current_embeds.clone()
print(f"Embeds shape: {current_embeds.shape}")

sp = SamplingParams(n=1, max_tokens=20, temperature=0.1, top_p=1.0)

# Baseline gen
out = llm.generate({"prompt_embeds": current_embeds}, sampling_params=sp, use_tqdm=False)
comp = get_vllm_text_output(out)[0]
analysis, tox_scores = tc.predict(comp)
tox_val = tox_scores[0]
print(f"Base: comp='{comp[:60]}' tox={tox_val:.4f}")

# One TIDE step
print("Computing gradient (N=8)...")
grad = backward(llm, current_embeds, sp, tc, mu=0.1, N=8)
grad = normalize_grad(grad)
current_embeds = current_embeds - 1.5 * grad
current_embeds = project_cosine(current_embeds, original_embeds, 0.2)
cs = F.cosine_similarity(current_embeds.flatten(), original_embeds.flatten(), dim=0)
print(f"CosSim: {cs:.4f}")

out2 = llm.generate({"prompt_embeds": current_embeds}, sampling_params=sp, use_tqdm=False)
comp2 = get_vllm_text_output(out2)[0]
analysis2, tox_scores2 = tc.predict(comp2)
tox_val2 = tox_scores2[0]
print(f"TIDE: comp='{comp2[:60]}' tox={tox_val2:.4f}")

print("PIPELINE WORKS!")
