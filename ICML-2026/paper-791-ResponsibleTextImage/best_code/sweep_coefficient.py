#!/usr/bin/env python3
"""Coefficient sweep for ALGO-02: find optimal coefficient for delta minimization."""
import torch, torch.nn.functional as F, os, json, numpy as np, sys
from tqdm import tqdm
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
from transformers import CLIPProcessor, CLIPModel

MODEL_PATH  = "/paper_data"
FEMALE_CKPT = "/repo/checkpoints/external_concept_female.pt"
MALE_CKPT   = "/repo/checkpoints/external_concept_male.pt"
OUTPUT_DIR  = "/repo/sweep_output"
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"

# Validation config (small subset for speed)
VAL_OCCS = int(os.environ.get("VAL_OCCUPATIONS", "6"))
VAL_SAMPLES = int(os.environ.get("VAL_SAMPLES", "8"))
INFERENCE_STEPS = int(os.environ.get("INFERENCE_STEPS", "20"))
GUIDANCE = float(os.environ.get("GUIDANCE_SCALE", "4.5"))
SEED = int(os.environ.get("SEED", "42"))
TARGET_LAYERS = list(range(11, 28))
TARGET_HEADS = [10, 12, 14]

# Coefficients to sweep
COEFFICIENTS = [5, 10, 20, 40, 80, 120, 160, 200]

ALL_OCCUPATIONS = [
    "doctor","nurse","teacher","professor","engineer","scientist",
    "lawyer","judge","CEO","manager","secretary","receptionist",
    "accountant","architect","artist","athlete","author","baker",
    "banker","bartender","biologist","builder","butcher","chef",
    "chemist","cleaner","clerk","coach","dentist","designer",
    "detective","developer","director","driver","economist","editor"
][:VAL_OCCS]

torch.manual_seed(SEED); np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load models once
print("Loading PixArt-Alpha...")
pipe = PixArtAlphaPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True, local_files_only=True).to(DEVICE)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

female_ckpt = torch.load(FEMALE_CKPT, map_location="cpu")
male_ckpt = torch.load(MALE_CKPT, map_location="cpu")

# External Heads classes (same as eval_final.py)
class LoadedExternalHeads:
    def __init__(self, sd, tl, nh=16, hd=72):
        self.target_layers = tl; self.num_heads = nh; self.head_dim = hd; self.external_heads = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith("external_heads."): nk = nk[len("external_heads."):]
            self.external_heads[nk] = v
    def get(self, li, hi, sl, dev, dt, th=None):
        if th is not None and hi not in th: return torch.zeros(sl, self.head_dim, device=dev, dtype=dt)
        if li not in self.target_layers: return torch.zeros(sl, self.head_dim, device=dev, dtype=dt)
        k = f"layer_{li}_head_{hi}"
        if k not in self.external_heads: return torch.zeros(sl, self.head_dim, device=dev, dtype=dt)
        return self.external_heads[k].to(device=dev, dtype=dt)
    def all_heads(self, li, sl, dev, dt, th=None):
        return torch.stack([self.get(li, h, sl, dev, dt, th) for h in range(self.num_heads)], dim=0)

class ExternalHeadProcessor:
    def __init__(self, orig, li, attn, ext, coeff, th=None):
        self.orig = orig; self.li = li; self.attn = attn; self.ext = ext; self.coeff = coeff; self.th = th
        self.nh = getattr(attn, "heads", None)
        idim = attn.to_q.out_features if hasattr(attn.to_q, "out_features") else attn.to_q.weight.shape[0]
        self.hd = idim // self.nh
    def _bf(self, attn, d):
        to = attn.to_out
        if to is None: return d
        if isinstance(to, torch.nn.ModuleList):
            y = d
            for m in to: y = F.linear(y, m.weight, bias=None) if isinstance(m, torch.nn.Linear) else m(y)
            return y
        if isinstance(to, torch.nn.Sequential):
            f0 = to[0]
            if isinstance(f0, torch.nn.Linear):
                y = F.linear(d, f0.weight, bias=None)
                for m in list(to)[1:]: y = m(y)
                return y
            return to(d)
        if isinstance(to, torch.nn.Linear): return F.linear(d, to.weight, bias=None)
        return to(d)
    def __call__(self, attn, hs, encoder_hidden_states=None, attention_mask=None, **kw):
        oo = self.orig(attn, hs, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **kw)
        B, N, _ = hs.shape; dev, dt = hs.device, hs.dtype
        ext = self.ext.all_heads(self.li, N, dev, dt, self.th).unsqueeze(0).expand(B, -1, -1, -1)
        H, dh = ext.shape[1], ext.shape[3]
        dc = ext.transpose(1, 2).reshape(B, N, H * dh)
        return oo + self.coeff * self._bf(attn, dc)

def save_procs(pipe, ls): return {l: pipe.transformer.transformer_blocks[l].attn2.get_processor() for l in ls}
def reset_procs(pipe, ls, ps):
    for l in ls: pipe.transformer.transformer_blocks[l].attn2.set_processor(ps[l])
def setup_procs(pipe, ls, ps, ext, coeff, th):
    for l in ls:
        b = pipe.transformer.transformer_blocks[l]
        b.attn2.set_processor(ExternalHeadProcessor(ps[l], l, b.attn2, ext, coeff, th))

female_heads = LoadedExternalHeads(female_ckpt, TARGET_LAYERS)
male_heads = LoadedExternalHeads(male_ckpt, TARGET_LAYERS)
baseline_procs = save_procs(pipe, TARGET_LAYERS)

def classify_gender(img):
    texts = ["a photo of a male person","a photo of a man","a photo of a female person","a photo of a woman"]
    inp = clip_processor(text=texts, images=img, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        logits = clip_model(**inp).logits_per_image[0]
        return "male" if logits[:2].mean() > logits[2:].mean() else "female"

def clip_score_fn(img, prompt):
    inp = clip_processor(text=[prompt], images=img, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        out = clip_model(**inp)
        ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return (ie @ te.T).item()

def gen(pipe, prompt, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt=prompt, num_inference_steps=INFERENCE_STEPS, generator=g, guidance_scale=GUIDANCE).images[0]

# Pre-generate assignment and seeds (same for all coefficients)
import random
random.seed(SEED)
assignments = []
for oi in range(VAL_OCCS):
    for si in range(VAL_SAMPLES):
        assignments.append("female" if random.random() < 0.5 else "male")
total_N = len(assignments)

# Generate baseline once (not needed for delta but good for reference)
print(f"\nSweeping {len(COEFFICIENTS)} coefficients on {VAL_OCCS} occs x {VAL_SAMPLES} samples = {total_N} images each...")
results = []

for coeff in COEFFICIENTS:
    print(f"\n--- Coefficient = {coeff} ---")
    gender_preds = []
    clip_scores = []
    
    for idx in tqdm(range(total_N), desc=f"coeff={coeff}"):
        oi = idx // VAL_SAMPLES
        si = idx % VAL_SAMPLES
        cs = SEED + oi * VAL_SAMPLES + si
        assigned = assignments[idx]
        occ = ALL_OCCUPATIONS[oi]
        prompt = f"a photo of a {occ}"
        
        ext = female_heads if assigned == "female" else male_heads
        setup_procs(pipe, TARGET_LAYERS, baseline_procs, ext, coeff, TARGET_HEADS)
        img = gen(pipe, prompt, cs)
        reset_procs(pipe, TARGET_LAYERS, baseline_procs)
        
        gender_preds.append(classify_gender(img))
        clip_scores.append(clip_score_fn(img, prompt) * 100)
    
    n_m = sum(1 for g in gender_preds if g == "male")
    n_f = total_N - n_m
    G = 2
    delta = (max(n_m, n_f) / (total_N / G) - 1) / (1 - 1 / G) if total_N > 0 else float("nan")
    clp = np.mean(clip_scores)
    
    results.append({"coefficient": coeff, "delta": float(delta), "clip": float(clp), "male": n_m, "female": n_f})
    print(f"  delta={delta:.4f}, CLIP={clp:.2f}, male={n_m}, female={n_f}")

# Report
print("\n" + "="*60)
print("COEFFICIENT SWEEP RESULTS")
print("="*60)
for r in sorted(results, key=lambda x: x["delta"]):
    print(f"  coeff={r[coefficient]:5.1f}  delta={r[delta]:.4f}  CLIP={r[clip]:.2f}  m={r[male]} f={r[female]}")

best = min(results, key=lambda x: x["delta"])
print(f"\nBest coefficient: {best[coefficient]} (delta={best[delta]:.4f}, CLIP={best[clip]:.2f})")

with open(f"{OUTPUT_DIR}/sweep_results.json", "w") as f:
    json.dump({"best": best, "all": results, "config": {"val_occs": VAL_OCCS, "val_samples": VAL_SAMPLES, "coefficients": COEFFICIENTS}}, f, indent=2)
print(f"\nResults saved to {OUTPUT_DIR}/sweep_results.json")
