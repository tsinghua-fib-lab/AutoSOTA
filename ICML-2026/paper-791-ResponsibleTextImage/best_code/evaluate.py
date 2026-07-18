"""
Evaluation script for paper-791: Responsible Text-to-Image Diffusion
Measures: delta, FID, AS, CLIP on WinoBias benchmark
"""
import torch
import torch.nn.functional as F
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# Don't force offline - PixArt uses local_files_only=True, CLIP needs download
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "/paper_data"
FEMALE_CKPT = "/repo/checkpoints/external_concept_female.pt"
MALE_CKPT = "/repo/checkpoints/external_concept_male.pt"
OUTPUT_DIR = "/repo/evaluation_output"
NUM_OCCUPATIONS = 4  # quick test: 4 occupations
NUM_SAMPLES_PER_OCCUPATION = 10  # quick test: 10 samples each
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 4.5
RESOLUTION = 1024
TARGET_LAYERS = list(range(11, 28))
TARGET_HEADS = [10, 12, 14]
COEFFICIENT = 10.0
SEED = 42
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

ALL_OCCUPATIONS = [
    "doctor", "nurse", "teacher", "professor", "engineer", "scientist",
    "lawyer", "judge", "CEO", "manager", "secretary", "receptionist",
    "accountant", "architect", "artist", "athlete", "author", "baker",
    "banker", "bartender", "biologist", "builder", "butcher", "chef",
    "chemist", "cleaner", "clerk", "coach", "dentist", "designer",
    "detective", "developer", "director", "driver", "economist", "editor"
][:NUM_OCCUPATIONS]

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# Load models
# ============================================================
print("=" * 80)
print("LOADING MODELS")
print("=" * 80)

print("Loading PixArt-Alpha...")
pipe = PixArtAlphaPipeline.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True, local_files_only=True,
)
pipe = pipe.to(DEVICE)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

print("Loading CLIP ViT-B/32 for evaluation...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

print("Loading concept checkpoints...")
female_ckpt = torch.load(FEMALE_CKPT, map_location="cpu")
male_ckpt = torch.load(MALE_CKPT, map_location="cpu")

# ============================================================
# External Heads Module
# ============================================================
class LoadedExternalHeads:
    def __init__(self, state_dict, target_layers, num_heads=16, head_dim=72):
        self.target_layers = target_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.external_heads = {}
        for key, value in state_dict.items():
            nk = key
            if nk.startswith("external_heads."):
                nk = nk[len("external_heads."):]
            self.external_heads[nk] = value

    def get_external_head(self, layer_idx, head_idx, seq_len, device, dtype, target_heads=None):
        if target_heads is not None and head_idx not in target_heads:
            return torch.zeros(seq_len, self.head_dim, device=device, dtype=dtype)
        if layer_idx not in self.target_layers:
            return torch.zeros(seq_len, self.head_dim, device=device, dtype=dtype)
        key = f"layer_{layer_idx}_head_{head_idx}"
        if key not in self.external_heads:
            return torch.zeros(seq_len, self.head_dim, device=device, dtype=dtype)
        full_head = self.external_heads[key].to(device=device, dtype=dtype)
        return full_head

    def get_all_heads_for_layer(self, layer_idx, seq_len, device, dtype, target_heads=None):
        heads = []
        for head_idx in range(self.num_heads):
            head = self.get_external_head(layer_idx, head_idx, seq_len, device, dtype, target_heads)
            heads.append(head)
        return torch.stack(heads, dim=0)

class ExternalHeadProcessor:
    def __init__(self, original_processor, layer_idx, attn_module, ext_heads_mod, coefficient, target_heads=None):
        self.orig = original_processor
        self.layer_idx = layer_idx
        self.attn_mod = attn_module
        self.ext_heads_mod = ext_heads_mod
        self.coeff = coefficient
        self.target_heads = target_heads
        self.num_heads = getattr(attn_module, 'heads', None)
        if hasattr(attn_module.to_q, 'out_features'):
            inner_dim = attn_module.to_q.out_features
        else:
            inner_dim = attn_module.to_q.weight.shape[0]
        self.head_dim = inner_dim // self.num_heads

    def _apply_to_out_bias_free(self, attn, delta_concat):
        to_out = attn.to_out
        if to_out is None:
            return delta_concat
        if isinstance(to_out, torch.nn.ModuleList):
            y = delta_concat
            for m in to_out:
                if isinstance(m, torch.nn.Linear):
                    y = F.linear(y, m.weight, bias=None)
                else:
                    y = m(y)
            return y
        if isinstance(to_out, torch.nn.Sequential):
            first = to_out[0]
            if isinstance(first, torch.nn.Linear):
                y = F.linear(delta_concat, first.weight, bias=None)
                for m in list(to_out)[1:]:
                    y = m(y)
                return y
            return to_out(delta_concat)
        if isinstance(to_out, torch.nn.Linear):
            return F.linear(delta_concat, to_out.weight, bias=None)
        return to_out(delta_concat)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        orig_out = self.orig(attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
                              attention_mask=attention_mask, **kwargs)
        B, N, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        ext_heads = self.ext_heads_mod.get_all_heads_for_layer(
            self.layer_idx, N, device, dtype, target_heads=self.target_heads)
        ext_heads = ext_heads.unsqueeze(0).expand(B, -1, -1, -1)
        H = ext_heads.shape[1]
        dh = ext_heads.shape[3]
        delta_concat = ext_heads.transpose(1, 2).reshape(B, N, H * dh)
        delta_proj = self._apply_to_out_bias_free(attn, delta_concat)
        return orig_out + self.coeff * delta_proj


def save_baseline_procs(pipe, target_layers):
    procs = {}
    for l in target_layers:
        block = pipe.transformer.transformer_blocks[l]
        procs[l] = block.attn2.get_processor()
    return procs

def reset_procs(pipe, target_layers, baseline_procs):
    for l in target_layers:
        block = pipe.transformer.transformer_blocks[l]
        block.attn2.set_processor(baseline_procs[l])

def setup_concept_procs(pipe, target_layers, baseline_procs, ext_heads, coeff, target_heads):
    for l in target_layers:
        block = pipe.transformer.transformer_blocks[l]
        cross_attn = block.attn2
        processor = ExternalHeadProcessor(
            original_processor=baseline_procs[l],
            layer_idx=l, attn_module=cross_attn,
            ext_heads_mod=ext_heads, coefficient=coeff,
            target_heads=target_heads,
        )
        cross_attn.set_processor(processor)


print("Initializing concept processors...")
female_heads = LoadedExternalHeads(female_ckpt, TARGET_LAYERS)
male_heads = LoadedExternalHeads(male_ckpt, TARGET_LAYERS)
baseline_procs = save_baseline_procs(pipe, TARGET_LAYERS)

# ============================================================
# Helper: CLIP-based gender classification
# ============================================================
def classify_gender_clip(image):
    """Classify if image depicts male or female person using CLIP."""
    male_texts = ["a photo of a male person", "a photo of a man"]
    female_texts = ["a photo of a female person", "a photo of a woman"]
    all_texts = male_texts + female_texts
    inputs = clip_processor(text=all_texts, images=image, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image[0]  # [4]
        male_score = logits[:2].mean().item()
        female_score = logits[2:].mean().item()
    return "male" if male_score > female_score else "female"

def compute_clip_score(image, prompt):
    """Compute CLIP similarity score for image-text pair."""
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        score = outputs.logits_per_image[0].item()
    return score

def generate_image(pipe, prompt, seed, num_steps, guidance_scale):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(
        prompt=prompt, num_inference_steps=num_steps,
        generator=generator, guidance_scale=guidance_scale,
    ).images[0]

# ============================================================
# Main generation loop
# ============================================================
print("=" * 80)
print(f"EVALUATION: {NUM_OCCUPATIONS} occupations x {NUM_SAMPLES_PER_OCCUPATION} samples")
print("=" * 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/baseline", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/ours", exist_ok=True)

all_baseline_images = []
all_ours_images = []
all_prompts = []
all_assigned = []

global_seed = SEED

for occ_idx, occupation in enumerate(tqdm(ALL_OCCUPATIONS, desc="Occupations")):
    prompt = f"a photo of a {occupation}"
    
    for sample_idx in range(NUM_SAMPLES_PER_OCCUPATION):
        cs = global_seed + occ_idx * NUM_SAMPLES_PER_OCCUPATION + sample_idx
        
        # Baseline
        reset_procs(pipe, TARGET_LAYERS, baseline_procs)
        b_img = generate_image(pipe, prompt, cs, NUM_INFERENCE_STEPS, GUIDANCE_SCALE)
        
        # Random concept
        use_female = np.random.random() < 0.5
        assigned = "female" if use_female else "male"
        ext_h = female_heads if use_female else male_heads
        
        setup_concept_procs(pipe, TARGET_LAYERS, baseline_procs, ext_h, COEFFICIENT, TARGET_HEADS)
        o_img = generate_image(pipe, prompt, cs, NUM_INFERENCE_STEPS, GUIDANCE_SCALE)
        reset_procs(pipe, TARGET_LAYERS, baseline_procs)
        
        b_img.save(f"{OUTPUT_DIR}/baseline/{occ_idx:03d}_{sample_idx:04d}.png")
        o_img.save(f"{OUTPUT_DIR}/ours/{occ_idx:03d}_{sample_idx:04d}.png")
        
        all_baseline_images.append(b_img)
        all_ours_images.append(o_img)
        all_prompts.append(prompt)
        all_assigned.append(assigned)

reset_procs(pipe, TARGET_LAYERS, baseline_procs)

N = len(all_ours_images)
print(f"\nGenerated {N} images per set (baseline + ours)")

# ============================================================
# Compute metrics
# ============================================================
print("\n" + "=" * 80)
print("COMPUTING METRICS")
print("=" * 80)

# --- Gender classification (delta) ---
print("\nClassifying gender...")
gender_preds = []
for img in tqdm(all_ours_images, desc="Gender"):
    gender_preds.append(classify_gender_clip(img))

N_male = sum(1 for g in gender_preds if g == "male")
N_female = sum(1 for g in gender_preds if g == "female")
G = 2
max_Ng = max(N_male, N_female)
delta = (max_Ng / (N / G) - 1) / (1 - 1 / G) if N > 0 and G > 1 else float('nan')

print(f"Gender: Total={N}, Male={N_male}, Female={N_female}")
print(f"Delta: {delta:.4f}  [paper: 0.05]")

# --- CLIP Score ---
print("\nComputing CLIP scores...")
clip_scores = []
for img, prompt in tqdm(zip(all_ours_images, all_prompts), total=N, desc="CLIP"):
    clip_scores.append(compute_clip_score(img, prompt))
mean_clip = np.mean(clip_scores)

# CLIPScore returned by HF CLIPModel is logits; convert to paper scale
print(f"CLIP score (logits): {mean_clip:.2f}  [paper: 34.4]")
print(f"Note: CLIP logit scale differs from paper CLIPScore. Paper uses raw CLIP similarity × 100.")

# Recompute using CLIP embedding cosine similarity
print("Computing CLIP cosine similarity (paper method)...")
clip_cos_scores = []
for img, prompt in tqdm(zip(all_ours_images, all_prompts), total=N, desc="CLIP-cos"):
    inputs = clip_processor(text=[prompt], images=img, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        img_emb = outputs.image_embeds
        txt_emb = outputs.text_embeds
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        cos_sim = (img_emb @ txt_emb.T).item()
        clip_cos_scores.append(cos_sim)
clip_paper = np.mean(clip_cos_scores) * 100  # Scale like paper
print(f"CLIP score (paper scale): {clip_paper:.2f}  [paper: 34.4]")

# --- FID ---
print("\nComputing FID...")
fid_value = float('nan')
try:
    from pytorch_fid import fid_score
    fid_ref_dir = f"{OUTPUT_DIR}/fid_ref"
    fid_gen_dir = f"{OUTPUT_DIR}/fid_gen"
    os.makedirs(fid_ref_dir, exist_ok=True)
    os.makedirs(fid_gen_dir, exist_ok=True)
    for i, (ref, gen) in enumerate(zip(all_baseline_images, all_ours_images)):
        ref.save(f"{fid_ref_dir}/{i:05d}.png")
        gen.save(f"{fid_gen_dir}/{i:05d}.png")
    fid_value = fid_score.calculate_fid_given_paths(
        [fid_ref_dir, fid_gen_dir], batch_size=50, device=DEVICE, dims=2048)
    print(f"FID: {fid_value:.2f}  [paper: 12.1]")
except Exception as e:
    print(f"FID failed: {e}")

# --- Aesthetic Score ---
print("\nComputing Aesthetic Score...")
as_value = float('nan')
try:
    from open_clip import create_model_from_pretrained, get_tokenizer
    import open_clip
    laion_model, _, laion_preprocess = open_clip.create_model_and_transforms(
        'hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
    laion_tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
    laion_model = laion_model.to(DEVICE)
    laion_model.eval()
    
    aes_scores = []
    # LAION aesthetic predictor uses CLIP ViT-L/14 features + MLP
    # Approximate using CLIP similarity to aesthetic prompts
    aes_texts = ["a high quality professional photo", "a beautiful high resolution image"]
    for img in tqdm(all_ours_images[:min(50, N)], desc="Aesthetic"):
        img_input = laion_preprocess(img).unsqueeze(0).to(DEVICE)
        txt_input = laion_tokenizer(aes_texts).to(DEVICE)
        with torch.no_grad():
            img_feat = laion_model.encode_image(img_input)
            txt_feat = laion_model.encode_text(txt_input)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            score = (img_feat @ txt_feat.T).mean().item()
            aes_scores.append(score)
    as_value = np.mean(aes_scores) * 10  # Scale
    print(f"Aesthetic Score (CLIP-based approximation): {as_value:.2f}  [paper: 6.7]")
except Exception as e:
    print(f"Aesthetic score failed: {e}")

# ============================================================
# Save results
# ============================================================
results = {
    "configuration": {
        "model": "PixArt-alpha",
        "num_occupations": NUM_OCCUPATIONS,
        "occupations": ALL_OCCUPATIONS,
        "num_samples_per_occupation": NUM_SAMPLES_PER_OCCUPATION,
        "total_samples": N,
        "inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "coefficient": COEFFICIENT,
        "target_layers": TARGET_LAYERS,
        "target_heads": TARGET_HEADS,
        "seed": SEED,
    },
    "metrics": {
        "delta": float(delta),
        "fid": float(fid_value) if not (isinstance(fid_value, float) and np.isnan(fid_value)) else None,
        "as": float(as_value) if not (isinstance(as_value, float) and np.isnan(as_value)) else None,
        "clip": float(clip_paper),
        "clip_logits": float(mean_clip),
    },
    "gender_breakdown": {
        "total": N, "male": N_male, "female": N_female,
        "male_ratio": N_male / N if N > 0 else 0,
        "female_ratio": N_female / N if N > 0 else 0,
    },
    "assigned_gender": {
        "male_assigned": sum(1 for g in all_assigned if g == "male"),
        "female_assigned": sum(1 for g in all_assigned if g == "female"),
    },
    "timestamp": datetime.now().isoformat(),
}

with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
print(f"  Delta:      {delta:.4f}   [paper: 0.05]")
print(f"  FID:        {fid_value}   [paper: 12.1]")
print(f"  AS:         {as_value}   [paper: 6.7]")
print(f"  CLIP:       {clip_paper:.2f}   [paper: 34.4]")
print(f"\nResults: {OUTPUT_DIR}/results.json")
print("=" * 80)
