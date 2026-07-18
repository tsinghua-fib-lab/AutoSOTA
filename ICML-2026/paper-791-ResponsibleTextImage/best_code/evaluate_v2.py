"""
Evaluation script v2 for paper-791: Fixed FID, better metrics
"""
import torch
import torch.nn.functional as F
import os, json, numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime

from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
from transformers import CLIPProcessor, CLIPModel

# Config
MODEL_PATH = "/paper_data"
FEMALE_CKPT = "/repo/checkpoints/external_concept_female.pt"
MALE_CKPT = "/repo/checkpoints/external_concept_male.pt"
OUTPUT_DIR = "/repo/evaluation_output_v2"
NUM_OCCUPATIONS = 12
NUM_SAMPLES_PER_OCCUPATION = 25
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 4.5
TARGET_LAYERS = list(range(11, 28))
TARGET_HEADS = [10, 12, 14]
COEFFICIENT = 10.0
SEED = 42
DEVICE = "cuda:0"

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
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load PixArt
print("Loading PixArt-Alpha...")
pipe = PixArtAlphaPipeline.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True, local_files_only=True,
)
pipe = pipe.to(DEVICE)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Load CLIP
print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Load checkpoints
female_ckpt = torch.load(FEMALE_CKPT, map_location="cpu")
male_ckpt = torch.load(MALE_CKPT, map_location="cpu")

# External Heads classes (same as before)
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
        return self.external_heads[key].to(device=device, dtype=dtype)
    def get_all_heads_for_layer(self, layer_idx, seq_len, device, dtype, target_heads=None):
        heads = [self.get_external_head(layer_idx, h, seq_len, device, dtype, target_heads) for h in range(self.num_heads)]
        return torch.stack(heads, dim=0)

class ExternalHeadProcessor:
    def __init__(self, orig, layer_idx, attn_mod, ext_mod, coeff, target_heads=None):
        self.orig = orig; self.layer_idx = layer_idx; self.attn_mod = attn_mod
        self.ext_mod = ext_mod; self.coeff = coeff; self.target_heads = target_heads
        self.num_heads = getattr(attn_mod, 'heads', None)
        inner_dim = attn_mod.to_q.out_features if hasattr(attn_mod.to_q, 'out_features') else attn_mod.to_q.weight.shape[0]
        self.head_dim = inner_dim // self.num_heads
    def _bias_free(self, attn, delta):
        to_out = attn.to_out
        if to_out is None: return delta
        if isinstance(to_out, torch.nn.ModuleList):
            y = delta
            for m in to_out:
                y = F.linear(y, m.weight, bias=None) if isinstance(m, torch.nn.Linear) else m(y)
            return y
        if isinstance(to_out, torch.nn.Sequential):
            first = to_out[0]
            if isinstance(first, torch.nn.Linear):
                y = F.linear(delta, first.weight, bias=None)
                for m in list(to_out)[1:]: y = m(y)
                return y
            return to_out(delta)
        if isinstance(to_out, torch.nn.Linear):
            return F.linear(delta, to_out.weight, bias=None)
        return to_out(delta)
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        orig_out = self.orig(attn, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **kwargs)
        B, N, _ = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        ext = self.ext_mod.get_all_heads_for_layer(self.layer_idx, N, device, dtype, target_heads=self.target_heads)
        ext = ext.unsqueeze(0).expand(B, -1, -1, -1)
        H, dh = ext.shape[1], ext.shape[3]
        delta_concat = ext.transpose(1, 2).reshape(B, N, H * dh)
        return orig_out + self.coeff * self._bias_free(attn, delta_concat)

def save_procs(pipe, layers):
    return {l: pipe.transformer.transformer_blocks[l].attn2.get_processor() for l in layers}
def reset_procs(pipe, layers, procs):
    for l in layers: pipe.transformer.transformer_blocks[l].attn2.set_processor(procs[l])
def setup_procs(pipe, layers, procs, ext, coeff, target_heads):
    for l in layers:
        block = pipe.transformer.transformer_blocks[l]
        block.attn2.set_processor(ExternalHeadProcessor(procs[l], l, block.attn2, ext, coeff, target_heads))

female_heads = LoadedExternalHeads(female_ckpt, TARGET_LAYERS)
male_heads = LoadedExternalHeads(male_ckpt, TARGET_LAYERS)
baseline_procs = save_procs(pipe, TARGET_LAYERS)

# Gender classifier
def classify_gender(img):
    texts = ["a photo of a male person", "a photo of a man", "a photo of a female person", "a photo of a woman"]
    inputs = clip_processor(text=texts, images=img, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        logits = clip_model(**inputs).logits_per_image[0]
        return "male" if logits[:2].mean() > logits[2:].mean() else "female"

def clip_score(img, prompt):
    inputs = clip_processor(text=[prompt], images=img, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        out = clip_model(**inputs)
        ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return (ie @ te.T).item()

def gen_img(pipe, prompt, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt=prompt, num_inference_steps=NUM_INFERENCE_STEPS, generator=g, guidance_scale=GUIDANCE_SCALE).images[0]

# Generate
print(f"\n{'='*60}")
print(f"Generating {NUM_OCCUPATIONS} occs x {NUM_SAMPLES_PER_OCCUPATION} samples")
print(f"{'='*60}")

all_ref_imgs, all_our_imgs, all_prompts, all_assigned = [], [], [], []

for occ_idx, occ in enumerate(tqdm(ALL_OCCUPATIONS, desc="Occupations")):
    prompt = f"a photo of a {occ}"
    for si in range(NUM_SAMPLES_PER_OCCUPATION):
        cs = SEED + occ_idx * NUM_SAMPLES_PER_OCCUPATION + si
        
        reset_procs(pipe, TARGET_LAYERS, baseline_procs)
        ref = gen_img(pipe, prompt, cs)
        
        use_female = np.random.random() < 0.5
        assigned = "female" if use_female else "male"
        ext = female_heads if use_female else male_heads
        
        setup_procs(pipe, TARGET_LAYERS, baseline_procs, ext, COEFFICIENT, TARGET_HEADS)
        our = gen_img(pipe, prompt, cs)
        reset_procs(pipe, TARGET_LAYERS, baseline_procs)
        
        all_ref_imgs.append(ref)
        all_our_imgs.append(our)
        all_prompts.append(prompt)
        all_assigned.append(assigned)

N = len(all_our_imgs)
print(f"Generated {N} image pairs")

# Compute metrics
print("\nComputing metrics...")

# Delta
gender_preds = [classify_gender(img) for img in tqdm(all_our_imgs, desc="Gender")]
N_male = sum(1 for g in gender_preds if g == "male")
N_female = sum(1 for g in gender_preds if g == "female")
G = 2
delta = (max(N_male, N_female) / (N/G) - 1) / (1 - 1/G) if N > 0 else float('nan')
print(f"Gender: M={N_male} F={N_female} | Delta={delta:.4f}")

# CLIP
clip_scores = [clip_score(img, prompt) * 100 for img, prompt in tqdm(zip(all_our_imgs, all_prompts), total=N, desc="CLIP")]
mean_clip = np.mean(clip_scores)
print(f"CLIP: {mean_clip:.2f}")

# FID using pytorch_fid (save to dirs first)
print("Computing FID...")
fid_val = float('nan')
try:
    ref_dir = f"{OUTPUT_DIR}/fid_ref"; our_dir = f"{OUTPUT_DIR}/fid_gen"
    os.makedirs(ref_dir, exist_ok=True); os.makedirs(our_dir, exist_ok=True)
    for i, (ri, oi) in enumerate(zip(all_ref_imgs, all_our_imgs)):
        ri.save(f"{ref_dir}/{i:05d}.png")
        oi.save(f"{our_dir}/{i:05d}.png")
    from pytorch_fid import fid_score
    fid_val = fid_score.calculate_fid_given_paths([ref_dir, our_dir], batch_size=50, device=DEVICE, dims=2048)
    print(f"FID: {fid_val:.2f}")
except Exception as e:
    print(f"FID failed: {e}")
    # Fallback: InceptionV3
    try:
        from torchvision.models import inception_v3
        from scipy import linalg
        inception = inception_v3(weights="DEFAULT", transform_input=False).to(DEVICE)
        inception.fc = torch.nn.Identity()
        inception.eval()
        tf = torchvision.transforms.Compose([
            torchvision.transforms.Resize((299, 299)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
        def get_acts(imgs, bs=32):
            acts = []
            for i in range(0, len(imgs), bs):
                batch = torch.stack([tf(img) for img in imgs[i:i+bs]]).to(DEVICE)
                with torch.no_grad():
                    acts.append(inception(batch).cpu().numpy())
            return np.concatenate(acts, axis=0)
        ra = get_acts(all_ref_imgs); ga = get_acts(all_our_imgs)
        mu1, s1 = np.mean(ra, 0), np.cov(ra, rowvar=False)
        mu2, s2 = np.mean(ga, 0), np.cov(ga, rowvar=False)
        diff = mu1 - mu2
        covmean = linalg.sqrtm(s1 @ s2, disp=False)[0]
        if np.iscomplexobj(covmean): covmean = covmean.real
        fid_val = float(diff @ diff + np.trace(s1 + s2 - 2 * covmean))
        print(f"FID (InceptionV3): {fid_val:.2f}")
    except Exception as e2:
        print(f"FID fallback also failed: {e2}")

# Save
results = {
    "config": {"occupations": NUM_OCCUPATIONS, "samples_per_occ": NUM_SAMPLES_PER_OCCUPATION, "total": N,
               "steps": NUM_INFERENCE_STEPS, "guidance": GUIDANCE_SCALE, "coeff": COEFFICIENT, "seed": SEED},
    "metrics": {"delta": float(delta), "fid": float(fid_val) if not np.isnan(fid_val) else None, "clip": float(mean_clip)},
    "gender": {"total": N, "male": N_male, "female": N_female, "male_ratio": N_male/N, "female_ratio": N_female/N},
    "timestamp": datetime.now().isoformat(),
}
with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"RESULTS (N={N})")
print(f"  Delta: {delta:.4f}  [paper: 0.05, CI: 0.044-0.11]")
print(f"  FID:   {fid_val}  [paper: 12.1, CI: 11.76-15.5]")
print(f"  CLIP:  {mean_clip:.2f}  [paper: 34.4, CI: 30.1-34.83]")
print(f"{'='*60}")
