#!/usr/bin/env python3
"""Paper-791 evaluation: Delta, CLIP on WinoBias (PixArt-alpha + gender concept vectors)"""
import torch, torch.nn.functional as F, os, json, numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
from transformers import CLIPProcessor, CLIPModel

# === Configuration ===
MODEL_PATH     = os.environ.get("PIXART_MODEL_PATH", "/paper_data")
FEMALE_CKPT    = os.environ.get("FEMALE_CKPT", "/repo/checkpoints/external_concept_female.pt")
MALE_CKPT      = os.environ.get("MALE_CKPT", "/repo/checkpoints/external_concept_male.pt")
OUTPUT_DIR     = os.environ.get("OUTPUT_DIR", "/repo/evaluation_output")
NUM_OCCS       = int(os.environ.get("NUM_OCCUPATIONS", "12"))
NUM_SAMPLES    = int(os.environ.get("NUM_SAMPLES", "25"))
INFERENCE_STEPS = int(os.environ.get("INFERENCE_STEPS", "20"))
GUIDANCE       = float(os.environ.get("GUIDANCE_SCALE", "4.5"))
COEFFICIENT    = float(os.environ.get("COEFFICIENT", "10.0"))
SEED           = int(os.environ.get("SEED", "42"))
DEVICE         = "cuda:0" if torch.cuda.is_available() else "cpu"

# WinoBias occupations (36 total)
ALL_OCCUPATIONS = [
    "doctor","nurse","teacher","professor","engineer","scientist",
    "lawyer","judge","CEO","manager","secretary","receptionist",
    "accountant","architect","artist","athlete","author","baker",
    "banker","bartender","biologist","builder","butcher","chef",
    "chemist","cleaner","clerk","coach","dentist","designer",
    "detective","developer","director","driver","economist","editor"
][:NUM_OCCS]
TARGET_LAYERS = list(range(11, 28))
TARGET_HEADS  = [10, 12, 14]

torch.manual_seed(SEED); np.random.seed(SEED)
os.makedirs(f"{OUTPUT_DIR}/baseline", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/ours", exist_ok=True)

# Load PixArt
print("Loading PixArt-Alpha...")
pipe = PixArtAlphaPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
pipe = pipe.to(DEVICE)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Load CLIP
print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Load concept checkpoints
female_ckpt = torch.load(FEMALE_CKPT, map_location="cpu")
male_ckpt   = torch.load(MALE_CKPT, map_location="cpu")

# === External Heads classes ===
class LoadedExternalHeads:
    def __init__(self, sd, tl, nh=16, hd=72):
        self.target_layers=tl; self.num_heads=nh; self.head_dim=hd; self.external_heads={}
        for k,v in sd.items():
            nk=k; 
            if nk.startswith("external_heads."): nk=nk[len("external_heads."):]
            self.external_heads[nk]=v
    def get(self, li, hi, sl, dev, dt, th=None):
        if th is not None and hi not in th: return torch.zeros(sl,self.head_dim,device=dev,dtype=dt)
        if li not in self.target_layers: return torch.zeros(sl,self.head_dim,device=dev,dtype=dt)
        k=f"layer_{li}_head_{hi}"
        if k not in self.external_heads: return torch.zeros(sl,self.head_dim,device=dev,dtype=dt)
        return self.external_heads[k].to(device=dev,dtype=dt)
    def all_heads(self, li, sl, dev, dt, th=None):
        return torch.stack([self.get(li,h,sl,dev,dt,th) for h in range(self.num_heads)], dim=0)

class ExternalHeadProcessor:
    def __init__(self, orig, li, attn, ext, coeff, th=None):
        self.orig=orig; self.li=li; self.attn=attn; self.ext=ext; self.coeff=coeff; self.th=th
        self.nh=getattr(attn,'heads',None)
        idim=attn.to_q.out_features if hasattr(attn.to_q,'out_features') else attn.to_q.weight.shape[0]
        self.hd=idim//self.nh
    def _bf(self, attn, d):
        to=attn.to_out
        if to is None: return d
        if isinstance(to,torch.nn.ModuleList):
            y=d
            for m in to: y=F.linear(y,m.weight,bias=None) if isinstance(m,torch.nn.Linear) else m(y)
            return y
        if isinstance(to,torch.nn.Sequential):
            f0=to[0]
            if isinstance(f0,torch.nn.Linear):
                y=F.linear(d,f0.weight,bias=None)
                for m in list(to)[1:]: y=m(y)
                return y
            return to(d)
        if isinstance(to,torch.nn.Linear): return F.linear(d,to.weight,bias=None)
        return to(d)
    def __call__(self, attn, hs, encoder_hidden_states=None, attention_mask=None, **kw):
        oo=self.orig(attn,hs,encoder_hidden_states=encoder_hidden_states,attention_mask=attention_mask,**kw)
        B,N,_=hs.shape; dev,dt=hs.device,hs.dtype
        ext=self.ext.all_heads(self.li,N,dev,dt,self.th).unsqueeze(0).expand(B,-1,-1,-1)
        H,dh=ext.shape[1],ext.shape[3]
        dc=ext.transpose(1,2).reshape(B,N,H*dh)
        return oo+self.coeff*self._bf(attn,dc)

def save_procs(pipe,ls): return {l:pipe.transformer.transformer_blocks[l].attn2.get_processor() for l in ls}
def reset_procs(pipe,ls,ps):
    for l in ls: pipe.transformer.transformer_blocks[l].attn2.set_processor(ps[l])
def setup_procs(pipe,ls,ps,ext,coeff,th):
    for l in ls:
        b=pipe.transformer.transformer_blocks[l]
        b.attn2.set_processor(ExternalHeadProcessor(ps[l],l,b.attn2,ext,coeff,th))

female_heads = LoadedExternalHeads(female_ckpt, TARGET_LAYERS)
male_heads   = LoadedExternalHeads(male_ckpt, TARGET_LAYERS)
baseline_procs = save_procs(pipe, TARGET_LAYERS)

# Helpers
def classify_gender(img):
    texts=["a photo of a male person","a photo of a man","a photo of a female person","a photo of a woman"]
    inp=clip_processor(text=texts,images=img,return_tensors="pt",padding=True).to(DEVICE)
    with torch.no_grad():
        logits=clip_model(**inp).logits_per_image[0]
        return "male" if logits[:2].mean()>logits[2:].mean() else "female"

def clip_score(img, prompt):
    inp=clip_processor(text=[prompt],images=img,return_tensors="pt",padding=True).to(DEVICE)
    with torch.no_grad():
        out=clip_model(**inp)
        ie=out.image_embeds/out.image_embeds.norm(dim=-1,keepdim=True)
        te=out.text_embeds/out.text_embeds.norm(dim=-1,keepdim=True)
        return (ie@te.T).item()

def gen(pipe,prompt,seed):
    g=torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt=prompt,num_inference_steps=INFERENCE_STEPS,generator=g,guidance_scale=GUIDANCE).images[0]

# === Generate ===
print(f"\nGenerating {NUM_OCCS} occs x {NUM_SAMPLES} samples ({NUM_OCCS*NUM_SAMPLES} total)")
all_ref,all_our,all_prompts,all_assigned=[],[],[],[]
for oi,occ in enumerate(tqdm(ALL_OCCUPATIONS,desc="Occupations")):
    prompt=f"a photo of a {occ}"
    for si in range(NUM_SAMPLES):
        cs=SEED+oi*NUM_SAMPLES+si
        reset_procs(pipe,TARGET_LAYERS,baseline_procs)
        ref=gen(pipe,prompt,cs)
        uf=np.random.random()<0.5; assigned="female" if uf else "male"
        ext=female_heads if uf else male_heads
        setup_procs(pipe,TARGET_LAYERS,baseline_procs,ext,COEFFICIENT,TARGET_HEADS)
        our=gen(pipe,prompt,cs)
        reset_procs(pipe,TARGET_LAYERS,baseline_procs)
        ref.save(f"{OUTPUT_DIR}/baseline/{oi:03d}_{si:04d}.png")
        our.save(f"{OUTPUT_DIR}/ours/{oi:03d}_{si:04d}.png")
        all_ref.append(ref); all_our.append(our); all_prompts.append(prompt); all_assigned.append(assigned)

N=len(all_our); print(f"Generated {N} image pairs")

# === Metrics ===
print("\nComputing metrics...")
gender_preds=[classify_gender(img) for img in tqdm(all_our,desc="Gender")]
n_m=sum(1 for g in gender_preds if g=="male"); n_f=N-n_m
G=2; delta=(max(n_m,n_f)/(N/G)-1)/(1-1/G) if N>0 else float('nan')

clip_scores=[clip_score(img,p)*100 for img,p in tqdm(zip(all_our,all_prompts),total=N,desc="CLIP")]
clp=np.mean(clip_scores)

results={
    "config":{"model":"PixArt-alpha","occupations":NUM_OCCS,"occ_list":ALL_OCCUPATIONS,
              "samples_per_occ":NUM_SAMPLES,"total":N,"inference_steps":INFERENCE_STEPS,
              "guidance":GUIDANCE,"coefficient":COEFFICIENT,"target_layers":TARGET_LAYERS,
              "target_heads":TARGET_HEADS,"seed":SEED},
    "metrics":{"delta":float(delta),"clip":float(clp)},
    "gender":{"total":N,"male":n_m,"female":n_f,"male_ratio":n_m/N,"female_ratio":n_f/N},
    "assigned":{"male":sum(1 for a in all_assigned if a=="male"),"female":sum(1 for a in all_assigned if a=="female")},
    "timestamp":datetime.now().isoformat(),
}
with open(f"{OUTPUT_DIR}/results.json","w") as f: json.dump(results,f,indent=2)

print(f"\n{'='*50}\nRESULTS (N={N})\n  Delta: {delta:.4f}  [paper: 0.05, CI: 0.044-0.11]")
print(f"  CLIP:  {clp:.2f}  [paper: 34.4, CI: 30.1-34.83]\nResults: {OUTPUT_DIR}/results.json\n{'='*50}")
