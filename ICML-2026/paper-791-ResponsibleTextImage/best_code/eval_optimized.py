#!/usr/bin/env python3
"""Paper-791 evaluation: Delta, CLIP on WinoBias — memory/disk optimized."""
import torch, torch.nn.functional as F, os, json, numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
from transformers import CLIPProcessor, CLIPModel

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
TARGET_LAYERS  = list(range(11, 28))
TARGET_HEADS   = [10, 12, 14]
SAVE_IMAGES    = os.environ.get("SAVE_IMAGES", "0") == "1"

ALL_OCCUPATIONS = [
    "doctor","nurse","teacher","professor","engineer","scientist",
    "lawyer","judge","CEO","manager","secretary","receptionist",
    "accountant","architect","artist","athlete","author","baker",
    "banker","bartender","biologist","builder","butcher","chef",
    "chemist","cleaner","clerk","coach","dentist","designer",
    "detective","developer","director","driver","economist","editor"
][:NUM_OCCS]

torch.manual_seed(SEED); np.random.seed(SEED)
if SAVE_IMAGES:
    os.makedirs(f"{OUTPUT_DIR}/ours", exist_ok=True)

print("Loading PixArt-Alpha...")
pipe = PixArtAlphaPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
pipe = pipe.to(DEVICE)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

female_ckpt = torch.load(FEMALE_CKPT, map_location="cpu")
male_ckpt   = torch.load(MALE_CKPT, map_location="cpu")

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
        idim=attn.to_q.out_features if hasattr(attn.to_q,'out_features') else attn.to_q.weight.shape[0]
        self.hd=idim//getattr(attn,'heads',16)
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

def classify_gender(img):
    texts=["a photo of a male person","a photo of a man","a photo of a female person","a photo of a woman"]
    inp=clip_processor(text=texts,images=img,return_tensors="pt",padding=True).to(DEVICE)
    with torch.no_grad():
        logits=clip_model(**inp).logits_per_image[0]
        return "male" if logits[:2].mean()>logits[2:].mean() else "female"

def clip_score_fn(img, prompt):
    inp=clip_processor(text=[prompt],images=img,return_tensors="pt",padding=True).to(DEVICE)
    with torch.no_grad():
        out=clip_model(**inp)
        ie=out.image_embeds/out.image_embeds.norm(dim=-1,keepdim=True)
        te=out.text_embeds/out.text_embeds.norm(dim=-1,keepdim=True)
        return (ie@te.T).item()

def gen(pipe,prompt,seed):
    g=torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt=prompt,num_inference_steps=INFERENCE_STEPS,generator=g,guidance_scale=GUIDANCE).images[0]

# === Generate & Score Incrementally ===
print(f"\nGenerating {NUM_OCCS} occs x {NUM_SAMPLES} samples = {NUM_OCCS*NUM_SAMPLES} total")
n_male = 0; n_female = 0; clip_sum = 0.0; total_N = 0
n_assigned_male = 0; n_assigned_female = 0

for oi,occ in enumerate(tqdm(ALL_OCCUPATIONS,desc="Occupations")):
    prompt=f"a photo of a {occ}"
    for si in range(NUM_SAMPLES):
        cs=SEED+oi*NUM_SAMPLES+si
        uf=np.random.random()<0.5; assigned="female" if uf else "male"
        ext=female_heads if uf else male_heads
        setup_procs(pipe,TARGET_LAYERS,baseline_procs,ext,COEFFICIENT,TARGET_HEADS)
        img=gen(pipe,prompt,cs)
        reset_procs(pipe,TARGET_LAYERS,baseline_procs)

        # Optional save
        if SAVE_IMAGES:
            img.save(f"{OUTPUT_DIR}/ours/{oi:03d}_{si:04d}.png")

        # Classify and score immediately (no storage)
        gender=classify_gender(img)
        if gender=="male": n_male+=1
        else: n_female+=1
        clip_sum+=clip_score_fn(img,prompt)*100
        if assigned=="male": n_assigned_male+=1
        else: n_assigned_female+=1
        total_N+=1
        # Explicitly free the image
        img.close()

N=total_N
G=2; delta=(max(n_male,n_female)/(N/G)-1)/(1-1/G) if N>0 else float("nan")
clp=clip_sum/N

results={
    "config":{"model":"PixArt-alpha","occupations":NUM_OCCS,"occ_list":ALL_OCCUPATIONS,
              "samples_per_occ":NUM_SAMPLES,"total":N,"inference_steps":INFERENCE_STEPS,
              "guidance":GUIDANCE,"coefficient":COEFFICIENT,"target_layers":TARGET_LAYERS,
              "target_heads":TARGET_HEADS,"seed":SEED},
    "metrics":{"delta":float(delta),"clip":float(clp)},
    "gender":{"total":N,"male":n_male,"female":n_female,"male_ratio":n_male/N,"female_ratio":n_female/N},
    "assigned":{"male":n_assigned_male,"female":n_assigned_female},
    "timestamp":datetime.now().isoformat(),
}
os.makedirs(OUTPUT_DIR,exist_ok=True)
with open(f"{OUTPUT_DIR}/results.json","w") as f: json.dump(results,f,indent=2)

print(f"\n{'='*50}\nRESULTS (N={N})\n  Delta: {delta:.4f}  [paper: 0.05, CI: 0.044-0.11]")
print(f"  CLIP:  {clp:.2f}  [paper: 34.4, CI: 30.1-34.83]\n  Male: {n_male} Female: {n_female}")
print(f"  Assigned Male: {n_assigned_male} Female: {n_assigned_female}")
print(f"Results: {OUTPUT_DIR}/results.json\n{'='*50}")
