from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from torch import nn
ROOT=Path('/workspace/c09-single-cell'); OUT=ROOT/'experiments'/'masked_cell_transfer_v2_results.json'
SEED=17
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device='cuda' if torch.cuda.is_available() else 'cpu'
df=pd.read_parquet(ROOT/'data'/'de_train.parquet')
meta=['cell_type','sm_name','sm_lincs_id','SMILES','control']; genes=[c for c in df.columns if c not in meta]
sources=['NK cells','T cells CD4+','T cells CD8+','T regulatory cells']; targets=['B cells','Myeloid cells']; cells=sources+targets; cell_idx={c:i for i,c in enumerate(cells)}
raw={(r.sm_name,r.cell_type):r[genes].to_numpy(np.float32) for _,r in df.iterrows()}
complete=[c for c,g in df.groupby('sm_name') if set(cells)<=set(g.cell_type)]
source_rows=df[df.cell_type.isin(sources)][genes].to_numpy(np.float32)
pca=PCA(n_components=96,svd_solver='randomized',random_state=SEED).fit(source_rows)
pc_mean=pca.mean_.astype(np.float32); comps=pca.components_.astype(np.float32)
def project(x): return (x-pc_mean)@comps.T
def inverse(z): return z@comps+pc_mean
all_pc=np.stack([project(raw[k]) for k in raw]); pc_mu=all_pc.mean(0); pc_sd=all_pc.std(0)+1e-4
def zscore(x): return (project(x)-pc_mu)/pc_sd
def anchor(comp):
 vals=[raw[(comp,c)] for c in sources if (comp,c) in raw]; x=np.stack(vals); mean=x.mean(0); med=np.median(x,0); strength=np.sqrt(np.mean(mean*mean)); nk=raw.get((comp,'NK cells'),mean); snk=np.sqrt(np.mean(nk*nk)); return (.75*nk+.25*mean) if strength>=1.987 and snk/strength>=1.5 else (mean if strength>=1.987 else med)
def features(comp,target):
 blocks=[]; masks=[]
 for s in sources:
  if s==target or (comp,s) not in raw: blocks.append(np.zeros(96,np.float32)); masks.append(0.)
  else: blocks.append(zscore(raw[(comp,s)])); masks.append(1.)
 one=np.zeros(6,np.float32); one[cell_idx[target]]=1
 return np.concatenate(blocks+[np.asarray(masks,np.float32),one])
class Net(nn.Module):
 def __init__(self,d):
  super().__init__(); self.net=nn.Sequential(nn.Linear(d,384),nn.LayerNorm(384),nn.GELU(),nn.Dropout(.1),nn.Linear(384,192),nn.GELU(),nn.Dropout(.1),nn.Linear(192,96))
 def forward(self,x): return self.net(x)
def make_train(heldout):
 xs=[];ys=[];ws=[]
 for comp,g in df.groupby('sm_name'):
  avail=set(g.cell_type)
  for t in cells:
   if t not in avail or (t in targets and comp in heldout): continue
   if sum(s in avail and s!=t for s in sources)<3: continue
   xs.append(features(comp,t)); ys.append((project(raw[(comp,t)])-project(anchor(comp)))/pc_sd); ws.append(8.0 if t in targets else 1.0)
 return np.asarray(xs,np.float32),np.asarray(ys,np.float32),np.asarray(ws,np.float32)
def train_predict(train,heldout,seed):
 torch.manual_seed(seed); x,y,w=make_train(heldout); xt=torch.from_numpy(x).to(device); yt=torch.from_numpy(y).to(device); wt=torch.from_numpy(w).to(device); model=Net(x.shape[1]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=2e-3)
 best=None;best_loss=1e9;pat=0
 for epoch in range(500):
  model.train(); order=torch.randperm(len(xt),device=device); total=0.
  for ix in order.split(64):
   pr=model(xt[ix]); per=((pr-yt[ix])**2).mean(1); loss=(per*wt[ix]).sum()/wt[ix].sum(); opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();total+=float(loss)*len(ix)
  cur=total/len(xt)
  if cur<best_loss-1e-5: best_loss=cur;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};pat=0
  else: pat+=1
  if pat>=35: break
 model.load_state_dict(best);model.eval(); out={}
 with torch.no_grad():
  for c in heldout:
   for t in targets:
    rz=model(torch.from_numpy(features(c,t)[None]).to(device)).cpu().numpy()[0]; out[(c,t)]=anchor(c)+inverse(rz*pc_sd)-pc_mean
 return out,best_loss,epoch+1
kf=KFold(5,shuffle=True,random_state=SEED); fold_records=[]; pred={t:{} for t in targets}
for fold,(_,va) in enumerate(kf.split(complete)):
 held=[complete[i] for i in va]; ensemble=[]; meta=[]
 for seed in [17,29,43]:
  o,loss,epochs=train_predict(None,held,seed); ensemble.append(o);meta.append({'seed':seed,'loss':loss,'epochs':epochs})
 for c in held:
  for t in targets: pred[t][c]=np.mean([o[(c,t)] for o in ensemble],0)
 fold_records.append({'fold':fold,'heldout':held,'runs':meta}); print('fold',fold,held,meta,flush=True)
report={'device':device,'complete_compounds':complete,'folds':fold_records,'targets':{}}
rng=np.random.default_rng(SEED)
for t in targets:
 y=np.stack([raw[(c,t)] for c in complete]); base=np.stack([anchor(c) for c in complete]); ph=np.stack([pred[t][c] for c in complete]); eb=np.sqrt(np.mean((base-y)**2,1)); en=np.sqrt(np.mean((ph-y)**2,1)); d=eb-en; boots=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(20000)]); rec={'anchor':float(eb.mean()),'model':float(en.mean()),'delta':float(d.mean()),'wins':int((d>0).sum()),'ci95':[float(x) for x in np.quantile(boots,[.025,.975])],'per_compound':{c:float(v) for c,v in zip(complete,d)}};report['targets'][t]=rec;print(t,rec,flush=True)
OUT.write_text(json.dumps(report,indent=2),encoding='utf-8')

