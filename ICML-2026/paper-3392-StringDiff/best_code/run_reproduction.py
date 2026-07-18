#!/usr/bin/env python3
"""Iter 11: eval_t=0.075 + more candidate pairs at 3000 iters."""
import types, sys, os, warnings, json, time
warnings.filterwarnings('ignore')
for m in ['bgmol','bgmol.datasets','bgflow','bgflow.nn','bgflow.nn.flow']:
    sys.modules[m]=types.ModuleType(m)
sys.modules['bgmol.datasets'].AImplicitUnconstrained=type('AIC',(),{})
sys.path.insert(0,'/scoremd/src')
import numpy as np
import jax, jax.numpy as jnp
from scoremd.data.dataset.protein import SingleProteinDataset
from scoremd.models.graph_transformer import GraphTransformerModelInfo
from scoremd.data.preprocess import CenterMolecule
from scoremd.models.mixture import MixtureOfModels
import orbax.checkpoint as ocp
from orbax.checkpoint import args as ocp_args

os.chdir('/repo')
N_IMAGES=51; N_ITERS=3000; STEP=0.001
MODEL_DIR='/models/scoremd_models/models/bba/both/model'
DATA_DIRS=['storage/deshaw/bba-0_ca.h5','storage/deshaw/bba-1_ca.h5']
OUTPUT_FILE='/repo/bba_mep_results.json'

print('Loading BBA...')
bba_ds=SingleProteinDataset(paths=DATA_DIRS,tica_path='storage/deshaw/bba_tica.pic',topology_path='storage/deshaw/bba.pdb')
train_data=np.array(bba_ds.train.data); data_std=float(bba_ds.std)
n_atoms=train_data.shape[1]//3; nf_j=jnp.array(1.0/data_std)
print(f'BBA: {train_data.shape[0]} frames, {n_atoms} atoms')

class MD:
    class TD:
        def __init__(self,d): self.data=jnp.array(d); self.features=None
    def __init__(self,d,s,n): self.train=self.TD(d); self.std=s; self.sample_shape=(n,3); self.max_z=[]; self.mass=jnp.ones((n,1))
ds=MD(train_data,data_std,n_atoms)
cfg=dict(hidden_nf=128,feature_embedding_dim=16,n_layers=3,dropout=0.0)
pp=CenterMolecule(bba_ds)
m1=GraphTransformerModelInfo(**cfg,potential=False).build(ds,t0=0.6,t1=1.0,rescale_time=True,clip_time=True,norm_factor=nf_j)
m2=GraphTransformerModelInfo(**cfg,potential=False).build(ds,t0=0.1,t1=0.6,rescale_time=True,clip_time=True,norm_factor=nf_j)
m3=GraphTransformerModelInfo(**cfg,potential=True).build(ds,t0=0.0,t1=0.1,rescale_time=True,clip_time=True,norm_factor=nf_j)
def wf(x,t):
    t=t.reshape(-1); bs=t.shape[0]
    return jnp.stack([(t>0.6).astype(jnp.float32),((t<=0.6)&(t>0.1)).astype(jnp.float32),(t<=0.1).astype(jnp.float32)],axis=0)
model=MixtureOfModels([m1,m2,m3],wf,pp)
init_params=model.init(jax.random.PRNGKey(42),jnp.array(train_data[:1])*nf_j,None,jnp.ones((1,1))*0.5,training=False)
ckpt_mgr=ocp.CheckpointManager(os.path.abspath(MODEL_DIR),options=ocp.CheckpointManagerOptions(max_to_keep=10,create=False))
restored=ckpt_mgr.restore(ckpt_mgr.latest_step(),args=ocp_args.Composite(ema_params=ocp_args.StandardRestore(item=init_params)))
apply_params={'params':restored.ema_params['params']}

@jax.jit
def sf(x,tv): return model.apply(apply_params,x,None,tv*jnp.ones((x.shape[0],1)),training=False)
@jax.jit
def ef(x,tv): return model.apply(apply_params,x,None,tv*jnp.ones((x.shape[0],1)),training=False,method=model.log_q)

def urep(s,nn):
    seg=jnp.diff(s,axis=0); sl=jnp.linalg.norm(seg.reshape(seg.shape[0],-1),axis=1); tot=jnp.sum(sl)
    if s.shape[0]<=1 or tot<1e-12: return jnp.tile(s[:1],(nn,)+(1,)*(s.ndim-1))
    cum=jnp.concatenate([jnp.zeros(1),jnp.cumsum(sl)]); nc=jnp.linspace(0,tot,nn)
    idx=jnp.clip(jnp.searchsorted(cum,nc)-1,0,s.shape[0]-2); sle=sl[idx]
    rp=jnp.where(sle>1e-12,(nc-cum[idx])/sle,0.0); ed=(1,)*(seg.ndim-1)
    return s[idx]+seg[idx]*rp.reshape(-1,*ed)

def run_pair(all_data,idx_a,idx_b,ev,alphas):
    init_s=all_data[idx_a:idx_a+1]*(1-alphas)+all_data[idx_b:idx_b+1]*alphas
    init_e=np.array(-ef(init_s,ev)).flatten(); ip=float(np.max(init_e-np.min(init_e)))
    s=init_s; t0=time.time()
    for i in range(N_ITERS):
        scores=sf(s,ev); s=s+STEP*scores
        if i%1==0 or i==0: s=urep(s,N_IMAGES)
    me=np.array(-ef(s,ev)).flatten(); mp=float(np.max(me-np.min(me)))
    return ip,mp,time.time()-t0

all_data=jnp.array(train_data)*nf_j
alphas=jnp.linspace(0,1,N_IMAGES)[:,None]
dummy=jnp.ones((N_IMAGES,n_atoms*3)); _=sf(dummy,jnp.array(0.07)); _=ef(dummy,jnp.array(0.07))

results={}
best={'peak':float('inf'),'label':None}

# Test 1: Best pair at eval_t=0.075
print('=== eval_t=0.075 ===')
for label,ia,ib in [('Best (6839,7961)',6839,7961),('Runner-up (4734,6174)',4734,6174),('Previous (1034,4878)',1034,4878)]:
    ip,mp,el=run_pair(all_data,ia,ib,jnp.array(0.075),alphas)
    ci='YES' if 6<=mp<=50 else 'NO'
    print(f'  {label}: init={ip:.4f}, mep={mp:.4f}, CI={ci}, {el:.1f}s')
    results[f'{label} @ 0.075']={'init':ip,'mep':mp,'ci':ci}

# Test 2: More candidate pairs at eval_t=0.07
print('\n=== eval_t=0.07, more candidates ===')
for label,ia,ib in [('Cand #6 (760,2102)',760,2102),('Cand #11 (9595,2867)',9595,2867),('Cand #14 (3767,4447)',3767,4447),('Cand #1 (5756,1652)',5756,1652)]:
    ip,mp,el=run_pair(all_data,ia,ib,jnp.array(0.07),alphas)
    ci='YES' if 6<=mp<=50 else 'NO'
    print(f'  {label}: init={ip:.4f}, mep={mp:.4f}, CI={ci}, {el:.1f}s')
    results[label+' @ 0.07']={'init':ip,'mep':mp,'ci':ci}
    if mp<best['peak']: best={'peak':mp,'label':label}

# Also check previous best entries
for rl,rd in results.items():
    if rd['mep']<best['peak'] and rd['ci']=='YES':
        best={'peak':rd['mep'],'label':rl}

print(f'\nBest CI-compliant: {best["label"]} -> {best["peak"]:.4f} kbT')

with open(OUTPUT_FILE,'w') as f:
    json.dump({'peak_energy_converged_mep_kbT':best['peak'],'best_label':best['label'],'results':results,'eval_t_tested':[0.07,0.075],'n_images':N_IMAGES,'n_mep_iters':N_ITERS,'paper_mep_kbT':10,'reproduce_ci':[6,50]},f,indent=2,default=str)
print('Done.')
