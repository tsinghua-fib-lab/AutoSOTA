"""Reproduce the stalled checkpoint gradient failure and verify the repair."""
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from upstream_models import ROOT, make_team3
from team3_stable_training import log_gem

def main():
    torch.set_num_threads(8); torch.cuda.set_device(0)
    class Pool: kernel_size=8
    evidence={}
    x=torch.full((2,3,32),1e-6,device='cuda',requires_grad=True)
    p=torch.tensor([7.4093],device='cuda',requires_grad=True)
    old=torch.nn.functional.avg_pool1d(x.pow(p),8).pow(1/p)
    old.sum().backward()
    evidence['clamped_pool_old']={'finite_forward':bool(torch.isfinite(old).all()),'finite_x_gradient':bool(torch.isfinite(x.grad).all()),'finite_p_gradient':bool(torch.isfinite(p.grad).all())}
    x.grad=None;p.grad=None
    new=log_gem(Pool(),x,p)
    new.sum().backward()
    evidence['clamped_pool_log']={'finite_forward':bool(torch.isfinite(new).all()),'finite_x_gradient':bool(torch.isfinite(x.grad).all()),'finite_p_gradient':bool(torch.isfinite(p.grad).all()),'value':new.mean().item()}
    torch.manual_seed(33)
    z=(torch.rand(4,5,64,device='cuda')*3+1e-5).requires_grad_(); q=torch.tensor([4.3],device='cuda',requires_grad=True)
    ref=torch.nn.functional.avg_pool1d(z.double().pow(q.double()),8).pow(1/q.double())
    got=log_gem(Pool(),z,q)
    gr=torch.autograd.grad(ref.sum(),(z,q),retain_graph=True); gg=torch.autograd.grad(got.sum(),(z,q))
    evidence['float64_reference']={'forward_max_abs':float((ref-got).abs().max()),'input_gradient_max_abs':float((gr[0]-gg[0]).abs().max()),'p_gradient_abs':float((gr[1]-gg[1]).abs().max())}
    fold=2; directory=ROOT/'experiments/campaign14h_width_f2_snr_mix_pre24/fold2'
    ck=torch.load(directory/'Fold_2_last_model.pth',map_location='cpu')['model_state_dict']
    table=pd.read_csv(ROOT/'data/packed/train.csv'); rows=np.flatnonzero(table.fold_48.values!=fold)[:64]
    waves=np.load(ROOT/'data/packed/train.npy',mmap_mode='r');batch=torch.tensor(np.array(waves[rows]),device='cuda');target=torch.tensor(table.target.values[rows],device='cuda',dtype=torch.float32)
    for stable in [False,True]:
        model=make_team3(directory/'average_spectrum.pt',32,.8,stable_gem=stable).cuda().train();model.load_state_dict(ck)
        # Match the previous BF16 adapter, which already cast GeM to FP32.
        if not stable:
            for m in model.modules():
                if m.__class__.__name__=='GeM':
                    import types
                    def previous(self,value,p=3,eps=1e-6):
                        with torch.cuda.amp.autocast(enabled=False):
                            return torch.nn.functional.avg_pool1d(value.float().clamp(min=eps).pow(p.float()),self.kernel_size).pow(1/p.float())
                    m.gem=types.MethodType(previous,m)
        torch.manual_seed(48);torch.cuda.manual_seed_all(48)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred=model(batch).reshape(-1);loss=torch.nn.functional.binary_cross_entropy_with_logits(pred,target)
        loss.backward()
        bad=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        evidence['real_checkpoint_'+('log' if stable else 'old')]={'loss':float(loss),'nonfinite_gradient_tensors':len(bad),'first_bad':bad[:5]}
        if stable:
            assert not bad
            opt=torch.optim.AdamW(model.parameters(),lr=.0003); before={n:p.detach().clone() for n,p in model.named_parameters()}; opt.step()
            evidence['real_checkpoint_log']['changed_parameters']=sum(not torch.equal(p,before[n]) for n,p in model.named_parameters())
        del model
    assert evidence['clamped_pool_log']['finite_p_gradient']
    assert evidence['float64_reference']['forward_max_abs']<1e-5
    assert evidence['float64_reference']['input_gradient_max_abs']<1e-5
    path=ROOT/'reports/campaign_final_gem_audit.json';path.write_text(json.dumps(evidence,indent=2));print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
