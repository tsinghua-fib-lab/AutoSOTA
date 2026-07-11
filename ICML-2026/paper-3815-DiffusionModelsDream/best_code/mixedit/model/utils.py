import torch
import torch.nn.functional as F


def get_model_fn(model, train=False):
    input_dim = (model.module if hasattr(model, "module") else model).input_dim
    def model_fn(x, t):
        if train:
            model.train()
        else:
            model.eval()
        return model(x[...,:input_dim], t)
    return model_fn

def get_drift_fn(model, sde, train=False, sampling=False, **kwargs):
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        def drift_fn(x, t):
            probs = F.softmax(model_fn(x, t).to(torch.float32), dim=-1)
            probs = torch.cat([probs, torch.zeros((*probs.shape[:-1], x.shape[-1]-probs.shape[-1]), device=x.device)], dim=-1)
            
            drift = sde.manifold.weighted_sum(probs, x)
            drift = sde.scale_by_coeff(drift, t)
            drift = sde.manifold.to_tangent(drift, x)
            return drift

    return drift_fn

def get_model_fn_mixed(model, train=False):
    # Handle DDP wrapping
    module = model.module if hasattr(model, "module") else model
    input_dim = module.input_dim
    continuous_dim = getattr(module, "continuous_dim", None) 

    def model_fn(x_disc, x_cont, t_disc, t_cont):
        if train:
            model.train()
        else:
            model.eval()
        
        # Slice discrete input to vocab size if necessary (handling manifold dim)
        x_d = x_disc[..., :input_dim]
        
        # Slice continuous input if dimension is known
        x_c = x_cont
        if continuous_dim is not None:
            x_c = x_c[..., :continuous_dim]
            
        return model(x_d, x_c, t_disc, t_cont)
    
    return model_fn


def get_drift_fn_mixed(model, sde_disc, sde_cont, train=False, sampling=False, **kwargs):
    if sampling:
        assert not train, "Must sample in eval mode"
        
    model_fn = get_model_fn_mixed(model, train=train)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        def drift_fn(x_disc, x_cont, t_disc, t_cont):
            # 1. Forward Pass
            # logits_disc: [B, L, V_disc]
            # eps_cont:    [B, L, V_cont]
            logits_disc, eps_cont = model_fn(x_disc, x_cont, t_disc, t_cont)
            
            # --- Discrete Drift (RDLM Logic) ---
            # Softmax logits to get probabilities
            probs = F.softmax(logits_disc.to(torch.float32), dim=-1)
            
            # Pad probabilities if x_disc has extra dimensions (e.g. mask token in Mixture Path)
            # x_disc shape is usually [B, L, V+1] (hypersphere embedding)
            if x_disc.shape[-1] > probs.shape[-1]:
                pad = torch.zeros((*probs.shape[:-1], x_disc.shape[-1]-probs.shape[-1]), device=x_disc.device)
                probs = torch.cat([probs, pad], dim=-1)
            
            # Compute manifold vector field
            drift_d = sde_disc.manifold.weighted_sum(probs, x_disc)
            drift_d = sde_disc.scale_by_coeff(drift_d, t_disc)
            drift_d = sde_disc.manifold.to_tangent(drift_d, x_disc)
            
            # --- Continuous Drift (Gaussian SDE Logic) ---
            # Convert predicted epsilon -> SDE Drift
            drift_c = sde_cont.get_drift(x_cont, t_cont, eps_cont)
            
            return drift_d, drift_c

    return drift_fn