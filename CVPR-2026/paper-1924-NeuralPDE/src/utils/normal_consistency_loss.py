# normal_consistency_loss.py

import torch

def normal_consitency_loss(preds, band_pos, normals_gt, k_select=None, create_graph=False):
    """
    L_NC = E[ |<∇_q Phi(q), n_gt(q)>| ] en réutilisant 'preds' déjà calculé.
    Exige que 'band_pos' ait requires_grad=True AU MOMENT du forward de 'preds'.
    preds: (B,K,N), band_pos: (B,N,3), normals_gt: (B,N,3)
    """
    B, N, _ = band_pos.shape
    outK = preds.mean(dim=1, keepdim=True) if k_select is None else preds[:, k_select:k_select+1, :]

    must_retain = True if create_graph else False

    s = outK.sum()  
    grads = torch.autograd.grad(s, band_pos,
                            retain_graph=must_retain,
                            create_graph=create_graph)[0]  

    grads = grads / (grads.norm(dim=-1, keepdim=True).clamp_min(1e-12))  

    proj = (grads * normals_gt).sum(dim=-1).abs() 
    return proj.mean()

def normal_consitency_loss_eval(preds, band_pos, normals_gt, k_select=None, create_graph=False):
    """
    L_NC = E[ |<∇_q Phi(q), n_gt(q)>| ] en réutilisant 'preds' déjà calculé.
    Exige que 'band_pos' ait requires_grad=True AU MOMENT du forward de 'preds'.
    preds: (B,K,N), band_pos: (B,N,3), normals_gt: (B,N,3)
    """
    B, N, _ = band_pos.shape
    outK = preds.mean(dim=1, keepdim=True) if k_select is None else preds[:, k_select:k_select+1, :]

    s = outK.sum() 
    grads = torch.autograd.grad(s, band_pos,
                                retain_graph=False,
                                create_graph=create_graph)[0] 

    grads = grads / (grads.norm(dim=-1, keepdim=True).clamp_min(1e-12)) 

    proj = (grads * normals_gt).sum(dim=-1).abs() 
    return proj.mean()