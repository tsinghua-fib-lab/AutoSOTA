import torch

def conjugate_gradient(fn_fvp, g, nsteps=10, residual_tol=1e-10):
    x = torch.zeros_like(g) 
    r = g.clone()
    p = r.clone()
    rdotr = torch.dot(r, r)

    for i in range(nsteps):
        z = fn_fvp(p)
        alpha = rdotr / torch.dot(p, z)
        x += alpha * p
        r -= alpha * z
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x