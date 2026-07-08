# code from https://github.com/HxSun08/Alt-Diff/blob/main/classification/newlayer.py

import torch
from torch import nn
import math
import time

def relu(s):
    ss = s
    for i in range(len(s)):
        if s[i] < 0:
            ss[i] = 0
    return ss

def sgn(s):
    ss = torch.zeros(len(s))
    for i in range(len(s)):
        if s[i]<=0:
            ss[i] = 0
        else:
            ss[i] = 1
    return ss

def proj(s):
    ss = s
    for i in range(len(s)):
        if s[i] < 0:
            ss[i] = (ss[i] + math.sqrt(ss[i] ** 2 + 4 * 0.001)) / 2
    return ss

def alt_diff(Pi, qi, Ai, bi, Gi, hi, device="cuda"):
    
    n, m, d = qi.shape[0], bi.shape[0], hi.shape[0]
    xk = torch.zeros(n).to(device).to(torch.float64)
    sk = torch.zeros(d).to(device).to(torch.float64)
    lamb = torch.zeros(m).to(device).to(torch.float64)
    nu = torch.zeros(d).to(device).to(torch.float64)
    
    
    dxk = torch.zeros((n, n)).to(device).to(torch.float64)
    dsk = torch.zeros((d, n)).to(device).to(torch.float64)
    dlamb = torch.zeros((m, n)).to(device).to(torch.float64)
    dnu = torch.zeros((d, n)).to(device).to(torch.float64)
    
    rho = 1
    thres = 1e-5
    R = - torch.linalg.inv(Pi + rho * Ai.T @ Ai + rho * Gi.T @ Gi)
    
    res = [1000, -100]
    
    ATb = rho * Ai.T @ bi.double()
    GTh = rho * Gi.T @ hi
    begin2 = time.time()

    while abs((res[-1]-res[-2])/res[-2]) > thres:
        iter_time_start = time.time()
        #print((Ai.T @ lamb).shape)
        xk = R @ (qi + Ai.T @ lamb + Gi.T @ nu - ATb + rho * Gi.T @ sk - GTh)
        
        dxk = R @ (torch.eye(n).to(device) + Ai.T @ dlamb + Gi.T @ dnu + rho * Gi.T @ dsk)
        
        sk = relu(- (1 / rho) * nu - (Gi @ xk - hi))
        dsk = (-1 / rho) * sgn(sk).to(device).reshape(d,1) @ torch.ones((1, n)).to(device) * (dnu + rho * Gi @ dxk)

        lamb = lamb + rho * (Ai @ xk - bi)
        dlamb = dlamb + rho * (Ai @ dxk)

        nu = nu + rho * (Gi @ xk + sk - hi)
        dnu = dnu + rho * (Gi @ dxk + dsk)

        res.append(0.5 * (xk.T @ Pi @ xk) + qi.T @ xk)      

    return (xk, dxk)

class _AltDiffFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, q, G, h, A, b):
        B, n, _ = Q.shape
        device = Q.device

        xs = []
        dxs = []

        with torch.no_grad():
            for i in range(B):
                Pi = Q[i]
                qi = q[i]
                Gi = G[i]
                hi = h[i]

                Ai = A[i] if (A is not None and A.dim() == 3) else A
                bi = b[i] if (b is not None and b.dim() == 2) else b

                xk, dxk = alt_diff(Pi, qi, Ai, bi, Gi, hi, device=str(device))
                xs.append(xk)
                dxs.append(dxk)

        x = torch.stack(xs, dim=0)
        dx = torch.stack(dxs, dim=0)
        ctx.save_for_backward(dx)
        return x.to(Q.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (dx,) = ctx.saved_tensors
        grad_out = grad_out.to(dx.dtype)

        # dx is Jacobian ∂x/∂q, so grad_q = dx/dq @ grad_out
        grad_q = torch.bmm(dx.transpose(1, 2), grad_out.unsqueeze(-1)).squeeze(-1)

        return None, grad_q, None, None, None, None


class AltDiffLayer(nn.Module):
    def forward(self, Q, q, G, h, A=None, b=None):
        out_dtype = q.dtype
        Q = Q.double(); q = q.double(); G = G.double(); h = h.double()
        device = Q.device
        B, n, _ = Q.shape

        no_eq = (A is None) or (b is None) or (A.numel() == 0) or (b.numel() == 0)
        if no_eq:
            A = torch.empty((0, n), device=device, dtype=torch.float64)
            b = torch.empty((0,), device=device, dtype=torch.float64)
        else:
            A = A.double()
            b = b.double()

            mA = A.shape[-2] if A.dim() == 3 else A.shape[0]
            mb = b.shape[-1] if b.dim() == 2 else b.shape[0]
            assert mA == mb, f"A has {mA} rows but b has {mb} elems"

        x = _AltDiffFn.apply(Q, q, G, h, A, b)
        return x.to(out_dtype)