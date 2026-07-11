# evaluate2D_patched.py
# 2-D evaluators: damped wave (time) and Poisson–Gauss (steady). Includes IC/BC where applicable.
import numpy as np
import torch, math, sympy
from torch.nn import functional as F
torch.set_grad_enabled(False)  # Disable gradients by default
torch.set_num_threads(1)  # Prevent thread explosion
# ---------- helpers ----------
def _ensure_vec(y, like: torch.Tensor):
    if isinstance(y, torch.Tensor):
        if y.ndim == 0: return torch.full_like(like, y.item())
        if y.shape == like.shape: return y.to(dtype=like.dtype, device=like.device)
        try: return (y + torch.zeros_like(like)).to(dtype=like.dtype, device=like.device)
        except Exception: return torch.full_like(like, float(y.detach().cpu().numpy()))
    else:
        return torch.full_like(like, float(y))

# def _safe_grad(y, x, create_graph=True):
#     if not (isinstance(y, torch.Tensor) and y.requires_grad): return torch.zeros_like(x)
#     g = torch.autograd.grad(y, x, create_graph=create_graph, allow_unused=True)[0]
#     return torch.zeros_like(x) if g is None else g
def _safe_grad(y, x, create_graph=True):
    if not (isinstance(y, torch.Tensor) and y.requires_grad): 
        return torch.zeros_like(x)
    try:
        # Only use create_graph for second derivatives, not first
        g = torch.autograd.grad(y.sum(), x, 
                                retain_graph=True,  # This is cheaper than create_graph
                                create_graph=create_graph, 
                                allow_unused=True)[0]
        return g if g is not None else torch.zeros_like(x)
    except:
        return torch.zeros_like(x)

def _lambdify_to_torch_2d(sympy_expr, X, Y):
    import numpy as _np, torch as _th
    if sympy_expr is None: return _th.zeros_like(X)
    f_np = sympy.lambdify(["x","y"], sympy_expr, "numpy")
    vals = f_np(_np.asarray(X.detach().cpu().numpy()),
                _np.asarray(Y.detach().cpu().numpy()))
    return _th.as_tensor(vals, dtype=X.dtype, device=X.device)

def _lambdify_to_torch_3d(sympy_expr, X, Y, T):
    import numpy as _np, torch as _th
    if sympy_expr is None: return _th.zeros_like(X)
    f_np = sympy.lambdify(["x","y","t"], sympy_expr, "numpy")
    vals = f_np(_np.asarray(X.detach().cpu().numpy()),
                _np.asarray(Y.detach().cpu().numpy()),
                _np.asarray(T.detach().cpu().numpy()))
    return _th.as_tensor(vals, dtype=X.dtype, device=X.device)

# ---------- truth u(x,y,t) for IC/BC targets ----------
def _truth_u(ptype, pparams):
    if ptype == "damped-wave-2d":
        k     = float(pparams.get("k", 0.5))
        omega = float(pparams.get("omega", 0.4))
        gamma = float(pparams.get("gamma", 0.3))
        x0    = float(pparams.get("x0", 0.0))
        y0    = float(pparams.get("y0", 0.0))
        def u(X,Y,T):
            r = torch.sqrt(torch.clamp((X-x0)**2 + (Y-y0)**2, min=0.0))
            return torch.cos(k*r - omega*T) * torch.exp(-gamma*T)
        return u

    if ptype == "poisson-2d":
        centers = pparams.get("centers", [(0.5,0.5)])
        sigma   = float(pparams.get("sigma", 0.12))
        def u(X,Y,T):
            mask = torch.sin(math.pi*X)*torch.sin(math.pi*Y)
            gsum = 0.0
            for (cx,cy) in centers:
                gsum = gsum + torch.exp(-((X-cx)**2 + (Y-cy)**2)/(2*sigma**2))
            return mask*gsum
        return u

    def zero(X,Y,T): return torch.zeros_like(X)
    return zero

# ---------- PDE residuals ----------
def _torch_DampedWave2d(expr_callable, Xb, Yb, Tb, c2, gamma, F_sym):
    with torch.enable_grad(): 
        Xb.requires_grad_(True); Yb.requires_grad_(True); Tb.requires_grad_(True)
        u   = _ensure_vec(expr_callable(Xb, Yb, Tb), Xb)
        ut  = _safe_grad(u.sum(), Tb, create_graph=True)
        utt = _safe_grad(ut.sum(), Tb, create_graph=True)
        ux  = _safe_grad(u.sum(), Xb, create_graph=True)
        uxx = _safe_grad(ux.sum(), Xb, create_graph=True)
        uy  = _safe_grad(u.sum(), Yb, create_graph=True)
        uyy = _safe_grad(uy.sum(), Yb, create_graph=True)
        F_t = _lambdify_to_torch_3d(F_sym, Xb, Yb, Tb) if F_sym is not None else torch.zeros_like(Xb)
        R = utt + 2.0*gamma*ut - c2*(uxx + uyy) - F_t
        return F.mse_loss(R, torch.zeros_like(R))

def _torch_Poisson2d(expr_callable, Xb, Yb, F_sym):
    with torch.enable_grad(): 
        Xb.requires_grad_(True); Yb.requires_grad_(True)
        T0 = torch.zeros_like(Xb)
        u   = _ensure_vec(expr_callable(Xb, Yb, T0), Xb)
        ux  = _safe_grad(u.sum(), Xb, create_graph=True)
        uxx = _safe_grad(ux.sum(), Xb, create_graph=True)
        uy  = _safe_grad(u.sum(), Yb, create_graph=True)
        uyy = _safe_grad(uy.sum(), Yb, create_graph=True)
        fxy = _lambdify_to_torch_2d(F_sym, Xb, Yb) if F_sym is not None else torch.zeros_like(Xb)
        R = (uxx + uyy) - fxy
        return F.mse_loss(R, torch.zeros_like(R))

# ---------- evaluator ----------
def make_evaluator_2d(ptype, pparams, F_str, w_data=1.0, w_pde=1.0, w_ic=1.0, w_bc=1.0, mask_poisson=True):
    F_sym = None
    try:
        if F_str is not None and str(F_str).strip() != "0":
            F_sym = sympy.sympify(F_str)
    except Exception:
        F_sym = None

    truth = _truth_u(ptype, pparams)
    xr = pparams.get("x_range", (0.0, 1.0))
    yr = pparams.get("y_range", (0.0, 1.0))
    tr = pparams.get("t_range", (0.0, 1.0))

    def deap_evaluator(individual, toolbox, X_train, y, d):
        expr = toolbox.compile(expr=individual)
        # QUICK SANITY CHECK - ADD THIS:
        try:
            test_vals = [expr(0.5, 0.5, 0.0), expr(0.25, 0.25, 0.1)]
            for v in test_vals:
                if isinstance(v, torch.Tensor):
                    v = v.item()
                if not (-1e4 < float(v) < 1e4):  # Quick bounds check
                    return (1e6,)
        except:
            return (1e6,)
        def expr_effective(X, Y, T):
            out = _ensure_vec(expr(X, Y, T), X)
            if ptype == "poisson-2d" and mask_poisson:
                out = out * torch.sin(math.pi*X)*torch.sin(math.pi*Y)  # enforce zero Dirichlet
            return out

        # Data MSE
        if y is not None and len(y) > 0:
            xs, ys, ts = X_train
            Xv = torch.tensor(xs, dtype=torch.float64)
            Yv = torch.tensor(ys, dtype=torch.float64)
            Tv = torch.tensor(ts, dtype=torch.float64)
            y_pred_t = expr_effective(Xv, Yv, Tv)
            y_pred = y_pred_t.detach().cpu().numpy()
            y_pred = np.clip(y_pred, -1e4, 1e4)
            mse_data = float(((y_pred - np.array(y))**2).mean())
        else:
            mse_data = 0.0
        # Collocation
        Xb = torch.linspace(float(xr[0]), float(xr[1]), 11, dtype=torch.float64)
        Yb = torch.linspace(float(yr[0]), float(yr[1]), 11, dtype=torch.float64)

        if ptype == "poisson-2d":
            Xg, Yg = torch.meshgrid(Xb, Yb, indexing="xy")
            Xcol = Xg.reshape(-1); Ycol = Yg.reshape(-1)
            pde = _torch_Poisson2d(expr_effective, Xcol, Ycol, F_sym)
            ic_mse = 0.0
            bc_mse = 0.0
        else:
            Tb = torch.linspace(float(tr[0]), float(tr[1]), 20, dtype=torch.float64)
            n = min(len(Xb), len(Yb), len(Tb))
            Xcol = Xb[:n].clone(); Ycol = Yb[:n].clone(); Tcol = Tb[:n].clone()
            if ptype == "damped-wave-2d":
                c2 = float(pparams.get("c2", 1.0))
                gamma = float(pparams.get("gamma", 0.3))
                pde = _torch_DampedWave2d(expr_effective, Xcol, Ycol, Tcol, c2, gamma, F_sym)
            else:
                return (1e12,)

            # IC (t=0)
            Xi, Yi = torch.meshgrid(
                torch.linspace(float(xr[0]), float(xr[1]), 11, dtype=torch.float64),
                torch.linspace(float(yr[0]), float(yr[1]), 11, dtype=torch.float64),
                indexing="xy"
            )
            Ti0 = torch.zeros_like(Xi)
            u_ic_pred = _ensure_vec(expr_effective(Xi, Yi, Ti0), Xi)
            with torch.no_grad():
                u_ic_true = truth(Xi, Yi, Ti0)
            ic_mse = F.mse_loss(u_ic_pred, u_ic_true).item()

            # BC (x walls)
            Tj = torch.linspace(float(tr[0]), float(tr[1]), 15, dtype=torch.float64)
            Yline = torch.linspace(float(yr[0]), float(yr[1]), 11, dtype=torch.float64)
            XL = torch.full_like(Tj, float(xr[0])); XR = torch.full_like(Tj, float(xr[1]))
            uL = _ensure_vec(expr_effective(XL, Yline, Tj), XL)
            uR = _ensure_vec(expr_effective(XR, Yline, Tj), XR)
            with torch.no_grad():
                uL_true = truth(XL, Yline, Tj)
                uR_true = truth(XR, Yline, Tj)
            bc_mse = 0.5*(F.mse_loss(uL, uL_true).item() + F.mse_loss(uR, uR_true).item())

        total = w_pde*float(pde.item()) + w_data*mse_data + w_ic*ic_mse + w_bc*bc_mse
        if not np.isfinite(total): total = 1e4
        
        import gc
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
        return (total,)
    return deap_evaluator

__all__ = ["make_evaluator_2d"]
