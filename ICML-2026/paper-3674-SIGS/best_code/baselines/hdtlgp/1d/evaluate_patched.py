# evaluate_patched.py
import numpy as np
import torch
import sympy
from torch.nn import functional as F
import math

# ---------- helpers ----------
def _ensure_vec(y, like: torch.Tensor):
    if isinstance(y, torch.Tensor):
        if y.ndim == 0:
            return torch.full_like(like, y.item())
        if y.shape == like.shape:
            return y.to(dtype=like.dtype, device=like.device)
        try:
            return (y + torch.zeros_like(like)).to(dtype=like.dtype, device=like.device)
        except Exception:
            return torch.full_like(like, float(y.detach().cpu().numpy()))
    else:
        return torch.full_like(like, float(y))

def _safe_grad(y, x, create_graph=True):
    if not (isinstance(y, torch.Tensor) and y.requires_grad):
        return torch.zeros_like(x)
    g = torch.autograd.grad(y, x, create_graph=create_graph, allow_unused=True)[0]
    if g is None:
        return torch.zeros_like(x)
    return g

def _lambdify_to_torch(sympy_expr, X, T=None):
    import numpy as _np, torch as _th
    if sympy_expr is None:
        return _th.zeros_like(X)
    if T is None:
        f_np = sympy.lambdify(["x"], sympy_expr, "numpy")
        vals = f_np(_np.asarray(X.detach().cpu().numpy()))
    else:
        f_np = sympy.lambdify(["x","t"], sympy_expr, "numpy")
        vals = f_np(_np.asarray(X.detach().cpu().numpy()),
                    _np.asarray(T.detach().cpu().numpy()))
    return _th.as_tensor(vals, dtype=X.dtype, device=X.device)

# ---------- truth u(x,t) used for IC/BC targets ----------
def _truth_u(ptype, pparams):
    pi = math.pi
    if ptype == "diffusion-1d":
        A = float(pparams.get("A", 3.974))
        D = float(pparams.get("D", 0.697))
        L = float(pparams.get("L", 1.397))
        lam1 = (pi*pi*D)/(L*L)
        lam3 = (9*pi*pi*D)/(L*L)
        lam5 = (25*pi*pi*D)/(L*L)
        def u(X,T):
            return (
                A*torch.sin(pi*X/L)*torch.exp(-lam1*T)
              - A*torch.sin(3*pi*X/L)*torch.exp(-lam3*T)
              + A*torch.sin(5*pi*X/L)*torch.exp(-lam5*T)
            )
        return u

    if ptype == "wave-1d":
        coeffs = pparams.get("coeffs", [1.4e-1, 4.6e-3, 2.3e-4, 1.1e-4])
        c2 = float(pparams.get("c2", 0.14**2))
        c  = math.sqrt(max(c2, 0.0))
        def u(X,T):
            s = torch.zeros_like(X, dtype=torch.float64)
            for n, a in enumerate(coeffs, start=1):
                s = s + a*torch.sin(n*pi*X)*torch.cos(c*n*pi*T)
            return (pi/16.0)*s
        return u

    if ptype == "burgers-1d":
        uL = float(pparams.get("u_L", 1.46))
        uR = float(pparams.get("u_R", 0.26))
        x0 = float(pparams.get("x0", 0.33))
        nu = float(pparams.get("nu", 0.01))
        s  = 0.5*(uL+uR)
        A  = 0.5*(uL-uR)
        m  = 0.5*(uL+uR)
        alpha = (uL-uR)/(4.0*nu)
        def u(X,T):
            return m - A*torch.tanh((X - x0 - s*T)*alpha)
        return u

    # fallback (zero)
    def zero_u(X,T): return torch.zeros_like(X)
    return zero_u

# ---------- PDE residuals ----------
def _torch_Diffusion1d(expr_callable, Xb, Tb, D, F_sym):
    Xb.requires_grad_(True); Tb.requires_grad_(True)
    u = _ensure_vec(expr_callable(Xb, Tb), Xb)
    ut  = _safe_grad(u.sum(), Tb, create_graph=True)
    ux  = _safe_grad(u.sum(), Xb, create_graph=True)
    uxx = _safe_grad(ux.sum(), Xb, create_graph=True)
    F_t = _lambdify_to_torch(F_sym, Xb, Tb) if F_sym is not None else torch.zeros_like(Xb)
    R = ut - D*uxx - F_t
    return F.mse_loss(R, torch.zeros_like(R))

def _torch_Wave1d(expr_callable, Xb, Tb, c2, F_sym):
    Xb.requires_grad_(True); Tb.requires_grad_(True)
    u = _ensure_vec(expr_callable(Xb, Tb), Xb)
    ut  = _safe_grad(u.sum(), Tb, create_graph=True)
    utt = _safe_grad(ut.sum(), Tb, create_graph=True)
    ux  = _safe_grad(u.sum(), Xb, create_graph=True)
    uxx = _safe_grad(ux.sum(), Xb, create_graph=True)
    F_t = _lambdify_to_torch(F_sym, Xb, Tb) if F_sym is not None else torch.zeros_like(Xb)
    R = utt - c2*uxx - F_t
    return F.mse_loss(R, torch.zeros_like(R))

def _torch_Burgers1d(expr_callable, Xb, Tb, nu, F_sym):
    Xb.requires_grad_(True); Tb.requires_grad_(True)
    u = _ensure_vec(expr_callable(Xb, Tb), Xb)
    ut  = _safe_grad(u.sum(), Tb, create_graph=True)
    ux  = _safe_grad(u.sum(), Xb, create_graph=True)
    uxx = _safe_grad(ux.sum(), Xb, create_graph=True)
    F_t = _lambdify_to_torch(F_sym, Xb, Tb) if F_sym is not None else torch.zeros_like(Xb)
    R = ut + u*ux - nu*uxx - F_t
    return F.mse_loss(R, torch.zeros_like(R))

# ---------- DEAP evaluator (paper-style loss) ----------
def make_evaluator_1d(ptype, pparams, F_str, w_data=1.0, w_pde=1.0, w_ic=1.0, w_bc=1.0):
    F_sym = None
    try:
        if F_str is not None and str(F_str).strip() != "0":
            F_sym = sympy.sympify(F_str)
    except Exception:
        F_sym = None

    truth = _truth_u(ptype, pparams)

    def deap_evaluator(individual, toolbox, X_train, y, d):
        expr = toolbox.compile(expr=individual)

        # ----- Data MSE
        if y is not None and len(y) > 0:
            xs, ts = X_train
            Xv = torch.tensor(xs, dtype=torch.float64)
            Tv = torch.tensor(ts, dtype=torch.float64)
            y_pred_t = _ensure_vec(expr(Xv, Tv), Xv)
            y_pred = y_pred_t.detach().cpu().numpy()
            y_pred = np.clip(y_pred, -1e6, 1e6)
            #print(f"DEBUG: y_pred range [{np.min(y_pred)}, {np.max(y_pred)}], y range [{np.min(y)}, {np.max(y)}]")
            mse_data = float(((y_pred - np.array(y))**2).mean())
            # if mse_data < 0.01:
            #     mse_data = 0.0
        else:
            mse_data = 0.0

        # ----- Collocation grids
        xs, ts = X_train
        xmin, xmax = float(np.min(xs)), float(np.max(xs))
        tmin, tmax = float(np.min(ts)), float(np.max(ts))

        x_col = torch.linspace(xmin, xmax, 51, dtype=torch.float64)
        t_col = torch.linspace(tmin, tmax, 51, dtype=torch.float64)
        n = min(len(x_col), len(t_col))
        Xb = x_col[:n].clone().requires_grad_(True)
        Tb = t_col[:n].clone().requires_grad_(True)

        def expr_callable(X, T):
            return _ensure_vec(expr(X, T), X)

        # ----- PDE residual
        if ptype == "diffusion-1d":
            D = float(pparams.get("D", 1.0))
            pde = _torch_Diffusion1d(expr_callable, Xb, Tb, D, F_sym)
        elif ptype == "wave-1d":
            c2 = float(pparams.get("c2", 0.14**2))
            pde = _torch_Wave1d(expr_callable, Xb, Tb, c2, F_sym)
        elif ptype == "burgers-1d":
            nu = float(pparams.get("nu", 0.01))
            pde = _torch_Burgers1d(expr_callable, Xb, Tb, nu, F_sym)
        else:
            return (1e12,)

        # ----- IC penalty: t=0
        Xi = torch.linspace(xmin, xmax, 201, dtype=torch.float64)
        Ti0 = torch.zeros_like(Xi)
        u_ic_pred = _ensure_vec(expr(Xi, Ti0), Xi)
        with torch.no_grad():
            u_ic_true = truth(Xi, Ti0)
        ic_mse = F.mse_loss(u_ic_pred, u_ic_true).item()

        # ----- BC penalty: x=xmin and x=xmax over t-grid
        Tj = torch.linspace(tmin, tmax, 201, dtype=torch.float64)
        XL = torch.full_like(Tj, xmin)
        XR = torch.full_like(Tj, xmax)
        u_bcL_pred = _ensure_vec(expr(XL, Tj), Tj)
        u_bcR_pred = _ensure_vec(expr(XR, Tj), Tj)
        with torch.no_grad():
            u_bcL_true = truth(XL, Tj)
            u_bcR_true = truth(XR, Tj)
        bc_mse = 0.5*(F.mse_loss(u_bcL_pred, u_bcL_true).item()
                      + F.mse_loss(u_bcR_pred, u_bcR_true).item())

        total = w_pde*float(pde.item()) + w_data*mse_data + w_ic*ic_mse + w_bc*bc_mse
        if not np.isfinite(total):
            total = 1e12
        print(f"[EVAL] data={mse_data:.3e}, pde={pde.item():.3e}, ic={ic_mse:.3e}, bc={bc_mse:.3e} => total={total:.3e}")    
        return (total,)
        
    return deap_evaluator

# re-export
__all__ = [
    "make_evaluator_1d",
    "_torch_Diffusion1d",
    "_torch_Wave1d",
    "_torch_Burgers1d",
]
