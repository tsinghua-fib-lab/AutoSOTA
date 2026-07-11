# Dataset2D_patched.py
# Manufactured 2-D datasets for: damped wave (time-dependent) and Poisson–Gauss
# (elliptic) with 2/3/4 centers. Includes a 1-center helper if you want it.
import math, random, numpy as np
import sympy as sp

def dataset_damped_wave_2d(k=0.5, omega=0.4, c=None, gamma=0.3,
                           x0=0.0, y0=0.0,
                           x_pad=3.0, y_pad=3.0,
                           t_range=(0.0, 4.0), n_x=48, n_y=48, n_t=48, sample=None, return_meta=False):
    """
    u(x,y,t) = cos(k*sqrt((x-x0)^2+(y-y0)^2) - omega t) * exp(-gamma t)
    PDE: utt + 2*gamma*ut - c^2 (uxx + uyy) = F(x,y,t)
    """
    if c is None: c = omega / max(k, 1e-12)
    xr = (x0 - x_pad, x0 + x_pad)
    yr = (y0 - y_pad, y0 + y_pad)
    ts = np.linspace(t_range[0], t_range[1], n_t)

    def u_num(x, y, t):
        r = math.hypot(x - x0, y - y0)
        return math.cos(k*r - omega*t) * math.exp(-gamma*t)

    X_list, Y_list, T_list, y_list = [], [], [], []
    if sample is None:
        for x in np.linspace(*xr, n_x):
            for y in np.linspace(*yr, n_y):
                for t in ts:
                    X_list.append(float(x)); Y_list.append(float(y)); T_list.append(float(t))
                    y_list.append(float(u_num(x,y,t)))
    else:
        for _ in range(sample):
            x = random.uniform(*xr); y = random.uniform(*yr); t = random.uniform(*t_range)
            X_list += [x]; Y_list += [y]; T_list += [t]; y_list += [u_num(x,y,t)]

    # symbolic F
    x, y, t = sp.symbols('x y t', real=True)
    r = sp.sqrt((x - sp.Float(x0))**2 + (y - sp.Float(y0))**2)
    u = sp.cos(sp.Float(k)*r - sp.Float(omega)*t) * sp.exp(-sp.Float(gamma)*t)
    u_t  = sp.diff(u, t)
    F_sym = sp.simplify(sp.diff(u, t, 2) + 2*sp.Float(gamma)*u_t - sp.Float(c)**2 * (sp.diff(u, x, 2) + sp.diff(u, y, 2)))
    F_str = str(F_sym)

    pparams = {"k":k, "omega":omega, "c2": c**2, "gamma":gamma, "x0":x0, "y0":y0,
               "x_range":xr, "y_range":yr, "t_range":t_range}
    X_train = [X_list, Y_list, T_list]
    if return_meta:
        return X_train, y_list, pparams, F_str, "damped-wave-2d"
    return X_train, y_list

# ----- Poisson–Gauss builders -----
def _poisson_u_sym(centers, sigma):
    x, y = sp.symbols('x y', real=True)
    mask = sp.sin(sp.pi*x)*sp.sin(sp.pi*y)   # homogeneous Dirichlet
    gsum = sum(sp.exp(-((x-sp.Float(cx))**2+(y-sp.Float(cy))**2)/(2*sp.Float(sigma)**2)) for (cx,cy) in centers)
    u = mask * gsum
    return u

def _poisson_make_dataset(centers, sigma=0.12, n_x=100, n_y=100, sample=None, return_meta=False):
    xr, yr = (0.0, 1.0), (0.0, 1.0)
    def u_num(x, y):
        mask = math.sin(math.pi*x)*math.sin(math.pi*y)
        gsum = sum(math.exp(-((x-cx)**2+(y-cy)**2)/(2*sigma**2)) for (cx,cy) in centers)
        return mask * gsum

    X_list, Y_list, T_list, y_list = [], [], [], []
    if sample is None:
        for x in np.linspace(*xr, n_x):
            for y in np.linspace(*yr, n_y):
                X_list.append(float(x)); Y_list.append(float(y)); T_list.append(0.0)
                y_list.append(float(u_num(x,y)))
    else:
        for _ in range(sample):
            x = random.uniform(*xr); y = random.uniform(*yr)
            X_list += [x]; Y_list += [y]; T_list += [0.0]; y_list += [u_num(x,y)]

    u_sym = _poisson_u_sym(centers, sigma)
    F_sym = sp.simplify(-(sp.diff(u_sym, 'x', 2) + sp.diff(u_sym, 'y', 2)))
    F_str = str(F_sym)

    pparams = {"centers": centers, "sigma": sigma, "x_range": xr, "y_range": yr, "t_range": (0.0,0.0)}
    X_train = [X_list, Y_list, T_list]
    if return_meta:
        return X_train, y_list, pparams, F_str, "poisson-2d"
    return X_train, y_list

# one-center helper (not in the 4-problem list, but handy)
def dataset_poisson_gauss_2d_onecenter(cx=0.5, cy=0.5, sigma=0.12, **kw):
    return _poisson_make_dataset([(cx,cy)], sigma=sigma, **kw)

def dataset_poisson_gauss_2d_2centers(sigma=0.12, centers=None, **kw):
    if centers is None:
        centers = [(0.3,0.8), (0.7,0.2)]
    return _poisson_make_dataset(centers, sigma=sigma, **kw)

def dataset_poisson_gauss_2d_3centers(sigma=0.12, centers=None, **kw):
    if centers is None:
        centers = [(0.3,0.8), (0.7,0.8), (0.5,0.2)]
    return _poisson_make_dataset(centers, sigma=sigma, **kw)

def dataset_poisson_gauss_2d_4centers(sigma=0.12, centers=None, **kw):
    if centers is None:
        centers = [(0.3,0.8), (0.7,0.2), (0.5,0.2), (0.4,0.6)]
    return _poisson_make_dataset(centers, sigma=sigma, **kw)
