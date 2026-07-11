# Manufactured 1D datasets + meta for Diffusion, Wave, Burgers (with symbolic F)
import math, random, numpy as np
import sympy as sp

def dataset1D_from_u(u_callable, x_range, t_range, n_x=128, n_t=128, n_sample=None):
    xs = np.linspace(x_range[0], x_range[1], n_x)
    ts = np.linspace(t_range[0], t_range[1], n_t)
    if n_sample is None:
        X, T = np.meshgrid(xs, ts, indexing="xy")
        X_list = X.reshape(-1).tolist(); T_list = T.reshape(-1).tolist()
    else:
        X_list, T_list = [], []
        for _ in range(n_sample):
            X_list.append(random.uniform(*x_range)); T_list.append(random.uniform(*t_range))
    y_list = [float(u_callable(x, t)) for x, t in zip(X_list, T_list)]
    return [X_list, T_list], y_list

def dataset_diffusion_1d(A=3.974, D=0.697, L=1.397, n_x=128, n_t=128, n_sample=None, return_meta=False):
    # numeric u for data
    def u_num(x, t):
        return (
            A*math.sin(math.pi*x/L)*math.exp(-(1**2*math.pi**2*D*t)/(L**2))
          - A*math.sin(3*math.pi*x/L)*math.exp(-(3**2*math.pi**2*D*t)/(L**2))
          + A*math.sin(5*math.pi*x/L)*math.exp(-(5**2*math.pi**2*D*t)/(L**2))
        )
    X_train, y_train = dataset1D_from_u(u_num, (0.0, L), (0.0, 1.0), n_x, n_t, n_sample)

    # symbolic F: ut - D u_xx
    x, t = sp.symbols('x t', real=True)
    u_sym = (
        sp.Float(A)*sp.sin(sp.pi*x/sp.Float(L))*sp.exp(-(1**2*sp.pi**2*sp.Float(D)*t)/(sp.Float(L)**2))
      - sp.Float(A)*sp.sin(3*sp.pi*x/sp.Float(L))*sp.exp(-(3**2*sp.pi**2*sp.Float(D)*t)/(sp.Float(L)**2))
      + sp.Float(A)*sp.sin(5*sp.pi*x/sp.Float(L))*sp.exp(-(5**2*sp.pi**2*sp.Float(D)*t)/(sp.Float(L)**2))
    )
    F_sym = sp.simplify(sp.diff(u_sym, t) - sp.Float(D)*sp.diff(u_sym, x, 2))
    F_str = str(F_sym)

    pparams = {"A": A, "D": D, "L": L}
    if return_meta:
        return X_train, y_train, pparams, F_str, "diffusion-1d"
    return X_train, y_train

def dataset_wave_1d(coeffs=(1.4e-1, 4.6e-3, 2.3e-4, 1.1e-4), c=0.14, n_x=128, n_t=128, n_sample=None, return_meta=False):
    # numeric u for data
    def u_num(x, t):
        s = 0.0
        for n, a in enumerate(coeffs, start=1):
            s += a*math.sin(n*math.pi*x)*math.cos(c*n*math.pi*t)
        return (math.pi/16.0)*s
    X_train, y_train = dataset1D_from_u(u_num, (-5.0, 5.0), (0.0, 5.0), n_x, n_t, n_sample)

    # symbolic F: utt - c^2 u_xx
    x, t = sp.symbols('x t', real=True)
    u_sym = sp.Float(sp.pi)/sp.Integer(16) * sum(
        sp.Float(coeffs[n-1]) * sp.sin(n*sp.pi*x) * sp.cos(sp.Float(c)*n*sp.pi*t)
        for n in range(1, len(coeffs)+1)
    )
    F_sym = sp.simplify(sp.diff(u_sym, t, 2) - sp.Float(c)**2 * sp.diff(u_sym, x, 2))
    F_str = str(F_sym)

    pparams = {"coeffs": list(coeffs), "c2": c**2}
    if return_meta:
        return X_train, y_train, pparams, F_str, "wave-1d"
    return X_train, y_train

def dataset_burgers_1d(u_L=1.46, u_R=0.26, x0=0.33, nu=0.01, n_x=128, n_t=128, n_sample=None, return_meta=False):
    s = 0.5*(u_L + u_R)
    # numeric u for data
    def u_num(x, t):
        return 0.5*(u_L+u_R) - 0.5*(u_L-u_R)*math.tanh((x - x0 - s*t)*(u_L-u_R)/(4*nu))
    X_train, y_train = dataset1D_from_u(u_num, (-5.0, 5.0), (0.0, 2.0), n_x, n_t, n_sample)

    # symbolic F: ut + u ux - nu u_xx
    x, t = sp.symbols('x t', real=True)
    s_sym  = sp.Rational(1,2)*(sp.Float(u_L)+sp.Float(u_R))
    A_sym  = sp.Rational(1,2)*(sp.Float(u_L)-sp.Float(u_R))
    m_sym  = sp.Rational(1,2)*(sp.Float(u_L)+sp.Float(u_R))
    alpha  = (sp.Float(u_L)-sp.Float(u_R))/(4*sp.Float(nu))
    u_sym  = m_sym - A_sym*sp.tanh((x - sp.Float(x0) - s_sym*t)*alpha)
    F_sym  = sp.simplify(sp.diff(u_sym, t) + u_sym*sp.diff(u_sym, x) - sp.Float(nu)*sp.diff(u_sym, x, 2))
    F_str  = str(F_sym)

    pparams = {"nu": nu, "u_L": u_L, "u_R": u_R, "x0": x0}
    if return_meta:
        return X_train, y_train, pparams, F_str, "burgers-1d"
    return X_train, y_train
