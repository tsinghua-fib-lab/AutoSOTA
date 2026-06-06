# OLLA_NIPS/src/constraints/german_credit.py

import numpy as np
import pandas as pd
import torch

from sklearn.datasets import fetch_openml
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from torch.func import vmap, grad

# --- Default Experiment Settings ---
EXPERIMENT_SETTINGS = {
    'n_steps': 1000,
    'n_particles': 1,
    'is_single_chain': True,
}

# --- Sampler Hyperparameters ---
SAMPLER_SETTINGS = {
    'OLLA':      {'step_size': 5e-4, 'alpha': 100.0, 'epsilon': 1.0, 'proj_damp': 0.0},
    'OLLA-H':    {'step_size': 5e-4, 'alpha': 100.0, 'epsilon': 1.0, 'num_hutchinson_samples': 0, 'proj_damp': 0.0},
    'CLangevin': {'step_size': 5e-4, 'proj_iters': 10, 'proj_tol': 1e-3, 'proj_damp': 0.1},
    'CHMC':      {'step_size': 5e-3, 'gamma': 1.0, 'proj_iters': 10, 'proj_tol': 1e-3, 'proj_damp': 0.5},
    'CGHMC':     {'step_size': 5e-3, 'gamma': 1.0, 'proj_iters': 10, 'proj_tol': 1e-3, 'proj_damp': 0.5},
}


# --- Globals for the Current Problem ---
dim = None
h_fns = []
g_fns = []
potential_fn = None
prior_param = 1e-3

# --- Private module-level variables for data and model structure ---
_X_ext, _y, _A = None, None, None
_X_train, _X_test, _y_train, _y_test, _A_train, _A_test = None, None, None, None, None, None
_pos_g1, _pos_g0, _neg_g1, _neg_g0 = [], [], [], []
_S_PLUS_IDX, _S_MINUS_IDX = np.array([]), np.array([])
_input_dim, _H1, _H2 = None, 32, 16
_anchor_idx, _num_cols_count = None, 0
DELTA_MARGIN = 1e-0

# --- Neural Network Helpers ---

def _unpack(v: torch.Tensor):
    """Unpacks a flat parameter vector into NN matrices and vectors."""
    idx = 0
    s = v.shape[:-1]
    W1 = v[..., idx : idx + _H1*_input_dim].reshape(*s, _H1, _input_dim); idx += _H1*_input_dim
    b1 = v[..., idx : idx + _H1];                                         idx += _H1
    W2 = v[..., idx : idx + _H2*_H1].reshape(*s, _H2, _H1);               idx += _H2*_H1
    b2 = v[..., idx : idx + _H2];                                         idx += _H2
    w3 = v[..., idx : idx + _H2];                                         idx += _H2
    alpha = v[..., idx]; b0 = v[..., idx+1]
    return W1, b1, W2, b2, w3, alpha, b0

def _logits_torch(v: torch.Tensor, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Computes the NN output (logits) for parameters and data."""
    W1, b1, W2, b2, w3, alpha, b0 = _unpack(v)
    # Add a dimension for broadcasting the bias vectors correctly over the data points.
    h1 = torch.relu(torch.einsum('nD,...HD->...nH', X, W1) + b1.unsqueeze(-2))
    h2 = torch.relu(torch.einsum('...nH,...KH->...nK', h1, W2) + b2.unsqueeze(-2))
    return torch.einsum('...nK,...K->...n', h2, w3) + alpha.unsqueeze(-1) * A + b0.unsqueeze(-1)

# --- Potential Function (Negative Log-Posterior) ---

def nn_potential_fn(v: torch.Tensor) -> torch.Tensor:
    """Computes the negative log-posterior for the Bayesian NN."""
    W1, b1, W2, b2, w3, alpha, b0 = _unpack(v)
    prior = -0.5 * prior_param * ((W1*W1).sum(dim=(-1,-2)) + (b1*b1).sum(dim=-1) +
                    (W2*W2).sum(dim=(-1,-2)) + (b2*b2).sum(dim=-1) +
                    (w3*w3).sum(dim=-1) + (alpha*alpha) + (b0*b0))

    logits = _logits_torch(v, _X_ext, _A)
    loglik = (torch.nn.functional.logsigmoid(logits) * _y +
              torch.nn.functional.logsigmoid(-logits) * (1.0 - _y)).sum(dim=-1)
              
    return -(loglik + prior)

# --- Constraint Definitions ---

def _h_TPR(v: torch.Tensor) -> torch.Tensor:
    """Equality: True Positive Rate parity."""
    f = torch.sigmoid(_logits_torch(v, _X_ext, _A))
    tpr1 = f[..., _pos_g1].mean(dim=-1) if len(_pos_g1) > 0 else 0.0
    tpr0 = f[..., _pos_g0].mean(dim=-1) if len(_pos_g0) > 0 else 0.0
    return tpr1 - tpr0

def _h_FPR(v: torch.Tensor) -> torch.Tensor:
    """Equality: False Positive Rate parity."""
    f = torch.sigmoid(_logits_torch(v, _X_ext, _A))
    fpr1 = f[..., _neg_g1].mean(dim=-1) if len(_neg_g1) > 0 else 0.0
    fpr0 = f[..., _neg_g0].mean(dim=-1) if len(_neg_g0) > 0 else 0.0
    return fpr1 - fpr0

def _g_mono_grad(v: torch.Tensor) -> torch.Tensor:
    """Inequality: Monotonicity via gradients on anchor points."""
    def logit_single(v1, x, a): return _logits_torch(v1, x.unsqueeze(0), a.unsqueeze(0)).squeeze()
    grad_x = grad(logit_single, argnums=1)
    
    G = torch.stack([grad_x(v, _X_ext[i], _A[i]) for i in _anchor_idx])

    G_num = G[..., :_num_cols_count]
    
    vals = []
    if _S_PLUS_IDX.size > 0: vals.append((-G_num[..., _S_PLUS_IDX]).amax(dim=(-2,-1)) - DELTA_MARGIN)
    if _S_MINUS_IDX.size > 0: vals.append((G_num[..., _S_MINUS_IDX]).amax(dim=(-2,-1)) - DELTA_MARGIN)
    
    return torch.stack(vals, dim=-1).amax(dim=-1) if vals else torch.zeros(v.shape[:-1], device=v.device)


# --- Dynamic Problem Builder ---

def build_constraints(n_dim=1000, seed=2025):
    """Builds the German Credit problem with a dynamically sized NN."""
    global dim,h_fns,g_fns,potential_fn,_X_ext,_y,_A,_pos_g1,_pos_g0,_neg_g1,_neg_g0
    global _X_train,_X_test,_y_train,_y_test,_A_train,_A_test
    global _S_PLUS_IDX,_S_MINUS_IDX,_input_dim,_num_cols_count,_anchor_idx
    
    rng = np.random.default_rng(seed)
    
    df_raw = fetch_openml('credit-g', version=1, as_frame=True, parser='auto')
    df = pd.concat([df_raw.data, df_raw.target.rename('class')], axis=1)
    y_np = (df['class'].str.lower() == 'bad').astype(int).to_numpy()
    A_np = df['personal_status'].str.startswith('male').astype(int).to_numpy()
    num_cols = ['duration', 'credit_amount', 'existing_credits', 'age']
    X_num = df[num_cols].apply(lambda x: (x - x.mean()) / (x.std() or 1.0)).to_numpy()
    _num_cols_count = len(num_cols)

    # Solve approximate hash_dim from n
    fixed = _H1 + _H2*_H1 + _H2 + _H2 + 2
    raw_D = (int(n_dim) - fixed) / max(_H1, 1)
    base_D = _num_cols_count
    hash_dim = int(max(0, round(raw_D - base_D)))

    if hash_dim == 0:
        raise ValueError(f"Requested n_dim={n_dim} is too small for the fixed NN structure; hash_dim=0. Increase n_dim (recommend n_dim >=1000).")

    cat_cols = ['checking_status', 'savings_status', 'employment']
    tokens = df[cat_cols].astype(str).apply(lambda row: [f'{c}={row[c]}' for c in cat_cols], axis=1)
    hasher = FeatureHasher(n_features=hash_dim, input_type='string')
    X_cat = hasher.transform(tokens).toarray()
    X_ext_np = np.concatenate([X_num, X_cat], axis=1)
    _input_dim = X_ext_np.shape[1]
    dim = _H1*_input_dim + _H1 + _H2*_H1 + _H2 + _H2 + 2
    print("Total number of NN parameters (dim) close to n_dim:", dim)

    indices=np.arange(len(y_np))
    train_idx,test_idx=train_test_split(indices,test_size=0.2,random_state=seed,stratify=y_np)

    _X_ext=torch.tensor(X_ext_np,dtype=torch.float64)
    _y=torch.tensor(y_np,dtype=torch.float64)
    _A=torch.tensor(A_np,dtype=torch.float64)

    _X_train,_X_test=_X_ext[train_idx],_X_ext[test_idx]
    _y_train,_y_test=_y[train_idx],_y[test_idx]
    _A_train,_A_test=_A[train_idx],_A[test_idx]

    _pos_g1, _pos_g0 = np.where((A_np==1)&(y_np==1))[0], np.where((A_np==0)&(y_np==1))[0]
    _neg_g1, _neg_g0 = np.where((A_np==1)&(y_np==0))[0], np.where((A_np==0)&(y_np==0))[0]
    _S_PLUS_IDX = np.array([num_cols.index(n) for n in ['duration', 'credit_amount', 'existing_credits']])
    _S_MINUS_IDX = np.array([num_cols.index('age')])
    _anchor_idx = rng.choice(len(y_np), size=min(128, len(y_np)), replace=False) 

    h_fns = [_h_TPR, _h_FPR]
    g_fns = [_g_mono_grad]

    # h_fns =[]
    # g_fns =[]
        
    potential_fn = nn_potential_fn



# --- Initial Sample Generation ---
def _solve_alpha_bisection(s, res_fn):
    lo, hi = -50.0, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi);
        if res_fn(mid, s) >= 0.0: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)

def generate_samples(num_samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigmoid = lambda u: 1.0 / (1.0 + np.exp(-u))

    def base_scores(W1, b1, W2, b2, w3, b0):
        H1 = np.maximum(0.0, _X_ext.numpy() @ W1.T + b1)
        H2 = np.maximum(0.0, H1 @ W2.T + b2)
        return H2 @ w3 + b0

    samples = np.zeros((num_samples, dim))
    for i in range(num_samples):
        W1 = 0.02*rng.standard_normal((_H1,_input_dim)); b1=0.02*rng.standard_normal(_H1)
        W2 = 0.02*rng.standard_normal((_H2,_H1)); b2=0.02*rng.standard_normal(_H2)
        w3 = 0.02*rng.standard_normal(_H2)
        p_hat = _y.numpy().mean(); b0 = np.log(p_hat/(1-p_hat)) if 0<p_hat<1 else 0.0

        if _S_PLUS_IDX.size>0: W1[:,_S_PLUS_IDX] = np.abs(W1[:,_S_PLUS_IDX])+1e-3
        if _S_MINUS_IDX.size>0: W1[:,_S_MINUS_IDX] = -np.abs(W1[:,_S_MINUS_IDX])-1e-3
        W2[:], w3[:] = np.maximum(W2, 1e-3), np.maximum(w3, 1e-3)
        
        s = base_scores(W1, b1, W2, b2, w3, b0)
        alpha = 0.0
        if _h_TPR in h_fns:
            res_fn = lambda a, s_in: (sigmoid(s_in[_pos_g1]+a).mean()-sigmoid(s_in[_pos_g0]).mean())
            alpha = _solve_alpha_bisection(s, res_fn)
        # Newton solver for TPR+FPR is omitted for simplicity.
        
        v = np.concatenate([p.ravel() for p in [W1,b1,W2,b2,w3,np.array([alpha,b0])]])
        samples[i] = v
        
    return samples

