"""Utilities for training monitoring, selection heuristics, serialization, and constraints."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from typing import Any, Dict, Tuple

import numpy as np
import torch


# ============================================================
# Optional NaN detection hooks
# ============================================================

def forward_hook(module: torch.nn.Module, input, output):
    """Callback that raises if the module emits NaNs in weights/values."""
    if isinstance(output, tuple) and len(output) == 2:
        weights, values = output
        if torch.isnan(weights).any():
            raise RuntimeError(f"[NaN] in weights of {module.__class__.__name__}")
        if torch.isnan(values).any():
            raise RuntimeError(f"[NaN] in values of {module.__class__.__name__}")


def grad_hook(grad):
    """Gradient hook that validates gradients remain finite."""
    if torch.isnan(grad).any():
        raise RuntimeError("[NaN] in gradient")
    return grad


def attach_nan_hooks(model: torch.nn.Module):
    """Attach forward/backward NaN detectors to a model's leaf modules."""
    for _, module in model.named_modules():
        if len(list(module.children())) == 0:
            module.register_forward_hook(forward_hook)
    for _, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(grad_hook)

def moded_max(
    scores: torch.Tensor,            # (S, B)
    candidates: torch.Tensor,        # (1, B, d)
    dim: int = 1,
    temp: float = 5.0,
    max_effective_temp: float = 5000.0,
    mode: str = "soft",              # "soft" | "hard" | "ste"
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Compute maximum over candidates using soft, hard, or straight-through estimator (STE) modes.
    
    This function provides three modes for selecting from a set of candidates:
    - **soft**: Uses temperature-scaled softmax to compute a weighted combination.
      Provides smooth gradients and is fully differentiable.
    - **hard**: Selects the argmax (single best candidate). Forward pass is discrete,
      no gradients through the selection.
    - **ste**: Straight-through estimator combining hard forward pass with soft backward pass.
      Uses hard selection in forward but gradients from soft selection in backward.
    
    The effective temperature is adaptively scaled based on the span (max - min) of scores
    to ensure numerical stability and meaningful soft selections even when scores have
    varying ranges.
    
    Args:
        scores: Tensor of shape (#samples, #candidates) containing scores for each candidate.
        candidates: Tensor of shape (1, #candidates, #dim) containing the candidate points.
        dim: Dimension to maximize over (must be 1 in current implementation).
        temp: Base softmax temperature. Lower values make selection sharper (closer to hard).
        max_effective_temp: Maximum allowed effective temperature to prevent numerical issues.
        mode: Selection mode - 'soft', 'hard', or 'ste'.
    
    Returns:
        choice: Tensor of shape (#samples, #dim) - selected point(s).
          In soft mode, this is a weighted combination of candidates.
          In hard mode, this is the candidate with maximum score.
          In ste mode, forward is hard but gradients flow from soft.
        v: Tensor of shape (#samples,) - selected values.
          In soft mode, this is the weighted average of scores.
          In hard/ste mode, this is the maximum score value.
        aux: Dictionary containing diagnostic information:
            - 'weights': (#samples, #candidates) - softmax weights (soft/ste only, else None)
            - 'idx': (#samples,) - argmax indices (hard/ste only, else None)  
            - 'mean_max': float - mean of max weight per sample (1.0 for hard, <1.0 for soft)
            - 'eff_temp': (#samples, 1) - effective temperature used (soft/ste only, else None)
    
    Raises:
        AssertionError: If dim != 1 or candidates shape is not (1, #candidates, #dim).
        ValueError: If mode is not one of 'soft', 'hard', 'ste'.
    
    Examples:
        >>> scores = torch.tensor([[1.0, 5.0, 3.0]])
        >>> candidates = torch.tensor([[[0.0], [1.0], [2.0]]])
        >>> choice, v, aux = moded_max(scores, candidates, mode='hard')
        >>> print(aux['idx'])  # Should be [1] (argmax)
        >>> choice, v, aux = moded_max(scores, candidates, mode='soft', temp=1.0)
        >>> print(aux['weights'].sum())  # Should be 1.0
    """
    assert dim == 1, "This implementation assumes choices are on dim=1."
    assert candidates.dim() == 3 and candidates.size(0) == 1, \
        "Expected candidates with shape (1, #candidates, #dim)."

    candidates_squeezed = candidates.squeeze(0)  # (B, d)
    aux: Dict[str, Any] = {"weights": None, "idx": None,
                           "mean_max": float("nan"), "eff_temp": None}

    if mode == "soft":
        span = (scores.amax(dim=dim, keepdim=True) -
                scores.amin(dim=dim, keepdim=True)).clamp_min(1e-3)  # (S,1)
        eff_temp = (temp / span).clamp(max=max_effective_temp, min=temp)  # (S,1)

        z = scores * eff_temp
        z = z - z.amax(dim=dim, keepdim=True)
        w = torch.softmax(z, dim=dim)                     # (S,B)
        v = (w * scores).sum(dim=dim)                     # (S,)
        choice = w @ candidates_squeezed                  # (S,d)

        aux["weights"]  = w
        aux["mean_max"] = float(w.max(dim=dim).values.mean().item())
        aux["eff_temp"] = eff_temp
        return choice, v, aux

    elif mode == "hard":
        v, idx = scores.max(dim=dim)                      # (S,), (S,)
        choice = candidates_squeezed[idx]                 # (S,d)
        aux["idx"] = idx
        aux["mean_max"] = 1.0
        return choice, v, aux

    elif mode == "ste":
        choice_soft, v_soft, aux_soft = moded_max(
            scores, candidates, dim=dim,
            temp=temp, max_effective_temp=max_effective_temp,
            mode="soft"
        )
        choice_hard, v_hard, aux_hard = moded_max(
            scores, candidates, dim=dim,
            temp=temp, max_effective_temp=max_effective_temp,
            mode="hard"
        )

        # Straight-through: forward=hard, backward=soft
        choice = choice_soft + (choice_hard - choice_soft).detach()
        v      = v_soft      + (v_hard      - v_soft).detach()

        aux["weights"]  = aux_soft["weights"]
        aux["eff_temp"] = aux_soft["eff_temp"]
        aux["idx"]      = aux_hard["idx"]
        aux["mean_max"] = aux_soft["mean_max"]
        return choice, v, aux

    else:
        raise ValueError(f"mode must be 'soft', 'hard', or 'ste', got {mode!r}")

def moded_min(
    scores: torch.Tensor,            # (S, B)
    candidates: torch.Tensor,        # (1, B, d)
    dim: int = 1,
    temp: float = 5.0,
    max_effective_temp: float = 5000.0,
    mode: str = "soft",              # "soft" | "hard" | "ste"
):
    """
    Compute minimum over candidates using soft, hard, or straight-through estimator (STE) modes.
    
    This function mirrors moded_max but computes minimization instead of maximization.
    It is implemented by negating scores and calling moded_max, ensuring numerical
    stability and consistency.
    
    The three modes work as follows:
    - **soft**: Uses temperature-scaled softmin (softmax on negated scores) to compute
      a weighted combination favoring low-scoring candidates.
    - **hard**: Selects the argmin (candidate with minimum score). Discrete selection.
    - **ste**: Straight-through estimator with hard forward pass (argmin) and soft
      backward pass (softmin gradients).
    
    Args:
        scores: Tensor of shape (#samples, #candidates) containing scores for each candidate.
        candidates: Tensor of shape (1, #candidates, #dim) containing the candidate points.
        dim: Dimension to minimize over (must be 1 in current implementation).
        temp: Base softmax temperature. Lower values make selection sharper.
        max_effective_temp: Maximum allowed effective temperature.
        mode: Selection mode - 'soft', 'hard', or 'ste'.
    
    Returns:
        choice: Tensor of shape (#samples, #dim) - selected point(s).
          In soft mode, weighted combination with higher weights on low scores.
          In hard mode, the candidate with minimum score.
        v: Tensor of shape (#samples,) - selected values.
          In soft mode, weighted average of scores (softmin).
          In hard mode, the minimum score value.
        aux: Dictionary with same structure as moded_max:
            - 'weights': softmin weights (interpret as probability over candidates)
            - 'idx': argmin indices (which candidate has minimum score)
            - 'mean_max': mean of max weight (note: still called 'mean_max' for consistency)
            - 'eff_temp': effective temperature used
    
    Implementation Note:
        This function achieves numerical stability by converting the minimization to
        maximization: min(scores) = -max(-scores). All computations are delegated
        to moded_max, ensuring consistent behavior and numerical properties.
    
    Examples:
        >>> scores = torch.tensor([[5.0, 1.0, 3.0]])
        >>> candidates = torch.tensor([[[0.0], [1.0], [2.0]]])
        >>> choice, v, aux = moded_min(scores, candidates, mode='hard')
        >>> print(aux['idx'])  # Should be [1] (argmin)
        >>> print(v)  # Should be [1.0] (minimum score)
    """

    # Convert min → max on negated scores
    neg_scores = -scores

    # Call moded_max on the negated scores
    choice, neg_vals, aux = moded_max(
        neg_scores,
        candidates,
        dim=dim,
        temp=temp,
        max_effective_temp=max_effective_temp,
        mode=mode,
    )

    # Convert values back:
    #   neg_vals = max_j (-scores)
    #            = -min_j scores
    vals = -neg_vals

    # (aux["idx"] is already argmin, because it is argmax(-scores))
    # (aux["weights"] are softmin weights over original scores)

    return choice, vals, aux

def _to_serializable(obj):
    """
    Recursively convert Python objects to JSON-serializable types.
    
    This function handles conversion of various Python types (including PyTorch tensors,
    NumPy arrays, and other objects) into basic JSON-serializable types (int, float, str,
    list, dict, bool, None).
    
    Conversion rules:
    - **Scalar torch.Tensor**: Converted to Python number via .item()
    - **Non-scalar torch.Tensor**: Converted to nested list via .tolist()
    - **numpy.ndarray**: Converted to nested list via .tolist()
    - **numpy scalar types**: Converted to Python number via .item()
    - **dict**: Recursively converted (preserves keys, converts values)
    - **list/tuple**: Recursively converted to list
    - **callable**: Converted to string representation (module.class.name)
    - **torch.device**: Converted to string (e.g., "cpu", "cuda:0")
    - **torch.optim.Optimizer**: Converted to string "OPTIMIZER:<class_name>"
    - **Primitives** (int, float, str, bool, None): Pass through unchanged
    
    This function is particularly useful for:
    - Saving model configurations to JSON files
    - Creating hashable representations of complex objects (see hash_dict)
    - Logging and serialization of experiment parameters
    
    Args:
        obj: Any Python object to convert.
    
    Returns:
        A JSON-serializable version of the object (int, float, str, list, dict, bool, or None).
    
    Examples:
        >>> _to_serializable(torch.tensor(3.14))
        3.14
        >>> _to_serializable(torch.tensor([1, 2, 3]))
        [1, 2, 3]
        >>> _to_serializable({"a": torch.tensor(1.0), "b": np.array([2, 3])})
        {'a': 1.0, 'b': [2, 3]}
        >>> _to_serializable(torch.device("cuda:0"))
        'cuda:0'
    
    Note:
        - Torch tensors are automatically detached and moved to CPU before conversion.
        - Tuples are converted to lists (JSON doesn't distinguish between them).
        - Gradient information is not preserved.
    """

    # Torch tensor
    if isinstance(obj, torch.Tensor):
        # Scalar tensor → return Python number
        if obj.dim() == 0:
            return obj.item()
        # Non-scalar → list
        return obj.detach().cpu().tolist()

    # Numpy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Numpy scalar
    if isinstance(obj, (np.generic,)):
        return obj.item()

    # Dict → recursive
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}

    # List / Tuple → recursive
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    
    import types

    if callable(obj):
        if isinstance(obj, types.FunctionType):
            # Distinguish L22, L33, inverse_L22x, etc.
            return obj.__module__ + "." + obj.__qualname__
        # Fallback for callable classes / instances
        module = getattr(obj, "__module__", obj.__class__.__module__)
        qualname = getattr(obj, "__qualname__", obj.__class__.__qualname__)
        return module + "." + qualname

    
    if isinstance(obj, torch.device):
        return str(obj)    # "cuda:0" or "cpu"
    
    if isinstance(obj, torch.optim.Optimizer):
        return f"OPTIMIZER:{obj.__class__.__name__}"


    # Basic Python primitive
    return obj

def hash_dict(d):
    """
    Compute a deterministic SHA-256 hash of a dictionary.
    
    This function creates a canonical, order-independent hash of a dictionary by:
    1. Converting all values to JSON-serializable types via _to_serializable()
    2. Encoding to canonical JSON with sorted keys
    3. Computing SHA-256 hash of the resulting string
    
    The hash is deterministic and independent of:
    - Key insertion order in the dictionary
    - Torch tensor device (CPU/CUDA)
    - Numpy vs Torch representation (equivalent arrays produce same hash)
    
    This is useful for:
    - Cache keys based on configuration dictionaries
    - Detecting changes in experiment parameters
    - Creating unique identifiers for parameter combinations
    - Checkpointing and reproducibility
    
    Args:
        d: Dictionary to hash. Can contain nested structures with torch tensors,
           numpy arrays, and other types supported by _to_serializable().
    
    Returns:
        str: Hexadecimal SHA-256 hash digest (64 characters).
    
    Examples:
        >>> d1 = {"a": 1, "b": 2, "c": 3}
        >>> d2 = {"c": 3, "a": 1, "b": 2}  # Different order
        >>> hash_dict(d1) == hash_dict(d2)  # Same hash
        True
        >>> d3 = {"tensor": torch.tensor([1.0, 2.0])}
        >>> d4 = {"tensor": torch.tensor([1.0, 2.0])}
        >>> hash_dict(d3) == hash_dict(d4)  # Same tensors
        True
        >>> len(hash_dict(d1))  # SHA-256 produces 64 hex characters
        64
    
    Note:
        - Floating point precision matters: torch.tensor(1.0) and torch.tensor(1.00001)
          will produce different hashes.
        - The hash is stable across Python sessions but may change if the underlying
          serialization format changes in future versions.
    """
    serializable = _to_serializable(d)

    # canonical JSON encoding: sorted keys + no whitespace
    json_str = json.dumps(serializable, sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def is_in_jupyter_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except Exception:
        return False

IN_JUPYTER = is_in_jupyter_notebook()

def linear_kernel(X,Y, dim=1):
    """Compute dot-product kernel along specified dimension."""
    return ((X * Y)).sum(dim=dim)

def quadratic_cost(choice, dim=1):
    """Return half the squared-norm (quadratic) cost for choices."""
    return (choice**2).sum(dim=dim)/2

def loader(paths=None, root=None):
    """
    Load saved mechanism artifacts (samples and finals) for inspection.

    Parameters
    ----------
    paths : Sequence[str] | None
        Specific files to load. If None, the function scans `root` for artifacts.
    root : str | None
        Directory to scan. Defaults to `config.WRITING_ROOT` when unspecified.

    Returns
    -------
    Dict[str, Tuple[torch.Tensor, list]]
        Maps each directory to a tuple with the sample tensor and a list of final models.
    """
    names = {}
    if paths is None:
        if root is None:
            from config import WRITING_ROOT as root
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root)
                # parts = rel.split(os.sep)
                key = dirpath#parts[0]
                if fn.endswith("sample.pt"):
                    key = dirpath
                    names.setdefault(key, [None, []])
                    names[key][0] = full
                if fn.endswith('final.pt'):
                    key = os.path.dirname(dirpath)
                    names.setdefault(key, [None, []])
                    names[key][1].append(full)
    
    optim_mod = importlib.import_module("optim")
    for k, v in optim_mod.__dict__.items():
        if not k.startswith("_"):
            setattr(sys.modules["__main__"], k, v)
    
    results = {}
    for key, (sample_name, models_name) in names.items():
        if sample_name is None or len(models_name) == 0:
            continue
        sample = torch.load(sample_name, map_location="cpu", weights_only=False)["all_sample"]
        models = [torch.load(f, map_location="cpu", weights_only=False) for f in models_name]
        results[key] = (sample, models)
    return results


def generate_dir_and_name(prefix, *args, **kwargs):
    """
    Create a structured directory path by combining prefix with args/kwargs metadata.

    Args:
        prefix: Base directory name prefix.
        *args: Positional metadata (classes/functions or strings).
        **kwargs: Named metadata that will appear as `key:value` segments.
    """
    dir = prefix
    for arg in args:
        if isinstance(arg, str):
            dir += f"{arg},"
        else:
            dir += f"{arg.__name__}_"
    for k, v in sorted(kwargs.items()):
        if callable(v):
            name = getattr(v, "__name__", type(v).__name__)
            dir += f"{k}:{name},"
        else:
            dir += f"{k}:{v},"
    dir = dir + "/"
    os.makedirs(dir, exist_ok=True)
    return dir

def IR_constraint(mechanism, mechanism_data, kappa = 100):
    """Individual rationality constraint: returns indirect utilities `v`."""
    v = mechanism_data["v"]
    return v
   
def model_min_constraint(mechanism, mechanism_data, kappa=100):
    """Enforce lower bounds on remaining `Y` intercepts."""
    if mechanism.y_min is None:
        return torch.zeros_like(mechanism.Y_rest)
    return mechanism.Y_rest - mechanism.y_min
   
def model_max_constraint(mechanism, mechanism_data, kappa=100):
    """Enforce upper bounds on remaining `Y` intercepts."""
    if mechanism.y_max is None:
        return torch.zeros_like(mechanism.Y_rest)
    return mechanism.y_max - mechanism.Y_rest

@torch.no_grad()
def greedy_neg_order(R: torch.Tensor, k: int | None = None):
    """
    Heuristic ordering that brings the most negative correlations to the top-left.

    Parameters
    ----------
    R : torch.Tensor
        Square matrix (n, n) of pairwise scores/correlations.
    k : int | None
        Size of the leading block to optimize (defaults to max(n//5,2)).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Permuted matrix and the permutation vector.
    """
    n = R.size(0)
    if k is None: k = max(n // 5, 2)

    Rz = R.clone(); Rz.fill_diagonal_(0.0)

    # start from the most negative pair
    tri = torch.triu_indices(n, n, 1)
    idx = torch.argmin(R[tri[0], tri[1]])
    sel = [tri[0, idx].item(), tri[1, idx].item()]
    rem = set(range(n)) - set(sel)

    # greedily add the item with most negative avg corr to selected
    while rem:
        cand = torch.tensor(list(rem), device=R.device)
        j = cand[torch.argmin(Rz[cand][:, sel].mean(dim=1))].item()
        sel.append(j); rem.remove(j)

    perm = torch.tensor(sel, device=R.device)
    perm_rev = torch.flip(perm, dims=[0])

    # flip if it makes the first k block more negative (avg off-diag)
    def block_mean(p):
        if k < 2: return torch.tensor(0.0, device=R.device)
        idx = p[:k]
        B = Rz.index_select(0, idx).index_select(1, idx)
        return (B.sum() / 2) / (k * (k - 1) / 2)

    if block_mean(perm) > block_mean(perm_rev):
        perm = perm_rev

    R_perm = R.index_select(0, perm).index_select(1, perm)
    return R_perm, perm

def Lpp(X, Y, p=2):
    """Compute the L_p^p cost (sum over dimensions) between X and Y."""
    diff = (X - Y).abs()
    return (diff ** p).sum(dim=-1) / p


def inverse_Lppx(X, Z, p=2):
    """Given grad w.r.t. x of L_p^p(X, Y), return the associated Y."""
    assert p > 1
    d = torch.sign(Z) * Z.abs()**(1.0 / (p - 1))
    return X - d

def nLpp(X, Y, p=2):
    """Negative L_p^p cost (useful for concave counterparts)."""
    return -Lpp(X, Y, p=p)

def inverse_nLppx(X, Z, p=2):
    """Inverse gradient of the negative L_p^p cost."""
    assert p > 1
    d = torch.sign(Z) * Z.abs()**(1/(p - 1))
    return X + d

def Lpp_1d(x, y, p=2):
    """1D version of L_p^p between scalars x and y."""
    diff = (x - y).abs()
    return diff ** p / p

def L22_1d(x, y):
    return Lpp_1d(x, y, p=2)

def L33_1d(x, y):
    return Lpp_1d(x, y, p=3)

def L44_1d(x, y):
    return Lpp_1d(x, y, p=4)

def L55_1d(x, y):
    return Lpp_1d(x, y, p=5)


def L22(X, Y):
    """Convenience wrapper for squared Euclidean cost L_2^2."""
    return Lpp(X, Y, p=2)

def L33(X, Y):
    """L_3^3 cost."""
    return Lpp(X, Y, p=3)

def L44(X, Y):
    """L_4^4 cost."""
    return Lpp(X, Y, p=4)

def L55(X, Y):
    """L_5^5 cost."""
    return Lpp(X, Y, p=5)

def inverse_L22x(X, Z):
    """Inverse gradient of L_2^2."""
    return inverse_Lppx(X, Z, p=2)

def inverse_L33x(X, Z):
    """Inverse gradient of L_3^3."""
    return inverse_Lppx(X, Z, p=3)

def inverse_L44x(X, Z):
    """Inverse gradient of L_4^4."""
    return inverse_Lppx(X, Z, p=4)

def inverse_L55x(X, Z):
    """Inverse gradient of L_5^5."""
    return inverse_Lppx(X, Z, p=5)

def project_to_box(x, R):
    """Project tensor `x` componentwise onto the [-R, R] box."""
    return x.clamp(-R, R)


def nL22_1d(x, y):
    """Negative squared distance in 1D."""
    return -Lpp_1d(x, y, p=2)

def inverse_nL22x(X, Z):
    """Inverse gradient for negative L_2^2 in 1D."""
    return inverse_nLppx(X, Z, p=2)
