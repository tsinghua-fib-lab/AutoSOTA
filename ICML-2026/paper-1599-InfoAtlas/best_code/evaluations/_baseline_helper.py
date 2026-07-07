"""
Shared baseline MI estimation helper for all evaluation scripts.

Provides a uniform calling convention for the third-party estimators in
``baselines/`` (see ``baselines/NOTICE.md``).
"""
import sys
import os

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Path setup: make baseline internal imports work
#   - `import optimizer`          -> finds baselines/optimizer.py
#   - `import estimators.X`      -> finds baselines/X (via sys.modules alias)
#   - `from nde.X import ...`    -> finds baselines/nde/X.py
# ---------------------------------------------------------------------------
_project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_baselines_dir = os.path.join(_project_root, "baselines")

# Put our project root at the FRONT of sys.path so our local baselines/
# package is found before any system-installed 'baselines' (e.g. OpenAI).
sys.path.insert(0, _baselines_dir)
sys.path.insert(0, _project_root)

# Clear any previously cached wrong baselines module
sys.modules.pop("baselines", None)

import baselines  # noqa: E402
sys.modules.setdefault("estimators", baselines)

# ---------------------------------------------------------------------------

AVAILABLE_BASELINES = baselines.AVAILABLE_BASELINES

# ---------------------------------------------------------------------------
# InfoNet V1 model management (pre-trained baseline, needs explicit init)
# ---------------------------------------------------------------------------
_infonet_v1_model = None
_infonet_v1_device = None


def init_infonet_v1(config_path, ckpt_path, device="cuda:0"):
    """Load InfoNet V1 model for use as a baseline. Must be called before
    using ``estimate_mi_baseline("InfoNet", ...)``."""
    global _infonet_v1_model, _infonet_v1_device
    from baselines.InfoNet_V1.infer import load_model
    _infonet_v1_model = load_model(config_path, ckpt_path, device=device)
    _infonet_v1_device = torch.device(device)
    print(f"[InfoNet V1] Model loaded from {ckpt_path}")


class _Hyperparams:
    """Simple hyperparams container, same pattern as in 2dtrack_final.py."""
    def __init__(self):
        self.critic = "neural"
        self.lr = 5e-4
        self.bs = 500
        self.n_bridges = 4
        self.n_neg = 4
        self.wd = 1e-5
        self.encode_x = False
        self.encode_y = False
        self.max_iteration = 1500


def estimate_mi_baseline(method_name, X, Y, device="cuda:0"):
    """
    Estimate MI for a single (X, Y) pair using a baseline method.

    Args:
        method_name: one of AVAILABLE_BASELINES
        X: [N, dx] tensor or numpy array
        Y: [N, dy] tensor or numpy array
        device: torch device string

    Returns:
        float: estimated MI value
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y).float()

    X = X.float().to(device)
    Y = Y.float().to(device)

    n, dx = X.shape
    _, dy = Y.shape
    d = dx + dy
    architecture_critic = [d, 500, 500, 500, 1]

    if method_name == "KSG":
        from baselines.KSG import KSG
        ksg_estimator = KSG(k_neighbors=5, tree_type="kd_tree", tree_kwargs={})
        mi_est = ksg_estimator(X.cpu().numpy(), Y.cpu().numpy(), std=False)
        return float(mi_est)

    elif method_name == "MINE":
        from baselines.MINE import MINE
        hyperparams = _Hyperparams()
        estimator = MINE(None, None, architecture_critic, hyperparams).to(device)
        estimator.learn(X, Y)
        return estimator.MI(X, Y)

    elif method_name == "InfoNCE":
        from baselines.InfoNCE import InfoNCE
        hyperparams = _Hyperparams()
        estimator = InfoNCE(None, None, architecture_critic, hyperparams).to(device)
        estimator.learn(X, Y)
        return estimator.MI(X, Y)

    elif method_name == "SMILE":
        from baselines.SMILE import SMILE
        hyperparams = _Hyperparams()
        estimator = SMILE(None, None, architecture_critic, hyperparams).to(device)
        estimator.learn(X, Y)
        return estimator.MI(X, Y)

    elif method_name == "MINDE":
        from baselines.MINDE import MINDE
        hyperparams = _Hyperparams()
        hyperparams.t_patience = 500
        hyperparams.dim = dx
        hyperparams.device = device
        hyperparams.importance_sampling = True
        estimator = MINDE(None, None, None, hyperparams).to(device)
        estimator.learn(X, Y)
        return estimator.MI(X, Y)

    elif method_name == "DoE":
        from baselines.DoE import DoE
        hyperparams = _Hyperparams()
        estimator = DoE(None, None, architecture_critic, hyperparams).to(device)
        estimator.learn(X, Y)
        return estimator.MI(X, Y)

    elif method_name == "FLE":
        from baselines.FLE import FLE
        hyperparams = _Hyperparams()
        estimator = FLE(None, None, architecture_critic, hyperparams).to(device)
        estimator.learn(X, Y)
        return estimator.MI(X, Y)

    elif method_name == "FastMI":
        from baselines.FastMI import FastMI
        hyperparams = _Hyperparams()
        estimator = FastMI(None, None, architecture_critic, hyperparams).to(device)
        estimator.learn(X, Y)
        return estimator.MI(X, Y)

    elif method_name == "MIENF":
        from baselines.MIENF import MIENF
        hyperparams = _Hyperparams()
        estimator = MIENF(None, None, architecture_critic, hyperparams).to(device)
        estimator.learn(X, Y)
        return estimator.MI(X, Y)

    elif method_name == "MRE":
        from baselines.MRE import MRE
        hyperparams = _Hyperparams()
        estimator = MRE(None, None, architecture_critic, hyperparams).to(device)
        estimator.learn(X, Y)
        return estimator.MI(X, Y)

    elif method_name == "VCE":
        from baselines.VCE import VCE
        hyperparams = _Hyperparams()
        hyperparams.nde_type = "FM"
        hyperparams.K_components = 5
        estimator = VCE(None, None, architecture_critic, hyperparams).to(device)
        estimator.learn(X, Y)
        return estimator.MI(X, Y, mode="mc")

    elif method_name == "InfoNet":
        if _infonet_v1_model is None:
            raise RuntimeError(
                "InfoNet V1 model not initialized. "
                "Call evaluations._baseline_helper.init_infonet_v1(config_path, ckpt_path, device) first."
            )
        dev = _infonet_v1_device or torch.device(device)
        from baselines.InfoNet_V1.infer import estimate_mi, compute_smi_mean
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        Y_np = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else Y
        dx, dy = X_np.shape[1], Y_np.shape[1]
        if dx == 1 and dy == 1:
            return estimate_mi(_infonet_v1_model, X_np[:, 0], Y_np[:, 0], device=dev)
        else:
            return compute_smi_mean(X_np, Y_np, _infonet_v1_model, device=dev,
                                    proj_num=32, batchsize=8)

    else:
        raise ValueError(
            f"Unknown baseline method: {method_name}. "
            f"Available: {AVAILABLE_BASELINES}"
        )
