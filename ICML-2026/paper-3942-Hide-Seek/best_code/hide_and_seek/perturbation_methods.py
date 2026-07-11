import os
import sys
import uuid

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor


_RUN_MACHINE_CACHE = {}


def shuffle_tensor_cols(X_tensor, replace=False, random_state=None):
    """
    Shuffle each column of a tensor independently.

    Args:
        X_tensor (torch.Tensor): Input tensor of shape (n_samples, n_features)
        replace (bool): Whether to sample with replacement
        random_state (int): Seed for reproducibility

    Returns:
        torch.Tensor: Tensor with each column shuffled independently
    """
    if random_state is not None:
        torch.manual_seed(random_state)

    shuffled = X_tensor.clone()
    n_samples, n_features = X_tensor.shape

    for col in range(n_features):
        if replace:
            indices = torch.randint(0, n_samples, (n_samples,))
        else:
            indices = torch.randperm(n_samples)
        shuffled[:, col] = X_tensor[indices, col]

    return shuffled


def _import_gaussian_knockoffs_class(): #not used in experiments
    """
    Import GaussianKnockoffs either from an installed package or from local source.
    """
    try:
        from DeepKnockoffs import GaussianKnockoffs
        return GaussianKnockoffs
    except Exception as exc:
        here = os.path.dirname(os.path.abspath(__file__))
        # parent of the DeepKnockoffs package, so `from DeepKnockoffs.* import` resolves
        # (the package's own modules do `from DeepKnockoffs.mmd import ...` internally)
        local_module_dir = os.path.join(
            here,
            "..",
            "packages",
            "knockoffs",
            "deepknockoffs-master",
            "DeepKnockoffs",
        )
        local_module_dir = os.path.normpath(local_module_dir)
        if os.path.isdir(local_module_dir) and local_module_dir not in sys.path:
            sys.path.insert(0, local_module_dir)
        try:
            from DeepKnockoffs.gaussian import GaussianKnockoffs
            return GaussianKnockoffs
        except Exception as local_exc:
            raise ImportError(
                "Could not import GaussianKnockoffs. Install DeepKnockoffs or ensure the "
                "local source path is available: "
                "knockoffs/deepknockoffs-master/DeepKnockoffs/DeepKnockoffs"
            ) from local_exc


def _import_knockoff_machine_class():
    """
    Import KnockoffMachine either from an installed package or from local source.
    """
    try:
        from DeepKnockoffs import KnockoffMachine
        return KnockoffMachine
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        # parent of the DeepKnockoffs package, so `from DeepKnockoffs.* import` resolves
        # (the package's own modules do `from DeepKnockoffs.mmd import ...` internally)
        local_module_dir = os.path.join(
            here,
            "..",
            "packages",
            "knockoffs",
            "deepknockoffs-master",
            "DeepKnockoffs",
        )
        local_module_dir = os.path.normpath(local_module_dir)
        if os.path.isdir(local_module_dir) and local_module_dir not in sys.path:
            sys.path.insert(0, local_module_dir)
        try:
            from DeepKnockoffs.machine import KnockoffMachine
            return KnockoffMachine
        except Exception as local_exc:
            raise ImportError(
                "Could not import KnockoffMachine. Install DeepKnockoffs or ensure the "
                "local source path is available: "
                "knockoffs/deepknockoffs-master/DeepKnockoffs/DeepKnockoffs"
            ) from local_exc


def begin_knockoff_run(seed=None):
    """
    Create a run-scoped cache key for knockoff machine reuse during one training run.
    """
    seed_tag = "none" if seed is None else str(seed)
    return f"ko_run_{seed_tag}_{uuid.uuid4().hex}"


def end_knockoff_run(run_cache_key):
    """
    Clear cached knockoff machine for the given training run.
    """
    if run_cache_key in _RUN_MACHINE_CACHE:
        entry = _RUN_MACHINE_CACHE.pop(run_cache_key)
        if isinstance(entry, dict):
            entry.clear()


def _build_knockoff_machine_pars(
    x_np,
    deep_epochs,
    deep_batch_size,
    deep_dim_h,
    deep_family,
):
    n_samples, n_features = x_np.shape
    batch_size = int(max(2, min(deep_batch_size, n_samples)))
    epoch_length = int(max(1, min(50, max(1, n_samples // batch_size))))

    # Keep test split valid for tiny datasets so training covariance can be estimated.
    if n_samples >= 20:
        test_size = max(10, int(0.1 * n_samples))
        test_size = min(test_size, n_samples - 2)
    elif n_samples >= 4:
        test_size = 1
    else:
        test_size = 0

    pars = {
        "family": deep_family,
        "p": int(n_features),
        "epochs": int(max(1, deep_epochs)),
        "epoch_length": epoch_length,
        "batch_size": batch_size,
        "test_size": int(max(0, test_size)),
        "lr": 0.01,
        "lr_milestones": [int(max(1, deep_epochs))],
        "dim_h": int(max(n_features, deep_dim_h)),
        "target_corr": 0.5 * np.eye(n_features),
        "LAMBDA": 25.0,
        "DELTA": 1.0,
        "GAMMA": 1.0,
        "alphas": np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
    }
    return pars


def _train_machine_once(
    x_np,
    random_state,
    deep_epochs,
    deep_batch_size,
    deep_dim_h,
    deep_family,
):
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    KnockoffMachine = _import_knockoff_machine_class()
    pars = _build_knockoff_machine_pars(
        x_np=x_np,
        deep_epochs=deep_epochs,
        deep_batch_size=deep_batch_size,
        deep_dim_h=deep_dim_h,
        deep_family=deep_family,
    )
    machine = KnockoffMachine(pars)
    machine.train(x_np)
    return machine


def create_knockoffs(
    X_tensor,
    random_state=None,
    method="sdp",
    use_deep_machine=True,
    run_cache_key=None,
    deep_epochs=10,
    deep_batch_size=1_000,
    deep_dim_h=None,
    deep_family="continuous",
):
    """
    Create knockoffs using DeepKnockoffs machine (default) or Gaussian fallback.

    Deep mode trains the knockoff machine once per run when `run_cache_key` is provided.
    Later calls in the same run reuse the trained machine and only call `generate`.

    Args:
        X_tensor (torch.Tensor): Input tensor of shape (n_samples, n_features)
        random_state (int): Seed for reproducibility (numpy/random)
        method (str): Gaussian method. Strict mode requires "sdp" when Gaussian is used.
        use_deep_machine (bool): If True, use DeepKnockoffs machine first
        run_cache_key (str|None): Run-scoped cache key from begin_knockoff_run()
        deep_epochs (int): DeepKnockoffs training epochs
        deep_batch_size (int): DeepKnockoffs batch size
        deep_dim_h (int|None): DeepKnockoffs hidden width, defaults to 4 * n_features
        deep_family (str): Data family passed to DeepKnockoffs machine

    Returns:
        torch.Tensor: Knockoff tensor with same shape/device as X_tensor
    """
    if not isinstance(X_tensor, torch.Tensor):
        raise TypeError("X_tensor must be a torch.Tensor")
    if X_tensor.ndim != 2:
        raise ValueError("X_tensor must be 2D (n_samples, n_features)")

    input_device = X_tensor.device
    x_np = X_tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    if deep_dim_h is None:
        deep_dim_h = int(max(8, 4 * x_np.shape[1]))

    if use_deep_machine:
        cache_entry = None
        data_signature = (x_np.shape, str(x_np.dtype), deep_epochs, deep_batch_size, deep_dim_h, deep_family)

        if run_cache_key is not None:
            cache_entry = _RUN_MACHINE_CACHE.get(run_cache_key)

        if cache_entry is None or cache_entry.get("signature") != data_signature:
            machine = _train_machine_once(
                x_np=x_np,
                random_state=random_state,
                deep_epochs=deep_epochs,
                deep_batch_size=deep_batch_size,
                deep_dim_h=deep_dim_h,
                deep_family=deep_family,
            )
            if run_cache_key is not None:
                _RUN_MACHINE_CACHE[run_cache_key] = {
                    "machine": machine,
                    "signature": data_signature,
                }
        else:
            machine = cache_entry["machine"]

        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        xk_np = machine.generate(x_np)
        if xk_np.shape != x_np.shape:
            raise ValueError(
                f"Knockoff shape mismatch: got {xk_np.shape}, expected {x_np.shape}"
            )
        return torch.tensor(xk_np, dtype=torch.float32, device=input_device)

    if method != "sdp":
        raise ValueError(
            "Gaussian mode is strict and only supports method='sdp'."
        )

    SigmaHat = np.cov(x_np, rowvar=False)
    mu = np.mean(x_np, axis=0)

    GaussianKnockoffs = _import_gaussian_knockoffs_class()
    generator = GaussianKnockoffs(SigmaHat, mu=mu, method="sdp")

    xk_np = generator.generate(x_np)
    if xk_np.shape != x_np.shape:
        raise ValueError(
            f"Knockoff shape mismatch: got {xk_np.shape}, expected {x_np.shape}"
        )

    return torch.tensor(xk_np, dtype=torch.float32, device=input_device)


class RFConditionalSampler:
    """
    Conditional resampler that approximates X_j ~ P(X_j | X_{-j}) via the
    leaf co-occurrence structure of a Random Forest trained to predict X_j
    from the other features.

    The heavy work (forest fit + leaf extraction) happens in fit_and_prepare,
    which is called exactly once. sample_all is cheap and can be called every
    epoch.
    """

    def __init__(self, 
                 n_estimators=100, 
                 min_samples_leaf=30,
                 random_state=None):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.forest = None
        self.y = None
        self.train_leaves = None       # shape (N, T), int32
        self.leaf_to_indices = None    # list length T; dict[leaf_id] -> np.ndarray of row indices
        self.leaf_sizes = None         # shape (N, T): size of the leaf each row landed in, per tree

    def fit_and_prepare(self, X, j):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        n_samples, n_features = X.shape
        if n_features < 2:
            raise ValueError("Need at least two features for conditional resampling")
        if len(X) < 2 * self.min_samples_leaf: #To ensure some conditional sampling. 2 is a heuristic. 
            raise ValueError("Not enough samples for RF conditional resampling. Either reduce min_samples_leaf or increase the number of training samples.")
        
        y = X[:, j].astype(np.float64, copy=True)
        X_minus = np.delete(X, j, axis=1)

        self.forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.forest.fit(X_minus, y)

        train_leaves = self.forest.apply(X_minus).astype(np.int32)   # (n, T) node indices (only leaves) for each sample per tree
        self.train_leaves = train_leaves
        self.y = y

        n, T = train_leaves.shape # number of samples, number of trees
        leaf_to_indices = []
        leaf_sizes = np.empty((n, T), dtype=np.int32)
        for t in range(T):
            col = train_leaves[:, t]
            order = np.argsort(col, kind="stable")
            sorted_leaves = col[order]
            change = np.concatenate(([True], sorted_leaves[1:] != sorted_leaves[:-1]))
            starts = np.flatnonzero(change) # start indices of each unique leaf in the sorted order
            ends = np.concatenate((starts[1:], [n])) # end indices (exclusive) of each unique leaf in the sorted order
            tree_map = {}
            sizes_for_tree = np.empty(n, dtype=np.int32)
            for s, e in zip(starts, ends):
                leaf_id = int(sorted_leaves[s])
                members = order[s:e] # original row indices of samples in this leaf
                tree_map[leaf_id] = members
                sizes_for_tree[members] = e - s # for every sample in this tree, the size of the leaf it is in
            leaf_to_indices.append(tree_map)
            leaf_sizes[:, t] = sizes_for_tree

        self.leaf_to_indices = leaf_to_indices
        self.leaf_sizes = leaf_sizes

    def sample_all(self, random_state=None):
        if self.train_leaves is None:
            raise RuntimeError("fit_and_prepare must be called before sample_all")

        rng = np.random.default_rng(random_state)
        n, T = self.train_leaves.shape

        weights = self.leaf_sizes.astype(np.float64) # for a sample i in tree j, the size of the leaf it is in
        totals = weights.sum(axis=1, keepdims=True)
        cdf = np.cumsum(weights / totals, axis=1)
        r = rng.random(n)[:, None]
        chosen_tree = np.minimum((cdf < r).sum(axis=1), T - 1) # minimum is to account for floating point error (e.g. 0.99999)

        chosen_leaf = self.train_leaves[np.arange(n), chosen_tree] # chosen leaf for the chosen tree per sample
        sampled_idx = np.empty(n, dtype=np.int64)
        for i in range(n):
            members = self.leaf_to_indices[int(chosen_tree[i])][int(chosen_leaf[i])]
            sampled_idx[i] = members[rng.integers(len(members))]

        return self.y[sampled_idx]


def fit_rf_samplers(X_tensor, n_estimators=100, min_samples_leaf=30, 
                    random_state=None):
    """
    Fit one RFConditionalSampler per column of X_tensor. Called exactly once
    before the epoch loop.
    """
    if isinstance(X_tensor, torch.Tensor):
        X_np = X_tensor.detach().cpu().numpy()
    else:
        X_np = np.asarray(X_tensor)
    samplers = []
    for j in range(X_np.shape[1]):
        rs = None if random_state is None else int(random_state) + j
        s = RFConditionalSampler(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=rs,
        )
        s.fit_and_prepare(X_np, j)
        samplers.append(s)
    return samplers


def _sample_with_rf_samplers(X_tensor, rf_samplers, random_state):
    if not isinstance(X_tensor, torch.Tensor):
        raise TypeError("X_tensor must be a torch.Tensor")
    if X_tensor.ndim != 2:
        raise ValueError("X_tensor must be 2D (n_samples, n_features)")
    n_samples, n_features = X_tensor.shape
    if len(rf_samplers) != n_features:
        raise ValueError(
            f"rf_samplers length {len(rf_samplers)} does not match number of features {n_features}"
        )

    out = np.empty((n_samples, n_features), dtype=np.float32)
    for j, sampler in enumerate(rf_samplers):
        if len(sampler.y) != n_samples:
            raise ValueError(
                "RFConditionalSampler was fit on a dataset of different length than X_tensor"
            )
        rs = None if random_state is None else int(random_state) + j
        out[:, j] = sampler.sample_all(random_state=rs).astype(np.float32, copy=False)

    return torch.tensor(out, dtype=X_tensor.dtype, device=X_tensor.device)


def perturb_X(
    X_tensor,
    method,
    random_state=None,
    *,
    replace=True,
    knockoff_run_cache_key=None,
    rf_samplers=None,
):
    """
    Dispatch perturbation to one of the supported methods.

    Args:
        X_tensor (torch.Tensor): input tensor (n_samples, n_features)
        method (str): one of 'draw_marginal', 'knock_off', 'conditional_rf'
        random_state (int|None): seed for the perturbation step
        replace (bool): for draw_marginal, whether to sample with replacement
        knockoff_run_cache_key (str|None): run cache key for knock_off
        rf_samplers (list[RFConditionalSampler]|None): pre-fit samplers for conditional_rf

    Returns:
        torch.Tensor with same shape/dtype/device as X_tensor
    """
    if method == 'draw_marginal':
        return shuffle_tensor_cols(X_tensor, replace=replace, random_state=random_state)
    if method == 'knock_off':
        return create_knockoffs(
            X_tensor,
            random_state=random_state,
            run_cache_key=knockoff_run_cache_key,
        )
    if method == 'conditional_rf':
        if rf_samplers is None:
            raise ValueError(
                "conditional_rf requires pre-fit rf_samplers (call fit_rf_samplers first)."
            )
        return _sample_with_rf_samplers(X_tensor, rf_samplers, random_state)
    raise ValueError(
        f"Unsupported perturbation_method: {method!r}. "
        "Use 'draw_marginal', 'knock_off', or 'conditional_rf'."
    )
