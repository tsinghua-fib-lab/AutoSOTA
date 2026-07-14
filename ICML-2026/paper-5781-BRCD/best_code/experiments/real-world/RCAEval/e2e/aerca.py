# aerca_wrapper.py
# Usage:
#   potential_rcs = run_aerca_root_cause(df_n_no_time, df_a_no_time)
# Optional:
#   potential_rcs = run_aerca_root_cause(df_n, df_a, topk=5, repo_root="/path/to/AERCA")

from __future__ import annotations
import os, sys, types
import numpy as np
import pandas as pd
from RCAEval.io.time_series import drop_extra


F_NODE = "F-node" 

# Only select the metrics that are in both datasets
def _match_columns(n_df, a_df):
    cols = _list_intersection(n_df.columns, a_df.columns)
    return (n_df[cols], a_df[cols])

def run_aerca_root_cause(
    df_normal: pd.DataFrame,
    df_failure: pd.DataFrame,
    *,
    repo_root="RCAEval/e2e/AERCA",   # folder that contains: args/, models/, utils/
    topk: int = 5,
    device: str | None = None,
    seed: int | None = None,
):
    """
    Train AERCA on normal data, calibrate thresholds, then analyze the failure period and
    return a ranked list of root-cause variable names using the repo’s nonlinear defaults.

    Returns
    -------
    List[str]
        Column names ranked by root-cause score (highest first).
    """

    # ---------- 0) Wiring to the repo (robust to their internal imports) ----------
    if repo_root is None:
        # assume this file sits beside the cloned repo folder named "AERCA"
        #   your_layout/
        #     AERCA/               <- repo root (has args/, models/, utils/)
        #     aerca_wrapper.py     <- this file
        repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AERCA")
    repo_root = os.path.abspath(repo_root)
    if not (os.path.isdir(repo_root) and
            os.path.isdir(os.path.join(repo_root, "args")) and
            os.path.isdir(os.path.join(repo_root, "models")) and
            os.path.isdir(os.path.join(repo_root, "utils"))):
        raise RuntimeError(f"repo_root does not look like the AERCA repo: {repo_root}")

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Fix the slightly odd absolute imports inside models/aerca.py
    import importlib
    sys.modules["RCAEval.e2e.AERCA.models.senn"]  = importlib.import_module("RCAEval.e2e.AERCA.models.senn")
    sys.modules["RCAEval.e2e.AERCA.utils.utils"]  = importlib.import_module("RCAEval.e2e.AERCA.utils.utils")

    # Now import repo modules
    from RCAEval.e2e.AERCA.args.nonlinear_args import create_arg_parser
    from RCAEval.e2e.AERCA.utils import utils
    from RCAEval.e2e.AERCA.models.aerca import AERCA

    # ---------- 1) Validate inputs ----------
    if not isinstance(df_normal, pd.DataFrame) or not isinstance(df_failure, pd.DataFrame):
        raise TypeError("Both inputs must be pandas.DataFrame.")
    if list(df_normal.columns) != list(df_failure.columns):
        raise ValueError("Columns (names/order) must match between normal and failure DataFrames.")
    if df_normal.shape[0] < 50 or df_failure.shape[0] < 10:
        # A gentle guard; adjust if needed
        raise ValueError("Normal/failure series are too short. Provide longer time windows.")

    Xn = df_normal.to_numpy(dtype=np.float32, copy=False)
    Xa = df_failure.to_numpy(dtype=np.float32, copy=False)
    num_vars = Xn.shape[1]

    # ---------- 2) Nonlinear defaults from repo args ----------
    # (We parse with no CLI args to get their defaults, then override what we must.)
    parser = create_arg_parser()
    args = parser.parse_args([])  # defaults from args/nonlinear_args.py
    opts = vars(args)

    # Always treat as nonlinear dataset configuration
    opts["dataset_name"]   = "nonlinear"
    opts["num_vars"]       = num_vars
    # Respect their default window_size/stride/regularizers/etc from nonlinear_args.py
    # but allow device+seed override if user asked
    if device is not None:
        opts["device"] = device
    else:
        # Use CUDA if available else CPU (repo default is "cuda")
        try:
            import torch
            opts["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            opts["device"] = "cpu"
    if seed is not None:
        opts["seed"] = seed

    # ---------- 3) Reproducibility ----------
    utils.set_seed(opts["seed"])

    # ---------- 4) Build & train the AERCA model on NORMAL data ----------
    # The model’s _training routine expects a numpy array of shape (N, T, P).
    Xn_batch = np.asarray([Xn], dtype=np.float32)

    model = AERCA(
        num_vars=opts["num_vars"],
        hidden_layer_size=opts["hidden_layer_size"],
        num_hidden_layers=opts["num_hidden_layers"],
        device=opts["device"],
        window_size=opts["window_size"],
        stride=opts["stride"],
        encoder_alpha=opts["encoder_alpha"],
        decoder_alpha=opts["decoder_alpha"],
        encoder_gamma=opts["encoder_gamma"],
        decoder_gamma=opts["decoder_gamma"],
        encoder_lambda=opts["encoder_lambda"],
        decoder_lambda=opts["decoder_lambda"],
        beta=opts["beta"],
        lr=opts["lr"],
        epochs=opts["epochs"],
        recon_threshold=opts["recon_threshold"],
        data_name=opts["dataset_name"],
        causal_quantile=opts["causal_quantile"],
        root_cause_threshold_encoder=opts["root_cause_threshold_encoder"],
        root_cause_threshold_decoder=opts["root_cause_threshold_decoder"],
        risk=opts["risk"],
        initial_level=opts["initial_level"],
        num_candidates=opts["num_candidates"],
    )

    # Train on normal data
    model._training(Xn_batch)

    # Calibrate all thresholds on NORMAL data (as in the paper/repo)
    model._get_recon_threshold(Xn_batch)
    model._get_root_cause_threshold_encoder(Xn_batch)
    model._get_root_cause_threshold_decoder(Xn_batch)

    # ---------- 5) Compute POT thresholds (encoder z-scores) on NORMAL data ----------
    # Faithful to _testing_root_cause(): POT on z(us_normal)
    # We recompute encoder residuals for normal to obtain the per-variable POT cutoffs.
    #   us_all_z = (-(u - mean) / std)
    from RCAEval.e2e.AERCA.utils.utils import pot  # repo’s POT implementation

    us_normal_all = []
    with _torch_no_grad_on_device(opts["device"]):
        for x in Xn_batch:
            # _testing_step returns: loss, nexts_hat, nexts, enc_coeffs, dec_coeffs, kl, preproc_label, us
            us = model._testing_step(x, add_u=False)[-1].cpu().numpy()
            us_normal_all.append(us)
    us_normal_all = np.concatenate(us_normal_all, axis=0)        # (T', P)
    z_normal_all  = (-(us_normal_all - model.us_mean_encoder) / (model.us_std_encoder + 1e-12))

    pot_thresh = np.zeros(num_vars, dtype=np.float32)
    for j in range(num_vars):
        pot_val, _ = pot(z_normal_all[:, j], opts["risk"], opts["initial_level"], opts["num_candidates"])
        pot_thresh[j] = float(pot_val)

    # ---------- 6) Score the FAILURE data ----------
    # Compute encoder residual z-scores for failure, then exceed the POT thresholds.
    Xa_batch = np.asarray([Xa], dtype=np.float32)
    with _torch_no_grad_on_device(opts["device"]):
        us_fail = model._testing_step(Xa_batch[0], add_u=False)[-1].cpu().numpy()  # (Tf', P)

    z_fail = (-(us_fail - model.us_mean_encoder) / (model.us_std_encoder + 1e-12))  # (Tf', P)

    # Aggregate to a variable-level score (faithful spirit: exceedance-based)
    #   - primary: how far above the POT cutoff (max exceedance)
    #   - tie-breaker: frequency of exceedances
    exceed = (z_fail - pot_thresh[None, :])
    exceed_pos = np.clip(exceed, 0.0, None)
    max_exceed = exceed_pos.max(axis=0)           # (P,)
    freq_exceed = (exceed_pos > 0).mean(axis=0)   # (P,)

    # Final score: emphasize magnitude, use frequency as a tiebreaker
    score = max_exceed + 0.10 * freq_exceed

    # ---------- 7) Rank and return ----------
    cols = list(df_normal.columns)
    order = np.argsort(score)[::-1]
    ranked_cols = [cols[i] for i in order]
    # Return top-k (or full ranking if topk <= 0)
    return ranked_cols[:topk] if topk and topk > 0 else ranked_cols


# ---------- helpers ----------
class _torch_no_grad_on_device:
    """Context manager that enables torch.no_grad() and pins default device."""
    def __init__(self, device_str: str):
        self.device_str = device_str
        self._cm = None
    def __enter__(self):
        try:
            import torch
            self._cm = torch.no_grad()
            self._cm.__enter__()
            return self
        except Exception:
            return self
    def __exit__(self, exc_type, exc, tb):
        try:
            if self._cm is not None:
                self._cm.__exit__(exc_type, exc, tb)
        finally:
            return False


def drop_constant(df):
    return df.loc[:, (df != df.iloc[0]).any()]

def add_fnode_and_concat(normal_df, anomalous_df):
    normal_df[F_NODE] = "0"
    anomalous_df[F_NODE] = "1"
    return pd.concat([normal_df, anomalous_df])

def _select_lat(df, per):
    return df.filter(regex=(".*(?<!lat_\d{2})$|_lat_" + str(per) + "$"))


# Convert all memeory columns to MBs
def _scale_down_mem(df):
    def update_mem(x):
        if not x.name.endswith("_mem"):
            return x
        x /= 1e6
        x = x.astype(int)
        return x

    return df.apply(update_mem)

_list_intersection = lambda l1, l2: [x for x in l1 if x in l2]
_rm_time = lambda df: df.loc[:, ~df.columns.isin(["time"])]

# Only used for sock-shop and real outage datasets
def preprocess_sock_shop(n_df, a_df, per, dk_select_useful=False):
    _process = lambda df: _select_lat(_scale_down_mem(_rm_time(df)), per)

    n_df = _process(n_df)
    a_df = _process(a_df)

    n_df = drop_constant(n_df)
    a_df = drop_constant(a_df)

    n_df, a_df = _match_columns(n_df, a_df)

    df = add_fnode_and_concat(n_df, a_df)

    if dk_select_useful is True:
        df = _select_useful_cols(df)

    n_df = df[df[F_NODE] == "0"].drop(columns=[F_NODE])
    a_df = df[df[F_NODE] == "1"].drop(columns=[F_NODE])

    return (n_df, a_df)

def _select_useful_cols(df):
    i = df.loc[:, df.columns != F_NODE].std() > 1
    cols = i[i].index.tolist()
    if len(cols) == 1:
        return None
    elif len(cols) == len(df.columns):
        return df
    print(f"Drop {len(df.columns) - len(cols)} columns, left with {len(cols)}")

    return df[cols]

def aerca(
    data,
    inject_time,
    dk_select_useful=False,
    verbose=False,
    dataset=None,
    seed=None,
    **kwargs,
):
    normal_df = data[data["time"] < inject_time]
    anomal_df = data[data["time"] >= inject_time]
    normal_df = normal_df.drop(columns=["time"])
    anomal_df = anomal_df.drop(columns=["time"])

    if dk_select_useful is True:
        normal_df = drop_extra(normal_df)
        anomal_df = drop_extra(anomal_df)

    # if dataset == real outages:
    if dataset == "sock-shop":
        normal_df, anomal_df = preprocess_sock_shop(normal_df, anomal_df, 90, dk_select_useful)
    elif dataset is not None:
        from RCAEval.io.time_series import convert_mem_mb, drop_constant, drop_time, preprocess

        normal_df = drop_constant(convert_mem_mb(drop_time(normal_df)))
        anomal_df = drop_constant(convert_mem_mb(drop_time(anomal_df)))

        normal_df, anomal_df = _match_columns(normal_df, anomal_df)

        #df = add_fnode_and_concat(normal_df, anomal_df)
        if dk_select_useful is True:
            df = _select_useful_cols(df)

        #normal_df = df[df[F_NODE] == "0"].drop(columns=[F_NODE])
        #anomal_df = df[df[F_NODE] == "1"].drop(columns=[F_NODE])

    if seed is not None:
        np.random.seed(seed)

    rc = run_aerca_root_cause(normal_df, anomal_df)
    # return rc
    return {
        "ranks": rc,
    }
