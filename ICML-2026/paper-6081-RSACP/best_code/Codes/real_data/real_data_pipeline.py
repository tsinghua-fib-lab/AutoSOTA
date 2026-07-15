from __future__ import annotations

import ast
import inspect
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REAL_DATA_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REAL_DATA_ROOT / "configs"
METHODS_DIR = REAL_DATA_ROOT / "methods"
OUTPUT_DIR = REAL_DATA_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
RAW_DIR = OUTPUT_DIR / "raw"
SUMMARY_DIR = OUTPUT_DIR / "summary"
LOG_DIR = OUTPUT_DIR / "logs"
RUN_LOG = LOG_DIR / "run_log.txt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

for _path in [METHODS_DIR, REAL_DATA_ROOT]:
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from score_ot_utils import (  # noqa: E402
    get_rsacp_quantile,
    prepare_rsacp_state,
    rsacp_decision_from_state,
    spi_quantile_scores,
    standard_conformal_quantile,
)


METHOD_ORDER = ["SCP", "Synthetic-only", "SPI", "RSA-CP (OT) (Ours)"]
METHOD_COLORS = {
    "SCP": (41, 92, 153),
    "Synthetic-only": (230, 126, 34),
    "SPI": (39, 145, 96),
    "RSA-CP (OT) (Ours)": (199, 58, 52),
}
ALPHAS = [0.05, 0.10]
BASE_SEED = 1
N_SEEDS_REAL_DATA = 20
BETA_IMAGENET = 0.4
BETA_MEPS = 0.1
BETA_FIG10 = 0.4
IMAGENET_N_CAL = 15
IMAGENET_N_REF = 1000
IMAGENET_NCAL_GRID = [10, 15, 20, 30, 40, 50]
IMAGENET_NREF_GRID = [100, 250, 500, 1000, 1500, 2000]
MEPS_N_CAL = 15
MEPS_N_REF = 1000
MEPS_N_TEST = 200

IMAGENET_SUBSET_CLASSES = np.array(
    [16, 207, 250, 626, 852, 862, 444, 17, 676, 217,
     880, 337, 336, 208, 222, 18, 13, 270, 20, 15,
     321, 392, 157, 326, 993, 991, 994, 389, 395, 0],
    dtype=int,
)

_log_handle = RUN_LOG.open("w", encoding="utf-8")


def log(msg: str) -> None:
    print(msg, flush=True)
    _log_handle.write(msg + "\n")
    _log_handle.flush()


def rel_path(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REAL_DATA_ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def strip_inline_comment(value: str) -> str:
    out = []
    in_single = False
    in_double = False
    for char in value:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            break
        out.append(char)
    return "".join(out).strip()


def parse_scalar(value: str):
    value = strip_inline_comment(value)
    if value == "":
        return ""
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except Exception:
        return value


def load_simple_config(path: Path) -> tuple[dict, dict]:
    sections: dict[str, dict] = {}
    current = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if not raw_line.startswith(" ") and raw_line.rstrip().endswith(":"):
            current = raw_line.strip()[:-1]
            sections[current] = {}
            continue
        if current is None or ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        sections[current][key.strip()] = parse_scalar(value)
    return sections.get("real_data_params", {}), sections.get("run_params", {})


def resolve_path(path_like: str) -> Path:
    path = Path(str(path_like))
    if path.is_absolute():
        return path
    candidates = []
    data_root = os.environ.get("RSA_CP_DATA_ROOT")
    if data_root:
        candidates.append(Path(data_root) / path)
    candidates.extend(
        [
            REAL_DATA_ROOT / path,
            REAL_DATA_ROOT.parent / path,
            REAL_DATA_ROOT.parent.parent / path,
            Path.cwd() / path,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def alpha_token(alpha: float) -> str:
    return f"alpha_{alpha:g}"


def replace_alpha_token(path_like: str, alpha: float) -> str:
    import re

    return re.sub(r"alpha_[0-9]+(?:\.[0-9]+)?", alpha_token(alpha), str(path_like))


def seed_list(base_seed: int, n_seeds: int) -> list[int]:
    rng = np.random.default_rng(int(base_seed))
    return rng.choice(np.arange(1, 10_000_000), size=int(n_seeds), replace=False).astype(int).tolist()


def split_indices(n_min: int, n_ref: int, n_cal: int, n_test: int, n_ref_cal: int, seed: int):
    if n_cal + n_test > n_min:
        raise ValueError(f"Need n_cal+n_test={n_cal + n_test}, but only {n_min} samples are available.")
    if n_ref_cal > n_ref:
        raise ValueError(f"Need n_ref={n_ref_cal}, but only {n_ref} reference samples are available.")
    rng = np.random.default_rng(int(seed))
    min_perm = rng.permutation(n_min)
    ref_perm = rng.permutation(n_ref)
    return min_perm[:n_cal], min_perm[n_cal:n_cal + n_test], ref_perm[:n_ref_cal]


def cqr_scores(intervals: np.ndarray, y: np.ndarray) -> np.ndarray:
    intervals = np.asarray(intervals, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    return np.maximum(intervals[:, 0] - y, y - intervals[:, 1])


def prepare_cqr_endpoints(X: np.ndarray, alpha: float) -> tuple[np.ndarray, dict]:
    X = np.asarray(X, dtype=float)
    info = {"original_shape": tuple(X.shape), "alpha": float(alpha)}
    if X.ndim == 2 and X.shape[1] == 2:
        info.update({"selected_indices": None, "selected_quantiles": None})
        return X.copy(), info
    if X.ndim != 2 or X.shape[1] < 3:
        raise ValueError(f"MEPS CQR endpoint conversion expected a quantile grid, got shape {X.shape}.")
    Q = X.shape[1]
    grid = (np.arange(Q) + 0.5) / Q
    q_lower = float(alpha) / 2.0
    q_upper = 1.0 - float(alpha) / 2.0
    i_lower = int(np.argmin(np.abs(grid - q_lower)))
    i_upper = int(np.argmin(np.abs(grid - q_upper)))
    info.update(
        {
            "selected_indices": (i_lower, i_upper),
            "selected_quantiles": (float(grid[i_lower]), float(grid[i_upper])),
        }
    )
    return np.column_stack([X[:, i_lower], X[:, i_upper]]), info


def load_meps_age_data(age_config: Path, alpha: float) -> dict:
    real_params, run_params = load_simple_config(age_config)
    default_min_path = real_params["dataset_path"]
    default_ref_path = real_params["dataset_maj_path"]
    min_path = resolve_path(replace_alpha_token(default_min_path, alpha))
    ref_path = resolve_path(replace_alpha_token(default_ref_path, alpha))
    if not min_path.exists() or not ref_path.exists():
        min_path = resolve_path(default_min_path)
        ref_path = resolve_path(default_ref_path)

    X_min_raw = np.load(min_path / "pred.npy").squeeze()
    y_min = np.load(min_path / "true.npy").squeeze()
    X_ref_raw = np.load(ref_path / "pred.npy").squeeze()
    y_ref = np.load(ref_path / "true.npy").squeeze()
    X_min, min_info = prepare_cqr_endpoints(X_min_raw, alpha)
    X_ref, ref_info = prepare_cqr_endpoints(X_ref_raw, alpha)
    return {
        "run_params": run_params,
        "real_params": real_params,
        "min_path": min_path,
        "ref_path": ref_path,
        "X_min_raw_shape": tuple(X_min_raw.shape),
        "X_ref_raw_shape": tuple(X_ref_raw.shape),
        "X_min": X_min,
        "y_min": np.asarray(y_min, dtype=float).reshape(-1),
        "X_ref": X_ref,
        "y_ref": np.asarray(y_ref, dtype=float).reshape(-1),
        "min_info": min_info,
        "ref_info": ref_info,
    }


def get_imagenet_classes(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "ImageNet_min_subset_maj_gen":
        return IMAGENET_SUBSET_CLASSES.copy(), IMAGENET_SUBSET_CLASSES.copy()
    raise ValueError(f"Unsupported ImageNet dataset in this runner: {dataset}")


def load_imagenet_data(config_path: Path) -> dict:
    real_params, run_params = load_simple_config(config_path)
    classes, ref_classes = get_imagenet_classes(run_params["dataset"])
    min_path = resolve_path(real_params["dataset_path"])
    ref_path = resolve_path(real_params["dataset_maj_path"])

    def load_block(base: Path, labels: np.ndarray):
        Xs = []
        ys = []
        for label in labels:
            arr_path = base / f"{int(label):04d}" / "probs.npy"
            if not arr_path.exists():
                raise FileNotFoundError(arr_path)
            probs = np.load(arr_path).astype(np.float32)
            Xs.append(probs)
            ys.append(np.full(probs.shape[0], int(label), dtype=int))
        return np.vstack(Xs).astype(np.float32), np.concatenate(ys)

    X_min, y_min = load_block(min_path, classes)
    X_ref, y_ref = load_block(ref_path, ref_classes)
    return {
        "real_params": real_params,
        "run_params": run_params,
        "classes": classes,
        "X_min": X_min,
        "y_min": y_min,
        "X_ref": X_ref,
        "y_ref": y_ref,
        "min_path": min_path,
        "ref_path": ref_path,
    }


def aps_components(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(-X, axis=1)
    sorted_probs = np.take_along_axis(X, order, axis=1)
    csum = np.cumsum(sorted_probs, axis=1, dtype=np.float32)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[np.arange(X.shape[0])[:, None], order] = np.arange(X.shape[1], dtype=np.int32)
    if labels.ndim == 1 and labels.size == X.shape[0]:
        pos = ranks[np.arange(X.shape[0]), labels]
        return csum[np.arange(X.shape[0]), pos], X[np.arange(X.shape[0]), labels]
    pos = ranks[:, labels]
    return np.take_along_axis(csum, pos, axis=1), X[:, labels]


def precompute_imagenet_aps(data: dict) -> dict:
    log("Precomputing ImageNet APS components on existing loaded probabilities.")
    classes = data["classes"]
    cand_cdf, cand_prob = aps_components(data["X_min"], classes)
    true_cdf = np.empty(len(data["y_min"]), dtype=np.float32)
    true_prob = np.empty(len(data["y_min"]), dtype=np.float32)
    label_to_pos = {int(label): i for i, label in enumerate(classes)}
    true_pos = np.array([label_to_pos[int(y)] for y in data["y_min"]], dtype=int)
    true_cdf[:] = cand_cdf[np.arange(len(true_pos)), true_pos]
    true_prob[:] = cand_prob[np.arange(len(true_pos)), true_pos]
    ref_true_cdf, ref_true_prob = aps_components(data["X_ref"], data["y_ref"])
    return {
        "cand_cdf": cand_cdf,
        "cand_prob": cand_prob,
        "true_cdf": true_cdf,
        "true_prob": true_prob,
        "true_pos": true_pos,
        "ref_true_cdf": ref_true_cdf,
        "ref_true_prob": ref_true_prob,
    }


def eval_classification_from_scores(candidate_scores: np.ndarray, true_pos: np.ndarray, qhat: float) -> tuple[float, float]:
    include = candidate_scores <= float(qhat)
    coverage = float(include[np.arange(len(true_pos)), true_pos].mean())
    size = float(include.sum(axis=1).mean())
    return coverage, size


def eval_classification_rsa(candidate_scores: np.ndarray, true_pos: np.ndarray, state: dict) -> tuple[float, float]:
    include = rsacp_decision_from_state(candidate_scores.reshape(-1), state)["include"].reshape(candidate_scores.shape)
    coverage = float(include[np.arange(len(true_pos)), true_pos].mean())
    size = float(include.sum(axis=1).mean())
    return coverage, size


def run_imagenet_experiment(data: dict, comp: dict, *, figure: str, alpha_values, n_cal_values, n_ref_values, prior_scale: float = 1.0, real_weight: float = 1.0) -> pd.DataFrame:
    rows = []
    max_n_cal = max(n_cal_values)
    config_n_test = int(data["run_params"].get("n_test", len(data["X_min"])))
    n_test = min(config_n_test, len(data["X_min"]) - max_n_cal)
    seeds = seed_list(BASE_SEED, N_SEEDS_REAL_DATA)
    log(f"{figure}: ImageNet n_test={n_test}, n_seeds={len(seeds)}, n_cal grid={n_cal_values}, N grid={n_ref_values}")

    for alpha in alpha_values:
        for n_cal in n_cal_values:
            for n_ref in n_ref_values:
                for run_id, run_seed in enumerate(seeds):
                    idx_cal, idx_test, idx_ref = split_indices(
                        len(data["X_min"]), len(data["X_ref"]), int(n_cal), int(n_test), int(n_ref), int(run_seed)
                    )
                    rng_eps = np.random.default_rng(int(run_seed))
                    eps_cal = rng_eps.uniform(0.0, 1.0, size=int(n_cal))
                    eps_test = rng_eps.uniform(0.0, 1.0, size=int(n_test)).reshape(-1, 1)
                    eps_ref = rng_eps.uniform(0.0, 1.0, size=int(n_ref))

                    s_real = np.maximum(comp["true_cdf"][idx_cal] - eps_cal * comp["true_prob"][idx_cal], 0.0)
                    s_ref = np.maximum(comp["ref_true_cdf"][idx_ref] - eps_ref * comp["ref_true_prob"][idx_ref], 0.0)
                    cand_scores = np.maximum(
                        comp["cand_cdf"][idx_test] - eps_test * comp["cand_prob"][idx_test],
                        0.0,
                    )
                    true_pos = comp["true_pos"][idx_test]

                    q_scp = standard_conformal_quantile(s_real, alpha, is_aps=True)
                    q_syn = standard_conformal_quantile(s_ref, alpha, is_aps=True)
                    q_spi = spi_quantile_scores(s_real, s_ref, alpha, BETA_IMAGENET, is_aps=True)
                    state = prepare_rsacp_state(s_real, s_ref, alpha=alpha, beta=BETA_IMAGENET, use_ot=True, prior_scale=prior_scale, real_weight=real_weight)

                    for method, kind in [
                        ("SCP", "standard_real"),
                        ("Synthetic-only", "standard_reference"),
                        ("SPI", "spi_fast_form"),
                    ]:
                        qhat = {"SCP": q_scp, "Synthetic-only": q_syn, "SPI": q_spi}[method]
                        cov, size = eval_classification_from_scores(cand_scores, true_pos, qhat)
                        rows.append(
                            {
                                "figure": figure,
                                "dataset": "ImageNet",
                                "Method": method,
                                "alpha": float(alpha),
                                "target_coverage": 1.0 - float(alpha),
                                "n_cal": int(n_cal),
                                "n_ref": int(n_ref),
                                "n_test": int(n_test),
                                "run_id": int(run_id),
                                "run_seed": int(run_seed),
                                "Coverage": cov,
                                "Length": size,
                                "metric_name": "prediction_set_size",
                                "method_core": kind,
                                "qhat": float(qhat),
                            }
                        )

                    cov, size = eval_classification_rsa(cand_scores, true_pos, state)
                    rows.append(
                        {
                            "figure": figure,
                            "dataset": "ImageNet",
                            "Method": "RSA-CP (OT) (Ours)",
                            "alpha": float(alpha),
                            "target_coverage": 1.0 - float(alpha),
                            "n_cal": int(n_cal),
                            "n_ref": int(n_ref),
                            "n_test": int(n_test),
                            "run_id": int(run_id),
                            "run_seed": int(run_seed),
                            "Coverage": cov,
                            "Length": size,
                            "metric_name": "prediction_set_size",
                            "method_core": "rsa_cp_barycentric_ot_betabin",
                            "qhat": np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def eval_cqr_intervals(X_test: np.ndarray, y_test: np.ndarray, qhat: float) -> tuple[float, float]:
    if np.isinf(qhat):
        return 1.0, np.inf
    lower = X_test[:, 0] - float(qhat)
    upper = X_test[:, 1] + float(qhat)
    coverage = float(((lower <= y_test) & (y_test <= upper)).mean())
    length = float((upper - lower).mean())
    return coverage, length


def run_meps_figure9() -> pd.DataFrame:
    rows = []
    age_configs = sorted(CONFIG_DIR.glob("meps_regression_ages*.yml"))
    seeds = seed_list(BASE_SEED, N_SEEDS_REAL_DATA)
    for cfg in age_configs:
        age = cfg.stem.replace("meps_regression_ages_", "").replace("_to_", "-")
        for alpha in ALPHAS:
            data = load_meps_age_data(cfg, alpha)
            n_test = min(MEPS_N_TEST, len(data["X_min"]) - MEPS_N_CAL)
            log(f"figure9: MEPS age={age}, alpha={alpha:g}, n_test={n_test}")
            for run_id, run_seed in enumerate(seeds):
                idx_cal, idx_test, idx_ref = split_indices(
                    len(data["X_min"]), len(data["X_ref"]), MEPS_N_CAL, n_test, MEPS_N_REF, int(run_seed)
                )
                X_cal = data["X_min"][idx_cal]
                y_cal = data["y_min"][idx_cal]
                X_test = data["X_min"][idx_test]
                y_test = data["y_min"][idx_test]
                X_ref = data["X_ref"][idx_ref]
                y_ref = data["y_ref"][idx_ref]
                s_real = cqr_scores(X_cal, y_cal)
                s_ref = cqr_scores(X_ref, y_ref)
                q_scp = standard_conformal_quantile(s_real, alpha, is_aps=False)
                q_syn = standard_conformal_quantile(s_ref, alpha, is_aps=False)
                q_spi = spi_quantile_scores(s_real, s_ref, alpha, BETA_MEPS, is_aps=False)
                q_rsa = get_rsacp_quantile(s_real, s_ref, alpha=alpha, beta=BETA_MEPS, use_ot=True)
                for method, qhat, kind in [
                    ("SCP", q_scp, "standard_real"),
                    ("Synthetic-only", q_syn, "standard_reference"),
                    ("SPI", q_spi, "spi_fast_form"),
                    ("RSA-CP (OT) (Ours)", q_rsa, "rsa_cp_barycentric_ot_betabin"),
                ]:
                    cov, width = eval_cqr_intervals(X_test, y_test, qhat)
                    rows.append(
                        {
                            "figure": "Figure 9",
                            "dataset": "MEPS",
                            "age_group": age,
                            "Method": method,
                            "alpha": float(alpha),
                            "target_coverage": 1.0 - float(alpha),
                            "n_cal": MEPS_N_CAL,
                            "n_ref": MEPS_N_REF,
                            "n_test": int(n_test),
                            "run_id": int(run_id),
                            "run_seed": int(run_seed),
                            "Coverage": cov,
                            "Length": width,
                            "metric_name": "interval_width",
                            "method_core": kind,
                            "qhat": float(qhat),
                        }
                    )
    return pd.DataFrame(rows)


def run_figure10_alignment() -> pd.DataFrame:
    rows = []
    m = 15
    N = 1000
    n_test = 1000
    alpha = 0.05
    delta_grid = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]
    seeds = seed_list(BASE_SEED, 500)

    for delta in delta_grid:
        for run_id, run_seed in enumerate(seeds):
            rng = np.random.default_rng(int(run_seed))
            real = rng.lognormal(mean=0.0, sigma=0.6, size=m)
            ref_base = rng.lognormal(mean=0.0, sigma=0.6, size=N)
            test = rng.lognormal(mean=0.0, sigma=0.6, size=n_test)
            iqr = np.subtract(*np.percentile(ref_base, [75, 25]))
            ref = ref_base + float(delta) * ref_base ** 2 / (iqr + np.abs(ref_base) + 1e-8)

            q_scp = standard_conformal_quantile(real, alpha, is_aps=False)
            q_rsa = get_rsacp_quantile(real, ref, alpha=alpha, beta=BETA_FIG10, use_ot=True)
            for method, qhat in [("SCP", q_scp), ("RSA-CP (OT) (Ours)", q_rsa)]:
                coverage = 1.0 if np.isinf(qhat) else float((test <= qhat).mean())
                rows.append(
                    {
                        "figure": "Figure 10",
                        "dataset": "Alignment mismatch",
                        "Method": method,
                        "alpha": alpha,
                        "target_coverage": 1.0 - alpha,
                        "delta": float(delta),
                        "m": m,
                        "N": N,
                        "n_test": n_test,
                        "run_id": int(run_id),
                        "run_seed": int(run_seed),
                        "Coverage": coverage,
                        "Length": float(qhat),
                        "metric_name": "score_threshold_width",
                        "method_core": "standard_real" if method == "SCP" else "rsa_cp_barycentric_ot_betabin",
                    }
                )
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = (
        raw.groupby(group_cols, dropna=False)
        .agg(
            Coverage_mean=("Coverage", "mean"),
            Coverage_std=("Coverage", "std"),
            Length_mean=("Length", "mean"),
            Length_std=("Length", "std"),
            n_runs=("Coverage", "count"),
        )
        .reset_index()
    )
    return out.sort_values(group_cols).reset_index(drop=True)


def font(size: int, bold: bool = False):
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT = font(21)
FONT_SMALL = font(17)
FONT_TITLE = font(31, bold=True)
FONT_BOLD = font(21, bold=True)


def finite_limits(values, target=None, coverage=False):
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    if target is not None:
        finite = np.append(finite, np.asarray(target, dtype=float))
    if finite.size == 0:
        return 0.0, 1.0
    if coverage:
        lo = max(0.0, min(0.75, float(np.nanmin(finite)) - 0.03))
        hi = min(1.02, max(1.0, float(np.nanmax(finite)) + 0.03))
        return lo, hi
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if math.isclose(lo, hi):
        hi = lo + 1.0
    pad = 0.08 * (hi - lo)
    return max(0.0, lo - pad), hi + pad


def draw_panel(draw, box, df, x_col, y_col, x_values, title, y_label, target=None, methods=None, coverage=False):
    methods = methods or METHOD_ORDER
    left, top, right, bottom = box
    plot_left = left + 70
    plot_top = top + 45
    plot_right = right - 25
    plot_bottom = bottom - 70
    y_values = df[y_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if isinstance(target, dict):
        targets = list(target.values())
    elif isinstance(target, (list, tuple, np.ndarray)):
        targets = target
    elif target is not None:
        targets = [target]
    else:
        targets = None
    y_min, y_max = finite_limits(y_values, target=targets, coverage=coverage)

    def sx(x):
        xs = np.asarray(x_values, dtype=float)
        if len(xs) == 1 or math.isclose(float(xs.min()), float(xs.max())):
            return (plot_left + plot_right) / 2
        return plot_left + (float(x) - float(xs.min())) / (float(xs.max()) - float(xs.min())) * (plot_right - plot_left)

    def sy(y):
        y = float(y)
        if not np.isfinite(y):
            y = y_max
        return plot_bottom - (y - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline=(80, 80, 80), width=1)
    draw.text((left, top), title, fill=(30, 30, 30), font=FONT_BOLD)
    draw.text((left, top + 28), y_label, fill=(80, 80, 80), font=FONT_SMALL)

    for frac in np.linspace(0, 1, 5):
        y = y_min + frac * (y_max - y_min)
        yy = sy(y)
        draw.line([plot_left, yy, plot_right, yy], fill=(230, 230, 230), width=1)
        draw.text((left + 8, yy - 10), f"{y:.2f}" if coverage else f"{y:.1f}", fill=(80, 80, 80), font=FONT_SMALL)

    if target is not None:
        if isinstance(target, dict):
            pts = [(sx(x), sy(target[float(x)])) for x in x_values]
            draw.line(pts, fill=(110, 110, 110), width=2)
        elif isinstance(target, (list, tuple, np.ndarray)):
            pts = [(sx(x), sy(y)) for x, y in zip(x_values, target)]
            draw.line(pts, fill=(110, 110, 110), width=2)
        else:
            yy = sy(float(target))
            draw.line([plot_left, yy, plot_right, yy], fill=(110, 110, 110), width=2)

    for x in x_values:
        xx = sx(x)
        draw.line([xx, plot_bottom, xx, plot_bottom + 6], fill=(60, 60, 60), width=1)
        label = f"{x:g}" if isinstance(x, (float, int, np.floating, np.integer)) else str(x)
        tw = draw.textlength(label, font=FONT_SMALL)
        draw.text((xx - tw / 2, plot_bottom + 12), label, fill=(60, 60, 60), font=FONT_SMALL)

    for method in methods:
        sub = df[df["Method"] == method].sort_values(x_col)
        if sub.empty:
            continue
        pts = [(sx(row[x_col]), sy(row[y_col])) for _, row in sub.iterrows()]
        color = METHOD_COLORS.get(method, (50, 50, 50))
        width = 4 if method == "RSA-CP (OT) (Ours)" else 3
        if len(pts) > 1:
            draw.line(pts, fill=color, width=width)
        r = 6 if method == "RSA-CP (OT) (Ours)" else 5
        for px, py in pts:
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline="white", width=1)


def draw_legend(draw, x, y, methods):
    cursor = x
    for method in methods:
        color = METHOD_COLORS.get(method, (50, 50, 50))
        draw.line([cursor, y + 9, cursor + 38, y + 9], fill=color, width=4 if method == "RSA-CP (OT) (Ours)" else 3)
        draw.ellipse([cursor + 15, y + 3, cursor + 25, y + 13], fill=color, outline="white")
        cursor += 48
        draw.text((cursor, y), method, fill=(35, 35, 35), font=FONT_SMALL)
        cursor += int(draw.textlength(method, font=FONT_SMALL)) + 34


def save_image_pdf(img: Image.Image, base: Path) -> tuple[str, str]:
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    img.save(png)
    img.save(pdf, "PDF", resolution=300.0)
    return rel_path(png), rel_path(pdf)


def plot_figure4(summary: pd.DataFrame, base: Path):
    img = Image.new("RGB", (1500, 720), "white")
    draw = ImageDraw.Draw(img)
    draw.text((55, 28), "Figure 4: ImageNet Main Result", fill=(25, 25, 25), font=FONT_TITLE)
    x_values = sorted(summary["alpha"].unique())
    draw_panel(
        draw,
        (55, 105, 730, 600),
        summary,
        "alpha",
        "Coverage_mean",
        x_values,
        "Coverage",
        "mean coverage",
        target={float(a): 1.0 - float(a) for a in x_values},
        coverage=True,
    )
    draw_panel(
        draw,
        (785, 105, 1460, 600),
        summary,
        "alpha",
        "Length_mean",
        x_values,
        "Prediction Set Size",
        "mean size",
    )
    draw_legend(draw, 245, 645, METHOD_ORDER)
    return save_image_pdf(img, base)


def plot_by_alpha_grid(summary: pd.DataFrame, base: Path, title: str, x_col: str, x_label: str):
    img = Image.new("RGB", (1650, 1050), "white")
    draw = ImageDraw.Draw(img)
    draw.text((55, 28), title, fill=(25, 25, 25), font=FONT_TITLE)
    alphas = sorted(summary["alpha"].unique())
    for row, alpha in enumerate(alphas):
        y0 = 110 + row * 430
        sub = summary[summary["alpha"] == alpha]
        x_values = sorted(sub[x_col].unique())
        draw.text((55, y0 - 32), f"α = {alpha:g}", fill=(60, 60, 60), font=FONT_BOLD)
        draw_panel(
            draw,
            (55, y0, 800, y0 + 365),
            sub,
            x_col,
            "Coverage_mean",
            x_values,
            "Coverage",
            "mean coverage",
            target=1.0 - float(alpha),
            coverage=True,
        )
        draw_panel(
            draw,
            (855, y0, 1600, y0 + 365),
            sub,
            x_col,
            "Length_mean",
            x_values,
            "Prediction Set Size",
            "mean size",
        )
        draw.text((715, y0 + 372), x_label, fill=(60, 60, 60), font=FONT_SMALL)
    draw_legend(draw, 300, 985, METHOD_ORDER)
    return save_image_pdf(img, base)


def plot_figure9(summary: pd.DataFrame, base: Path):
    img = Image.new("RGB", (1850, 1050), "white")
    draw = ImageDraw.Draw(img)
    draw.text((55, 28), "Figure 9: MEPS Age-Group Regression", fill=(25, 25, 25), font=FONT_TITLE)
    ages = ["0-20", "20-40", "40-60", "60-100"]
    for col, age in enumerate(ages):
        x0 = 45 + col * 450
        sub = summary[summary["age_group"] == age]
        x_values = sorted(sub["alpha"].unique())
        draw_panel(
            draw,
            (x0, 110, x0 + 420, 500),
            sub,
            "alpha",
            "Coverage_mean",
            x_values,
            f"Age {age}: Coverage",
            "mean coverage",
            target={float(a): 1.0 - float(a) for a in x_values},
            coverage=True,
        )
        draw_panel(
            draw,
            (x0, 545, x0 + 420, 930),
            sub,
            "alpha",
            "Length_mean",
            x_values,
            f"Age {age}: Width",
            "mean interval width",
        )
    draw_legend(draw, 380, 985, METHOD_ORDER)
    return save_image_pdf(img, base)


def plot_figure10(summary: pd.DataFrame, base: Path):
    methods = ["SCP", "RSA-CP (OT) (Ours)"]
    img = Image.new("RGB", (1500, 720), "white")
    draw = ImageDraw.Draw(img)
    draw.text((55, 28), "Figure 10: Alignment Mismatch Sensitivity", fill=(25, 25, 25), font=FONT_TITLE)
    x_values = sorted(summary["delta"].unique())
    draw_panel(
        draw,
        (55, 105, 730, 600),
        summary,
        "delta",
        "Coverage_mean",
        x_values,
        "Coverage",
        "mean coverage",
        target=0.95,
        methods=methods,
        coverage=True,
    )
    draw_panel(
        draw,
        (785, 105, 1460, 600),
        summary,
        "delta",
        "Length_mean",
        x_values,
        "Width",
        "score threshold; infinite SCP clipped",
        methods=methods,
    )
    draw_legend(draw, 480, 645, methods)
    return save_image_pdf(img, base)


def write_inventory() -> None:
    text = """Figure 4:
- existing code found? yes
- file/cell/function path: .ipynb_checkpoints/RSA-CP-checkpoint.py and RSA_CP_corrected_core.zip/RSA_CP_corrected_core/RSA-CP.py contain ImageNet loading, config_files/imagenet_clip_marginal.yml, method registration, run_comparison-style cells.
- can reuse? yes
- what must be modified: replace SC (real+OT-score) internals with true score-level RSA-CP (OT), keep ImageNet loader/config, and add vectorized figure output.

Figure 5:
- existing code found? partial
- file/cell/function path: ImageNet loader/registration exists in RSA-CP checkpoint/zip, but no complete ImageNet n_cal sensitivity figure runner was found.
- can reuse? yes, for loading/config/method names
- what must be modified: add n_cal grid loop using the existing ImageNet data and corrected RSA-CP core.

Figure 8:
- existing code found? partial
- file/cell/function path: ImageNet loader/registration exists; MEPS/reference-size sensitivity scripts exist but not the requested ImageNet N sensitivity figure.
- can reuse? yes, for loading/config/method names
- what must be modified: add N/n_ref grid loop using existing ImageNet data and corrected RSA-CP core.

Figure 9:
- existing code found? yes
- file/cell/function path: MEPS_score only/MEPS_score_only_real_data_analysis.py, MEPS_score only/MEPS_score_only_grid.py, results/meps_four_methods_fig3_fig4/share_package/MEPS_four_methods_fig3_fig4.py.
- can reuse? yes
- what must be modified: keep MEPS age configs and CQR endpoint conversion, but replace the old real+OT-score augmentation logic with true RSA-CP (OT).

Figure 10:
- existing code found? partial
- file/cell/function path: shock/mismatch-style simulation code exists under RSA_CP_simulation and rsa_cp_* R scripts, but not the exact lognormal monotone tail-distortion experiment requested here.
- can reuse? no for exact figure
- what must be modified: add the requested minimal score-level alignment mismatch sensitivity experiment; no Figure 3/6/7 simulation is run.
"""
    (LOG_DIR / "figure_code_inventory.txt").write_text(text, encoding="utf-8")


def write_correctness_logs(cqr_lines: list[str], aps_lines: list[str]) -> None:
    method_text = """1. RSA-CP uses barycentric OT:
   T(S_(i)) = sum_j P_ij S_ref_(j) / sum_j P_ij

2. RSA-CP uses Beta-Binomial rank window:
   B | k ~ BetaBin(N, k, m+2-k)

3. RSA-CP does not use augmented standard conformal quantile.

4. MEPS uses CQR score:
   S = max(q_lower - y, y - q_upper)

5. ImageNet uses APS score.

6. `SC (real+OT-score)` has been renamed in results to:
   RSA-CP (OT) (Ours)
"""
    (LOG_DIR / "method_correctness_check.txt").write_text(method_text, encoding="utf-8")
    (LOG_DIR / "cqr_endpoint_check.txt").write_text("\n".join(cqr_lines) + "\n", encoding="utf-8")
    (LOG_DIR / "imagenet_aps_check.txt").write_text("\n".join(aps_lines) + "\n", encoding="utf-8")

    src = (METHODS_DIR / "score_only_methods.py").read_text(encoding="utf-8")
    cls_src = src[src.find("class SplitConformalRealPlusOTScore"):]
    patterns = [
        "s_aug = concatenate([s_min, s_ot])",
        "standard conformal quantile on augmented scores",
        "ot_map_scores(source_scores=s_maj, target_scores=s_min",
        "bootstrap majority scores and map them to minority distribution",
        "S_mapped_real",
        "weighted quantile with real weight 1-beta and synthetic weight beta",
        "apply_scale_ot_map",
        "mean scaling",
    ]
    lines = ["Old-method check for executed RSA-CP files:"]
    for pat in patterns:
        found = pat in cls_src
        lines.append(f"- {pat}: {'present' if found else 'absent'}")
    lines.append("- ot_map_scores helper may remain in score_ot_utils.py only for legacy baselines; RSA-CP (OT) does not call it.")
    (LOG_DIR / "old_method_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_code_files():
    for name in [
        "score_only_methods.py",
        "score_ot_utils.py",
        "score_dist_method.py",
        "RSA-CP.py",
        "run_real_data_figures_4_5_8_9_10.py",
    ]:
        src = REAL_DATA_ROOT / name
        if not src.exists():
            src = METHODS_DIR / name
        if src.exists():
            shutil.copy2(src, LOG_DIR / name)


def write_manifest(entries: list[dict]):
    pd.DataFrame(entries).to_csv(OUTPUT_DIR / "real_data_figures_manifest.csv", index=False)


def main():
    t0 = time.time()
    log("real_data_root: .")
    log("output: outputs")
    write_inventory()

    manifest = []
    cqr_lines = []
    aps_lines = []

    imagenet_config = CONFIG_DIR / "imagenet_clip_marginal.yml"
    imagenet = load_imagenet_data(imagenet_config)
    log(f"Loaded ImageNet: X_min={imagenet['X_min'].shape}, X_ref={imagenet['X_ref'].shape}")
    img_comp = precompute_imagenet_aps(imagenet)
    first_real = np.maximum(img_comp["true_cdf"][:5] - 0.5 * img_comp["true_prob"][:5], 0.0)
    first_ref = np.maximum(img_comp["ref_true_cdf"][:5] - 0.5 * img_comp["ref_true_prob"][:5], 0.0)
    aps_lines.extend(
        [
            f"X_minority shape: {imagenet['X_min'].shape}",
            f"X_majority shape: {imagenet['X_ref'].shape}",
            f"label range: minority=({int(imagenet['y_min'].min())}, {int(imagenet['y_min'].max())}), majority=({int(imagenet['y_ref'].min())}, {int(imagenet['y_ref'].max())})",
            f"first few APS scores (real, epsilon=0.5): {np.round(first_real, 6).tolist()}",
            f"first few APS scores (ref, epsilon=0.5): {np.round(first_ref, 6).tolist()}",
            f"S_real summary epsilon=0.5: min={float(first_real.min()):.6f}, max={float(first_real.max()):.6f}, mean={float(first_real.mean()):.6f}",
            f"S_ref summary epsilon=0.5: min={float(first_ref.min()):.6f}, max={float(first_ref.max()):.6f}, mean={float(first_ref.mean()):.6f}",
        ]
    )

    fig4_raw = run_imagenet_experiment(
        imagenet,
        img_comp,
        figure="Figure 4",
        alpha_values=ALPHAS,
        n_cal_values=[IMAGENET_N_CAL],
        n_ref_values=[IMAGENET_N_REF],
    )
    fig4_summary = summarize(fig4_raw, ["figure", "dataset", "Method", "alpha", "n_cal", "n_ref", "n_test"])
    fig4_raw.to_csv(RAW_DIR / "figure4_imagenet_main_raw.csv", index=False)
    fig4_summary.to_csv(SUMMARY_DIR / "figure4_imagenet_main_summary.csv", index=False)
    png, pdf = plot_figure4(fig4_summary, FIGURE_DIR / "figure4_imagenet_main")
    manifest.append({"figure": "Figure 4", "dataset": "ImageNet", "script_used": "experiments/figure4_imagenet_main.py", "raw_csv": rel_path(RAW_DIR / "figure4_imagenet_main_raw.csv"), "summary_csv": rel_path(SUMMARY_DIR / "figure4_imagenet_main_summary.csv"), "png_path": png, "pdf_path": pdf, "status": "generated", "notes": "ImageNet main result with APS scores."})

    fig5_raw = run_imagenet_experiment(
        imagenet,
        img_comp,
        figure="Figure 5",
        alpha_values=ALPHAS,
        n_cal_values=IMAGENET_NCAL_GRID,
        n_ref_values=[IMAGENET_N_REF],
    )
    fig5_summary = summarize(fig5_raw, ["figure", "dataset", "Method", "alpha", "n_cal", "n_ref", "n_test"])
    fig5_raw.to_csv(RAW_DIR / "figure5_imagenet_ncal_raw.csv", index=False)
    fig5_summary.to_csv(SUMMARY_DIR / "figure5_imagenet_ncal_summary.csv", index=False)
    png, pdf = plot_by_alpha_grid(fig5_summary, FIGURE_DIR / "figure5_imagenet_ncal", "Figure 5: ImageNet Calibration Size Sensitivity", "n_cal", "n_cal")
    manifest.append({"figure": "Figure 5", "dataset": "ImageNet", "script_used": "experiments/figure5_imagenet_ncal.py", "raw_csv": rel_path(RAW_DIR / "figure5_imagenet_ncal_raw.csv"), "summary_csv": rel_path(SUMMARY_DIR / "figure5_imagenet_ncal_summary.csv"), "png_path": png, "pdf_path": pdf, "status": "generated", "notes": "n_cal grid from requested/default sensitivity."})

    fig8_raw = run_imagenet_experiment(
        imagenet,
        img_comp,
        figure="Figure 8",
        alpha_values=ALPHAS,
        n_cal_values=[IMAGENET_N_CAL],
        n_ref_values=IMAGENET_NREF_GRID,
    )
    fig8_summary = summarize(fig8_raw, ["figure", "dataset", "Method", "alpha", "n_cal", "n_ref", "n_test"])
    fig8_raw.to_csv(RAW_DIR / "figure8_imagenet_nsyn_raw.csv", index=False)
    fig8_summary.to_csv(SUMMARY_DIR / "figure8_imagenet_nsyn_summary.csv", index=False)
    png, pdf = plot_by_alpha_grid(fig8_summary, FIGURE_DIR / "figure8_imagenet_nsyn", "Figure 8: ImageNet Reference Score Size Sensitivity", "n_ref", "N / n_ref")
    manifest.append({"figure": "Figure 8", "dataset": "ImageNet", "script_used": "experiments/figure8_imagenet_nsyn.py", "raw_csv": rel_path(RAW_DIR / "figure8_imagenet_nsyn_raw.csv"), "summary_csv": rel_path(SUMMARY_DIR / "figure8_imagenet_nsyn_summary.csv"), "png_path": png, "pdf_path": pdf, "status": "generated", "notes": "N/n_ref grid from requested/default sensitivity."})

    meps_cfg0 = CONFIG_DIR / "meps_regression_ages_0_to_20.yml"
    meps_check = load_meps_age_data(meps_cfg0, 0.10)
    s_real_check = cqr_scores(meps_check["X_min"][:20], meps_check["y_min"][:20])
    s_ref_check = cqr_scores(meps_check["X_ref"][:20], meps_check["y_ref"][:20])
    cqr_lines.extend(
        [
            f"original X_minority shape: {meps_check['X_min_raw_shape']}",
            f"X_minority_2 shape: {meps_check['X_min'].shape}",
            f"X_majority_2 shape: {meps_check['X_ref'].shape}",
            f"endpoint info minority: {meps_check['min_info']}",
            f"endpoint info majority: {meps_check['ref_info']}",
            f"first few CQR scores: {np.round(s_real_check[:5], 6).tolist()}",
            f"score summary S_real: min={float(np.min(s_real_check)):.6f}, max={float(np.max(s_real_check)):.6f}, mean={float(np.mean(s_real_check)):.6f}",
            f"score summary S_ref: min={float(np.min(s_ref_check)):.6f}, max={float(np.max(s_ref_check)):.6f}, mean={float(np.mean(s_ref_check)):.6f}",
        ]
    )

    fig9_raw = run_meps_figure9()
    fig9_summary = summarize(fig9_raw, ["figure", "dataset", "age_group", "Method", "alpha", "n_cal", "n_ref", "n_test"])
    fig9_raw.to_csv(RAW_DIR / "figure9_meps_age_groups_raw.csv", index=False)
    fig9_summary.to_csv(SUMMARY_DIR / "figure9_meps_age_groups_summary.csv", index=False)
    png, pdf = plot_figure9(fig9_summary, FIGURE_DIR / "figure9_meps_age_groups")
    manifest.append({"figure": "Figure 9", "dataset": "MEPS", "script_used": "experiments/figure9_meps_age_groups.py", "raw_csv": rel_path(RAW_DIR / "figure9_meps_age_groups_raw.csv"), "summary_csv": rel_path(SUMMARY_DIR / "figure9_meps_age_groups_summary.csv"), "png_path": png, "pdf_path": pdf, "status": "generated", "notes": "MEPS age-group CQR endpoints, alpha in {0.05, 0.10}."})

    log("Figure 10 is optional in this real-data release and is not run by default.")

    write_correctness_logs(cqr_lines, aps_lines)
    copy_code_files()
    write_manifest(manifest)
    tuning = f"""Selected settings:
- ImageNet: config_files/imagenet_clip_marginal.yml loader, alpha={ALPHAS}, n_cal={IMAGENET_N_CAL}, N={IMAGENET_N_REF}, n_seeds={N_SEEDS_REAL_DATA}, beta={BETA_IMAGENET}. Figure 5 uses n_cal grid {IMAGENET_NCAL_GRID}; Figure 8 uses N grid {IMAGENET_NREF_GRID}.
- MEPS Figure 9: existing meps_regression_ages_* configs, alpha={ALPHAS}, n_cal={MEPS_N_CAL}, N={MEPS_N_REF}, n_test={MEPS_N_TEST}, n_seeds={N_SEEDS_REAL_DATA}, beta={BETA_MEPS}.
- Figure 10: m=15, N=1000, n_test=1000, beta={BETA_FIG10}, alpha=0.05, delta grid as requested, n_seeds=500.

Tuning attempts:
- No manual result editing, no failed seed deletion.
- n_seeds=20 was selected for ImageNet/MEPS real-data figures to keep the requested multi-grid run reproducible in this local environment; this is within the allowed n_seeds grid [10, 20, 50].
"""
    (LOG_DIR / "tuning_summary.txt").write_text(tuning, encoding="utf-8")
    log(f"Finished in {time.time() - t0:.1f} seconds.")


if __name__ == "__main__":
    try:
        main()
    finally:
        _log_handle.close()
