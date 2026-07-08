from __future__ import annotations
from pathlib import Path
import ast
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
import multiprocessing as mp
from typing import Optional, Any, Sequence, Dict, List

from .constants import (LV_PORTFOLIO_T_SCALE, LV_NEWSVENDOR_T_SCALE)

use_LaTeX = False  # Set to True if you have LaTeX installed and want publication-quality fonts
if use_LaTeX:
    plt.rcParams.update({
    "figure.figsize": (6.0, 4.0),
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.size": 18,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "legend.frameon": False,

    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    })
else:
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.figsize": (6.0, 4.0),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
    })

def parse_listlike_column_parallel(
    df: pd.DataFrame,
    col: str,
    n_workers: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> pd.Series:
    """
    Parallel version of parse_listlike_column using multiprocessing.Pool.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame (not copied here).
    col : str
        Column name to parse.
    n_workers : int, optional
        Number of worker processes (defaults to mp.cpu_count()).
    chunksize : int, optional
        Chunksize passed to pool.imap for better load balancing.

    Returns
    -------
    pd.Series
        Series of np.ndarray, aligned with df.index.
    """
    values = df[col].tolist()  # list is picklable, Series is not ideal
    n_workers = n_workers or mp.cpu_count()

    with mp.Pool(processes=n_workers) as pool:
        # imap is nicer for long iterables; chunksize ~ len(values)/(10*n_workers) is a reasonable default
        if chunksize is None and len(values) > 0:
            chunksize = max(1, len(values) // (10 * n_workers))

        parsed_iter = pool.imap(_parse_listlike, values, chunksize)
        parsed = list(parsed_iter)

    return pd.Series(parsed, index=df.index, name=col)


def _parse_listlike(s):
    """
    Robustly parse entries like "[0.1, 0.2, 0.3]" or
    "[np.float64(0.1), np.float64(0.2), ...]" into a 1D float numpy array.

    Crucially, we strip the 'np.float64(...)' wrappers *before* extracting
    numbers, so we do not accidentally pick up the '64' from 'float64'.
    Never raises; returns empty array on failure.
    """
    # Already an array
    if isinstance(s, np.ndarray):
        try:
            return s.astype(float)
        except Exception:
            return np.asarray([], dtype=float)

    # Plain Python containers
    if isinstance(s, (list, tuple)):
        try:
            return np.asarray(s, dtype=float)
        except Exception:
            return np.asarray([], dtype=float)

    # Single scalar
    if isinstance(s, (int, float)) and not pd.isna(s):
        return np.asarray([float(s)], dtype=float)

    # Anything non‑string: give up
    if not isinstance(s, str):
        return np.asarray([], dtype=float)

    txt = s.strip()
    if txt == "" or txt.lower() == "nan":
        return np.asarray([], dtype=float)

    _NP_FLOAT64_RE = re.compile(r"np\.float64\(")
    _NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    if "np.float64" in txt:
        # Remove the "np.float64(" wrapper and the matching ")"
        txt_clean = _NP_FLOAT64_RE.sub("", txt).replace(")", "")
        txt_clean = txt_clean.replace(")", "")
        try:
            obj = ast.literal_eval(txt_clean)
            if isinstance(obj, (list, tuple, np.ndarray)):
                return np.asarray(obj, dtype=float)
            return np.asarray([float(obj)], dtype=float)
        except Exception:
            # Fall through to the generic strategies below
            txt = txt_clean

    # --- 2) Try standard Python literal (e.g. "[0.1, 0.2]") -------------
    try:
        obj = ast.literal_eval(txt)
        if isinstance(obj, (list, tuple, np.ndarray)):
            return np.asarray(obj, dtype=float)
        return np.asarray([float(obj)], dtype=float)
    except Exception:
        pass

    # --- 3) Try JSON (rare, but cheap) -----------------------------------
    try:
        obj = json.loads(txt)
        if isinstance(obj, (list, tuple, np.ndarray)):
            return np.asarray(obj, dtype=float)
        return np.asarray([float(obj)], dtype=float)
    except Exception:
        pass

    # --- 4) Last resort: extract numeric literals with a regex -----------
    # We *first* strip "np.float64(" and ")" to avoid grabbing the "64".
    txt_regex = _NP_FLOAT64_RE.sub("", txt).replace(")", "")
    txt_regex = txt_regex.replace(")", "")
    nums = _NUM_RE.findall(txt_regex)
    try:
        arr = np.asarray([float(v) for v in nums], dtype=float)
        return arr
    except Exception:
        return np.asarray([], dtype=float)



def parse_listlike_column(df: pd.DataFrame, col: str) -> pd.Series:
    """Apply the robust parser to a given column of the DataFrame."""
    return df[col].apply(_parse_listlike)


# ---------- aggregation helpers ----------

def prepare_portfolio_syn_results(
    raw_df: pd.DataFrame,
    dataset: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare synthetic (portfolio/newsvendor) results for plotting.

    This is defensive against newer per-uuid CSVs which may contain extra metrics/outputs
    and may already include some config columns.

    Steps:
      1) (Optional) filter to a single dataset name.
      2) Parse list-like columns ('solution', 'out_of_sample_cost') when present.
      3) Derive per-replication out_of_sample_mean / out_of_sample_var.
      4) Define total_runtime = posterior_time + likelihood_time + solve_time.
      5) Aggregate across replications.
    """
    df = raw_df.copy()
    USE_PARALLEL_PARSE = True

    # Optional filter (useful when results.csv contains multiple datasets).
    if dataset is not None and "dataset" in df.columns:
        df = df[df["dataset"].astype(str) == str(dataset)].copy()

    # --- Parse list-like columns defensively ---
    if "solution" in df.columns:
        if USE_PARALLEL_PARSE:
            df["solution_vec"] = parse_listlike_column_parallel(df, "solution")
        else:
            df["solution_vec"] = parse_listlike_column(df, "solution")

    if "out_of_sample_cost" in df.columns:
        if USE_PARALLEL_PARSE:
            df["oos_vec"] = parse_listlike_column_parallel(df, "out_of_sample_cost")
        else:
            df["oos_vec"] = parse_listlike_column(df, "out_of_sample_cost")

        bad_mask = df["oos_vec"].apply(lambda v: getattr(v, "size", 0) == 0)
        if bad_mask.any():
            print(f"Warning: {bad_mask.sum()} rows have empty out_of_sample_cost after parsing.")

        df["out_of_sample_mean"] = df["oos_vec"].apply(
            lambda v: float(np.mean(v)) if getattr(v, "size", 0) > 0 else np.nan
        )
        df["out_of_sample_var"] = df["oos_vec"].apply(
            lambda v: float(np.var(v, ddof=1)) if getattr(v, "size", 0) > 1 else np.nan
        )
    else:
        # Allow precomputed scalar summaries (newer pipelines may skip logging full vectors).
        if not {"out_of_sample_mean", "out_of_sample_var"}.issubset(df.columns):
            raise ValueError(
                "prepare_portfolio_syn_results: expected 'out_of_sample_cost' OR "
                "both 'out_of_sample_mean' and 'out_of_sample_var' columns."
            )
        df["out_of_sample_mean"] = pd.to_numeric(df["out_of_sample_mean"], errors="coerce")
        df["out_of_sample_var"] = pd.to_numeric(df["out_of_sample_var"], errors="coerce")

    # --- Runtime (robust) ---
    for c in ["posterior_time", "likelihood_time", "solve_time", "setup_time", "dgp_time"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["total_runtime"] = (
        df.get("posterior_time", 0.0).fillna(0.0)
        + df.get("likelihood_time", 0.0).fillna(0.0)
        + df.get("solve_time", 0.0).fillna(0.0)
    )

    df["sampling_runtime"] = (
        df.get("posterior_time", 0.0).fillna(0.0)
        + df.get("likelihood_time", 0.0).fillna(0.0)
    )

    # --- Aggregation over replications ---
    group_cols = [
        "algorithm",
        "epsilon",
        "num_likelihood_samples",
        "num_posterior_samples",
        "contamination",
    ]
    existing_group_cols = [c for c in group_cols if c in df.columns]
    if not existing_group_cols:
        raise ValueError(f"prepare_portfolio_syn_results: none of the group columns {group_cols} are present.")

    agg_spec = dict(
        out_of_sample_mean=("out_of_sample_mean", "mean"),
        out_of_sample_mean_std=("out_of_sample_mean", "std"),
        out_of_sample_var=("out_of_sample_var", "mean"),
        out_of_sample_var_std=("out_of_sample_var", "std"),
        total_runtime_mean=("total_runtime", "mean"),
        total_runtime_std=("total_runtime", "std"),
        sampling_time_mean=("sampling_runtime", "mean"),
        sampling_time_std=("sampling_runtime", "std"),
    )
    if "replication" in df.columns:
        agg_spec["n_replications"] = ("replication", "nunique")
    else:
        agg_spec["n_rows"] = ("total_runtime", "size")

    agg_df = (
        df.groupby(existing_group_cols, dropna=False)
          .agg(**agg_spec)
          .reset_index()
    )

    print("Prepared df (per replication) with shape:", df.shape)
    print("Prepared agg_df (aggregated) with shape:", agg_df.shape)
    return df, agg_df

# ---------- plotting utilities ----------

def plot_oos_frontiers(
    agg_df: pd.DataFrame,
    title: str | None = None,
    special_epsilons: list[float] | None = None,
    out_path: Path | None = None,
    include_legend: bool = False,
):
    """
    Plot mean–variance 'frontiers' for each algorithm:
        x-axis: out-of-sample variance
        y-axis: out-of-sample mean
    One line per algorithm, sorted by epsilon.
    """
    if special_epsilons is None:
        special_epsilons = []

    LABEL_FS = 24
    TICK_FS = 18
    LEGEND_FS = 18
    ANNO_FS = 16

    algo_label = {
        "lv_bas": r"$\mathrm{LV}$",
        "kl_bdro": r"$\mathrm{KL\!-\!BDRO}$",
        "kl_empirical": r"$\mathrm{KL\!-\!Empirical}$",
        "kl_pp": r"$\mathrm{KL\!-\!BAS}_{\rm PP}$",
        "or_wdro": r"$\mathrm{OR\!-\!WDRO}$",
        "lv_reverse": r"$\mathrm{Rev\!-\!LV\!-\!BAS}$",
        "tv_ball": r"$\mathrm{TV\!-\!BAS}$",
    }

    # Colourblind-friendly (Okabe–Ito)
    algo_color = {
        "lv_bas": "#000000",        # black
        "kl_bdro": "#0072B2",       # blue
        "kl_empirical": "#E69F00",  # orange
        "kl_pp": "#009E73",         # bluish green
        "or_wdro": "#CC79A7",       # reddish purple
        "lv_reverse": "#D55E00",    # yellow
        "tv_ball": "#56B4E9",       # sky blue
    }
    contamination = agg_df["contamination"].iloc[0] if "contamination" in agg_df.columns else None

    if contamination is not None:
        title = f"contamination={contamination:.0%}"
        # title = r"Student-$t$-to-normal shift" use this for the entire-shift option
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    for algo_idx, (algo, g) in enumerate(agg_df.groupby("algorithm")):
        g_sorted = g.sort_values("epsilon")
        label = algo_label.get(str(algo), str(algo))
        c = algo_color.get(str(algo), None)
        ax.plot(
            g_sorted["out_of_sample_var"],
            g_sorted["out_of_sample_mean"],
            marker="o",
            markersize=9,
            linestyle="-",
            linewidth=2.5,
            label=label,
            color=c,
        )

        # Optional labels for a few epsilon values
        for eps_idx, eps in enumerate(special_epsilons):
            sel = g_sorted[np.isclose(g_sorted["epsilon"], eps)]
            if not sel.empty and algo == "lv_bas":
                r = sel.iloc[0]
                dx = 1 + 3 * (algo_idx % 3)
                dy = 4 + 6 * ((eps_idx % 3) - 1) + 2 * (algo_idx % 2)
                ax.annotate(
                    f"{eps:g}",
                    (r["out_of_sample_var"], r["out_of_sample_mean"]),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    fontsize=ANNO_FS,
                    color=c,
                )
        
        for eps_idx, eps in enumerate([1, 5, 10]):
            sel = g_sorted[np.isclose(g_sorted["epsilon"], eps)]
            if not sel.empty and algo in ["kl_bdro", "kl_empirical", "kl_pp"]:
                r = sel.iloc[0]
                dx = 1 + 3 * (algo_idx % 3)
                dy = 4 + 5 * (eps_idx - 1) + 2 * (algo_idx % 2)
                ax.annotate(
                    f"{eps:g}",
                    (r["out_of_sample_var"], r["out_of_sample_mean"]),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    fontsize=ANNO_FS,
                    color=c,
                )

    ax.set_xlabel(r"OOS variance ($\varepsilon$)", fontsize=LABEL_FS)
    ax.set_ylabel(r"OOS mean ($\varepsilon$)", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(4, 4), useMathText=True)
    if title is not None:
        ax.set_title(title, fontsize=LABEL_FS)
    if include_legend:
        ax.legend(fontsize=LEGEND_FS)
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        print("Saved frontier plot to", out_path)

    return fig, ax

def plot_oos_boxplots_by_epsilon(
    df: pd.DataFrame,
    metric: str = "out_of_sample_mean",
    epsilons: list[float] | None = None,
    out_dir: Path | None = None,
):
    """
    For each epsilon, draw a boxplot across replications comparing algorithms
    on the chosen metric ('out_of_sample_mean' or 'out_of_sample_var').
    """
    if epsilons is None:
        epsilons = sorted(df["epsilon"].unique())

    algorithms = sorted(df["algorithm"].unique())

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for eps in epsilons:
        sub = df[np.isclose(df["epsilon"], eps)]
        if sub.empty:
            continue

        data = []
        labels = []
        for algo in algorithms:
            vals = sub.loc[sub["algorithm"] == algo, metric].dropna().values
            if vals.size == 0:
                continue
            data.append(vals)
            labels.append(algo)

        if not data:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        bp = ax.boxplot(
            data,
            labels=labels,
            showmeans=True,
            meanline=True,
        )
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{metric.replace('_', ' ').title()} (ε = {eps:g})")
        fig.tight_layout()

        if out_dir is not None:
            fname = f"boxplot_{metric}_eps_{eps:g}.pdf"
            out_path = out_dir / fname
            fig.savefig(out_path, format="pdf", bbox_inches="tight")
            print("Saved", metric, "boxplot for ε=", eps, "to", out_path)

        plt.close(fig)


def plot_runtime_by_algorithm(
    df: pd.DataFrame,
    out_path: Path | None = None,
    log_scale: bool = False,
):
    """
    Bar plot comparing mean total runtime (posterior + solve) per algorithm.
    """
    df_local = df.copy()

    # Define total runtime consistently (and defensively) from components.
    df_local["total_runtime"] = (
        df_local.get("posterior_time", 0.0).fillna(0.0)
        + df_local.get("likelihood_time", 0.0).fillna(0.0)
        + df_local.get("solve_time", 0.0).fillna(0.0)
    )

    rt_df = (
        df_local.groupby("algorithm", dropna=False)
          .agg(
              total_runtime_mean=("total_runtime", "mean"),
              total_runtime_std=("total_runtime", "std"),
          )
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(rt_df))
    means = rt_df["total_runtime_mean"].values
    errs = rt_df["total_runtime_std"].values

    ax.bar(x, means, yerr=errs, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(rt_df["algorithm"])
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Mean total runtime per algorithm")

    if log_scale:
        ax.set_yscale("log")

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        print("Saved runtime plot to", out_path)

    return fig, ax

def make_and_save_runtime_table(
    df: pd.DataFrame,
    out_path: Path | None = None,
    group_cols: list[str] | None = None,
    include_components: bool = True,
) -> pd.DataFrame:
    """
    Create a runtime summary table (mean/sd) and optionally save it as CSV.

    By default, groups by ["algorithm"] and summarises:
      - total_runtime = posterior_time + solve_time (computed if missing)
      - (optional) posterior_time / solve_time / setup_time / likelihood_time / dgp_time if present

    Returns the summary DataFrame.
    """
    group_cols = group_cols or ["algorithm"]

    df_local = df.copy()

    # Define total runtime consistently from components (do not mutate or double-count).
    df_local["total_runtime"] = (
        df_local.get("posterior_time", 0.0).fillna(0.0)
        + df_local.get("likelihood_time", 0.0).fillna(0.0)
        + df_local.get("solve_time", 0.0).fillna(0.0)
    )

    df_local["sampling_runtime"] = (
        df_local.get("posterior_time", 0.0).fillna(0.0)
        + df_local.get("likelihood_time", 0.0).fillna(0.0)
    )

    agg = {}
    # count
    if "replication" in df_local.columns:
        agg["n_replications"] = ("replication", "nunique")
    else:
        agg["n_rows"] = ("total_runtime", "size")

    # always include total runtime stats
    agg["total_runtime_mean"] = ("total_runtime", "mean")
    agg["total_runtime_std"] = ("total_runtime", "std")
    agg["sampling_time_mean"] = ("sampling_runtime", "mean")
    agg["sampling_time_std"] = ("sampling_runtime", "std")

    if include_components:
        for col in ["posterior_time", "solve_time", "setup_time", "likelihood_time", "dgp_time"]:
            if col in df_local.columns:
                agg[f"{col}_mean"] = (col, "mean")
                agg[f"{col}_std"] = (col, "std")

    rt_table = df_local.groupby(group_cols, dropna=False).agg(**agg).reset_index()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rt_table.to_csv(out_path, index=False)
        print("Saved runtime table to", out_path)

    return rt_table

def summarise_runtime_vs_num_observations(
    df: pd.DataFrame,
    *,
    runtime_col: str = "total_runtime",
    include_cols: tuple[str, ...] = ("algorithm", "num_observations"),
) -> pd.DataFrame:
    """
    Return a tidy summary df with mean/sd runtime vs num_observations (per algorithm).
    Aggregates over replications (and over epsilon, if present).
    """
    if runtime_col not in df.columns:
        # Mirror earlier notebook choice: posterior + solve
        if not {"posterior_time", "solve_time"}.issubset(df.columns):
            raise ValueError(f"Missing {runtime_col!r} and cannot construct it (need posterior_time and solve_time).")
        df = df.copy()
        df[runtime_col] = df["posterior_time"].fillna(0.0) + df["solve_time"].fillna(0.0)

    out = df.copy()

    # Ensure numeric num_observations for sorting/plotting
    if "num_observations" in out.columns:
        out["num_observations"] = pd.to_numeric(out["num_observations"], errors="coerce")
        out = out.dropna(subset=["num_observations"])
        out["num_observations"] = out["num_observations"].astype(int)

    gb_cols = [c for c in include_cols if c in out.columns]
    if not gb_cols:
        raise ValueError(f"None of {include_cols} found in df columns.")

    summ = (
        out.groupby(gb_cols, dropna=False)
           .agg(
               runtime_mean=(runtime_col, "mean"),
               runtime_sd=(runtime_col, "std"),
               n=(runtime_col, "count"),
           )
           .reset_index()
           .sort_values([c for c in ("algorithm", "num_observations") if c in gb_cols])
    )
    return summ


def plot_runtime_vs_num_observations(
    df: pd.DataFrame,
    *,
    runtime_col: str = "total_runtime",
    title: str | None = None,
    logy: bool = False,
    out_path: Path | None = None,
    out_csv: Path | None = None,
):
    """
    Line plot: mean runtime vs num_observations (one line per algorithm), with ±1sd error bars.
    Also optionally saves the plotted summary as a CSV.
    """
    summ = summarise_runtime_vs_num_observations(df, runtime_col=runtime_col)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    for algo, g in summ.groupby("algorithm", dropna=False):
        g = g.sort_values("num_observations")
        ax.errorbar(
            g["num_observations"].to_numpy(),
            g["runtime_mean"].to_numpy(),
            yerr=g["runtime_sd"].to_numpy(),
            marker="o",
            linestyle="-",
            capsize=3,
            label=str(algo),
        )

    ax.set_xlabel("Number of observations")
    ax.set_ylabel("Runtime (seconds)")
    if title:
        ax.set_title(title)
    ax.legend()
    if logy:
        ax.set_yscale("log")

    fig.tight_layout()

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        summ.to_csv(out_csv, index=False)
        print("Saved runtime-vs-num_observations summary to", out_csv)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        print("Saved runtime-vs-num_observations plot to", out_path)

    return fig, ax, summ

def save_frontiers_per_n(
    df: pd.DataFrame,
    *,
    out_dir: str | Path,
    special_epsilons: list[float] | None = None,
):
    """
    Creates one mean–variance frontier plot per n and saves as PDFs:
    lv_portfolio_syn_frontiers_n_{n}.pdf

    Requires `plot_oos_frontiers(...)` to already be defined.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if special_epsilons is None:
        special_epsilons = []

    for n in sorted(df["num_observations"].dropna().unique()):
        sub = df.loc[df["num_observations"] == n].copy()

        agg_n = (
            sub.groupby(["algorithm", "epsilon"], dropna=False)
               .agg(
                   out_of_sample_mean=("out_of_sample_mean", "mean"),
                   out_of_sample_var=("out_of_sample_var", "mean"),
               )
               .reset_index()
        )

        out_path = out_dir / f"lv_portfolio_syn_frontiers_n_{int(n)}.pdf"
        fig, ax = plot_oos_frontiers(
            agg_n,
            title=None,
            special_epsilons=special_epsilons,
            out_path=out_path,
        )
        plt.close(fig)

def _eps_lv_to_kl(eps_lv: np.ndarray | float, t_scale: float = LV_PORTFOLIO_T_SCALE):
    eps_lv = np.asarray(eps_lv, dtype=float)
    return 0.5 * (t_scale * eps_lv) ** 2

def _eps_kl_to_lv(eps_kl: np.ndarray | float, t_scale: float = LV_PORTFOLIO_T_SCALE):
    eps_kl = np.asarray(eps_kl, dtype=float)
    eps_kl = np.maximum(eps_kl, 0.0)  # safety
    return np.sqrt(2.0 * eps_kl) / t_scale


def summarise_vs_epsilon(
    df: pd.DataFrame,
    *,
    gamma: float,
    t_scale: float = LV_PORTFOLIO_T_SCALE,
    convert_kl_eps_to_lv_scale: bool = True,
    epsilon_round: int = 12,
) -> pd.DataFrame:
    """
    Compute CE_gamma per replication and aggregate mean/sd vs epsilon for each algorithm.

    CE_gamma = mu - (gamma/2) * sigma^2
    where mu = out_of_sample_mean, sigma^2 = out_of_sample_var.

    If convert_kl_eps_to_lv_scale=True, then for algorithms whose name starts with "kl_",
    we map their stored epsilon (assumed to be epsilon_KL) to LV-scale epsilon for plotting via:
        eps_lv = sqrt(2 * eps_kl) / t_scale
    """
    required = {"algorithm", "epsilon", "out_of_sample_mean", "out_of_sample_var"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for CE plot: {sorted(missing)}")

    out = df.copy()
    out["epsilon"] = pd.to_numeric(out["epsilon"], errors="coerce")
    out = out.dropna(subset=["epsilon", "out_of_sample_mean", "out_of_sample_var", "algorithm"])

    out["ce_gamma"] = out["out_of_sample_mean"] - 0.5 * float(gamma) * out["out_of_sample_var"]
    out["SR"] = out["out_of_sample_mean"]/ np.sqrt(out["out_of_sample_var"])
    out["MSD"] = 0.5* (out["out_of_sample_mean"]+   np.sqrt(out["out_of_sample_var"]))

    # Common x-axis (LV epsilon scale). LV methods: use epsilon directly.
    out["epsilon_plot"] = out["epsilon"].astype(float)

    if convert_kl_eps_to_lv_scale:
        is_kl = out["algorithm"].astype(str).str.startswith("kl_")
        out.loc[is_kl, "epsilon_plot"] = _eps_kl_to_lv(out.loc[is_kl, "epsilon"].to_numpy(), t_scale=t_scale)
    
    # if convert_kl_eps_to_lv_scale:
    #     is_or = out["algorithm"].astype(str).str.startswith("or_")
    #     out.loc[is_or, "epsilon_plot"] = _eps_kl_to_lv(out.loc[is_or, "epsilon"].to_numpy(), t_scale=t_scale)

    # Make groupby stable under float noise
    out["epsilon_plot"] = out["epsilon_plot"].round(epsilon_round)

    summ = (
        out.groupby(["algorithm", "epsilon_plot"], dropna=False)
           .agg(
               ce_mean=("ce_gamma", "mean"),
               ce_sd=("ce_gamma", "std"),
               sr_mean=("SR", "mean"),
               sr_sd=("SR", "std"),
               msd_mean=("MSD", "mean"),
               msd_sd=("MSD", "std"),
               n=("ce_gamma", "count"),
           )
           .reset_index()
    )
    # 95% CI across replications (normal approx)
    summ["ce_se"] = summ["ce_sd"] / np.sqrt(summ["n"].clip(lower=1))
    summ["ce_ci_low"] = summ["ce_mean"] - 1.96 * summ["ce_se"]
    summ["ce_ci_high"] = summ["ce_mean"] + 1.96 * summ["ce_se"]

    summ["sr_se"] = summ["sr_sd"] / np.sqrt(summ["n"].clip(lower=1))
    summ["sr_ci_low"] = summ["sr_mean"] - 1.96 * summ["sr_se"]
    summ["sr_ci_high"] = summ["sr_mean"] + 1.96 * summ["sr_se"]

    summ["msd_se"] = summ["msd_sd"] / np.sqrt(summ["n"].clip(lower=1))
    summ["msd_ci_low"] = summ["msd_mean"] - 1.96 * summ["msd_se"]
    summ["msd_ci_high"] = summ["msd_mean"] + 1.96 * summ["msd_se"]

    return summ.sort_values(["algorithm", "epsilon_plot"])


def plot_ce_gamma_vs_epsilon(
    df: pd.DataFrame,
    *,
    gamma: float,
    t_scale: float = LV_PORTFOLIO_T_SCALE,
    convert_kl_eps_to_lv_scale: bool = True,
    show_ci: bool = True,
    title: str | None = None,
    out_path: Path | None = None,
    out_csv: Path | None = None,
):
    """
    One plot: y = CE_gamma, x = epsilon (LV scale), one line per algorithm.
    Adds a secondary top x-axis showing the corresponding KL epsilon mapping.
    Saves a vector PDF if out_path is given, and saves the plotted summary as CSV if out_csv is given.
    """
    summ = summarise_vs_epsilon(
        df,
        gamma=gamma,
        t_scale=t_scale,
        convert_kl_eps_to_lv_scale=convert_kl_eps_to_lv_scale,
    )

    fig, ax = plt.subplots(figsize=(7.8, 5.2))

    for algo, g in summ.groupby("algorithm", dropna=False):
        g = g.sort_values("epsilon_plot")
        x = g["epsilon_plot"].to_numpy()
        y = g["ce_mean"].to_numpy()
        ax.plot(x, y, marker="o", linestyle="-", label=str(algo))

        if show_ci:
            ax.fill_between(
                x,
                g["ce_ci_low"].to_numpy(),
                g["ce_ci_high"].to_numpy(),
                alpha=0.15,
                linewidth=0,
            )
        else:
            sd = g["ce_sd"].to_numpy()
            ax.fill_between(x, y - sd, y + sd, alpha=0.15, linewidth=0)

    ax.set_xlabel(r"LV Tolerance $\varepsilon_{\mathrm{LV}}$")
    ax.set_ylabel(rf"Certainty-equivalent $\mathrm{{CE}}_{{\gamma}}$  (γ = {gamma:g})")
    if title:
        ax.set_title(title)

    ax.legend(ncols=2, fontsize=9)
    fig.tight_layout()

    try:
        secax = ax.secondary_xaxis(
            "top",
            functions=(
                lambda eps_lv: _eps_lv_to_kl(eps_lv, t_scale=t_scale),
                lambda eps_kl: _eps_kl_to_lv(eps_kl, t_scale=t_scale),
            ),
        )
        secax.set_xlabel(r"KL tolerance $\varepsilon_{\mathrm{KL}} = \frac{1}{2}(t\varepsilon_{\mathrm{LV}})^2$")
    except Exception:
        pass

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        summ.to_csv(out_csv, index=False)
        print("Saved CE-vs-epsilon summary to", out_csv)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        print("Saved CE-vs-epsilon plot to", out_path)

    return fig, ax, summ

def plot_sr_vs_epsilon(
    df: pd.DataFrame,
    *,
    gamma: float,
    t_scale: float = LV_PORTFOLIO_T_SCALE,
    convert_kl_eps_to_lv_scale: bool = True,
    show_ci: bool = True,
    title: str | None = None,
    out_path: Path | None = None,
    out_csv: Path | None = None,
):
    """
    One plot: y = SR, x = epsilon (LV scale), one line per algorithm.
    Adds a secondary top x-axis showing the corresponding KL epsilon mapping.
    Saves a vector PDF if out_path is given, and saves the plotted summary as CSV if out_csv is given.
    """
    summ = summarise_vs_epsilon(
        df,
        gamma=gamma,
        t_scale=t_scale,
        convert_kl_eps_to_lv_scale=convert_kl_eps_to_lv_scale,
    )

    fig, ax = plt.subplots(figsize=(7.8, 5.2))

    for algo, g in summ.groupby("algorithm", dropna=False):
        g = g.sort_values("epsilon_plot")
        x = g["epsilon_plot"].to_numpy()
        y = g["sr_mean"].to_numpy()
        ax.plot(x, y, marker="o", linestyle="-", label=str(algo))

        if show_ci:
            ax.fill_between(
                x,
                g["sr_ci_low"].to_numpy(),
                g["sr_ci_high"].to_numpy(),
                alpha=0.15,
                linewidth=0,
            )
        else:
            sd = g["sr_sd"].to_numpy()
            ax.fill_between(x, y - sd, y + sd, alpha=0.15, linewidth=0)

    ax.set_xlabel(r"LV Tolerance $\varepsilon_{\mathrm{LV}}$")
    ax.set_ylabel(rf"Sharpe Ratio")
    if title:
        ax.set_title(title)

    ax.legend(ncols=2, fontsize=9)
    fig.tight_layout()

    try:
        secax = ax.secondary_xaxis(
            "top",
            functions=(
                lambda eps_lv: _eps_lv_to_kl(eps_lv, t_scale=t_scale),
                lambda eps_kl: _eps_kl_to_lv(eps_kl, t_scale=t_scale),
            ),
        )
        secax.set_xlabel(r"KL tolerance $\varepsilon_{\mathrm{KL}} = \frac{1}{2}(t\varepsilon_{\mathrm{LV}})^2$")
    except Exception:
        pass

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        summ.to_csv(out_csv, index=False)
        print("Saved SR-vs-epsilon summary to", out_csv)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        print("Saved SR-vs-epsilon plot to", out_path)

    return fig, ax, summ

def plot_msd_vs_epsilon(
    df: pd.DataFrame,
    *,
    t_scale: float = LV_NEWSVENDOR_T_SCALE,
    convert_kl_eps_to_lv_scale: bool = True,
    show_ci: bool = True,
    title: str | None = None,
    out_path: Path | None = None,
    out_csv: Path | None = None,
    include_legend: bool = False,
):
    """
    One plot: y = SR, x = epsilon (LV scale), one line per algorithm.
    Adds a secondary top x-axis showing the corresponding KL epsilon mapping.
    Saves a vector PDF if out_path is given, and saves the plotted summary as CSV if out_csv is given.
    """
    LABEL_FS = 24
    TICK_FS = 20
    LEGEND_FS = 20
    TOPLABEL_FS = 20

    algo_label = {
        "lv_bas": r"$\mathrm{LV}$",
        "kl_bdro": r"$\mathrm{KL\!-\!BDRO}$",
        "kl_empirical": r"$\mathrm{KL\!-\!Empirical}$",
        "kl_pp": r"$\mathrm{KL\!-\!BAS}_{\rm PP}$",
        "or_wdro": r"$\mathrm{OR\!-\!WDRO}$",
        "lv_reverse": r"$\mathrm{Rev\!-\!LV\!-\!BAS}$",
        "tv_ball": r"$\mathrm{TV\!-\!BAS}$",
    }

    # Colourblind-friendly (Okabe–Ito)
    algo_color = {
        "lv_bas": "#000000",        # black
        "kl_bdro": "#0072B2",       # blue
        "kl_empirical": "#E69F00",  # orange
        "kl_pp": "#009E73",         # bluish green
        "or_wdro": "#CC79A7",       # reddish purple
        "lv_reverse": "#D55E00",      # vermillion
        "tv_ball": "#56B4E9",        # sky blue
    }

    summ = summarise_vs_epsilon(
        df,
        gamma=0,
        t_scale=t_scale,
        convert_kl_eps_to_lv_scale=convert_kl_eps_to_lv_scale,
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    for algo, g in summ.groupby("algorithm", dropna=False):
        g = g.sort_values("epsilon_plot")
        x = g["epsilon_plot"].to_numpy()
        y = g["msd_mean"].to_numpy()
        label = algo_label.get(str(algo), str(algo))
        c0 = algo_color.get(str(algo), None)
        line, = ax.plot(
            x, y,
            marker="o",
            markersize=9,
            linestyle="-",
            linewidth=2.5,
            label=label,
            color=c0,
        )
        c = line.get_color()

        if show_ci:
            ax.fill_between(
                x,
                g["msd_ci_low"].to_numpy(),
                g["msd_ci_high"].to_numpy(),
                alpha=0.15,
                linewidth=0,
                color=c,
            )
        else:
            sd = g["msd_sd"].to_numpy()
            ax.fill_between(x, y - sd, y + sd, alpha=0.15, linewidth=0, color=c)

    ax.set_xlabel(r"LV Tolerance $\varepsilon_{\mathrm{LV}}$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$0.5\,(\mathrm{OOS\ mean} + \mathrm{OOS\ SD})$", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)

    if title:
        ax.set_title(title, fontsize=LABEL_FS)

    if include_legend:
        ax.legend(ncols=2, fontsize=LEGEND_FS)
    fig.tight_layout()

    try:
        secax = ax.secondary_xaxis(
            "top",
            functions=(
                lambda eps_lv: _eps_lv_to_kl(eps_lv, t_scale=t_scale),
                lambda eps_kl: _eps_kl_to_lv(eps_kl, t_scale=t_scale),
            ),
        )
        secax.set_xlabel(
            r"KL tolerance $\varepsilon_{\mathrm{KL}}$",
            fontsize=TOPLABEL_FS,
        )
        secax.tick_params(labelsize=TICK_FS)
    except Exception:
        pass

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        summ.to_csv(out_csv, index=False)
        print("Saved MSD-vs-epsilon summary to", out_csv)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        print("Saved MSD-vs-epsilon plot to", out_path)

    return fig, ax, summ

# ---------------------------------------------------------------------------
# Real-world LV-BAS (ε, γ) summaries
# ---------------------------------------------------------------------------

def prepare_realworld_results(
    raw_df: pd.DataFrame,
    dataset: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare and aggregate real-world experiment results (PyTorch pipeline).

    Expected conventions in results.csv:
      - Config columns (as available): dataset, scenario, eps_true, corruption, severity,
        algorithm, embedder, head_type, epsilon, gamma, replication.
      - Metric columns: prefixed with 'val_' and 'test_' (e.g. val_avg_acc, test_p95_loss, ...).

    Returns
    -------
    df : pd.DataFrame
        Filtered per-replication rows (one row per replication/seed).
    agg_df : pd.DataFrame
        Aggregated rows grouped by config columns with *_mean and *_std columns.
    """
    df = raw_df.copy()

    # Optional filtering
    if dataset is not None and "dataset" in df.columns:
        df = df[df["dataset"] == dataset]

    # Ensure ε/γ exist for plotting/grouping even for baselines (e.g. GroupDRO).
    if "epsilon" not in df.columns:
        df["epsilon"] = 0.0
    if "gamma" not in df.columns:
        df["gamma"] = np.nan

    # Identify real-world metrics via the val_/test_ prefixes.
    metric_cols = [c for c in df.columns if c.startswith(("val_", "test_"))]
    if not metric_cols:
        raise ValueError(
            "prepare_realworld_results: no columns starting with 'val_' or 'test_' found. "
            "This helper is intended for the real-world pipeline outputs."
        )

    # Coerce common hyperparams to numeric
    for c in ["epsilon", "gamma", "eps_true", "severity"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Coerce metrics to numeric
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Grouping columns (only those present)
    group_cols_candidates = [
        "dataset",
        "scenario",
        "eps_true",
        "corruption",
        "severity",
        "algorithm",
        "embedder",
        "head_type",
        "gamma",
        "epsilon",
    ]
    group_cols = [c for c in group_cols_candidates if c in df.columns]
    if "algorithm" not in group_cols:
        raise ValueError("prepare_realworld_results expects an 'algorithm' column in results.csv.")

    # Aggregation spec: mean/std for all metrics
    agg_spec: dict[str, tuple[str, str]] = {}
    for c in metric_cols:
        agg_spec[f"{c}_mean"] = (c, "mean")
        agg_spec[f"{c}_std"] = (c, "std")

    # Track number of replications aggregated (prefer the explicit 'replication' column if present)
    if "replication" in df.columns:
        agg_spec["n_replications"] = ("replication", "nunique")
    else:
        agg_spec["n_replications"] = (metric_cols[0], "size")

    agg_df = df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
    return df, agg_df


def select_realworld_minimax(
    agg_df: pd.DataFrame,
    *,
    split: str = "val",
    group_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Option-B heuristic selection of (ε, γ) using validation metrics only.

    Selection rule (blueprint):
      - If worst-group labels exist: select (ε,γ) maximising validation worst_group_acc.
      - Otherwise: select (ε,γ) minimising validation p90 loss (fallbacks: p95/max/avg loss).

    Operates on the aggregated dataframe produced by prepare_realworld_results
    (expects columns like 'val_worst_group_acc_mean', 'val_p90_loss_mean', etc).
    """
    out = agg_df.copy()

    # ------------------------------------------------------------------
    # Derived baseline: GroupDRO-B (fixed)
    # Define GroupDRO-B as the ε=0 slice of LV-Group (trained on bulk).
    # This keeps it as a single fixed point for plots/selection, without
    # inmsdducing a separate training run.
    # ------------------------------------------------------------------
    if "algorithm" in out.columns and "epsilon" in out.columns:
        if "rw_groupdro_b" not in out["algorithm"].astype(str).unique():
            src = out[
                (out["algorithm"].astype(str) == "rw_lv_empirical_fair")
                & np.isclose(pd.to_numeric(out["epsilon"], errors="coerce").astype(float), 0.0)
            ].copy()
            if not src.empty:
                src["algorithm"] = "rw_groupdro_b"
                out = pd.concat([out, src], ignore_index=True)

    def _col(metric_base: str) -> str:
        return f"{split}_{metric_base}_mean"

    # Primary metric per blueprint.
    wg_col = _col("worst_group_acc")
    if wg_col in out.columns and out[wg_col].notna().any():
        primary_col = wg_col
        maximise = True
    else:
        # Fallbacks when groups aren't defined / logged.
        primary_col = None
        for tail in ["p90_loss", "p95_loss", "max_loss", "avg_loss"]:
            cand = _col(tail)
            if cand in out.columns and out[cand].notna().any():
                primary_col = cand
                maximise = False
                break
        if primary_col is None:
            raise ValueError(
                "select_realworld_minimax could not find a suitable validation metric column. "
                f"Looked for {wg_col!r} or one of "
                f"{[_col(x) for x in ['p90_loss','p95_loss','max_loss','avg_loss']]}."
            )

    # Grouping: select per dataset/scenario/shift instance and per algorithm.
    if group_cols is None:
        group_cols = [
            c
            for c in [
                "dataset",
                "scenario",
                "eps_true",
                "corruption",
                "severity",
                "algorithm",
                "embedder",
                "head_type",
            ]
            if c in out.columns
        ]
    if "algorithm" not in group_cols and "algorithm" in out.columns:
        group_cols = list(group_cols) + ["algorithm"]

    # Secondary tie-breaker: prefer higher avg_acc if available, else lower avg_loss.
    secondary_cols: list[str] = []
    for base in ["avg_acc", "avg_loss"]:
        cand = _col(base)
        if cand in out.columns and cand != primary_col:
            secondary_cols.append(cand)

    # Deterministic selection within each group.
    selected_rows = []
    for _, g in out.groupby(list(group_cols), dropna=False):
        g = g.dropna(subset=[primary_col]).copy()
        if g.empty:
            continue

        # Make sure ε/γ are numeric for tie-break sorting.
        if "epsilon" in g.columns:
            g["epsilon"] = pd.to_numeric(g["epsilon"], errors="coerce")
        if "gamma" in g.columns:
            g["gamma"] = pd.to_numeric(g["gamma"], errors="coerce")

        sort_cols = [primary_col] + secondary_cols
        # Prefer "less conservative" settings on exact ties.
        if "epsilon" in g.columns:
            sort_cols.append("epsilon")
        if "gamma" in g.columns:
            sort_cols.append("gamma")

        ascending = [not maximise] + ([False] * len(secondary_cols))
        if "epsilon" in g.columns:
            ascending.append(True)
        if "gamma" in g.columns:
            ascending.append(True)

        # mergesort for stable determinism
        g_sorted = g.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        selected_rows.append(g_sorted.iloc[0])

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)

    meta = {
        "split": split,
        "primary_col": primary_col,
        "maximise": maximise,
        "group_cols": list(group_cols),
    }
    return selected_df, meta


def add_minimax_selected_flag(
    agg_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    *,
    key_cols: Sequence[str] | None = None,
    flag_col: str = "minimax_selected",
) -> pd.DataFrame:
    """
    Add a boolean flag to agg_df indicating which rows match the Option-B selection.

    Useful for highlighting points in Pareto plots without mutating original frames.
    """
    out = agg_df.copy()
    if key_cols is None:
        key_cols = [
            c
            for c in [
                "dataset",
                "scenario",
                "eps_true",
                "corruption",
                "severity",
                "algorithm",
                "embedder",
                "head_type",
                "gamma",
                "epsilon",
            ]
            if c in out.columns and c in selected_df.columns
        ]
    if not key_cols:
        raise ValueError("add_minimax_selected_flag: could not infer join keys to mark selection.")

    marker_df = selected_df[key_cols].drop_duplicates().copy()
    marker_df[flag_col] = True
    out = out.merge(marker_df, on=list(key_cols), how="left")
    out[flag_col] = out[flag_col].fillna(False).astype(bool)
    return out


def plot_realworld_metric_vs_epsilon_panels_gamma(
    agg_df: pd.DataFrame,
    *,
    metric: str,
    split: str = "test",
    dataset: str | None = None,
    scenario: str | None = None,
    eps_true: float | None = None,
    algorithms: Sequence[str] | None = None,
    gammas: Sequence[float] | None = None,
    title: str | None = None,
    out_path: Path | None = None,
    out_csv: Path | None = None,
):
    """
    Option-A style plot: curves vs ε with panels by γ.
    Uses aggregated columns like f"{split}_{metric}_mean" (and optional _std).
    """
    df = agg_df.copy()

    # Filtering
    if dataset is not None and "dataset" in df.columns:
        df = df[df["dataset"] == dataset]
    if scenario is not None and "scenario" in df.columns:
        df = df[df["scenario"] == scenario]
    if eps_true is not None and "eps_true" in df.columns:
        df = df[np.isclose(df["eps_true"].astype(float), float(eps_true))]

    if "epsilon" not in df.columns:
        raise ValueError("plot_realworld_metric_vs_epsilon_panels_gamma: expected an 'epsilon' column.")

    y_mean = f"{split}_{metric}_mean"
    y_std = f"{split}_{metric}_std"
    if y_mean not in df.columns:
        raise ValueError(f"plot_realworld_metric_vs_epsilon_panels_gamma: missing column {y_mean!r}.")

    # Determine gamma panels.
    if gammas is None:
        if "gamma" in df.columns and df["gamma"].notna().any():
            gammas = sorted(df["gamma"].dropna().unique().tolist())
        else:
            gammas = [np.nan]

    n_panels = len(gammas)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(6.5 * n_panels, 4.5),
        sharey=True if n_panels > 1 else False,
    )
    if n_panels == 1:
        axes = [axes]

    algo_allow = None if algorithms is None else set(map(str, algorithms))

    for ax, gval in zip(axes, gammas):
        if "gamma" in df.columns:
            if np.isnan(gval):
                panel = df[df["gamma"].isna()].copy()
                gamma_title = "gamma = N/A"
            else:
                # Include baselines with NaN gamma on all panels.
                panel = df[df["gamma"].isna() | np.isclose(df["gamma"].astype(float), float(gval))].copy()
                gamma_title = rf"$\gamma = {float(gval):g}$"
        else:
            panel = df.copy()
            gamma_title = "gamma (not recorded)"

        for algo, g_algo in panel.groupby("algorithm", dropna=False):
            if algo_allow is not None and str(algo) not in algo_allow:
                continue

            g_algo = g_algo.sort_values("epsilon")
            x = pd.to_numeric(g_algo["epsilon"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(g_algo[y_mean], errors="coerce").to_numpy(dtype=float)

            if y_std in g_algo.columns:
                sd = pd.to_numeric(g_algo[y_std], errors="coerce").to_numpy(dtype=float)
            else:
                sd = None

            mask = np.isfinite(x) & np.isfinite(y)
            if sd is not None:
                mask &= np.isfinite(sd)

            x = x[mask]
            y = y[mask]
            if sd is not None:
                sd = sd[mask]

            if x.size == 0:
                continue

            ax.plot(x, y, marker="o", linestyle="-", label=str(algo))
            if sd is not None and sd.size == x.size:
                ax.fill_between(x, y - sd, y + sd, alpha=0.15, linewidth=0)

        ax.set_title(gamma_title)
        ax.set_xlabel(r"$\epsilon$")
        ax.grid(True)

    axes[0].set_ylabel(metric.replace("_", " "))

    if title is None:
        parts = []
        if dataset is not None:
            parts.append(str(dataset))
        if scenario is not None:
            parts.append(str(scenario))
        if eps_true is not None:
            parts.append(f"eps_true={eps_true:g}")
        parts.append(f"{split}:{metric}")
        title = " | ".join(parts)

    fig.suptitle(title, y=0.995)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),  # <- pushes legend down a touch
            ncols=min(4, len(labels)),
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))  # <- more space for title+legend
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.92))


    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")

    return fig, axes

def plot_realworld_bulk_ablation_sweeps(
    agg_df: pd.DataFrame,
    *,
    metric: str,
    split: str = "test",
    dataset: str | None = None,
    scenario: str | None = None,
    eps_true: float | None = None,
    gammas: Sequence[float] | None = None,
    title: str | None = None,
    out_path: Path | None = None,
    out_csv: Path | None = None,
):
    """
    Plot CivilComments epsilon sweeps for the full bulk-filtered ablation.

    - x-axis: epsilon
    - one panel per gamma
    - includes the fixed GroupDRO-B baseline derived from LV-Group at epsilon=0
    - includes explicit bulk-filtered baselines ERM-B / CVaR-B / chi2-DRO-B
    """
    df = agg_df.copy()

    if dataset is not None and "dataset" in df.columns:
        df = df[df["dataset"] == dataset]
    if scenario is not None and "scenario" in df.columns:
        df = df[df["scenario"] == scenario]
    if eps_true is not None and "eps_true" in df.columns:
        df = df[np.isclose(df["eps_true"].astype(float), float(eps_true))]

    if "algorithm" in df.columns and "epsilon" in df.columns:
        if "rw_groupdro_b" not in df["algorithm"].astype(str).unique():
            src = df[
                (df["algorithm"].astype(str) == "rw_lv_empirical_fair")
                & np.isclose(pd.to_numeric(df["epsilon"], errors="coerce").astype(float), 0.0)
            ].copy()
            if not src.empty:
                src["algorithm"] = "rw_groupdro_b"
                df = pd.concat([df, src], ignore_index=True)

    if "epsilon" not in df.columns:
        raise ValueError("plot_realworld_bulk_ablation_sweeps: expected an 'epsilon' column.")

    y_mean = f"{split}_{metric}_mean"
    y_std = f"{split}_{metric}_std"
    if y_mean not in df.columns:
        raise ValueError(f"plot_realworld_bulk_ablation_sweeps: missing column {y_mean!r}.")

    if gammas is None:
        if "gamma" in df.columns and df["gamma"].notna().any():
            gammas = sorted(df["gamma"].dropna().unique().tolist())
        else:
            gammas = [np.nan]

    OKABE_ITO = {
        "black": "#000000",
        "orange": "#E69F00",
        "sky_blue": "#56B4E9",
        "bluish_green": "#009E73",
        "yellow": "#F0E442",
        "blue": "#0072B2",
        "vermillion": "#D55E00",
        "reddish_purple": "#CC79A7",
    }

    ALGO_ORDER = [
        "rw_chi2_dro",
        "rw_chi2_dro_b",
        "rw_cvar",
        "rw_cvar_b",
        "rw_erm",
        "rw_erm_b",
        "rw_groupdro",
        "rw_groupdro_b",
        "rw_lv_empirical",
        "rw_lv_empirical_fair",
    ]
    ALGO_DISPLAY = {
        "rw_chi2_dro": r"$\chi^2$-DRO",
        "rw_chi2_dro_b": r"$\chi^2$-DRO-B",
        "rw_cvar": "CVaR",
        "rw_cvar_b": "CVaR-B",
        "rw_erm": "ERM",
        "rw_erm_b": "ERM-B",
        "rw_groupdro": "GroupDRO",
        "rw_groupdro_b": "GroupDRO-B",
        "rw_lv_empirical": r"$\mathrm{LV-Empirical}$",
        "rw_lv_empirical_fair": r"$\mathrm{LV-Group}$",
    }
    ALGO_COLOUR = {
        "rw_chi2_dro": OKABE_ITO["vermillion"],
        "rw_chi2_dro_b": OKABE_ITO["vermillion"],
        "rw_cvar": OKABE_ITO["orange"],
        "rw_cvar_b": OKABE_ITO["orange"],
        "rw_erm": OKABE_ITO["black"],
        "rw_erm_b": "#7A7A7A",
        "rw_groupdro": OKABE_ITO["blue"],
        "rw_groupdro_b": OKABE_ITO["sky_blue"],
        "rw_lv_empirical": OKABE_ITO["reddish_purple"],
        "rw_lv_empirical_fair": OKABE_ITO["bluish_green"],
    }
    ALGO_LINESTYLE = {
        "rw_chi2_dro": "-",
        "rw_chi2_dro_b": "--",
        "rw_cvar": "-",
        "rw_cvar_b": "--",
        "rw_erm": "None",
        "rw_erm_b": "None",
        "rw_groupdro": "None",
        "rw_groupdro_b": "None",
        "rw_lv_empirical": "-",
        "rw_lv_empirical_fair": "-",
    }
    ALGO_MARKER = {
        "rw_chi2_dro": "o",
        "rw_chi2_dro_b": "o",
        "rw_cvar": "o",
        "rw_cvar_b": "o",
        "rw_erm": "o",
        "rw_erm_b": "*",
        "rw_groupdro": "o",
        "rw_groupdro_b": "*",
        "rw_lv_empirical": "o",
        "rw_lv_empirical_fair": "o",
    }
    ALGO_MARKERSIZE = {
        "rw_erm_b": 13.0,
        "rw_groupdro_b": 13.0,
    }

    n_panels = len(gammas)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(6.8 * n_panels, 4.8),
        sharey=True if n_panels > 1 else False,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, gval in zip(axes, gammas):
        if "gamma" in df.columns and (not np.isnan(gval)):
            panel = df[np.isclose(df["gamma"].astype(float), float(gval))].copy()
            gamma_title = rf"$\gamma = {float(gval):g}$"
        else:
            panel = df.copy()
            gamma_title = "gamma = N/A"

        for algo in ALGO_ORDER:
            sub = panel[panel["algorithm"].astype(str) == str(algo)].copy()
            if sub.empty:
                continue

            sub["epsilon"] = pd.to_numeric(sub["epsilon"], errors="coerce")
            sub = sub.sort_values("epsilon")

            x = pd.to_numeric(sub["epsilon"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(sub[y_mean], errors="coerce").to_numpy(dtype=float)
            if y_std in sub.columns:
                sd = pd.to_numeric(sub[y_std], errors="coerce").to_numpy(dtype=float)
            else:
                sd = None

            mask = np.isfinite(x) & np.isfinite(y)
            if sd is not None:
                mask &= np.isfinite(sd)

            x = x[mask]
            y = y[mask]
            if sd is not None:
                sd = sd[mask]

            if x.size == 0:
                continue

            colour = ALGO_COLOUR.get(str(algo), OKABE_ITO["black"])
            linestyle = ALGO_LINESTYLE.get(str(algo), "-")
            marker = ALGO_MARKER.get(str(algo), "o")
            markersize = ALGO_MARKERSIZE.get(str(algo), 6.5)
            label = ALGO_DISPLAY.get(str(algo), str(algo))

            ax.plot(
                x,
                y,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.5 if linestyle != "None" else 0.0,
                markersize=markersize,
                color=colour,
                label=label,
                alpha=0.95,
            )
            if sd is not None and sd.size == x.size and linestyle != "None":
                ax.fill_between(x, y - sd, y + sd, alpha=0.12, linewidth=0, color=colour)

        ax.set_title(gamma_title)
        ax.set_xlabel(r"$\epsilon$")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(metric.replace("_", " "))

    if title is None:
        parts = []
        if dataset is not None:
            parts.append(str(dataset))
        if scenario is not None:
            parts.append(str(scenario))
        if eps_true is not None:
            parts.append(f"eps_true={eps_true:g}")
        parts.append(f"{split}:{metric}: bulk ablation")
        title = " | ".join(parts)

    fig.suptitle(title, y=0.995)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            ncols=min(5, len(labels)),
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.92))

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")

    return fig, axes

def plot_realworld_pareto(
    agg_df: pd.DataFrame,
    *,
    x_metric: str,
    y_metric: str,
    split: str = "test",
    dataset: str | None = None,
    scenario: str | None = None,
    eps_true: float | None = None,
    algorithms: Sequence[str] | None = None,
    connect_by_epsilon: bool = True,
    selected_df: pd.DataFrame | None = None,
    highlight_selected: bool = True,
    title: str | None = None,  # kept for API compatibility (ignored by design)
    out_path: Path | None = None,
    out_csv: Path | None = None,
):
    """
    Pareto/trade-off plot in metric space.

    Changes vs older version:
      - Only plots a fixed shortlist of algorithms with paper-friendly display names.
      - Removes γ from legend (and avoids duplicate legend entries per γ).
      - No title (ICML-friendly; caption should explain).
      - Okabe–Ito colour palette.
      - Highlights “minimax selected” points (stars), with star colour matching the algorithm.
    """
    from matplotlib.lines import Line2D

    # ----------------------------
    # Display config (fixed shortlist)
    # ----------------------------
    ALGO_DISPLAY = {
        "rw_chi2_dro": r"$\chi^2$-DRO",
        "rw_chi2_dro_b": r"$\chi^2$-DRO-B",
        "rw_cvar": "CVaR",
        "rw_cvar_b": "CVaR-B",
        "rw_erm": "ERM",
        "rw_erm_b": "ERM-B",
        "rw_groupdro": "GroupDRO",
        "rw_groupdro_b": "GroupDRO-B",
        "rw_lv_empirical_fair": r"$\mathrm{LV-Group}$",
        "rw_lv_empirical": r"$\mathrm{LV-Empirical}$",
    }
    # Okabe–Ito palette (colourblind-friendly)
    OKABE_ITO = {
        "black": "#000000",
        "orange": "#E69F00",
        "sky_blue": "#56B4E9",
        "bluish_green": "#009E73",
        "yellow": "#F0E442",
        "blue": "#0072B2",
        "vermillion": "#D55E00",
        "reddish_purple": "#CC79A7",
    }
    ALGO_COLOUR = {
        "rw_erm": OKABE_ITO["black"],
        "rw_erm_b": "#7A7A7A",
        "rw_groupdro": OKABE_ITO["blue"],
        "rw_groupdro_b": OKABE_ITO["sky_blue"],
        "rw_cvar": OKABE_ITO["orange"],
        "rw_cvar_b": "#B27400",
        "rw_chi2_dro": OKABE_ITO["vermillion"],
        "rw_chi2_dro_b": "#9C3F00",
        "rw_lv_empirical_fair": OKABE_ITO["bluish_green"],
        "rw_lv_empirical": OKABE_ITO["reddish_purple"],
    }

    def _darken_colour(colour: str, factor: float = 0.65) -> tuple[float, float, float]:
        """Darken a Matplotlib colour (hex/name) by scaling RGB."""
        rgb = np.array(mpl.colors.to_rgb(colour), dtype=float)
        return tuple(np.clip(rgb * float(factor), 0.0, 1.0))

    # Font sizes (bigger + readable)
    LABEL_FS = 16
    TICK_FS = 14
    LEGEND_FS = 11

    df = agg_df.copy()

    if dataset is not None and "dataset" in df.columns:
        df = df[df["dataset"] == dataset]
    if scenario is not None and "scenario" in df.columns:
        df = df[df["scenario"] == scenario]
    if eps_true is not None and "eps_true" in df.columns:
        df = df[np.isclose(df["eps_true"].astype(float), float(eps_true))]

    # ------------------------------------------------------------------
    # Derived baseline: GroupDRO-B (fixed)
    # Define GroupDRO-B as the ε=0 slice of LV-Group (trained on bulk).
    # ------------------------------------------------------------------
    if "algorithm" in df.columns and "epsilon" in df.columns:
        if "rw_groupdro_b" not in df["algorithm"].astype(str).unique():
            src = df[
                (df["algorithm"].astype(str) == "rw_lv_empirical_fair")
                & np.isclose(pd.to_numeric(df["epsilon"], errors="coerce").astype(float), 0.0)
            ].copy()
            if not src.empty:
                src["algorithm"] = "rw_groupdro_b"
                df = pd.concat([df, src], ignore_index=True)

    x_col = f"{split}_{x_metric}_mean"
    y_col = f"{split}_{y_metric}_mean"
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"plot_realworld_pareto: missing {x_col!r} and/or {y_col!r}.")

    # Keep only the supported algorithms (and optionally intersect with user-specified algorithms)
    allowed = set(ALGO_DISPLAY.keys())
    if algorithms is not None:
        allowed = allowed.intersection(set(map(str, algorithms)))
    df = df[df["algorithm"].astype(str).isin(sorted(allowed))].copy()

    # If nothing remains, still return an empty plot (but don’t crash)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    if df.empty:
        ax.set_xlabel(x_metric.replace("_", " "), fontsize=LABEL_FS)
        ax.set_ylabel(y_metric.replace("_", " "), fontsize=LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.grid(True, alpha=0.3)
        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, format="pdf", bbox_inches="tight")
        return fig, ax

    # ----------------------------
    # Plot curves (label once per algorithm)
    # ----------------------------
    # We'll optionally show multiple γ curves per algorithm, but:
    #   - same colour
    #   - only the first γ gets a legend label
    if "gamma" in df.columns and df["gamma"].notna().any():
        gamma_vals_global = sorted(df["gamma"].dropna().unique().tolist())
    else:
        gamma_vals_global = [np.nan]

    for algo, g_algo in df.groupby("algorithm", dropna=False):
        algo = str(algo)
        colour = ALGO_COLOUR.get(algo, OKABE_ITO["black"])
        label = ALGO_DISPLAY.get(algo, algo.replace("rw_", ""))

        labelled_once = False
        for gval in gamma_vals_global:
            if "gamma" in g_algo.columns:
                if np.isnan(gval):
                    sub = g_algo[g_algo["gamma"].isna()].copy()
                else:
                    sub = g_algo[np.isclose(g_algo["gamma"].astype(float), float(gval))].copy()
            else:
                sub = g_algo.copy()

            if sub.empty:
                continue

            # For Pareto curves, connecting by epsilon is fine as long as epsilon exists.
            if connect_by_epsilon and "epsilon" in sub.columns:
                sub = sub.sort_values("epsilon")
                linestyle = "-"
            else:
                linestyle = "None"

            x = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if x.size == 0:
                continue

            this_label = label if not labelled_once else "_nolegend_"
            marker_style = "*" if algo == "rw_erm_b" else "o"
            line_style = "--" if algo in {"rw_chi2_dro_b", "rw_cvar_b"} else linestyle
            marker_size = 12.0 if algo == "rw_erm_b" else 6.5

            ax.plot(
                x,
                y,
                marker=marker_style,
                linestyle=line_style,
                linewidth=3.0 if line_style != "None" else 0.0,
                markersize=marker_size,
                color=colour,
                label=this_label,
                alpha=0.90 if not labelled_once else 0.35,
            )

            # Label a small set of ε values along each curve (darkened text; do NOT darken markers).
            if "epsilon" in sub.columns:
                mark_eps = np.array([0.1, 0.5, 0.9], dtype=float)
                eps = pd.to_numeric(sub["epsilon"], errors="coerce").to_numpy(dtype=float)
                mark_mask = np.isfinite(eps) & np.any(np.isclose(eps[:, None], mark_eps[None, :]), axis=1)
                if np.any(mark_mask):
                    xm = pd.to_numeric(sub.loc[mark_mask, x_col], errors="coerce").to_numpy(dtype=float)
                    ym = pd.to_numeric(sub.loc[mark_mask, y_col], errors="coerce").to_numpy(dtype=float)
                    em = eps[mark_mask]

                    mm = np.isfinite(xm) & np.isfinite(ym) & np.isfinite(em)
                    xm = xm[mm]
                    ym = ym[mm]
                    em = em[mm]

                    if xm.size:
                        dark = _darken_colour(colour)
                        for xi, yi, ei in zip(xm, ym, em):
                            ax.annotate(
                                f"{ei:.1f}",
                                xy=(xi, yi),
                                xytext=(2, 2),
                                textcoords="offset points",
                                ha="left",
                                va="bottom",
                                fontsize=10,
                                color=dark,
                                alpha=0.95,
                                zorder=5,
                            )

            labelled_once = True
    # ----------------------------
    # Highlight “minimax selected” points (stars, coloured by algorithm)
    # ----------------------------
    if selected_df is not None and highlight_selected:
        s = selected_df.copy()
        if dataset is not None and "dataset" in s.columns:
            s = s[s["dataset"] == dataset]
        if scenario is not None and "scenario" in s.columns:
            s = s[s["scenario"] == scenario]
        if eps_true is not None and "eps_true" in s.columns:
            s = s[np.isclose(s["eps_true"].astype(float), float(eps_true))]

        s = s[s["algorithm"].astype(str).isin(sorted(allowed))].copy()

        if x_col in s.columns and y_col in s.columns and (not s.empty):
            # plot per algorithm so stars inherit colour
            for algo, sub in s.groupby("algorithm", dropna=False):
                algo = str(algo)
                colour = ALGO_COLOUR.get(algo, OKABE_ITO["black"])

                xs = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(dtype=float)
                ys = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(xs) & np.isfinite(ys)
                xs = xs[mask]
                ys = ys[mask]
                if xs.size == 0:
                    continue

                ax.scatter(
                    xs,
                    ys,
                    marker="*",
                    s=300,
                    color=colour,
                    edgecolors=colour,
                    linewidths=0.8,
                    zorder=5,
                    label="_nolegend_",
                )

            # Add a single legend handle for the concept “minimax selected”
            star_handle = Line2D(
                [0],
                [0],
                marker="*",
                linestyle="None",
                markersize=12,
                markerfacecolor="none",
                markeredgecolor=OKABE_ITO["black"],
                markeredgewidth=2.0,
                label="minimax selected",
            )
            handles, labels = ax.get_legend_handles_labels()
            handles.append(star_handle)
            labels.append("minimax selected")
            ax.legend(handles, labels, fontsize=LEGEND_FS, ncols=4, frameon=True, loc='lower center', bbox_to_anchor=(0.5, 1.02))
        else:
            ax.legend(handles, labels, fontsize=LEGEND_FS, ncols=4, frameon=True, loc='lower center', bbox_to_anchor=(0.5, 1.02))
    else:
        ax.legend(handles, labels, fontsize=LEGEND_FS, ncols=4, frameon=True, loc='lower center', bbox_to_anchor=(0.5, 1.02))

    # ----------------------------
    # Axes styling (no title)
    # ----------------------------
    ax.set_xlabel("Mean accuracy", fontsize=LABEL_FS)
    ax.set_ylabel("Worst-group accuracy", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")

    return fig, ax

# ---------------------------------------------------------------------
# CIFAR-10-C plotting helpers (mixture shift; evaluation components)
#   Key design:
#     - facet into subplots by (epsilon, gamma) so curves are not tangled
#     - within each panel: colour = algorithm; linestyle/marker = metric
#     - legends are figure-level (not per-panel) to avoid eating the plot
# ---------------------------------------------------------------------

def _filter_cifar10c_agg(
    agg_df: "pd.DataFrame",
    *,
    scenario: str,
    corruption: str,
    severity: int | None,
    eps_true: float | None,
    dataset: str = "rw_cifar10c",
    algorithms: "Sequence[str] | None" = None,
    embedder: str | None = None,
    head_type: str | None = None,
    epsilon: float | None = None,
    gamma: float | None = None,
    x_col: str = "eps_true",
) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd

    if agg_df is None or getattr(agg_df, "empty", True):
        return agg_df.copy()

    df = agg_df.copy()

    if "dataset" not in df.columns:
        raise KeyError("Expected column 'dataset' in agg_df")
    df = df[df["dataset"].astype(str).str.lower() == str(dataset).lower()].copy()

    def _pick(*cands: str) -> str | None:
        for c in cands:
            if c in df.columns:
                return c
        return None

    scen_col = _pick("scenario", "rw_scenario")
    corr_col = _pick("corruption", "rw_corruption")
    sev_col = _pick("severity", "rw_severity")
    eps_true_col = _pick("eps_true", "rw_eps_true")

    if scen_col is None:
        raise KeyError("Expected column 'scenario' (or 'rw_scenario') in agg_df for CIFAR-10-C plots")
    if corr_col is None:
        raise KeyError("Expected column 'corruption' (or 'rw_corruption') in agg_df for CIFAR-10-C plots")
    if sev_col is None:
        raise KeyError("Expected column 'severity' (or 'rw_severity') in agg_df for CIFAR-10-C plots")
    if eps_true_col is None:
        raise KeyError("Expected column 'eps_true' (or 'rw_eps_true') in agg_df for CIFAR-10-C plots")

    df = df[df[scen_col].astype(str).str.lower() == str(scenario).lower()]
    df = df[df[corr_col].astype(str) == str(corruption)]

    x_norm = str(x_col).strip().lower()
    if x_norm in ("eps_true", "rw_eps_true"):
        if severity is None:
            raise ValueError("When plotting vs eps_true you must provide severity=...")
        df = df[df[sev_col].astype(int) == int(severity)]
    elif x_norm in ("severity", "rw_severity"):
        if eps_true is None:
            raise ValueError("When plotting vs severity you must provide eps_true=...")
        df = df[np.isclose(pd.to_numeric(df[eps_true_col], errors="coerce").astype(float), float(eps_true))]
    else:
        raise ValueError(f"Unsupported x_col={x_col!r} (use 'eps_true' or 'severity')")

    if algorithms is not None:
        df = df[df["algorithm"].isin(list(algorithms))]

    if embedder is not None and "embedder" in df.columns:
        df = df[df["embedder"].astype(str) == str(embedder)]
    if head_type is not None and "head_type" in df.columns:
        df = df[df["head_type"].astype(str) == str(head_type)]

    if epsilon is not None and "epsilon" in df.columns:
        df = df[np.isclose(pd.to_numeric(df["epsilon"], errors="coerce").astype(float), float(epsilon))]
    if gamma is not None and "gamma" in df.columns:
        df = df[np.isclose(pd.to_numeric(df["gamma"], errors="coerce").astype(float), float(gamma))]

    return df


def plot_cifar10c_metrics(
    agg_df: "pd.DataFrame",
    *,
    metrics: "Sequence[str]",
    metric_labels: "Sequence[str] | None" = None,
    scenario: str = "S1",
    corruption: str = "gaussian_noise",
    severity: int | None = 3,
    eps_true: float | None = None,
    dataset: str = "rw_cifar10c",
    algorithms: "Sequence[str] | None" = None,
    embedder: str | None = None,
    head_type: str | None = None,
    epsilon: float | None = None,
    gamma: float | None = None,
    x_col: str = "eps_true",
    out_dir: "Path | None" = None,
    filename: str | None = None,
    title: str | None = None,
    ylabel: str = "Accuracy",
) -> "Path | None":
    """
    CIFAR-10-C plotting with *facet subplots* by (epsilon, gamma).

    `metrics` are per-run metric keys; aggregated columns are expected as:
      - f"{metric}_mean" and (optionally) f"{metric}_std"
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from pathlib import Path

    metrics = list(metrics)
    if metric_labels is None:
        metric_labels = list(metrics)
    metric_labels = list(metric_labels)
    if len(metric_labels) != len(metrics):
        raise ValueError("metric_labels must have the same length as metrics")

    df = _filter_cifar10c_agg(
        agg_df,
        scenario=scenario,
        corruption=corruption,
        severity=severity,
        eps_true=eps_true,
        dataset=dataset,
        algorithms=algorithms,
        embedder=embedder,
        head_type=head_type,
        epsilon=epsilon,
        gamma=gamma,
        x_col=x_col,
    )
    if df.empty:
        print("[plot_cifar10c_metrics] No rows after filtering; nothing to plot.")
        return None

    def _pick(*cands: str) -> str | None:
        for c in cands:
            if c in df.columns:
                return c
        return None

    x_norm = str(x_col).strip().lower()
    if x_norm in ("eps_true", "rw_eps_true"):
        x_key = _pick("eps_true", "rw_eps_true")
        x_label = "True corruption fraction ε_true"
    elif x_norm in ("severity", "rw_severity"):
        x_key = _pick("severity", "rw_severity")
        x_label = "Severity"
    else:
        raise ValueError(f"Unsupported x_col={x_col!r} (use 'eps_true' or 'severity')")
    if x_key is None:
        raise KeyError(f"Could not find the x-axis column for x_col={x_col!r} in agg_df")

    # Resolve mean/std columns for each metric
    metric_cols: dict[str, tuple[str, str | None]] = {}
    for m in metrics:
        if f"{m}_mean" in df.columns:
            mean_col = f"{m}_mean"
            std_col = f"{m}_std" if f"{m}_std" in df.columns else None
        elif m in df.columns:
            mean_col = m
            std_guess = (m[:-5] + "_std") if m.endswith("_mean") else None
            std_col = std_guess if (std_guess is not None and std_guess in df.columns) else None
        else:
            raise KeyError(
                f"Missing metric column for {m!r}. Expected either {m!r} or {m + '_mean'!r} in agg_df."
            )
        metric_cols[m] = (mean_col, std_col)

    # Facet by epsilon and gamma if present
    if "epsilon" in df.columns:
        eps_levels = sorted(pd.to_numeric(df["epsilon"], errors="coerce").dropna().unique().astype(float).tolist())
    else:
        eps_levels = [None]

    if "gamma" in df.columns:
        gam_levels = sorted(pd.to_numeric(df["gamma"], errors="coerce").dropna().unique().astype(float).tolist())
    else:
        gam_levels = [None]

    nrows, ncols = len(eps_levels), len(gam_levels)

    # Algorithms (one colour each)
    algs = sorted(df["algorithm"].astype(str).unique().tolist())
    n_algs = len(algs)
    cmap_name = "tab20" if n_algs <= 20 else "hsv"
    cmap = plt.get_cmap(cmap_name)
    algo_color = {a: cmap(i / max(n_algs - 1, 1)) for i, a in enumerate(algs)}

    linestyles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "v", "D", "x", "+"]

    fig_w = max(7.8, 3.0 * ncols)
    fig_h = max(5.0, 2.6 * nrows)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    # Panels
    for i, epsv in enumerate(eps_levels):
        for j, gamv in enumerate(gam_levels):
            ax = axes[i][j]

            sub = df
            if epsv is not None and "epsilon" in sub.columns:
                sub = sub[np.isclose(pd.to_numeric(sub["epsilon"], errors="coerce").astype(float), float(epsv))]
            if gamv is not None and "gamma" in sub.columns:
                sub = sub[np.isclose(pd.to_numeric(sub["gamma"], errors="coerce").astype(float), float(gamv))]

            if sub.empty:
                ax.text(0.5, 0.5, "No runs", ha="center", va="center", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                for a in algs:
                    sa = sub[sub["algorithm"].astype(str) == a]
                    if sa.empty:
                        continue
                    sa = sa.sort_values(x_key)
                    x = pd.to_numeric(sa[x_key], errors="coerce").to_numpy(dtype=float)

                    for k, m in enumerate(metrics):
                        mean_col, std_col = metric_cols[m]
                        y = pd.to_numeric(sa[mean_col], errors="coerce").to_numpy(dtype=float)
                        yerr = (
                            pd.to_numeric(sa[std_col], errors="coerce").to_numpy(dtype=float)
                            if std_col is not None
                            else None
                        )
                        ax.errorbar(
                            x,
                            y,
                            yerr=yerr,
                            color=algo_color[a],
                            linestyle=linestyles[k % len(linestyles)],
                            marker=markers[k % len(markers)],
                            markersize=3.5,
                            linewidth=1.5,
                            capsize=2,
                            alpha=0.9,
                        )

                ax.grid(True, alpha=0.25)

            facet_title = []
            if epsv is not None:
                facet_title.append(f"ε={float(epsv):g}")
            if gamv is not None:
                facet_title.append(f"γ={float(gamv):g}")
            ax.set_title(", ".join(facet_title), fontsize=9)

            if i == nrows - 1:
                ax.set_xlabel(x_label)
            if j == 0:
                ax.set_ylabel(str(ylabel))

    # Title
    if title is None:
        if x_norm in ("eps_true", "rw_eps_true"):
            title = f"CIFAR-10-C ({scenario}, {corruption}, severity={int(severity)})"
        else:
            title = f"CIFAR-10-C ({scenario}, {corruption}, ε_true={float(eps_true):g})"
    fig.suptitle(title, y=0.995, fontsize=12)

    # Legends (figure-level)
    top_pad = 0.90
    if len(metrics) > 1:
        metric_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=linestyles[k % len(linestyles)],
                marker=markers[k % len(markers)],
                linewidth=1.8,
                markersize=5,
                label=str(lab),
            )
            for k, lab in enumerate(metric_labels)
        ]
        fig.legend(
            handles=metric_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=len(metric_handles),
            fontsize=9,
            frameon=False,
        )
        top_pad = 0.87

    alg_handles = [
        Line2D([0], [0], color=algo_color[a], linestyle="-", marker="o", linewidth=1.8, markersize=5, label=str(a))
        for a in algs
    ]
    if alg_handles:
        ncol_alg = min(6, max(1, int(np.ceil(len(algs) / 2))))
        nrow_alg = int(np.ceil(len(algs) / ncol_alg))
        bottom_pad = 0.08 + 0.035 * max(0, nrow_alg - 1)

        fig.legend(
            handles=alg_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=ncol_alg,
            fontsize=8,
            frameon=False,
        )
    else:
        bottom_pad = 0.08

    fig.tight_layout(rect=[0.02, bottom_pad, 0.98, top_pad])

    if out_dir is None:
        out_dir = Path("figures")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        tag = f"cifar10c__{str(scenario).lower()}__{corruption}"
        if x_norm in ("eps_true", "rw_eps_true"):
            tag += f"__sev{int(severity)}"
        else:
            tag += f"__epstrue{str(float(eps_true)).replace('.', 'p')}"
        metric_tag = "__".join([m.replace("/", "_") for m in metrics])
        filename = f"{tag}__x-{x_norm}__facet_eps_gamma__{metric_tag}.pdf"

    out_path = out_dir / filename
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_cifar10c_metrics] wrote {out_path}")
    return out_path


def plot_cifar10c_mix_acc(
    agg_df: "pd.DataFrame",
    *,
    scenario: str = "S1",
    corruption: str = "gaussian_noise",
    severity: int = 3,
    algorithms: "Sequence[str] | None" = None,
    embedder: str | None = None,
    head_type: str | None = None,
    epsilon: float | None = None,
    gamma: float | None = None,
    out_dir: "Path | None" = None,
) -> "Path | None":
    return plot_cifar10c_metrics(
        agg_df,
        metrics=["test_mix_acc_mean"],
        metric_labels=["mix"],
        scenario=scenario,
        corruption=corruption,
        severity=severity,
        algorithms=algorithms,
        embedder=embedder,
        head_type=head_type,
        epsilon=epsilon,
        gamma=gamma,
        out_dir=out_dir,
        ylabel="Accuracy",
        title="CIFAR-10-C: mixture accuracy",
    )


def plot_cifar10c_clean_vs_corrupt(
    agg_df: "pd.DataFrame",
    *,
    scenario: str = "S1",
    corruption: str = "gaussian_noise",
    severity: int = 3,
    algorithms: "Sequence[str] | None" = None,
    embedder: str | None = None,
    head_type: str | None = None,
    epsilon: float | None = None,
    gamma: float | None = None,
    out_dir: "Path | None" = None,
) -> "Path | None":
    return plot_cifar10c_metrics(
        agg_df,
        metrics=["test_clean_acc_mean", "test_corrupt_acc_mean"],
        metric_labels=["clean", "corrupted"],
        scenario=scenario,
        corruption=corruption,
        severity=severity,
        algorithms=algorithms,
        embedder=embedder,
        head_type=head_type,
        epsilon=epsilon,
        gamma=gamma,
        out_dir=out_dir,
        ylabel="Accuracy",
        title="CIFAR-10-C: clean vs corrupted components",
    )


def plot_cifar10c_inbulk_vs_outbulk(
    agg_df: "pd.DataFrame",
    *,
    scenario: str = "S1",
    corruption: str = "gaussian_noise",
    severity: int = 3,
    algorithms: "Sequence[str] | None" = None,
    embedder: str | None = None,
    head_type: str | None = None,
    epsilon: float | None = None,
    gamma: float | None = None,
    out_dir: "Path | None" = None,
) -> "Path | None":
    return plot_cifar10c_metrics(
        agg_df,
        metrics=["test_inbulk_acc_mean", "test_outbulk_acc_mean"],
        metric_labels=["in-bulk", "out-of-bulk"],
        scenario=scenario,
        corruption=corruption,
        severity=severity,
        algorithms=algorithms,
        embedder=embedder,
        head_type=head_type,
        epsilon=epsilon,
        gamma=gamma,
        out_dir=out_dir,
        ylabel="Accuracy",
        title="CIFAR-10-C: in-bulk vs out-of-bulk accuracy",
    )


def plot_cifar10c_worst_component(
    agg_df: "pd.DataFrame",
    *,
    scenario: str = "S1",
    corruption: str = "gaussian_noise",
    severity: int = 3,
    algorithms: "Sequence[str] | None" = None,
    embedder: str | None = None,
    head_type: str | None = None,
    epsilon: float | None = None,
    gamma: float | None = None,
    out_dir: "Path | None" = None,
) -> "Path | None":
    return plot_cifar10c_metrics(
        agg_df,
        metrics=["test_worst_component_acc"],
        metric_labels=["worst component"],
        scenario=scenario,
        corruption=corruption,
        severity=severity,
        algorithms=algorithms,
        embedder=embedder,
        head_type=head_type,
        epsilon=epsilon,
        gamma=gamma,
        out_dir=out_dir,
        ylabel="Accuracy",
        title="CIFAR-10-C: worst component accuracy",
    )


def summarise_and_plot_runtimes(
    results_path,
    out_dir,
    *,
    algorithm_order=None,
    dataset_filter=None,
    runtime_prefix: str = "runtime_",
):
    """
    Aggregate and plot runtime breakdowns from experiment records.

    Expected input:
      - JSONL file (one dict per line) OR a directory containing .jsonl files.
      - Records should include at least: 'dataset', 'algorithm', and runtime fields like
        'runtime_total_s', 'runtime_train_s', etc. (as produced by runner.py).

    Outputs:
      - CSV summary tables in out_dir
      - PNG plots (boxplots + stacked-bar stage breakdowns) in out_dir
    """
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    results_path = Path(results_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load records ----
    files = []
    if results_path.is_dir():
        files = sorted(results_path.rglob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files found under: {results_path}")
    else:
        files = [results_path]

    rows = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"No records loaded from: {results_path}")

    df = pd.DataFrame(rows)

    # ---- Basic filtering ----
    if dataset_filter is not None:
        df = df[df["dataset"].astype(str) == str(dataset_filter)].copy()

    # ---- Identify runtime columns ----
    runtime_cols = [c for c in df.columns if isinstance(c, str) and c.startswith(runtime_prefix) and c.endswith("_s")]
    if "runtime_total_s" in df.columns and "runtime_total_s" not in runtime_cols:
        runtime_cols.append("runtime_total_s")

    if not runtime_cols:
        raise ValueError(
            "No runtime columns found. Expected columns like 'runtime_total_s' or "
            f"'{runtime_prefix}*_*_s'."
        )

    for c in runtime_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- Algorithm order ----
    if algorithm_order is None:
        algs = sorted(df["algorithm"].astype(str).unique().tolist())
    else:
        algs = [a for a in algorithm_order if a in set(df["algorithm"].astype(str))]
        # Append any remaining algorithms not listed
        for a in sorted(df["algorithm"].astype(str).unique().tolist()):
            if a not in algs:
                algs.append(a)

    # ---- Stage ordering for breakdown plots ----
    stage_order = [
        "runtime_get_embedder_s",
        "runtime_prepare_splits_s",
        "runtime_make_cifar10c_splits_s",
        "runtime_make_sanity_splits_s",
        "runtime_ensure_image_cache_s",
        "runtime_ensure_text_score_cache_s",
        "runtime_load_cached_arrays_s",
        "runtime_load_scores_s",
        "runtime_load_true_scores_s",
        "runtime_build_score_s",
        "runtime_fit_gaussian_s",
        "runtime_bulk_select_s",
        "runtime_trueclass_scores_s",
        "runtime_class_thresholds_s",
        "runtime_lv_bas_mc_fit_gaussian_s",
        "runtime_lv_bas_mc_calibrate_tau_s",
        "runtime_lv_bas_mc_center_sampling_s",
        "runtime_center_sampling_s",
        "runtime_build_dataloaders_s",
        "runtime_tune_s",
        "runtime_train_s",
        "runtime_eval_val_s",
        "runtime_eval_test_s",
        "runtime_eval_test_inbulk_s",
    ]
    stage_cols = [c for c in stage_order if c in df.columns]
    # Add any additional runtime columns (except total) not in stage_order
    extra_stage_cols = [c for c in runtime_cols if c not in stage_cols and c != "runtime_total_s"]
    stage_cols = stage_cols + sorted(extra_stage_cols)

    # ---- Summary tables ----
    group_cols = ["dataset", "algorithm"]
    summary = (
        df.groupby(group_cols, dropna=False)[runtime_cols]
        .agg(["count", "median", "mean", "std"])
        .reset_index()
    )
    summary.to_csv(out_dir / "runtime_summary_by_dataset_algorithm.csv", index=False)

    # Also save a simpler median-only table for stage breakdowns
    median_stages = df.groupby(group_cols, dropna=False)[stage_cols + ["runtime_total_s"]].median(numeric_only=True).reset_index()
    median_stages.to_csv(out_dir / "runtime_median_stages_by_dataset_algorithm.csv", index=False)

    # ---- Plots ----
    datasets = sorted(df["dataset"].astype(str).unique().tolist())
    for ds in datasets:
        dfd = df[df["dataset"].astype(str) == ds].copy()
        if dfd.empty:
            continue

        # Boxplot: total runtime
        if "runtime_total_s" in dfd.columns:
            data = []
            labels = []
            for a in algs:
                x = dfd[dfd["algorithm"].astype(str) == a]["runtime_total_s"].dropna().values
                if x.size == 0:
                    continue
                data.append(x)
                labels.append(a)

            if data:
                fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels)), 4))
                ax.boxplot(data, labels=labels, showfliers=False)
                ax.set_ylabel("Runtime (s)")
                ax.set_title(f"{ds}: total runtime")
                ax.tick_params(axis="x", rotation=30)
                fig.tight_layout()
                fig.savefig(out_dir / f"runtime_total_boxplot__{ds}.png", dpi=200)
                plt.close(fig)

        # Boxplot: training runtime
        if "runtime_train_s" in dfd.columns:
            data = []
            labels = []
            for a in algs:
                x = dfd[dfd["algorithm"].astype(str) == a]["runtime_train_s"].dropna().values
                if x.size == 0:
                    continue
                data.append(x)
                labels.append(a)

            if data:
                fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels)), 4))
                ax.boxplot(data, labels=labels, showfliers=False)
                ax.set_ylabel("Runtime (s)")
                ax.set_title(f"{ds}: training runtime")
                ax.tick_params(axis="x", rotation=30)
                fig.tight_layout()
                fig.savefig(out_dir / f"runtime_train_boxplot__{ds}.png", dpi=200)
                plt.close(fig)

        # Stacked bar: median stage breakdown
        if stage_cols:
            med = (
                dfd.groupby("algorithm", dropna=False)[stage_cols]
                .median(numeric_only=True)
                .reindex(algs)
                .fillna(0.0)
            )
            med = med.loc[~med.index.isna()]
            med = med[med.sum(axis=1) > 0]

            if not med.empty:
                fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(med.index)), 5))
                bottom = np.zeros((len(med.index),), dtype=float)
                x = np.arange(len(med.index))

                for col in stage_cols:
                    vals = med[col].values.astype(float)
                    if np.all(vals == 0):
                        continue
                    ax.bar(x, vals, bottom=bottom, label=col.replace("runtime_", "").replace("_s", ""))
                    bottom = bottom + vals

                ax.set_xticks(x)
                ax.set_xticklabels(med.index.tolist(), rotation=30, ha="right")
                ax.set_ylabel("Median runtime (s)")
                ax.set_title(f"{ds}: median stage breakdown")
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)
                fig.tight_layout()
                fig.savefig(out_dir / f"runtime_stage_breakdown_median__{ds}.png", dpi=200)
                plt.close(fig)

    return summary, median_stages

def _safe_part(x: Any) -> str:
    s = str(x)
    s = s.replace("/", "_").replace(" ", "_").replace(":", "_")
    return s


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multiindex columns after groupby-agg."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    df = df.copy()
    df.columns = [
        "_".join([str(c) for c in tup if str(c) != ""])
        for tup in df.columns.to_list()
    ]
    return df


def extract_train_stats_df(raw_df: pd.DataFrame, dataset: Optional[str] = None) -> pd.DataFrame:
    """
    Pull out train_* + key identifiers from the raw results dataframe.
    Adds derived normalised-compute columns when possible.
    """
    df = raw_df.copy()

    if dataset is not None and "dataset" in df.columns:
        df = df[df["dataset"].astype(str).str.lower() == dataset.lower()].copy()

    train_cols = [c for c in df.columns if c.startswith("train_")]
    if not train_cols:
        # Nothing to do; caller should handle empty df.
        return pd.DataFrame()

    # Keep some identifiers if they exist
    base_cols = [
        c for c in [
            "uuid", "replication",
            "dataset", "algorithm",
            "epsilon", "gamma",
            "rw_split_seed", "rw_cal_fraction",
            "rw_scenario", "rw_eps_true", "rw_corruption", "rw_severity",
        ]
        if c in df.columns
    ]

    runtime_cols = [c for c in ["runtime_train_s", "runtime_total_s"] if c in df.columns]

    out = df[base_cols + train_cols + runtime_cols].copy()

    # Derived compute normalisations
    if "runtime_train_s" in out.columns and "train_steps" in out.columns:
        denom = out["train_steps"].replace(0, np.nan)
        out["train_runtime_per_step_s"] = out["runtime_train_s"] / denom

    if "runtime_train_s" in out.columns and "train_examples" in out.columns:
        denom = out["train_examples"].replace(0, np.nan)
        out["train_runtime_per_example_s"] = out["runtime_train_s"] / denom

    if "train_firstbatch_obj_minus_eps0" in out.columns:
        out["train_firstbatch_obj_minus_eps0_abs"] = out["train_firstbatch_obj_minus_eps0"].abs()

    # LV-BAS-Bin internal consistency ratio (should be ~1 when eps>0 and sup!=exp)
    if all(c in out.columns for c in ["train_firstbatch_obj_minus_eps0", "train_firstbatch_sup_minus_exp"]):
        # Prefer the logged train_epsilon if present; else fall back to the experiment epsilon.
        eps_col = "train_epsilon" if "train_epsilon" in out.columns else ("epsilon" if "epsilon" in out.columns else None)
        if eps_col is not None:
            eps = out[eps_col].astype(float)
            denom = eps * out["train_firstbatch_sup_minus_exp"].astype(float)
            denom = denom.replace(0.0, np.nan)
            out["train_lv_bas_bin_ratio"] = out["train_firstbatch_obj_minus_eps0"].astype(float) / denom

    return out


def summarise_train_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates per-(dataset, algorithm, gamma, epsilon) if present.
    Returns a flat-column dataframe.
    """
    if train_df.empty:
        return pd.DataFrame()

    group_cols = [c for c in ["dataset", "algorithm", "gamma", "epsilon"] if c in train_df.columns]
    if not group_cols:
        group_cols = [c for c in ["algorithm"] if c in train_df.columns]

    # Only summarise numeric columns that are actually informative here
    cand = [
        "train_steps",
        "train_examples",
        "runtime_train_s",
        "train_runtime_per_step_s",
        "train_runtime_per_example_s",
        "train_firstbatch_erm",
        "train_firstbatch_obj",
        "train_firstbatch_obj_eps0",
        "train_firstbatch_obj_minus_eps0",
        "train_firstbatch_obj_minus_eps0_abs",
        "train_firstbatch_sup_minus_exp",
        "train_lv_bas_bin_ratio",
    ]
    num_cols = [c for c in cand if c in train_df.columns]

    agg = (
        train_df
        .groupby(group_cols, dropna=False)[num_cols]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    agg = _flatten_columns(agg)
    return agg


def _boxplot_by_algorithm(
    df: pd.DataFrame,
    value_col: str,
    *,
    out_path: Path,
    title: str,
) -> None:
    if df.empty or value_col not in df.columns or "algorithm" not in df.columns:
        return

    # Drop NaNs
    d = df[["algorithm", value_col]].dropna().copy()
    if d.empty:
        return

    algs = sorted(d["algorithm"].astype(str).unique().tolist())
    data = [d.loc[d["algorithm"] == a, value_col].astype(float).values for a in algs]

    plt.figure(figsize=(max(8, 0.6 * len(algs)), 4.5))
    plt.boxplot(data, labels=algs, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(value_col)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _lineplot_eps_effect_panels_by_gamma(
    df: pd.DataFrame,
    y_col: str,
    *,
    out_path: Path,
    title: str,
    only_eps_used: bool = True,
) -> None:
    """
    Plot median y_col vs epsilon with separate panels for gamma (if present),
    and separate lines for each algorithm.
    """
    if df.empty or "algorithm" not in df.columns or "epsilon" not in df.columns or y_col not in df.columns:
        return

    d = df.copy()
    d["epsilon"] = d["epsilon"].astype(float)

    if only_eps_used and "train_epsilon_used" in d.columns:
        d = d[d["train_epsilon_used"].astype(bool)]

    if d.empty:
        return

    gammas = sorted(d["gamma"].astype(float).unique().tolist()) if "gamma" in d.columns else [None]
    n_panels = len(gammas)
    fig_w = max(8, 4.0 * n_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 4.0), sharey=True)

    if n_panels == 1:
        axes = [axes]

    for ax, g in zip(axes, gammas):
        dg = d if g is None else d[np.isclose(d["gamma"].astype(float), float(g))]

        for alg in sorted(dg["algorithm"].astype(str).unique().tolist()):
            da = dg[dg["algorithm"].astype(str) == alg]
            if da.empty:
                continue
            # median across replications at each epsilon
            agg = (
                da.groupby("epsilon", dropna=False)[y_col]
                .median()
                .reset_index()
                .sort_values("epsilon")
            )
            ax.plot(agg["epsilon"].values, agg[y_col].values, marker="o", label=alg)

        ax.set_xlabel("epsilon")
        ax.set_ylabel(y_col)
        if g is None:
            ax.set_title("all gammas")
        else:
            ax.set_title(f"gamma={float(g):g}")
        ax.grid(True, alpha=0.3)

    # Put legend on the last axis to avoid clutter
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def summarise_and_plot_train_stats(
    raw_df: pd.DataFrame,
    *,
    out_dir: Path,
    dataset: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end:
      - Extract train_* columns from raw results
      - Write train_stats_raw.csv + train_stats_agg.csv
      - Plot normalised compute + epsilon-effect diagnostics

    Returns: (train_raw_df, train_agg_df)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = extract_train_stats_df(raw_df, dataset=dataset)
    if train_df.empty:
        print("[train-stats] No train_* columns found. "
              "This usually means runner.py is not passing train_stats into train_head "
              "and/or the outputs are not written into results.csv.")
        return pd.DataFrame(), pd.DataFrame()

    raw_csv = out_dir / "train_stats_raw.csv"
    train_df.to_csv(raw_csv, index=False)
    print("[train-stats] wrote", raw_csv)

    agg_df = summarise_train_stats(train_df)
    agg_csv = out_dir / "train_stats_agg.csv"
    agg_df.to_csv(agg_csv, index=False)
    print("[train-stats] wrote", agg_csv)

    # If multiple datasets are present, make per-dataset plots for readability
    if "dataset" in train_df.columns and dataset is None:
        dataset_vals = sorted(train_df["dataset"].astype(str).unique().tolist())
    else:
        dataset_vals = [dataset if dataset is not None else "all"]

    for ds in dataset_vals:
        if "dataset" in train_df.columns and ds != "all":
            d = train_df[train_df["dataset"].astype(str) == str(ds)].copy()
            stem = _safe_part(ds)
        else:
            d = train_df.copy()
            stem = "all"

        if d.empty:
            continue

        # ---- Normalised compute plots ----
        if "train_runtime_per_step_s" in d.columns:
            _boxplot_by_algorithm(
                d, "train_runtime_per_step_s",
                out_path=out_dir / f"{stem}__train_runtime_per_step_s_boxplot.pdf",
                title=f"{stem}: train runtime per optimiser step (s)",
            )

        if "train_runtime_per_example_s" in d.columns:
            _boxplot_by_algorithm(
                d, "train_runtime_per_example_s",
                out_path=out_dir / f"{stem}__train_runtime_per_example_s_boxplot.pdf",
                title=f"{stem}: train runtime per example (s)",
            )

        if "train_steps" in d.columns:
            _boxplot_by_algorithm(
                d, "train_steps",
                out_path=out_dir / f"{stem}__train_steps_boxplot.pdf",
                title=f"{stem}: optimiser steps (train_steps)",
            )

        if "train_examples" in d.columns:
            _boxplot_by_algorithm(
                d, "train_examples",
                out_path=out_dir / f"{stem}__train_examples_boxplot.pdf",
                title=f"{stem}: examples processed (train_examples)",
            )

        # ---- Epsilon actually affects objective? ----
        if "train_firstbatch_obj_minus_eps0_abs" in d.columns:
            _lineplot_eps_effect_panels_by_gamma(
                d,
                "train_firstbatch_obj_minus_eps0_abs",
                out_path=out_dir / f"{stem}__train_abs_obj_minus_eps0_vs_epsilon_by_gamma.pdf",
                title=f"{stem}: |first-batch(obj - obj_eps0)| vs epsilon (eps-dependent algs)",
                only_eps_used=True,
            )

        # ---- LV-BAS-Bin specific diagnostics ----
        if "train_firstbatch_sup_minus_exp" in d.columns:
            # Plot this for *all* algorithms; only LV-BAS-Bin will have non-NaNs.
            _lineplot_eps_effect_panels_by_gamma(
                d,
                "train_firstbatch_sup_minus_exp",
                out_path=out_dir / f"{stem}__lvbas_bin_sup_minus_exp_vs_epsilon_by_gamma.pdf",
                title=f"{stem}: LV-BAS-Bin (sup - exp) vs epsilon",
                only_eps_used=False,
            )

        if "train_lv_bas_bin_ratio" in d.columns:
            _boxplot_by_algorithm(
                d, "train_lv_bas_bin_ratio",
                out_path=out_dir / f"{stem}__lvbas_bin_ratio_boxplot.pdf",
                title=f"{stem}: LV-BAS-Bin ratio = (obj-exp) / (eps*(sup-exp)) (should be ~1)",
            )

    return train_df, agg_df

# ---------------------------------------------------------------------------
# Sample-efficiency utilities (Monte Carlo budgets)
# ---------------------------------------------------------------------------

def add_num_total_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a derived integer column:
        num_total_samples := num_posterior_samples * num_likelihood_samples

    This matches your requested definition:
      - lv_bas / kl_pp: num_posterior_samples == 1 so product = num_likelihood_samples
      - kl_bdro: product is the meaningful "total model samples" proxy

    If the sampling columns are missing, returns NaN in num_total_samples.
    """
    out = df.copy()

    if "num_likelihood_samples" in out.columns:
        out["num_likelihood_samples"] = pd.to_numeric(out["num_likelihood_samples"], errors="coerce")
    else:
        out["num_likelihood_samples"] = np.nan

    if "num_posterior_samples" in out.columns:
        out["num_posterior_samples"] = pd.to_numeric(out["num_posterior_samples"], errors="coerce")
    else:
        out["num_posterior_samples"] = 1.0

    out["num_total_samples"] = out["num_likelihood_samples"].fillna(1.0) * out["num_posterior_samples"].fillna(1.0)
    out["num_total_samples"] = pd.to_numeric(out["num_total_samples"], errors="coerce")

    # Use nullable Int64 for clean group keys / labels.
    out["num_total_samples"] = out["num_total_samples"].round().astype("Int64")
    return out


def summarise_vs_epsilon_and_total_samples(
    df: pd.DataFrame,
    *,
    gamma: float = 0.0,
    t_scale: float = LV_NEWSVENDOR_T_SCALE,
    convert_kl_eps_to_lv_scale: bool = True,
    epsilon_round: int = 12,
) -> pd.DataFrame:
    """
    Like summarise_vs_epsilon(...), but additionally groups by num_total_samples.

    Output columns include:
      - oos_mean, oos_var (means across replications)
      - msd_mean, msd_sd, msd_ci_low, msd_ci_high
      - num_total_samples
      - epsilon_plot (LV epsilon scale if convert_kl_eps_to_lv_scale=True)
    """
    required = {"algorithm", "epsilon", "out_of_sample_mean", "out_of_sample_var"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for sample-efficiency summary: {sorted(missing)}")

    out = df.copy()
    out["epsilon"] = pd.to_numeric(out["epsilon"], errors="coerce")
    out["out_of_sample_mean"] = pd.to_numeric(out["out_of_sample_mean"], errors="coerce")
    out["out_of_sample_var"] = pd.to_numeric(out["out_of_sample_var"], errors="coerce")
    out = out.dropna(subset=["epsilon", "out_of_sample_mean", "out_of_sample_var", "algorithm"])

    out = add_num_total_samples(out)
    out = out.dropna(subset=["num_total_samples"])
    out["num_total_samples"] = out["num_total_samples"].astype(int)

    # Derived per-replication metrics
    var_clip = np.maximum(out["out_of_sample_var"].to_numpy(dtype=float), 0.0)
    out["ce_gamma"] = out["out_of_sample_mean"] - 0.5 * float(gamma) * var_clip
    out["SR"] = out["out_of_sample_mean"] / np.sqrt(var_clip + 1e-18)
    out["MSD"] = 0.5 * (out["out_of_sample_mean"] + np.sqrt(var_clip))

    # Common x-axis for plotting (LV epsilon scale).
    out["epsilon_plot"] = out["epsilon"].astype(float)
    # if convert_kl_eps_to_lv_scale:
    #     is_kl = out["algorithm"].astype(str).str.startswith("kl_")
    #     if is_kl.any():
    #         out.loc[is_kl, "epsilon_plot"] = _eps_kl_to_lv(
    #             out.loc[is_kl, "epsilon"].to_numpy(),
    #             t_scale=t_scale,
    #         )

    # out["epsilon_plot"] = out["epsilon_plot"].round(epsilon_round)

    summ = (
        out.groupby(["algorithm", "num_total_samples", "epsilon_plot"], dropna=False)
           .agg(
               oos_mean=("out_of_sample_mean", "mean"),
               oos_mean_sd=("out_of_sample_mean", "std"),
               oos_var=("out_of_sample_var", "mean"),
               oos_var_sd=("out_of_sample_var", "std"),
               ce_mean=("ce_gamma", "mean"),
               ce_sd=("ce_gamma", "std"),
               sr_mean=("SR", "mean"),
               sr_sd=("SR", "std"),
               msd_mean=("MSD", "mean"),
               msd_sd=("MSD", "std"),
               n=("MSD", "count"),
           )
           .reset_index()
           .sort_values(["algorithm", "num_total_samples", "epsilon_plot"])
    )

    # 95% CI across replications (normal approximation)
    n_eff = summ["n"].clip(lower=1).astype(float)
    summ["msd_se"] = summ["msd_sd"] / np.sqrt(n_eff)
    summ["msd_ci_low"] = summ["msd_mean"] - 1.96 * summ["msd_se"]
    summ["msd_ci_high"] = summ["msd_mean"] + 1.96 * summ["msd_se"]

    return summ


def sample_efficiency_summary(
    df: pd.DataFrame,
    *,
    t_scale: float = LV_NEWSVENDOR_T_SCALE,
    convert_kl_eps_to_lv_scale: bool = True,
    epsilon_round: int = 12,
) -> pd.DataFrame:
    """
    Produce a compact "sample efficiency" summary per (algorithm, num_total_samples).

    Principle:
      For each (algorithm, epsilon_plot), treat the largest available num_total_samples as
      a reference "best-sampled" estimate. Then measure how far smaller budgets are from it.

    Summary stats per (algorithm, num_total_samples):
      - mean_abs_delta_msd: mean over epsilon_plot of |MSD(m) - MSD(m_max)|
      - max_abs_delta_msd:  max over epsilon_plot of |MSD(m) - MSD(m_max)|
      - mean_rel_delta_msd: mean over epsilon_plot of |ΔMSD| / (|MSD_ref| + 1e-12)
      - mean_abs_delta_mean / var: analogous, on OOS mean and OOS variance
      - mean_msd_sd: average across epsilon_plot of across-replication MSD sd at that budget
    """
    summ = summarise_vs_epsilon_and_total_samples(
        df,
        gamma=0.0,
        t_scale=t_scale,
        convert_kl_eps_to_lv_scale=convert_kl_eps_to_lv_scale,
        epsilon_round=epsilon_round,
    )

    # Reference row = largest num_total_samples for each (algorithm, epsilon_plot)
    idx = summ.groupby(["algorithm", "epsilon_plot"], dropna=False)["num_total_samples"].idxmax()
    ref = (
        summ.loc[idx, ["algorithm", "epsilon_plot", "oos_mean", "oos_var", "msd_mean"]]
            .rename(columns={"oos_mean": "oos_mean_ref", "oos_var": "oos_var_ref", "msd_mean": "msd_ref"})
    )

    merged = summ.merge(ref, on=["algorithm", "epsilon_plot"], how="left")

    merged["abs_delta_msd"] = (merged["msd_mean"] - merged["msd_ref"]).abs()
    merged["abs_delta_mean"] = (merged["oos_mean"] - merged["oos_mean_ref"]).abs()
    merged["abs_delta_var"] = (merged["oos_var"] - merged["oos_var_ref"]).abs()
    merged["rel_delta_msd"] = merged["abs_delta_msd"] / (merged["msd_ref"].abs() + 1e-12)

    se = (
        merged.groupby(["algorithm", "num_total_samples"], dropna=False)
              .agg(
                  mean_abs_delta_msd=("abs_delta_msd", "mean"),
                  max_abs_delta_msd=("abs_delta_msd", "max"),
                  mean_rel_delta_msd=("rel_delta_msd", "mean"),
                  mean_abs_delta_mean=("abs_delta_mean", "mean"),
                  mean_abs_delta_var=("abs_delta_var", "mean"),
                  mean_msd_sd=("msd_sd", "mean"),
                  n_eps=("epsilon_plot", "nunique"),
              )
              .reset_index()
              .sort_values(["algorithm", "num_total_samples"])
    )
    return se


def plot_sample_efficiency(
    df: pd.DataFrame,
    *,
    t_scale: float = LV_NEWSVENDOR_T_SCALE,
    convert_kl_eps_to_lv_scale: bool = True,
    metric: str = "mean_abs_delta_msd",
    logx: bool = True,
    logy: bool = False,
    title: str | None = None,
    out_path: Path | None = None,
    out_csv: Path | None = None,
):
    """
    Combined sample-efficiency plot across algorithms:
        x = num_total_samples
        y = chosen metric from sample_efficiency_summary(...)
    """
    LABEL_FS = 20
    TICK_FS = 20
    LEGEND_FS = 20

    algo_label = {
        "lv_bas": r"$\mathrm{LV}$",
        "kl_bdro": r"$\mathrm{KL\!-\!BDRO}$",
        "kl_empirical": r"$\mathrm{KL\!-\!Empirical}$",
        "kl_pp": r"$\mathrm{KL\!-\!BAS}_{\rm PP}$",
        "or_wdro": r"$\mathrm{OR\!-\!WDRO}$",
        "lv_reverse": r"$\mathrm{Rev\!-\!LV\!-\!BAS}$",
        "tv_ball": r"$\mathrm{TV\!-\!BAS}$",
    }

    # Colourblind-friendly (Okabe–Ito)
    algo_color = {
        "lv_bas": "#000000",        # black
        "kl_bdro": "#0072B2",       # blue
        "kl_empirical": "#E69F00",  # orange
        "kl_pp": "#009E73",         # bluish green
        "or_wdro": "#CC79A7",       # reddish purple
        "lv_reverse": "#D55E00",     # vermillion
        "tv_ball": "#56B4E9",       # sky blue 
    }

    se = sample_efficiency_summary(
        df,
        t_scale=t_scale,
        convert_kl_eps_to_lv_scale=convert_kl_eps_to_lv_scale,
    )

    if metric not in se.columns:
        raise ValueError(f"Unknown metric {metric!r}. Available: {sorted(se.columns)}")

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for algo, g in se.groupby("algorithm", dropna=False):
        g = g.sort_values("num_total_samples")
        x = g["num_total_samples"].to_numpy(dtype=float)
        y = g[metric].to_numpy(dtype=float)
        label = algo_label.get(str(algo), str(algo))
        c = algo_color.get(str(algo), None)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=9,
            linestyle="-",
            linewidth=2.5,
            label=label,
            color=c,
        )

    ax.set_xlabel(r"Total model samples $M_{\mathrm{post}}\times M_{\mathrm{pred}}$", fontsize=LABEL_FS)
    ax.set_ylabel(r"MSD deviation $\Delta_{\mathrm{MSD}}$", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    # if title:
    #     ax.set_title(title, fontsize=LABEL_FS)
    ax.legend(loc="upper right", ncols=2, fontsize=LEGEND_FS)

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    fig.tight_layout()

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        se.to_csv(out_csv, index=False)
        print("Saved sample-efficiency summary to", out_csv)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        print("Saved sample-efficiency plot to", out_path)

    return fig, ax, se


def summarise_vs_epsilon_and_gamma_bulk(
    df: pd.DataFrame,
    *,
    gamma: float = 0.0,
    t_scale: float = LV_NEWSVENDOR_T_SCALE,
    convert_kl_eps_to_lv_scale: bool = True,
    epsilon_round: int = 12,
) -> pd.DataFrame:
    """
    Like summarise_vs_epsilon(...), but additionally groups by gamma_bulk.
    """
    required = {"algorithm", "epsilon", "gamma_bulk", "out_of_sample_mean", "out_of_sample_var"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for gamma-bulk summary: {sorted(missing)}")

    out = df.copy()
    out["epsilon"] = pd.to_numeric(out["epsilon"], errors="coerce")
    out["gamma_bulk"] = pd.to_numeric(out["gamma_bulk"], errors="coerce")
    out["out_of_sample_mean"] = pd.to_numeric(out["out_of_sample_mean"], errors="coerce")
    out["out_of_sample_var"] = pd.to_numeric(out["out_of_sample_var"], errors="coerce")
    out = out.dropna(subset=["algorithm", "epsilon", "gamma_bulk", "out_of_sample_mean", "out_of_sample_var"])

    var_clip = np.maximum(out["out_of_sample_var"].to_numpy(dtype=float), 0.0)
    out["ce_gamma"] = out["out_of_sample_mean"] - 0.5 * float(gamma) * var_clip
    out["SR"] = out["out_of_sample_mean"] / np.sqrt(var_clip + 1e-18)
    out["MSD"] = 0.5 * (out["out_of_sample_mean"] + np.sqrt(var_clip))

    out["epsilon_plot"] = out["epsilon"].astype(float)
    if convert_kl_eps_to_lv_scale:
        is_kl = out["algorithm"].astype(str).str.startswith("kl_")
        if is_kl.any():
            out.loc[is_kl, "epsilon_plot"] = _eps_kl_to_lv(
                out.loc[is_kl, "epsilon"].to_numpy(),
                t_scale=t_scale,
            )

    out["epsilon_plot"] = out["epsilon_plot"].round(epsilon_round)

    summ = (
        out.groupby(["algorithm", "gamma_bulk", "epsilon_plot"], dropna=False)
           .agg(
               oos_mean=("out_of_sample_mean", "mean"),
               oos_mean_sd=("out_of_sample_mean", "std"),
               oos_var=("out_of_sample_var", "mean"),
               oos_var_sd=("out_of_sample_var", "std"),
               ce_mean=("ce_gamma", "mean"),
               ce_sd=("ce_gamma", "std"),
               sr_mean=("SR", "mean"),
               sr_sd=("SR", "std"),
               msd_mean=("MSD", "mean"),
               msd_sd=("MSD", "std"),
               n=("MSD", "count"),
           )
           .reset_index()
           .sort_values(["algorithm", "gamma_bulk", "epsilon_plot"])
    )

    n_eff = summ["n"].clip(lower=1).astype(float)
    summ["msd_se"] = summ["msd_sd"] / np.sqrt(n_eff)
    summ["msd_ci_low"] = summ["msd_mean"] - 1.96 * summ["msd_se"]
    summ["msd_ci_high"] = summ["msd_mean"] + 1.96 * summ["msd_se"]

    return summ


def plot_frontiers_by_gamma_bulk(
    df: pd.DataFrame,
    *,
    algorithm: str = "lv_bas",
    t_scale: float = LV_NEWSVENDOR_T_SCALE,
    convert_kl_eps_to_lv_scale: bool = True,
    title: str | None = None,
    out_path: Path | None = None,
):
    """
    Mean–variance frontier for gamma_bulk sensitivity.

    Solid lines: different gamma_bulk values for `algorithm`.
    Dashed lines: all other baselines present in df.

    Also writes a second zoomed PDF (if out_path is provided) with OOS mean
    restricted to 700--900.
    """
    LABEL_FS = 18
    TICK_FS = 18
    LEGEND_FS = 14
    EPS_LABEL_FS = 10
    ZOOM_YMIN = 700.0
    ZOOM_YMAX = 900.0

    algo_label = {
        "lv_bas": r"$\mathrm{LV}$",
        "kl_bdro": r"$\mathrm{KL\!-\!BDRO}$",
        "kl_empirical": r"$\mathrm{KL\!-\!Empirical}$",
        "kl_pp": r"$\mathrm{KL\!-\!BAS}_{\rm PP}$",
        "or_wdro": r"$\mathrm{OR\!-\!WDRO}$",
        "lv_reverse": r"$\mathrm{Rev\!-\!LV\!-\!BAS}$",
        "tv_ball": r"$\mathrm{TV\!-\!BAS}$",
    }

    algo_color = {
        "lv_bas": "#000000",
        "kl_bdro": "#E15759",
        "kl_empirical": "#E69F00",
        "kl_pp": "#000000",
        "or_wdro": "#CC79A7",
        "lv_reverse": "#D55E00",
        "tv_ball": "#56B4E9",
    }

    gamma_mark_targets = [0.043, 0.05, 0.09, 0.15]
    epsilon_mark_targets = [0.01, 0.1, 0.5, 1.0]
    epsilon_label_offsets = {
        0.01: (6, 6),
        0.1: (6, -14),
        0.2: (6, 6),
        0.5: (6, -14),
        1.0: (-2, 3),
    }

    summ = summarise_vs_epsilon_and_gamma_bulk(
        df,
        gamma=0.0,
        t_scale=t_scale,
        convert_kl_eps_to_lv_scale=convert_kl_eps_to_lv_scale,
    )
    sub = summ[summ["algorithm"].astype(str) == str(algorithm)].copy()
    if sub.empty:
        raise ValueError(f"No rows for algorithm={algorithm!r}")

    gamma_values = sorted(float(v) for v in sub["gamma_bulk"].dropna().unique().tolist())
    n_g = len(gamma_values)

    cmap = plt.get_cmap("viridis")
    xs = np.linspace(0.1, 0.9, n_g) if n_g > 1 else np.array([0.5])
    gamma_colors = [mpl.colors.to_hex(cmap(x)) for x in xs]
    gamma_color = {g: gamma_colors[i] for i, g in enumerate(gamma_values)}

    base_summ = (
        summ[summ["algorithm"].astype(str) != str(algorithm)]
        .groupby(["algorithm", "epsilon_plot"], dropna=False)
        .agg(
            oos_mean=("oos_mean", "mean"),
            oos_var=("oos_var", "mean"),
        )
        .reset_index()
        .sort_values(["algorithm", "epsilon_plot"])
    )

    def _format_eps_label(eps: float) -> str:
        if np.isclose(eps, 1.0, atol=2e-3):
            return "1.0"
        return f"{eps:g}"

    def _mark_selected_eps(
        ax,
        g: pd.DataFrame,
        gamma_val: float,
        color: str,
        y_min: float | None = None,
        y_max: float | None = None,
    ):
        if not any(np.isclose(gamma_val, target, atol=2e-3, rtol=0.0) for target in gamma_mark_targets):
            return

        eps_vals = g["epsilon_plot"].to_numpy(dtype=float)
        x_vals = g["oos_var"].to_numpy(dtype=float)
        y_vals = g["oos_mean"].to_numpy(dtype=float)

        for eps_target in epsilon_mark_targets:
            idx = np.flatnonzero(np.isclose(eps_vals, eps_target, atol=2e-3, rtol=0.0))
            if idx.size == 0:
                continue
            i = int(idx[0])
            x0 = float(x_vals[i])
            y0 = float(y_vals[i])

            if y_min is not None and y0 < y_min:
                continue
            if y_max is not None and y0 > y_max:
                continue

            ax.scatter(
                [x0],
                [y0],
                s=120,
                facecolor=color,
                edgecolor="black",
                linewidth=1.0,
                zorder=6,
            )

            dx, dy = epsilon_label_offsets[eps_target]
            ha = "right" if np.isclose(eps_target, 1.0, atol=2e-3, rtol=0.0) else "left"
            ax.annotate(
                _format_eps_label(eps_target),
                xy=(x0, y0),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=EPS_LABEL_FS,
                color=color,
                ha=ha,
                va="bottom",
                clip_on=False,
                zorder=7,
            )

    def _add_external_legend(fig, ax_source):
        handles, labels = ax_source.get_legend_handles_labels()
        if not handles:
            return
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
            bbox_to_anchor=(0.03, 0.89, 0.94, 0.055),
            ncol=4,
            mode="expand",
            fontsize=LEGEND_FS,
            frameon=False,
            borderaxespad=0.0,
        )

    def _add_zoom_box(ax, y_min: float, y_max: float):
        x_min, x_max = ax.get_xlim()
        rect = mpl.patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor="grey",
            linestyle=":",
            linewidth=1.5,
            zorder=1,
        )
        ax.add_patch(rect)
        ax.set_xlim(x_min, x_max)

    def _add_zoom_callout(fig, ax_left, ax_right, y_min: float, y_max: float):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        x_min, x_max = ax_left.get_xlim()
        y_lo, y_hi = ax_left.get_ylim()

        # Start on the top edge of the zoom box, inside the full plot.
        x_start = 7.0e5
        x_start = min(max(x_start, x_min + 0.02 * (x_max - x_min)), x_max - 0.02 * (x_max - x_min))
        y_start = y_max

        start_disp = ax_left.transData.transform((x_start, y_start))
        start_fig = fig.transFigure.inverted().transform(start_disp)

        # End slightly to the left of the zoomed-panel y-label.
        ylabel_bbox = ax_right.yaxis.label.get_window_extent(renderer=renderer)
        end_disp = (
            ylabel_bbox.x0 - 18.0,
            0.5 * (ylabel_bbox.y0 + ylabel_bbox.y1),
        )
        end_fig = fig.transFigure.inverted().transform(end_disp)

        arrow = mpl.patches.FancyArrowPatch(
            posA=tuple(start_fig),
            posB=tuple(end_fig),
            transform=fig.transFigure,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=-0.28",
            mutation_scale=18,
            linewidth=1.8,
            color="grey",
            zorder=20,
        )
        fig.add_artist(arrow)

        # Put the text inside the full plot, above the arrow curve.
        text_x = x_start - 0.05 * (x_max - x_min)
        text_y = y_start + 0.12 * (y_hi - y_lo)
        ax_left.text(
            text_x,
            text_y,
            "zoomed-in",
            color="grey",
            fontsize=EPS_LABEL_FS + 8,
            ha="center",
            va="bottom",
            zorder=21,
            clip_on=False,
        )

    def _draw_panel(ax, *, y_min: float | None = None, y_max: float | None = None):
        for gamma_bulk, g in sub.groupby("gamma_bulk", dropna=False):
            g = g.sort_values("epsilon_plot")
            gamma_val = float(gamma_bulk)
            color = gamma_color.get(gamma_val, None)

            ax.plot(
                g["oos_var"].to_numpy(dtype=float),
                g["oos_mean"].to_numpy(dtype=float),
                marker="o",
                markersize=9,
                linestyle="-",
                linewidth=2.5,
                label=fr"LV with $\gamma={gamma_val:g}$",
                color=color,
                zorder=3,
            )

            _mark_selected_eps(ax, g, gamma_val, color=color, y_min=y_min, y_max=y_max)

        for base_algo, g in base_summ.groupby("algorithm", dropna=False):
            base_algo = str(base_algo)
            g = g.sort_values("epsilon_plot")
            ax.plot(
                g["oos_var"].to_numpy(dtype=float),
                g["oos_mean"].to_numpy(dtype=float),
                marker="o",
                markersize=7,
                linestyle="--",
                linewidth=2.2,
                label=algo_label.get(base_algo, base_algo),
                color=algo_color.get(base_algo, None),
                zorder=5,
            )

        ax.set_xlabel(r"OOS variance ($\varepsilon$)", fontsize=LABEL_FS)
        ax.set_ylabel(r"OOS mean ($\varepsilon$)", fontsize=LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(4, 4), useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(TICK_FS)
        ax.margins(x=0.02, y=0.08)

        if y_min is not None or y_max is not None:
            ymin_cur, ymax_cur = ax.get_ylim()
            ax.set_ylim(
                y_min if y_min is not None else ymin_cur,
                y_max if y_max is not None else ymax_cur,
            )

    full_title = title
    if full_title is None:
        full_title = rf"Mean--variance frontiers for {algo_label.get(str(algorithm), str(algorithm))} with varying $\gamma$"

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(15.6, 6.8), sharex=True)

    _draw_panel(ax_full)
    _add_zoom_box(ax_full, ZOOM_YMIN, ZOOM_YMAX)
    ax_full.set_title("Full plot", fontsize=LABEL_FS, pad=6)

    _draw_panel(ax_zoom, y_min=ZOOM_YMIN, y_max=ZOOM_YMAX)
    ax_zoom.set_xlim(ax_full.get_xlim())
    ax_zoom.set_title("Zoomed-in plot", fontsize=LABEL_FS, pad=6)

    fig.suptitle(full_title, fontsize=LABEL_FS + 1, y=0.975)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.84], w_pad=2.5)
    _add_external_legend(fig, ax_full)
    _add_zoom_callout(fig, ax_full, ax_zoom, ZOOM_YMIN, ZOOM_YMAX)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        print("Saved frontier-by-gamma-bulk plot to", out_path)

    return fig, (ax_full, ax_zoom), sub

def plot_msd_vs_epsilon_by_gamma_bulk(
    df: pd.DataFrame,
    *,
    algorithm: str = "lv_bas",
    t_scale: float = LV_NEWSVENDOR_T_SCALE,
    convert_kl_eps_to_lv_scale: bool = True,
    show_ci: bool = True,
    title: str | None = None,
    out_path: Path | None = None,
):
    """
    MSD-vs-epsilon for gamma_bulk sensitivity.

    Solid lines: different gamma_bulk values for `algorithm` (with optional CI bands).
    Dashed lines: all other baselines present in df, with no confidence bands.
    """
    LABEL_FS = 20
    TICK_FS = 20
    LEGEND_FS = 20
    TOPLABEL_FS = 20

    algo_label = {
        "lv_bas": r"$\mathrm{LV}$",
        "kl_bdro": r"$\mathrm{KL\!-\!BDRO}$",
        "kl_empirical": r"$\mathrm{KL\!-\!Empirical}$",
        "kl_pp": r"$\mathrm{KL\!-\!BAS}_{\rm PP}$",
        "or_wdro": r"$\mathrm{OR\!-\!WDRO}$",
        "lv_reverse": r"$\mathrm{Rev\!-\!LV\!-\!BAS}$",
        "tv_ball": r"$\mathrm{TV\!-\!BAS}$",
    }

    algo_color = {
        "lv_bas": "#000000",
        "kl_bdro": "#0072B2",
        "kl_empirical": "#E69F00",
        "kl_pp": "#009E73",
        "or_wdro": "#CC79A7",
        "lv_reverse": "#D55E00",
        "tv_ball": "#56B4E9",
    }

    summ = summarise_vs_epsilon_and_gamma_bulk(
        df,
        gamma=0.0,
        t_scale=t_scale,
        convert_kl_eps_to_lv_scale=convert_kl_eps_to_lv_scale,
    )
    sub = summ[summ["algorithm"].astype(str) == str(algorithm)].copy()
    if sub.empty:
        raise ValueError(f"No rows for algorithm={algorithm!r}")

    gamma_values = sorted(float(v) for v in sub["gamma_bulk"].dropna().unique().tolist())
    n_g = len(gamma_values)

    cmap = plt.get_cmap("viridis")
    xs = np.linspace(0.1, 0.9, n_g) if n_g > 1 else np.array([0.5])
    gamma_colors = [mpl.colors.to_hex(cmap(x)) for x in xs]
    gamma_color = {g: gamma_colors[i] for i, g in enumerate(gamma_values)}

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    for gamma_bulk, g in sub.groupby("gamma_bulk", dropna=False):
        g = g.sort_values("epsilon_plot")
        gamma_val = float(gamma_bulk)
        x = g["epsilon_plot"].to_numpy(dtype=float)
        y = g["msd_mean"].to_numpy(dtype=float)

        line, = ax.plot(
            x,
            y,
            marker="o",
            markersize=9,
            linestyle="-",
            linewidth=2.5,
            label=fr"$\gamma={gamma_val:g}$",
            color=gamma_color.get(gamma_val, None),
        )

        if show_ci and {"msd_ci_low", "msd_ci_high"}.issubset(g.columns):
            c = line.get_color()
            ax.fill_between(
                x,
                g["msd_ci_low"].to_numpy(dtype=float),
                g["msd_ci_high"].to_numpy(dtype=float),
                alpha=0.15,
                linewidth=0,
                color=c,
            )

    base_summ = (
        summ[summ["algorithm"].astype(str) != str(algorithm)]
        .groupby(["algorithm", "epsilon_plot"], dropna=False)
        .agg(
            msd_mean=("msd_mean", "mean"),
        )
        .reset_index()
        .sort_values(["algorithm", "epsilon_plot"])
    )

    for base_algo, g in base_summ.groupby("algorithm", dropna=False):
        base_algo = str(base_algo)
        g = g.sort_values("epsilon_plot")
        ax.plot(
            g["epsilon_plot"].to_numpy(dtype=float),
            g["msd_mean"].to_numpy(dtype=float),
            marker="o",
            markersize=7,
            linestyle="--",
            linewidth=2.2,
            label=algo_label.get(base_algo, base_algo),
            color=algo_color.get(base_algo, None),
        )

    ax.set_xlabel(r"LV Tolerance $\varepsilon_{\mathrm{LV}}$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$0.5\,(\mathrm{OOS\ mean} + \mathrm{OOS\ SD})$", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)

    if title:
        ax.set_title(title, fontsize=LABEL_FS)

    ax.legend(ncols=2, fontsize=LEGEND_FS)
    fig.tight_layout()

    try:
        secax = ax.secondary_xaxis(
            "top",
            functions=(
                lambda eps_lv: _eps_lv_to_kl(eps_lv, t_scale=t_scale),
                lambda eps_kl: _eps_kl_to_lv(eps_kl, t_scale=t_scale),
            ),
        )
        secax.set_xlabel(
            r"KL tolerance $\varepsilon_{\mathrm{KL}}$",
            fontsize=TOPLABEL_FS,
        )
        secax.tick_params(labelsize=TICK_FS)
    except Exception:
        pass

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        print("Saved MSD-by-gamma-bulk plot to", out_path)

    return fig, ax, sub

def plot_frontiers_by_total_samples_per_algorithm(
    df: pd.DataFrame,
    *,
    algorithm: str,
    t_scale: float = LV_NEWSVENDOR_T_SCALE,
    convert_kl_eps_to_lv_scale: bool = True,
    title: str | None = None,
    out_path: Path | None = None,
):
    """
    Mean–variance trade-off (frontier) for a *single* algorithm, with one curve per num_total_samples.
    """
    LABEL_FS = 18
    TICK_FS = 18
    LEGEND_FS = 18

    summ = summarise_vs_epsilon_and_total_samples(
        df,
        gamma=0.0,
        t_scale=t_scale,
        convert_kl_eps_to_lv_scale=convert_kl_eps_to_lv_scale,
    )
    sub = summ[summ["algorithm"].astype(str) == str(algorithm)].copy()
    if sub.empty:
        raise ValueError(f"No rows for algorithm={algorithm!r}")

    m_values = sorted(int(v) for v in sub["num_total_samples"].dropna().unique().tolist())
    # Sequential, colourblind-friendly colours that scale with M (no cycling)
    n_m = len(m_values)

    cmap = plt.get_cmap("viridis")  # sequential + colourblind-friendly
    xs = np.linspace(0.1, 0.9, n_m) if n_m > 1 else np.array([0.5])

    m_colors = [mpl.colors.to_hex(cmap(x)) for x in xs]
    m_color = {m: m_colors[i] for i, m in enumerate(m_values)}

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for m, g in sub.groupby("num_total_samples", dropna=False):
        g = g.sort_values("epsilon_plot")
        m_int = int(m)
        ax.plot(
            g["oos_var"].to_numpy(dtype=float),
            g["oos_mean"].to_numpy(dtype=float),
            marker="o",
            markersize=9,
            linestyle="-",
            linewidth=2.5,
            label=f"M={m_int}",
            color=m_color.get(m_int, None),
        )

    ax.set_xlabel(r"OOS variance ($\varepsilon$)", fontsize=LABEL_FS)
    ax.set_ylabel(r"OOS mean ($\varepsilon$)", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(4, 4), useMathText=True)
    ax.xaxis.get_offset_text().set_fontsize(TICK_FS)
    algo_label = {
        "lv_bas": r"$\mathrm{LV}$",
        "kl_bdro": r"$\mathrm{KL\!-\!BDRO}$",
        "kl_empirical": r"$\mathrm{KL\!-\!Empirical}$",
        "kl_pp": r"$\mathrm{KL\!-\!BAS}_{\rm PP}$",
        "or_wdro": r"$\mathrm{OR\!-\!WDRO}$",
        "lv_reverse": r"$\mathrm{Rev\!-\!LV\!-\!BAS}$",
        "tv_ball": r"$\mathrm{TV\!-\!BAS}$",
    }
    title = f"Algorithm: {algo_label[algorithm]}"
    ax.set_title(title, fontsize=LABEL_FS)

    if str(algorithm) == "lv_bas":
        ax.legend(ncols=2, fontsize=LEGEND_FS)

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        print("Saved frontier-by-total-samples plot to", out_path)

    return fig, ax, sub


def plot_msd_vs_epsilon_by_total_samples_per_algorithm(
    df: pd.DataFrame,
    *,
    algorithm: str,
    t_scale: float = LV_NEWSVENDOR_T_SCALE,
    convert_kl_eps_to_lv_scale: bool = True,
    show_ci: bool = True,
    title: str | None = None,
    out_path: Path | None = None,
):
    """
    MSD-vs-epsilon for a *single* algorithm, with one curve per num_total_samples.
    """
    LABEL_FS = 20
    TICK_FS = 20
    LEGEND_FS = 20

    summ = summarise_vs_epsilon_and_total_samples(
        df,
        gamma=0.0,
        t_scale=t_scale,
        convert_kl_eps_to_lv_scale=convert_kl_eps_to_lv_scale,
    )
    sub = summ[summ["algorithm"].astype(str) == str(algorithm)].copy()
    if sub.empty:
        raise ValueError(f"No rows for algorithm={algorithm!r}")

    m_values = sorted(int(v) for v in sub["num_total_samples"].dropna().unique().tolist())
    # Sequential, colourblind-friendly colours that scale with M (no cycling)
    n_m = len(m_values)

    cmap = plt.get_cmap("viridis")  # sequential + colourblind-friendly
    xs = np.linspace(0.1, 0.9, n_m) if n_m > 1 else np.array([0.5])

    m_colors = [mpl.colors.to_hex(cmap(x)) for x in xs]
    m_color = {m: m_colors[i] for i, m in enumerate(m_values)}

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for m, g in sub.groupby("num_total_samples", dropna=False):
        g = g.sort_values("epsilon_plot")
        # Keep LV-scaled x locations so the spacing along the axis is linear in eps_LV.
        # For KL-based algorithms we will *label* the axis in eps_KL later via a tick formatter.
        x = g["epsilon_plot"].to_numpy(dtype=float)
        y = g["msd_mean"].to_numpy(dtype=float)
        m_int = int(m)

        line, = ax.plot(
            x,
            y,
            marker="o",
            markersize=9,
            linestyle="-",
            linewidth=2.5,
            label=f"M={m_int}",
            color=m_color.get(m_int, None),
        )

        if show_ci and {"msd_ci_low", "msd_ci_high"}.issubset(g.columns):
            c = line.get_color()
            ax.fill_between(
                x,
                g["msd_ci_low"].to_numpy(dtype=float),
                g["msd_ci_high"].to_numpy(dtype=float),
                alpha=0.15,
                linewidth=0,
                color=c,
            )

    if str(algorithm) in ["kl_bdro", "kl_empirical", "kl_pp"]:
        ax.set_xlabel(r"KL tolerance $\varepsilon_{\mathrm{KL}}$", fontsize=LABEL_FS)
    else:
        ax.set_xlabel(r"LV Tolerance $\varepsilon_{\mathrm{LV}}$", fontsize=LABEL_FS)

    ax.set_ylabel(r"0.5(OOS mean + OOS SD)", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    
    # title = f"Algorithm: {str(algorithm)}"
    # ax.set_title(title, fontsize=LABEL_FS)

    if str(algorithm) == "lv_bas":
        ax.legend(ncols=2, fontsize=LEGEND_FS)

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        print("Saved MSD-by-total-samples plot to", out_path)

    return fig, ax, sub

def _ca_housing_lineplot(ax, df, x_col, y_col, title):
    algorithms = sorted(df["algorithm"].unique())

    # Convention: if y_col ends with "_mean", we look for matching CI columns
    ci_low_col = y_col.replace("_mean", "_ci_low") if isinstance(y_col, str) and y_col.endswith("_mean") else None
    ci_high_col = y_col.replace("_mean", "_ci_high") if isinstance(y_col, str) and y_col.endswith("_mean") else None
    has_ci = (ci_low_col in df.columns) and (ci_high_col in df.columns) if ci_low_col and ci_high_col else False

    for alg in algorithms:
        sub = df[df["algorithm"] == alg].sort_values(x_col)
        x = sub[x_col].values
        y = sub[y_col].values
        ax.plot(x, y, marker="o", label=alg)

        if has_ci:
            lo = sub[ci_low_col].values
            hi = sub[ci_high_col].values
            ax.fill_between(x, lo, hi, alpha=0.15, linewidth=0)

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)


def clopper_pearson_interval(k: int, n: int, alpha: float = 0.05):
    if n <= 0:
        return (np.nan, np.nan)
    if k <= 0:
        lo = 0.0
    else:
        lo = float(sp.stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    if k >= n:
        hi = 1.0
    else:
        hi = float(sp.stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


def plot_california_housing_mae_vs_cvar_trajectory(df: pd.DataFrame, output_dir: Path, gap_ratio: float = 0.0):
    """
    Pareto-style trajectories for California Housing DRO methods.

    x-axis: MAE
    y-axis: cvar loss (cvar_abs_error by default)

    Each point corresponds to one epsilon value, and points are connected in
    increasing epsilon order to show the trajectory.

    Layout matches the existing CA housing plots:
      - one figure per gamma_bulk

    Plots:
      - LV
      - CVaR
      - Wass
      - plus single points for ERM, Ridge, and Trivial (fixed w.r.t. epsilon)
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = {"algorithm", "epsilon", "gamma_bulk"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns in results df: {sorted(missing)}")

    # Metric columns (support a couple of common aliases)
    if "mae" in df.columns:
        mae_col = "mae"
    elif "MAE" in df.columns:
        mae_col = "MAE"
    else:
        raise ValueError("Missing MAE column (expected 'mae').")

    if "cvar_abs_error" in df.columns:
        cvar_col = "cvar_abs_error"
        cvar_label = "CVaR abs error"
    elif "cvarloss" in df.columns:
        cvar_col = "cvarloss"
        cvar_label = "cvar loss"
    else:
        raise ValueError("Missing cvar column (expected 'cvar_abs_error' or 'cvarloss').")

    # Trivial baselines (required by this plot)
    if "mae_trivial" not in df.columns:
        raise ValueError("Missing trivial baseline column 'mae_trivial'.")
    if "cvar_trivial_error" not in df.columns:
        raise ValueError("Missing trivial baseline column 'cvar_trivial_error'.")

    # If present, we will generate separate plots by calibrate_on_validation
    has_cal_flag = "calibrate_on_validation" in df.columns
    cal_flags = [None]
    if has_cal_flag:
        cal_flags = sorted(df["calibrate_on_validation"].dropna().unique().tolist())
        if not cal_flags:
            cal_flags = [None]

    # Normalise algorithm names
    def _norm_alg(s: object) -> str:
        return str(s).strip().lower().replace("-", "_").replace(" ", "_")

    # DRO mapping
    alg_map = {
        "lv_bas_ch": "LV",
        "cvar_lad": "CVaR",
        "wass_lad": "Wass",
        "wasserstein_lad": "Wass",
        "wasserstein": "Wass",
        "wass": "Wass",
        "chi2_lad": "Chi2",
        "chi2": "Chi2",
        "chi_square_lad": "Chi2",
        "chi_square": "Chi2",
        "kl_lad": "KL",
        "kl": "KL",
        "kl_dro": "KL",
        "kldro": "KL",
        "or_wdro": "OR-WDRO",
        "orwdro": "OR-WDRO",
    }
    dro_order = ["CVaR", "Wass", "Chi2", "KL", "OR-WDRO", "LV"]
    marker_map = {
        "LV": "*",
        "CVaR": "o",
        "Wass": "^",
        "Chi2": "<",
        "KL": ">",
        "OR-WDRO": "v",
        "ERM": "P",
        "Ridge": "D",
        "Trivial": "s",
    }

    # Okabe–Ito palette (user-specified mapping)
    algo_color = {
        "ERM": "#000000",        # black
        "LV": "#0072B2",       # blue
        "Wass": "#E69F00",        # orange
        "Chi2": "#8C564B",        # brown
        "KL": "#7F7F7F",          # grey
        "OR-WDRO": "#56B4E9",     # sky blue
        "CVaR": "#009E73", # bluish green
        "Trivial": "#CC79A7",             # reddish purple
        "Ridge": "#D55E00",           # yellow
    }

    def _darken_hex(hex_color: str, factor: float = 0.85) -> str:
        hc = str(hex_color).lstrip("#")
        if len(hc) != 6:
            return hex_color
        r = int(hc[0:2], 16)
        g = int(hc[2:4], 16)
        b = int(hc[4:6], 16)
        r = max(0, min(255, int(round(r * factor))))
        g = max(0, min(255, int(round(g * factor))))
        b = max(0, min(255, int(round(b * factor))))
        return f"#{r:02X}{g:02X}{b:02X}"

    mark_text_color = {
        "LV": _darken_hex(algo_color["LV"]),
        "CVaR": _darken_hex(algo_color["CVaR"]),
        "Wass": _darken_hex(algo_color["Wass"]),
        "Chi2": _darken_hex(algo_color["Chi2"]),
        "KL": _darken_hex(algo_color["KL"]),
        "OR-WDRO": _darken_hex(algo_color["OR-WDRO"]),
        "Ridge": _darken_hex(algo_color["Ridge"]),
    }

    # Baseline mapping (single point each)
    base_map = {
        "erm_lad": "ERM",
        "erm": "ERM",
        "erm_ridge": "Ridge",
        "ridge": "Ridge",
    }
    base_order = ["ERM", "Ridge", "Trivial"]

    # Subset of epsilons to annotate
    mark_eps = np.array([0.0, 0.05, 0.5, 2.0], dtype=float)
    ridge_mark_eps = np.array([1e-3, 1e-1, 10.0, 1000.0], dtype=float)

    # --- DRO-only dataframe (for trajectories)
    df0 = df.copy()
    df0["algorithm_norm"] = df0["algorithm"].apply(_norm_alg)
    df_dro = df0[df0["algorithm_norm"].isin(alg_map)].copy()
    df_dro["algorithm_show"] = df_dro["algorithm_norm"].map(alg_map)

    if df_dro.empty:
        raise ValueError(
            "No matching DRO methods found for the trajectory plot. "
            "Expected algorithms like 'lv_bas_ch', 'cvar_lad', 'wass_lad'."
        )

    # --- Baselines dataframe (ERM/Ridge points)
    df_base = df0[df0["algorithm_norm"].isin(base_map)].copy()
    if not df_base.empty:
        df_base["algorithm_show"] = df_base["algorithm_norm"].map(base_map)

    # Aggregate over replications for each epsilon (DRO trajectories)
    group_cols = ["algorithm_show", "epsilon", "gamma_bulk"]
    if has_cal_flag:
        group_cols = ["calibrate_on_validation"] + group_cols

    agg = (
        df_dro.groupby(group_cols, dropna=False)
        .agg(
            mae_mean=(mae_col, "mean"),
            mae_sd=(mae_col, "std"),
            cvar_mean=(cvar_col, "mean"),
            cvar_sd=(cvar_col, "std"),
            n_rep=(mae_col, "count"),
        )
        .reset_index()
    )
    agg["mae_sd"] = agg["mae_sd"].fillna(0.0)
    agg["cvar_sd"] = agg["cvar_sd"].fillna(0.0)
    # Aggregate Ridge trajectory (varies with epsilon)
    ridge_norm_keys = [k for k, v in base_map.items() if v == "Ridge"]
    df_ridge = df0[df0["algorithm_norm"].isin(ridge_norm_keys)].copy()
    ridge_agg = None
    if not df_ridge.empty:
        df_ridge["algorithm_show"] = "Ridge"
        ridge_group_cols = ["algorithm_show", "epsilon", "gamma_bulk"]
        if has_cal_flag:
            ridge_group_cols = ["calibrate_on_validation"] + ridge_group_cols
        ridge_agg = (
            df_ridge.groupby(ridge_group_cols, dropna=False)
            .agg(
                mae_mean=(mae_col, "mean"),
                mae_sd=(mae_col, "std"),
                cvar_mean=(cvar_col, "mean"),
                cvar_sd=(cvar_col, "std"),
                n_rep=(mae_col, "count"),
            )
            .reset_index()
        )
        ridge_agg["mae_sd"] = ridge_agg["mae_sd"].fillna(0.0)
        ridge_agg["cvar_sd"] = ridge_agg["cvar_sd"].fillna(0.0)

    # Aggregate baselines (single point per method per facet)
    base_group_cols = ["algorithm_show", "gamma_bulk"]
    if has_cal_flag:
        base_group_cols = ["calibrate_on_validation"] + base_group_cols

    base_agg = None
    if not df_base.empty:
        base_agg = (
            df_base.groupby(base_group_cols, dropna=False)
            .agg(
                mae_mean=(mae_col, "mean"),
                cvar_mean=(cvar_col, "mean"),
                n_rep=(mae_col, "count"),
            )
            .reset_index()
        )

    # Aggregate trivial baseline (single point per facet; deduplicate across methods/eps when possible)
    triv_group_cols = ["gamma_bulk"]
    if has_cal_flag:
        triv_group_cols = ["calibrate_on_validation"] + triv_group_cols

    df_triv = df0[
        triv_group_cols
        + ["mae_trivial", "cvar_trivial_error"]
        + (["replication"] if "replication" in df0.columns else [])
    ].copy()
    if "replication" in df_triv.columns:
        df_triv = df_triv.drop_duplicates(subset=["replication"] + triv_group_cols)
    triv_agg = (
        df_triv.groupby(triv_group_cols, dropna=False)
        .agg(
            mae_trivial_mean=("mae_trivial", "mean"),
            cvar_trivial_mean=("cvar_trivial_error", "mean"),
        )
        .reset_index()
    )

    gamma_vals = sorted(agg["gamma_bulk"].unique())

    for gamma in gamma_vals:
        agg_g = agg[agg["gamma_bulk"] == gamma]
        base_g = base_agg[base_agg["gamma_bulk"] == gamma] if base_agg is not None else None
        triv_g = triv_agg[triv_agg["gamma_bulk"] == gamma]
        ridge_g = ridge_agg[ridge_agg["gamma_bulk"] == gamma] if ridge_agg is not None else None

        cal_iter = cal_flags if has_cal_flag else [None]
        for cal_flag in cal_iter:
            if cal_flag is None:
                sub_g = agg_g
                sub_base = base_g
                sub_triv = triv_g
                sub_ridge = ridge_g
                cal_suffix = ""
            else:
                sub_g = agg_g[agg_g["calibrate_on_validation"] == cal_flag]
                if sub_g.empty:
                    continue
                sub_base = base_g[base_g["calibrate_on_validation"] == cal_flag] if base_g is not None else None
                sub_triv = triv_g[triv_g["calibrate_on_validation"] == cal_flag]
                sub_ridge = ridge_g[ridge_g["calibrate_on_validation"] == cal_flag] if ridge_g is not None else None
                cal_suffix = f"_cv{int(bool(cal_flag))}"

            # Global axis limits for this figure (so facets are comparable)
            x_list = [sub_g["mae_mean"].to_numpy(dtype=float)]
            y_list = [sub_g["cvar_mean"].to_numpy(dtype=float)]

            if sub_base is not None and not sub_base.empty:
                x_list.append(sub_base["mae_mean"].to_numpy(dtype=float))
                y_list.append(sub_base["cvar_mean"].to_numpy(dtype=float))

            if sub_triv is not None and not sub_triv.empty:
                x_list.append(sub_triv["mae_trivial_mean"].to_numpy(dtype=float))
                y_list.append(sub_triv["cvar_trivial_mean"].to_numpy(dtype=float))
            if sub_ridge is not None and not sub_ridge.empty:
                x_list.append(sub_ridge["mae_mean"].to_numpy(dtype=float))
                y_list.append(sub_ridge["cvar_mean"].to_numpy(dtype=float))

            x_vals = np.concatenate(x_list, axis=0)
            y_vals = np.concatenate(y_list, axis=0)
            x_vals = x_vals[np.isfinite(x_vals)]
            y_vals = y_vals[np.isfinite(y_vals)]
            if x_vals.size == 0 or y_vals.size == 0:
                continue

            x_min, x_max = float(x_vals.min()), float(x_vals.max())
            y_min, y_max = float(y_vals.min()), float(y_vals.max())
            x_pad = 0.05 * max(1e-12, x_max - x_min)
            y_pad = 0.05 * max(1e-12, y_max - y_min)

            n_rows = 1
            n_cols = 1
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(4.5 * n_cols, 3.8 * n_rows),
                squeeze=False,
            )
            gap = gap_ratio*100
            if use_LaTeX:
                fig.suptitle(rf"${gap:.0f}\%$ gap", fontsize=10, y=0.995)
                fig.text(
                    0.5,
                    0.955,
                    rf"Train: Eastern 50\%, Test: Western {50-gap:.0f}\%",
                    ha="center",
                    va="top",
                    fontsize=8,
                )
            else:
                fig.suptitle(f"{gap:.0f}% gap", fontsize=10, y=0.995)
                fig.text(
                    0.5,
                    0.955,
                    f"Train: Eastern 50%, Test: Western {50-gap:.0f}%",
                    ha="center",
                    va="top",
                    fontsize=8,
                )

            ax = axes[0, 0]
            cell = sub_g
            # Ridge trajectory (varies with epsilon)
            if sub_ridge is not None and not sub_ridge.empty:
                cell_r = sub_ridge
                if not cell_r.empty:
                    c = cell_r.sort_values("epsilon", ascending=True)

                    x = c["mae_mean"].to_numpy(dtype=float)
                    y = c["cvar_mean"].to_numpy(dtype=float)
                    e = c["epsilon"].to_numpy(dtype=float)

                    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
                    x, y, e = x[mask], y[mask], e[mask]
                    if x.size > 0:
                        ax.plot(
                            x,
                            y,
                            marker=marker_map.get("Ridge", "D"),
                            linewidth=1.5,
                            color=algo_color.get("Ridge", None),
                            label="Ridge",
                        )
                        for xx, yy, ee in zip(x, y, e):
                            if np.any(np.isclose(ee, ridge_mark_eps, rtol=0.0, atol=1e-12)):
                                ax.annotate(
                                    f"{ee:g}",
                                    (xx, yy),
                                    textcoords="offset points",
                                    xytext=(4, 2),
                                    fontsize=7,
                                    alpha=0.8,
                                    color=mark_text_color.get("Ridge", None),
                                )


            # Plot DRO trajectories (connect points by increasing epsilon)
            for alg_show in dro_order:
                c = cell[cell["algorithm_show"] == alg_show]
                if c.empty:
                    continue
                c = c.sort_values("epsilon", ascending=True)

                x = c["mae_mean"].to_numpy(dtype=float)
                y = c["cvar_mean"].to_numpy(dtype=float)
                e = c["epsilon"].to_numpy(dtype=float)

                mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
                x, y, e = x[mask], y[mask], e[mask]
                if x.size == 0:
                    continue

                ax.plot(
                    x,
                    y,
                    marker=marker_map.get(alg_show, "o"),
                    linewidth=1.5,
                    color=algo_color.get(alg_show, None),
                    label=alg_show,
                )

                # Only annotate a subset of epsilons
                for xx, yy, ee in zip(x, y, e):
                    if np.any(np.isclose(ee, mark_eps, rtol=0.0, atol=1e-12)):
                        lab = f"{ee:.2f}" if ee < 1.0 else f"{ee:g}"
                        ax.annotate(
                            lab,
                            (xx, yy),
                            textcoords="offset points",
                            xytext=(4, 2),
                            fontsize=7,
                            alpha=0.8,
                            color=mark_text_color.get(alg_show, None),
                        )

            # Add single baseline points (ERM, Ridge) if available
            if sub_base is not None and not sub_base.empty:
                cell_b = sub_base
                for name in ["ERM"]:
                    bb = cell_b[cell_b["algorithm_show"] == name]
                    if bb.empty:
                        continue
                    xx = float(bb["mae_mean"].iloc[0])
                    yy = float(bb["cvar_mean"].iloc[0])
                    ax.plot(
                        [xx],
                        [yy],
                        marker=marker_map.get(name, "x"),
                        linestyle="None",
                        markersize=7,
                        color=algo_color.get(name, None),
                        label=name,
                    )

            # Add trivial baseline point
            if sub_triv is not None and not sub_triv.empty:
                cell_t = sub_triv
                if not cell_t.empty:
                    xx = float(cell_t["mae_trivial_mean"].iloc[0])
                    yy = float(cell_t["cvar_trivial_mean"].iloc[0])
                    ax.plot(
                        [xx],
                        [yy],
                        marker=marker_map.get("Trivial", "P"),
                        linestyle="None",
                        markersize=7,
                        color=algo_color.get("Trivial", None),
                        label="Trivial",
                    )

            #ax.set_title(f"30% gap: (Train = 50% East ; Test = 20% West)")
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.grid(True, linewidth=0.5, alpha=0.3)
            ax.ticklabel_format(axis="both", style="sci", scilimits=(5, 5), useMathText=True)
            ax.xaxis.get_offset_text().set_fontsize(9)
            ax.yaxis.get_offset_text().set_fontsize(9)

            ax.set_xlabel("MAE", fontsize=10)
            ax.set_ylabel(cvar_label, fontsize=10)
            ax.tick_params(axis="both", which="major", labelsize=9)
            ax.tick_params(axis="both", which="minor", labelsize=9)
    

    # Legend inside the plot; no figure title
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(
            handles,
            labels,
            loc="best",
            framealpha=0.9,
            fontsize=9,
        )

    fig.tight_layout(pad=0.2)
    fig.savefig(output_dir / f"ca_housing_mae_vs_cvar_trajectory_gamma_{gamma}{cal_suffix}.pdf")
    plt.close(fig)
