"""
Notebook helpers: model ordering, prompt selection, metric aggregation.

These functions don't carry experimental logic; they assemble views over
the results DataFrames for downstream plotting and reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


# ============================================================
# ===================== MODEL ORDERING =======================
# ============================================================

def build_model_size_order(cfg: Config) -> tuple[list[int], dict[int, int]]:
    """Return ``(model_order, model_rank)`` for plots ordered by parameter count.

    Reads sizes from :class:`~aporia.config.ModelSpec.size_b` instead of
    parsing them out of model names.  Falls back to id order for models
    without a declared size.
    """
    model_order = cfg.model_order_by_size()
    model_rank = {m: i for i, m in enumerate(model_order)}
    return model_order, model_rank


# ============================================================
# ====================== PROMPT PICKING ======================
# ============================================================

def select_prompt_by_fraction(
    df_model: pd.DataFrame,
    mode: str = "balanced",
    require_valid_plot: bool = True,
    require_valid_geom: bool = False,
) -> pd.Series | None:
    """Pick one prompt per model based on genuine/hallucinated ratio.

    ``mode`` is one of ``"balanced"`` (frac_G closest to 0.5),
    ``"most_genuine"`` (largest frac_G), or
    ``"most_hallucinated"`` (smallest frac_G).
    """
    df_sel = df_model.copy()
    if require_valid_plot:
        df_sel = df_sel[df_sel["valid_plot"]]
    if require_valid_geom:
        df_sel = df_sel[df_sel["valid_geom"]]

    if df_sel.empty:
        return None

    if mode == "balanced":
        idx = (df_sel["frac_G"] - 0.5).abs().idxmin()
    elif mode == "most_genuine":
        idx = df_sel["frac_G"].idxmax()
    elif mode == "most_hallucinated":
        idx = df_sel["frac_G"].idxmin()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return df_sel.loc[idx]


def select_representative_prompts(
    results_df: pd.DataFrame,
    model_id: int,
    require_valid_plot: bool = True,
    require_valid_geom: bool = True,
) -> dict[str, pd.Series | None]:
    """Return balanced / most-genuine / most-hallucinated picks for one model."""
    df_model = results_df[results_df["model_id"] == model_id]
    return {
        "balanced":          select_prompt_by_fraction(df_model, "balanced",
                                                      require_valid_plot, require_valid_geom),
        "most_genuine":      select_prompt_by_fraction(df_model, "most_genuine",
                                                      require_valid_plot, require_valid_geom),
        "most_hallucinated": select_prompt_by_fraction(df_model, "most_hallucinated",
                                                      require_valid_plot, require_valid_geom),
    }


def reorder_selected_keys_by_model_size(
    selected_keys_dict: dict[str, list[tuple[int, int]]],
    model_rank: dict[int, int],
) -> dict[str, list[tuple[int, int]]]:
    """Reorder (model_id, prompt_id) tuples per panel by model size rank."""
    return {
        panel: sorted(keys, key=lambda k: model_rank[k[0]])
        for panel, keys in selected_keys_dict.items()
    }


# ============================================================
# ====================== AGGREGATION =========================
# ============================================================

def aggregate_metric_over_prompts(
    df: pd.DataFrame,
    metric: str = "f1",
    score_metric: str = "accuracy",
    agg_prompts: bool = True,
    agg_train_frac: bool = False,
    agg_models: bool = False,
) -> pd.DataFrame:
    """Aggregate a metric over the chosen axes.

    Returns
    -------
    pd.DataFrame with columns:
        - the non-aggregated axes (model_id / prompt_id / train_fraction)
        - metric_mean, metric_std
        - score_mean, score_std
        - mean_n_train, std_n_train
        - n_runs
    """
    group_cols: list[str] = []
    if not agg_models:
        group_cols.append("model_id")
    if not agg_prompts:
        group_cols.append("prompt_id")
    if not agg_train_frac:
        group_cols.append("train_fraction")

    agg_df = (
        df
        .groupby(group_cols)
        .agg(
            metric_mean=(metric, "mean"),
            metric_std=(metric, "std"),
            score_mean=(score_metric, "mean"),
            score_std=(score_metric, "std"),
            mean_n_train=("n_train", "mean"),
            std_n_train=("n_train", "std"),
            n_runs=(metric, "count"),
        )
        .reset_index()
    )
    return agg_df


# ============================================================
# ====================== MATPLOTLIB ==========================
# ============================================================

def matplotlib_latex_preamble(cfg: Config) -> str:
    """LaTeX preamble for ``plt.rcParams['text.latex.preamble']``.

    Generates a ``\\newcommand`` definition for each model's ``latex_tag``
    so that plots passing ``cfg.model_latextags`` as labels render
    correctly under ``text.usetex=True`` *without* needing the paper's
    ``camera-ready.tex`` to be on the path.

    Each macro expands to ``\\texttt{<name>}\\xspace`` where ``<name>``
    is the model's display ``name`` from the TOML config; mirrors the
    convention used in the paper.

    Use:

        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = ap.matplotlib_latex_preamble(cfg)
    """
    lines = [
        r"\usepackage{mathtools}",
        r"\usepackage{xspace}",
        "",
        "% Model-name macros generated from the active aporia config;",
        "% mirrors the \\newcommand definitions in camera-ready.tex so",
        "% notebooks can render plot labels without that file.",
    ]
    for m in cfg.models:
        if not m.latex_tag:
            continue
        tag = m.latex_tag.lstrip("\\")
        lines.append(rf"\newcommand{{\{tag}}}{{\texttt{{{m.name}}}\xspace}}")
    return "\n".join(lines)



# ============================================================
# ============== COSINE-SIMILARITY HELPERS ===================
# ============================================================

def cosine_similarity_matrix(vecs_list) -> np.ndarray:
    """Stack a list of unit vectors and return the |cos similarity| matrix."""
    V = np.stack(vecs_list)
    return np.abs(V @ V.T)


def collect_similarity_pairs(vecs: dict) -> dict:
    """Partition ordered pairs (i < j) of Fisher directions into:

      * ``within_model``    — same model_id, different prompt_id
      * ``cross_model_sp``  — different model_id, same prompt_id
      * ``cross_all``       — different model_id, different prompt_id

    The return value is ``{key: np.ndarray of similarities}``.
    """
    keys = list(vecs.keys())
    within_model, cross_model_sp, cross_all = [], [], []
    for i in range(len(keys)):
        mi, pi = keys[i]
        for j in range(i + 1, len(keys)):
            mj, pj = keys[j]
            sim = float(np.abs(np.dot(vecs[keys[i]], vecs[keys[j]])))
            if mi == mj:
                within_model.append(sim)
            elif pi == pj:
                cross_model_sp.append(sim)
            else:
                cross_all.append(sim)
    return {
        "within_model":   np.array(within_model),
        "cross_model_sp": np.array(cross_model_sp),
        "cross_all":      np.array(cross_all),
    }
