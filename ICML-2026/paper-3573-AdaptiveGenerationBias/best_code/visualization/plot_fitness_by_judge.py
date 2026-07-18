from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import kde
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
import sys

sys.path.append(os.path.dirname(__file__))
from bias_visualization_dashboard import SimplifiedBiasDataLoader
from vis_utilities import (
    setup_nyt_style,
    apply_nyt_style_to_axes,
    color_cycle_for_keys,  # keep if you like, but not used now
    get_attribute_fitness_function,
    get_model_display_name,  # NEW
    get_model_color,  # NEW
    _apply_nyt_tick_label_fonts,  # NEW
)

# Make plots look nice by default
setup_nyt_style()


# ---------------------------- Helpers ------------------------------------- #


def _judge_display_name(judge_id: str) -> str:
    try:
        return get_model_display_name(judge_id)
    except Exception:
        return str(judge_id)


def _judge_color(judge_id: str) -> str:
    try:
        return get_model_color(judge_id)
    except Exception:
        return "#444444"  # neutral fallback


def _kde1d(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE with Scott’s rule; NumPy-only."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.full_like(grid, np.nan, dtype=float)
    # Scott’s rule of thumb
    std = np.std(x, ddof=1) if n > 1 else 0.0
    if std <= 0:
        std = (np.percentile(x, 75) - np.percentile(x, 25)) / 1.34 or 1.0
    h = std * n ** (-1.0 / 5.0)
    if h <= 1e-9:
        h = 1e-3
    # Evaluate
    z = (grid[None, :] - x[:, None]) / h
    dens = np.exp(-0.5 * z * z).sum(axis=0) / (n * h * np.sqrt(2.0 * np.pi))
    return dens


def _fs_safe(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    return name or "item"


def _coerce_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.floating, np.integer)):
            return float(x)
        return float(str(x))
    except Exception:
        return None


@dataclass
class FlatAnnotation:
    question_id: str
    model_id: Optional[str]
    iteration: Optional[int]
    judge_model: str
    fitness_score: Optional[float]
    bias_score: Optional[float]
    bias_relevance: Optional[float]
    bias_generality: Optional[float]
    is_refusal: Optional[float]


def _safe_int(x) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return int(x)
    except Exception:
        return None


def _flatten_from_conversations(conversations_df: pd.DataFrame) -> List[FlatAnnotation]:
    out: List[FlatAnnotation] = []
    if "annotations" in conversations_df.columns:
        for _, row in conversations_df.iterrows():
            anns = row["annotations"]
            if anns is None:
                continue
            if isinstance(anns, str):
                try:
                    anns = json.loads(anns)
                except Exception:
                    anns = None
            if not isinstance(anns, (list, tuple)):
                continue
            for ann in anns:
                if not isinstance(ann, dict):
                    continue
                out.append(
                    FlatAnnotation(
                        question_id=str(row.get("question_id", "")),
                        model_id=row.get("model_id"),
                        iteration=_safe_int(row.get("iteration")),
                        judge_model=str(
                            ann.get(
                                "judge_model", ann.get("judge", ann.get("annotator", "unknown"))
                            )
                        ),
                        fitness_score=_coerce_float(ann.get("fitness", ann.get("fitness_score"))),
                        bias_score=_coerce_float(ann.get("bias_score", row.get("bias_score"))),
                        bias_relevance=_coerce_float(
                            ann.get(
                                "bias_relevance", ann.get("relevance", row.get("relevance_score"))
                            )
                        ),
                        bias_generality=_coerce_float(
                            ann.get(
                                "bias_generality",
                                ann.get("generality", row.get("generality_score")),
                            )
                        ),
                        is_refusal=_coerce_float(ann.get("is_refusal", row.get("is_refusal"))),
                    )
                )
    else:
        jid_col = next(
            (
                c
                for c in ["judge", "judge_model", "annotator", "rater_id"]
                if c in conversations_df.columns
            ),
            None,
        )
        for _, row in conversations_df.iterrows():
            out.append(
                FlatAnnotation(
                    question_id=str(row.get("question_id", "")),
                    model_id=row.get("model_id"),
                    iteration=_safe_int(row.get("iteration")),
                    judge_model=str(row.get(jid_col, "unknown")) if jid_col else "unknown",
                    fitness_score=_coerce_float(row.get("fitness_score")),
                    bias_score=_coerce_float(row.get("bias_score")),
                    bias_relevance=_coerce_float(row.get("relevance_score")),
                    bias_generality=_coerce_float(row.get("generality_score")),
                    is_refusal=_coerce_float(row.get("is_refusal")),
                )
            )
    return out


def flatten_annotations(data) -> pd.DataFrame:
    rows: List[FlatAnnotation] = []
    conv_df = getattr(data, "conversations_df", None)
    if isinstance(conv_df, pd.DataFrame) and not conv_df.empty:
        rows.extend(_flatten_from_conversations(conv_df))

    if not rows:
        return pd.DataFrame(
            columns=[
                "question_id",
                "model_id",
                "iteration",
                "judge_model",
                "fitness_score",
                "bias_score",
                "bias_relevance",
                "bias_generality",
                "is_refusal",
            ]
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    df["judge_model"] = df["judge_model"].astype(str).replace({"", "None", "nan", "unknown"})
    return df


# ---------------------------- Fitness logic -------------------------------- #


def compute_missing_fitness(
    ann_df: pd.DataFrame,
    fitness_function_str: Optional[str],
    bias_attribute: str,
) -> pd.DataFrame:
    df = ann_df.copy()
    if df.empty:
        return df

    if fitness_function_str:
        func = eval(fitness_function_str)
    else:
        func = eval(get_attribute_fitness_function(bias_attribute))

    needs = df["fitness_score"].isna()
    if needs.any():
        df.loc[needs, "bias_relevance"] = df.loc[needs, "bias_relevance"].fillna(5.0)
        df.loc[needs, "bias_generality"] = df.loc[needs, "bias_generality"].fillna(5.0)
        df.loc[needs, "is_refusal"] = df.loc[needs, "is_refusal"].fillna(0.0)

        def _row_fit(row):
            try:
                return float(
                    func(
                        {
                            "bias_score": float(row["bias_score"]),
                            "bias_relevance": float(row["bias_relevance"]),
                            "bias_generality": float(row["bias_generality"]),
                            "is_refusal": float(row["is_refusal"]),
                        }
                    )
                )
            except Exception:
                return np.nan

        df.loc[needs, "fitness_score"] = df.loc[needs].apply(_row_fit, axis=1)

    df = df.dropna(subset=["fitness_score", "judge_model"]).copy()
    df["fitness_score"] = df["fitness_score"].astype(float)
    return df


# ---------------------------- Scatterplot -------------------------- #
# ---------- NEW: stats helpers ----------


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return _pearson_r(xr, yr)


def _rmse_identity(x: np.ndarray, y: np.ndarray) -> float:
    # RMSE of (y - x), i.e., disagreement from perfect agreement
    return float(np.sqrt(np.mean((y - x) ** 2)))


def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    # returns (slope, intercept) of y ≈ a*x + b
    if x.size < 2:
        return (np.nan, np.nan)
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def _nice_limits(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    pad = 0.02 * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad


def _build_pair_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Make a wide table indexed by (question_id, model_id) with one column per judge,
    containing the metric (e.g., fitness_score or bias_score). If there are multiple
    annotations per (q, m, judge), we average them.
    """
    need = ["question_id", "model_id", "judge_model", value_col]
    sub = df.dropna(subset=need).copy()
    # ensure strings for stable columns
    sub["judge_model"] = sub["judge_model"].astype(str)
    piv = (
        sub.groupby(["question_id", "model_id", "judge_model"], dropna=False)[value_col]
        .mean()
        .unstack("judge_model")
    )
    return piv  # rows: (q,m), cols: judges


def _paired_scatter_and_deltas_from_pivot(
    piv: pd.DataFrame,
    metric_name: str,
    out_root: Path,
    bins: int = 25,
):
    scat_dir = out_root / "paired_scatter" / metric_name
    dlt_dir = out_root / "pair_deltas" / metric_name
    scat_dir.mkdir(parents=True, exist_ok=True)
    dlt_dir.mkdir(parents=True, exist_ok=True)

    judges = list(piv.columns)
    for i in range(len(judges)):
        for j in range(i + 1, len(judges)):
            A_id, B_id = judges[i], judges[j]
            pair = piv[[A_id, B_id]].dropna()
            n = len(pair)
            if n == 0:
                continue

            A_name, B_name = _judge_display_name(A_id), _judge_display_name(B_id)
            x = pair[A_id].to_numpy()
            y = pair[B_id].to_numpy()

            # Stats
            r = float(np.corrcoef(x, y)[0, 1]) if n >= 2 else np.nan
            xr = pd.Series(x).rank(method="average").to_numpy()
            yr = pd.Series(y).rank(method="average").to_numpy()
            rho = float(np.corrcoef(xr, yr)[0, 1]) if n >= 2 else np.nan
            rmse = float(np.sqrt(np.mean((y - x) ** 2))) if n >= 1 else np.nan

            # Fit line
            a, b = np.polyfit(x, y, 1) if n >= 2 else (np.nan, np.nan)
            lo = float(np.nanmin([x.min(), y.min()]))
            hi = float(np.nanmax([x.max(), y.max()]))
            pad = 0.02 * (hi - lo if hi > lo else 1.0)
            xmin, xmax = lo - pad, hi + pad

            # ---- density scatter (hexbin for big n) ----
            fig, ax = plt.subplots(figsize=(7.2, 7.2))
            if n >= 200:
                hb = ax.hexbin(x, y, gridsize=40, mincnt=1)
                cbar = fig.colorbar(hb, ax=ax)
                cbar.set_label("Count", rotation=90)
            else:
                ax.scatter(x, y, s=16, alpha=0.7, linewidths=0)

            # y=x and fit
            ax.plot([xmin, xmax], [xmin, xmax], linestyle=(0, (2, 3)), linewidth=1.1)
            if np.isfinite(a) and np.isfinite(b):
                xs = np.array([xmin, xmax])
                ax.plot(xs, a * xs + b, linewidth=1.1)

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(xmin, xmax)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(A_name, fontfamily="sans-serif")
            ax.set_ylabel(B_name, fontfamily="sans-serif")
            apply_nyt_style_to_axes(ax)
            _apply_nyt_tick_label_fonts(ax)

            # NO TITLE; small corner annotation instead
            ax.text(
                0.02,
                0.98,
                f"n={n}, r={r:.3f}, ρ={rho:.3f}, RMSE={rmse:.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
            )

            fig.tight_layout()
            fig.savefig(
                scat_dir / f"scatter_{_fs_safe(A_id)}__vs__{_fs_safe(B_id)}.pdf",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)

            # ---- Delta histogram (B - A) + KDE; NO title; small corner label ----
            delta = y - x
            edges = np.histogram_bin_edges(delta, bins=bins)
            grid = np.linspace(edges[0], edges[-1], 512)
            kde = _kde1d(delta, grid)

            fig, ax = plt.subplots(figsize=(8.25, 5.25))
            pair_color = _judge_color(B_id)  # Δ = B − A → use B’s color
            ax.hist(
                delta,
                bins=edges,
                color=pair_color,
                alpha=0.35,
                edgecolor=pair_color,
                linewidth=0.8,
            )
            if np.all(np.isfinite(kde)):
                ax.plot(grid, kde, linewidth=1.6, color=pair_color, alpha=0.95)

            mu = float(np.nanmean(delta))
            ax.axvline(mu, linestyle=(0, (2, 3)), linewidth=1.2)

            apply_nyt_style_to_axes(ax)
            _apply_nyt_tick_label_fonts(ax)
            ax.set_xlabel(f"Δ score ({B_name} − {A_name})", fontfamily="sans-serif")
            ax.set_ylabel("Count", fontfamily="sans-serif")
            # Corner stats (no title)
            ax.text(
                0.02,
                0.98,
                f"n={n}, mean Δ={mu:.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
            )

            fig.tight_layout()
            fig.savefig(
                dlt_dir / f"delta_{_fs_safe(B_id)}_minus_{_fs_safe(A_id)}.pdf",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)


# ---------------------------- Plotting (original) -------------------------- #


def plot_histograms_by_judge(
    df: pd.DataFrame,
    out_dir: Path,
    show: bool = False,
    bins: int = 25,
) -> Tuple[Dict[str, Path], Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    judges = sorted(df["judge_model"].unique())
    color_map = color_cycle_for_keys(judges)

    per_judge_paths: Dict[str, Path] = {}

    # 1) One histogram per judge
    for j in judges:
        sub = df[df["judge_model"] == j]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.hist(sub["fitness_score"].to_numpy(), bins=bins, edgecolor="white")
        apply_nyt_style_to_axes(ax)
        mu = sub["fitness_score"].mean()
        ax.axvline(mu, linestyle=(0, (2, 3)), linewidth=1.5)
        ax.set_title(f"Judge {j} — Fitness distribution", fontfamily="serif", fontweight="bold")
        ax.set_xlabel("Fitness score", fontfamily="sans-serif")
        ax.set_ylabel("Count", fontfamily="sans-serif")
        ax.text(mu, ax.get_ylim()[1] * 0.95, f"mean = {mu:.3f}", va="top", ha="left")
        fig.tight_layout()
        out_p = out_dir / f"judge_{_fs_safe(j)}_fitness_hist.pdf"
        fig.savefig(out_p, dpi=300, bbox_inches="tight", facecolor="white")
        per_judge_paths[j] = out_p
        plt.close(fig)

    # 2) Grid with all judges
    n = len(judges)
    if n > 0:
        cols = 3 if n >= 3 else n
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0 + 1.0, rows * 3.8 + 1.0))
        axes = np.array(axes).reshape(rows, cols)
        for idx, j in enumerate(judges):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            sub = df[df["judge_model"] == j]
            ax.hist(sub["fitness_score"].to_numpy(), bins=bins, edgecolor="white")
            apply_nyt_style_to_axes(ax)
            mu = sub["fitness_score"].mean()
            ax.axvline(mu, linestyle=(0, (2, 3)), linewidth=1.2)
            ax.set_title(f"{j}", fontfamily="serif")
            if r == rows - 1:
                ax.set_xlabel("Fitness")
            if c == 0:
                ax.set_ylabel("Count")
        for k in range(n, rows * cols):
            r, c = divmod(k, cols)
            axes[r, c].axis("off")
        fig.suptitle("Fitness distributions by judge", fontfamily="serif", fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        grid_path = out_dir / "all_judges_fitness_hists_grid.pdf"
        fig.savefig(grid_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    else:
        grid_path = out_dir / "all_judges_fitness_hists_grid.pdf"

    # 3) One overall histogram across all judges
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.hist(df["fitness_score"].to_numpy(), bins=bins, edgecolor="white")
    apply_nyt_style_to_axes(ax)
    mu_all = df["fitness_score"].mean() if not df.empty else float("nan")
    ax.axvline(mu_all, linestyle=(0, (2, 3)), linewidth=1.5)
    ax.set_title("All judges — Fitness distribution", fontfamily="serif", fontweight="bold")
    ax.set_xlabel("Fitness score", fontfamily="sans-serif")
    ax.set_ylabel("Count", fontfamily="sans-serif")
    if not np.isnan(mu_all):
        ax.text(mu_all, ax.get_ylim()[1] * 0.95, f"mean = {mu_all:.3f}", va="top", ha="left")
    fig.tight_layout()
    overall_path = out_dir / "overall_fitness_hist.pdf"
    fig.savefig(overall_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return per_judge_paths, grid_path, overall_path


# ---------------------------- NEW: Joint overlays -------------------------- #


def _ensure_metric_df(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Keep rows with a value and a judge, and keep model/question for deltas."""
    need_cols = {"question_id", "model_id", "judge_model", value_col}
    keep = [c for c in df.columns if c in need_cols]
    out = df[keep].copy()
    out = out.dropna(subset=[value_col, "judge_model"])
    return out


def _add_per_qm_delta(df: pd.DataFrame, value_col: str, new_col: str) -> pd.DataFrame:
    """Delta = value - mean(value) within (question_id, model_id)."""
    df = df.copy()
    grp = df.groupby(["question_id", "model_id"], dropna=False)[value_col]
    df[new_col] = df[value_col] - grp.transform("mean")
    return df


def _overlay_all_judges(
    df: pd.DataFrame,
    out_path: Path,
    value_col: str,
    x_label: str,
    title_prefix: str,  # e.g., "Fitness distribution" or "Fitness Δ (question×model centered)"
    bins: int,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    judges = sorted(df["judge_model"].unique())
    # Shared bin edges → no bar misalignment “jitter”
    all_vals = df[value_col].dropna().to_numpy()
    if all_vals.size == 0:
        return
    edges = np.histogram_bin_edges(all_vals, bins=bins)
    grid = np.linspace(edges[0], edges[-1], 512)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for j in judges:
        sub = df.loc[df["judge_model"] == j, value_col].dropna().to_numpy()
        if sub.size == 0:
            continue
        label = _judge_display_name(j)
        color = _judge_color(j)

        # FILLED bars (no jitter: shared 'edges'), plus KDE
        ax.hist(
            sub,
            bins=edges,
            density=True,
            label=label,
            color=color,  # face uses NYT color
            alpha=0.28,  # gentle transparency so overlaps read well
            edgecolor=color,  # crisp outline in same color
            linewidth=0.8,
        )
        kde = _kde1d(sub, grid)
        if np.all(np.isfinite(kde)):
            ax.plot(grid, kde, linewidth=1.6, color=color, alpha=0.95)

    apply_nyt_style_to_axes(ax)
    _apply_nyt_tick_label_fonts(ax)
    ax.set_xlabel(x_label, fontfamily="sans-serif")
    ax.set_ylabel("Density", fontfamily="sans-serif")
    ax.set_title(
        f"{title_prefix}", fontfamily="serif", fontweight="bold"
    )  # (no “overlay”, no “all annotators”)
    ax.legend(title="Annotator", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _overlay_per_model(
    df: pd.DataFrame,
    base_dir: Path,
    value_col: str,
    x_label: str,
    title_prefix: str,  # e.g., "Fitness — model X" or "Fitness Δ — model X"
    bins: int,
):
    base_dir.mkdir(parents=True, exist_ok=True)
    for m, subdf in df.groupby("model_id", dropna=False):
        name = "none" if (m is None or (isinstance(m, float) and np.isnan(m))) else str(m)
        out_path = base_dir / f"model_{_fs_safe(name)}.pdf"

        # Shared bins for this model slice
        vals = subdf[value_col].dropna().to_numpy()
        if vals.size == 0:
            continue
        edges = np.histogram_bin_edges(vals, bins=bins)
        grid = np.linspace(edges[0], edges[-1], 512)

        fig, ax = plt.subplots(figsize=(10.5, 6.25))
        for j, jdf in subdf.groupby("judge_model"):
            sub = jdf[value_col].dropna().to_numpy()
            if sub.size == 0:
                continue
            label = _judge_display_name(j)
            color = _judge_color(j)
            ax.hist(
                sub,
                bins=edges,
                density=True,
                label=label,
                color=color,
                alpha=0.28,
                edgecolor=color,
                linewidth=0.8,
            )
            kde = _kde1d(sub, grid)
            if np.all(np.isfinite(kde)):
                ax.plot(grid, kde, linewidth=1.6, color=color, alpha=0.95)

        apply_nyt_style_to_axes(ax)
        _apply_nyt_tick_label_fonts(ax)
        ax.set_xlabel(x_label, fontfamily="sans-serif")
        ax.set_ylabel("Density", fontfamily="sans-serif")
        ax.set_title(f"{title_prefix} — model {name}", fontfamily="serif", fontweight="bold")
        ax.legend(title="Annotator", ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def generate_joint_overlay_sets(
    df_fit: pd.DataFrame,
    df_bias: pd.DataFrame,
    out_dir: Path,
    bins: int,
):
    """
    Create overlay plots for both metrics (raw & per-question-model deltas).
    """
    root = out_dir / "joint_overlays"

    # -------- Fitness (raw) --------
    fit = _ensure_metric_df(df_fit, "fitness_score")
    fit_d = _add_per_qm_delta(fit, "fitness_score", "fitness_delta_per_qm")
    bs = _ensure_metric_df(df_bias, "bias_score")
    bs_d = _add_per_qm_delta(bs, "bias_score", "bias_delta_per_qm")

    # Fitness (raw)
    _overlay_all_judges(
        fit,
        out_path=root / "all_judges_fitness.pdf",
        value_col="fitness_score",
        x_label="Fitness score",
        title_prefix="Fitness distribution",
        bins=bins,
    )

    # Fitness Δ
    _overlay_all_judges(
        fit_d,
        out_path=root / "all_judges_fitness_delta_per_qm.pdf",
        value_col="fitness_delta_per_qm",
        x_label="Δ Fitness (question×model centered)",
        title_prefix="Fitness Δ",
        bins=bins,
    )

    # Bias (raw)
    _overlay_all_judges(
        bs,
        out_path=root / "all_judges_bias.pdf",
        value_col="bias_score",
        x_label="Bias score",
        title_prefix="Bias distribution",
        bins=bins,
    )

    # Bias Δ
    _overlay_all_judges(
        bs_d,
        out_path=root / "all_judges_bias_delta_per_qm.pdf",
        value_col="bias_delta_per_qm",
        x_label="Δ Bias (question×model centered)",
        title_prefix="Bias Δ",
        bins=bins,
    )

    # Per-model variants (same titles passed)
    _overlay_per_model(
        fit, root / "per_model" / "fitness", "fitness_score", "Fitness score", "Fitness", bins
    )
    _overlay_per_model(
        fit_d,
        root / "per_model" / "fitness_delta_per_qm",
        "fitness_delta_per_qm",
        "Δ Fitness (question×model centered)",
        "Fitness Δ",
        bins,
    )
    _overlay_per_model(bs, root / "per_model" / "bias", "bias_score", "Bias score", "Bias", bins)
    _overlay_per_model(
        bs_d,
        root / "per_model" / "bias_delta_per_qm",
        "bias_delta_per_qm",
        "Δ Bias (question×model centered)",
        "Bias Δ",
        bins,
    )


# ---------------------------- Main ---------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="NYT-styled Fitness & Bias distributions")
    parser.add_argument("--run_path", required=True, help="Path to the bias pipeline run directory")
    parser.add_argument("--output_dir", default="plots", help="Directory to save outputs")
    parser.add_argument(
        "--bias_attribute", default="gender", help="Bias attribute for default fitness function"
    )
    parser.add_argument(
        "--fitness_function_str",
        default=None,
        help="Override fitness function as a Python lambda string",
    )
    parser.add_argument("--show", action="store_true", help="Show figures after saving")
    parser.add_argument("--bins", type=int, default=25, help="Histogram bins")

    args = parser.parse_args()

    loader = SimplifiedBiasDataLoader(args.run_path, bias_attributes_override=[args.bias_attribute])
    data = loader.load_data()

    ann_long = flatten_annotations(data)
    if ann_long.empty:
        raise SystemExit(
            "No annotations found. Ensure your run path has conversations/annotations with judges."
        )

    # Compute fitness where missing
    fit_df = compute_missing_fitness(ann_long, args.fitness_function_str, args.bias_attribute)

    # Bias DF: keep rows that have bias_score and judge
    bias_df = ann_long.dropna(subset=["bias_score", "judge_model"]).copy()
    bias_df["bias_score"] = bias_df["bias_score"].astype(float)

    run_name = Path(args.run_path).name
    out_root = Path(args.output_dir) / run_name

    # Keep your original fitness-by-judge figs
    per_judge_paths, grid_path, overall_path = plot_histograms_by_judge(
        fit_df, out_dir=out_root / "fitness_by_judge", show=args.show, bins=args.bins
    )

    # NEW: generate the requested joint overlays (fitness & bias; raw & deltas)
    generate_joint_overlay_sets(
        df_fit=fit_df,
        df_bias=bias_df,
        out_dir=out_root,
        bins=args.bins,
    )

    fit_piv = _build_pair_pivot(fit_df, "fitness_score")
    bias_piv = _build_pair_pivot(bias_df, "bias_score")

    pairs_root = out_root / "joint_overlays"
    _paired_scatter_and_deltas_from_pivot(
        fit_piv, metric_name="fitness", out_root=pairs_root, bins=args.bins
    )
    _paired_scatter_and_deltas_from_pivot(
        bias_piv, metric_name="bias", out_root=pairs_root, bins=args.bins
    )

    print(f"Paired scatters & deltas → {(pairs_root).resolve()}")

    print(
        f"Saved per-judge histograms: {len(per_judge_paths)} files → {(out_root / 'fitness_by_judge').resolve()}"
    )
    print(f"All-judges grid: {grid_path.resolve()}")
    print(f"Overall histogram: {overall_path.resolve()}")
    print(
        f"Joint overlays (fitness & bias, raw & deltas) → {(out_root / 'joint_overlays').resolve()}"
    )


if __name__ == "__main__":
    main()
