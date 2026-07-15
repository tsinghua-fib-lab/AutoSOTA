import os
import sys
import json
import torch
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt
from appfl.algorithm.aggregator import BaseAggregator

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# gwas_config.py lives in the same directory as this file.
# If loaded from a non-standard path, set GWAS_PROJECT_DIR to the directory
# containing gwas_config.py (e.g. examples/resources/configs/gwas).
_gwas_demo_dir = os.environ.get("GWAS_PROJECT_DIR")
if _gwas_demo_dir:
    sys.path.insert(0, _gwas_demo_dir)
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwas_config import HIT_P_THRESHOLD  # noqa: E402

matplotlib.use("Agg")


CHROM_MAP = {"X": 23, "Y": 24, "XY": 25, "MT": 26, "M": 26}


def _normalize_chr(chrom):
    return chrom.astype(str).str.strip().str.upper().replace(CHROM_MAP).astype(int)


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _plot_manhattan(gwas_df, trait, threshold, out_path, label="APPFL"):
    n_snps = len(gwas_df)
    n_col = (
        "N_META"
        if "N_META" in gwas_df.columns
        else ("N" if "N" in gwas_df.columns else None)
    )
    n_samples = int(gwas_df[n_col].iloc[0]) if n_col else 0

    plot_df = gwas_df[["CHR", "BP", "P"]].copy()
    plot_df = plot_df.sort_values(["CHR", "BP"]).reset_index(drop=True)

    chrom_offsets = {}
    offset = 0
    tick_pos = []
    tick_labels = []
    for chrom, group in plot_df.groupby("CHR", sort=True):
        chrom = int(chrom)
        chrom_offsets[chrom] = offset
        bp = group["BP"].to_numpy(dtype=np.int64)
        tick_pos.append(offset + 0.5 * (bp.min() + bp.max()))
        tick_labels.append(str(chrom))
        offset += int(bp.max()) + 1_000_000

    plot_df["X"] = plot_df["BP"] + plot_df["CHR"].map(chrom_offsets)
    plot_df["LOGP"] = -np.log10(
        np.clip(plot_df["P"].to_numpy(dtype=np.float64), np.finfo(np.float64).tiny, 1.0)
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#1f5a96", "#d76f30"]
    for idx, (_, group) in enumerate(plot_df.groupby("CHR", sort=True)):
        ax.scatter(
            group["X"],
            group["LOGP"],
            s=4,
            color=colors[idx % 2],
            alpha=0.8,
            linewidths=0,
        )
    ax.axhline(-np.log10(threshold), color="#9b1c31", linestyle="--", linewidth=1.2)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("-log10(P)")
    ax.set_title(
        f"{trait} Manhattan Plot – {label}\nSNPs = {n_snps:,}  |  N = {n_samples:,}",
        fontsize=11,
    )
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_qq(p_values, trait, max_points, out_path):
    p_values = np.asarray(p_values, dtype=np.float64)
    p_values = p_values[np.isfinite(p_values)]
    p_values = np.clip(p_values, np.finfo(np.float64).tiny, 1.0)
    if p_values.size > max_points:
        p_values = np.sort(p_values)[:max_points]
    else:
        p_values = np.sort(p_values)

    n = p_values.size
    expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
    observed = -np.log10(p_values)
    upper = max(float(expected.max()), float(observed.max()), 1.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(expected, observed, s=8, color="#1f5a96", alpha=0.75, linewidths=0)
    ax.plot([0, upper], [0, upper], color="#9b1c31", linestyle="--", linewidth=1.2)
    ax.set_xlim(0, upper * 1.03)
    ax.set_ylim(0, upper * 1.03)
    ax.set_xlabel("Expected -log10(P)")
    ax.set_ylabel("Observed -log10(P)")
    ax.set_title(f"{trait} APPFL QQ Plot")
    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


class MetaAnalysisAggregator(BaseAggregator):
    def __init__(self, model=None, aggregator_configs=None, logger=None):
        self.model = model
        self.logger = logger
        self.aggregator_configs = aggregator_configs
        self.hit_threshold = float(
            aggregator_configs.get("hit_p_threshold", HIT_P_THRESHOLD)
        )
        self.qq_max_points = int(aggregator_configs.get("qq_max_points", 250000))
        self.output_dir = Path(
            aggregator_configs.get("output_dir", "GA4GH_Demo/server/output")
        ).resolve()
        self.data_dir = self.output_dir / "data"
        self.graphs_dir = self.output_dir / "graphs"
        self.logs_dir = self.output_dir / "logs"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.global_state = {
            "meta_ready": torch.tensor([0], dtype=torch.int64),
        }

    def get_parameters(self, **kwargs):
        return self.global_state

    def aggregate(self, local_models, **kwargs):
        client_ids = list(local_models.keys())
        self.logger.info(
            f"Running fixed-effect meta-analysis across {len(client_ids)} APPFL sites."
        )

        _meta_bytes = _to_numpy(local_models[client_ids[0]]["variant_meta"]).tobytes()
        variant_df = pd.DataFrame(json.loads(_meta_bytes.decode("utf-8")))

        bmi_beta_stack = np.vstack(
            [_to_numpy(local_models[cid]["bmi_beta"]) for cid in client_ids]
        )
        bmi_se_stack = np.vstack(
            [_to_numpy(local_models[cid]["bmi_se"]) for cid in client_ids]
        )
        t2d_beta_stack = np.vstack(
            [_to_numpy(local_models[cid]["t2d_beta"]) for cid in client_ids]
        )
        t2d_se_stack = np.vstack(
            [_to_numpy(local_models[cid]["t2d_se"]) for cid in client_ids]
        )
        maf_stack = np.vstack(
            [_to_numpy(local_models[cid]["maf"]) for cid in client_ids]
        )
        gwas_n = np.array(
            [int(_to_numpy(local_models[cid]["gwas_n"])[0]) for cid in client_ids],
            dtype=np.float64,
        )
        eval_n = np.array(
            [int(_to_numpy(local_models[cid]["eval_n"])[0]) for cid in client_ids],
            dtype=np.float64,
        )
        bmi_r2 = np.array(
            [
                float(_to_numpy(local_models[cid]["local_bmi_r2"])[0])
                for cid in client_ids
            ],
            dtype=np.float64,
        )
        t2d_auc = np.array(
            [
                float(_to_numpy(local_models[cid]["local_t2d_auc"])[0])
                for cid in client_ids
            ],
            dtype=np.float64,
        )

        n_variants = len(variant_df)
        if bmi_beta_stack.shape[1] != n_variants:
            raise ValueError(
                f"Client payload variant count ({bmi_beta_stack.shape[1]}) does not match "
                f"variant metadata from {client_ids[0]} ({n_variants})."
            )

        total_n = int(gwas_n.sum())
        meta_maf = np.average(maf_stack, axis=0, weights=gwas_n)
        bmi_df = self._meta_analyze_trait(
            bmi_beta_stack, bmi_se_stack, meta_maf, "BMI", total_n, variant_df
        )
        t2d_df = self._meta_analyze_trait(
            t2d_beta_stack, t2d_se_stack, meta_maf, "T2D", total_n, variant_df
        )

        bmi_path = self.data_dir / "appfl_meta_gwas_bmi.csv.gz"
        t2d_path = self.data_dir / "appfl_meta_gwas_t2d.csv.gz"
        hits_path = self.data_dir / "appfl_meta_gwas_hits.csv"
        metrics_path = self.data_dir / "appfl_site_pgs_metrics.csv"
        summary_path = self.data_dir / "appfl_meta_summary.csv"

        bmi_df.to_csv(bmi_path, index=False)
        t2d_df.to_csv(t2d_path, index=False)
        self._write_hits_table(bmi_df, t2d_df, hits_path)
        self._write_site_metrics(
            client_ids, gwas_n, eval_n, bmi_r2, t2d_auc, metrics_path, summary_path
        )

        _plot_manhattan(
            bmi_df,
            "BMI",
            self.hit_threshold,
            self.graphs_dir / "appfl_meta_gwas_bmi_manhattan.png",
            label="APPFL",
        )
        _plot_qq(
            bmi_df["P"].to_numpy(dtype=np.float64),
            "BMI",
            self.qq_max_points,
            self.graphs_dir / "appfl_meta_gwas_bmi_qq.png",
        )
        _plot_manhattan(
            t2d_df,
            "T2D",
            self.hit_threshold,
            self.graphs_dir / "appfl_meta_gwas_t2d_manhattan.png",
            label="APPFL",
        )
        _plot_qq(
            t2d_df["P"].to_numpy(dtype=np.float64),
            "T2D",
            self.qq_max_points,
            self.graphs_dir / "appfl_meta_gwas_t2d_qq.png",
        )

        self.global_state = {
            "meta_ready": torch.tensor([1], dtype=torch.int64),
            "num_clients": torch.tensor([len(client_ids)], dtype=torch.int64),
            "num_variants": torch.tensor([n_variants], dtype=torch.int64),
            "bmi_hits": torch.tensor(
                [int((bmi_df["P"] < self.hit_threshold).sum())], dtype=torch.int64
            ),
            "t2d_hits": torch.tensor(
                [int((t2d_df["P"] < self.hit_threshold).sum())], dtype=torch.int64
            ),
            "weighted_local_bmi_r2": torch.tensor(
                [float(np.average(bmi_r2, weights=eval_n))], dtype=torch.float64
            ),
            "weighted_local_t2d_auc": torch.tensor(
                [float(np.average(t2d_auc, weights=eval_n))], dtype=torch.float64
            ),
        }
        self.logger.info(
            f"APPFL meta-analysis outputs written to {self.output_dir}/{{data,graphs}}"
        )
        return self.global_state

    def _meta_analyze_trait(
        self, beta_stack, se_stack, maf, trait, total_n, variant_df
    ):
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(
                np.isfinite(se_stack) & (se_stack > 0), 1.0 / np.square(se_stack), 0.0
            )
        weight_sum = weights.sum(axis=0)
        beta = np.divide(
            np.sum(weights * beta_stack, axis=0),
            weight_sum,
            out=np.full(weight_sum.shape, np.nan, dtype=np.float64),
            where=weight_sum > 0,
        )
        se = np.sqrt(
            np.divide(
                1.0,
                weight_sum,
                out=np.full(weight_sum.shape, np.nan, dtype=np.float64),
                where=weight_sum > 0,
            )
        )
        stat = np.divide(
            beta,
            se,
            out=np.zeros(weight_sum.shape, dtype=np.float64),
            where=np.isfinite(se) & (se > 0),
        )
        p_value = np.clip(
            2.0 * norm.sf(np.abs(stat)),
            np.finfo(np.float64).tiny,
            1.0,
        )

        out_df = variant_df.copy()
        out_df["TRAIT"] = trait
        out_df["BETA"] = beta
        out_df["SE"] = se
        out_df["STAT"] = stat
        if trait == "T2D":
            out_df["OR"] = np.exp(np.clip(beta, -50, 50))
        out_df["P"] = p_value
        out_df["MAF"] = maf
        out_df["N_META"] = total_n
        return out_df

    def _write_hits_table(self, bmi_df, t2d_df, out_path):
        hit_tables = []
        for trait_df in [bmi_df, t2d_df]:
            hits = trait_df.loc[trait_df["P"] < self.hit_threshold].copy()
            if hits.empty:
                hits = trait_df.nsmallest(100, "P").copy()
                hits["HIT_SET"] = "top100"
            else:
                hits = hits.sort_values("P").copy()
                hits["HIT_SET"] = f"p<{self.hit_threshold:g}"
            hit_tables.append(hits)
        pd.concat(hit_tables, ignore_index=True).to_csv(out_path, index=False)

    def _write_site_metrics(
        self, client_ids, gwas_n, eval_n, bmi_r2, t2d_auc, metrics_path, summary_path
    ):
        site_metrics = pd.DataFrame(
            {
                "CLIENT_ID": client_ids,
                "GWAS_N": gwas_n.astype(int),
                "EVAL_N": eval_n.astype(int),
                "LOCAL_BMI_R2": bmi_r2,
                "LOCAL_T2D_AUROC": t2d_auc,
            }
        )
        site_metrics.to_csv(metrics_path, index=False)

        summary = pd.DataFrame(
            [
                {
                    "NUM_CLIENTS": len(client_ids),
                    "TOTAL_GWAS_N": int(gwas_n.sum()),
                    "TOTAL_EVAL_N": int(eval_n.sum()),
                    "WEIGHTED_LOCAL_BMI_R2": float(np.average(bmi_r2, weights=eval_n)),
                    "WEIGHTED_LOCAL_T2D_AUROC": float(
                        np.average(t2d_auc, weights=eval_n)
                    ),
                }
            ]
        )
        summary.to_csv(summary_path, index=False)
