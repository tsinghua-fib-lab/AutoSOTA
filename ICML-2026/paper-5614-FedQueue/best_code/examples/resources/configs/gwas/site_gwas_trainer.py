import os
import sys
import json
import torch
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import chi2, t
from pandas_plink import read_plink1_bin
from appfl.algorithm.trainer.base_trainer import BaseTrainer


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Add the gwas config directory to path so gwas_config is importable.
# When loaded directly from file, __file__ points to the gwas config directory and
# Path(__file__).parent resolves correctly.
# When APPFL sends this file to gRPC clients as source text, it is written to a
# temp path (~/.appfl/tmp/), so __file__-relative lookup breaks.
# In that case, set the GWAS_PROJECT_DIR env var to the directory containing gwas_config.py
# (e.g. export GWAS_PROJECT_DIR=/path/to/examples/resources/configs/gwas).
_gwas_demo_dir = os.environ.get("GWAS_PROJECT_DIR")
if _gwas_demo_dir:
    sys.path.insert(0, _gwas_demo_dir)
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwas_config import (  # noqa: E402
    USE_CUML,
    HIT_P_THRESHOLD,
    get_linear_regression,
    get_logistic_regression,
    apply_variant_scaling,
)


matplotlib.use("Agg")

if not USE_CUML:
    pass
from sklearn.metrics import r2_score, roc_auc_score, roc_curve  # noqa: E402


CHROM_MAP = {"X": 23, "Y": 24, "XY": 25, "MT": 26, "M": 26}


def _normalize_chr(chrom):
    return chrom.astype(str).str.strip().str.upper().replace(CHROM_MAP).astype(int)


def _fit_binary_model(X, y):
    for penalty in [None, "none"]:
        try:
            model = get_logistic_regression(
                fit_intercept=False,
                penalty=penalty,
                solver="lbfgs",
                max_iter=1000,
                tol=1e-8,
            )
            model.fit(X, y)
            return model
        except (TypeError, ValueError):
            continue
    raise RuntimeError("Failed to fit LogisticRegression for local T2D model.")


def _load_genotype_chunk(G, start, end):
    chunk = (
        G.isel(variant=slice(start, end))
        .compute(scheduler="single-threaded")
        .values.astype(np.float64, copy=False)
    )
    observed = np.isfinite(chunk)
    counts = observed.sum(axis=0)
    means = np.divide(
        np.nansum(chunk, axis=0),
        counts,
        out=np.zeros(chunk.shape[1], dtype=np.float64),
        where=counts > 0,
    )
    if not observed.all():
        row_idx, col_idx = np.where(~observed)
        chunk[row_idx, col_idx] = means[col_idx]
    maf = np.minimum(means / 2.0, 1.0 - means / 2.0)
    return chunk, maf


def _plot_manhattan(gwas_df, trait, threshold, out_path, label=""):
    n_snps = len(gwas_df)
    n_samples = int(gwas_df["N"].iloc[0]) if "N" in gwas_df.columns else 0

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
    ax.set_title(f"{trait} Local GWAS QQ Plot")
    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_bmi_scatter(y_true, y_pred, r2_value, out_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, s=8, alpha=0.4, color="#1f5a96", linewidths=0)
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="#9b1c31", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Observed BMI")
    ax.set_ylabel("Predicted BMI")
    ax.set_title(f"Local BMI PGS (R2 = {r2_value:.4f})")
    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_t2d_roc(y_true, probs, auc_value, out_path):
    fpr, tpr, _ = roc_curve(y_true, probs)  # codespell:ignore fpr
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(
        fpr,  # codespell:ignore fpr
        tpr,
        color="#1f5a96",
        linewidth=2.0,
        label=f"PGS + covariates (AUROC = {auc_value:.4f})",
    )
    ax.plot(
        [0, 1], [0, 1], color="#9b1c31", linestyle="--", linewidth=1.2, label="Chance"
    )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Local T2D PGS ROC")
    ax.grid(color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


class SiteGWASTrainer(BaseTrainer):
    def __init__(
        self,
        model=None,
        loss_fn=None,
        metric=None,
        train_dataset=None,
        val_dataset=None,
        train_configs=None,
        logger=None,
        client_id=None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metric=metric,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_configs=train_configs,
            logger=logger,
            client_id=client_id,
            **kwargs,
        )
        self.client_id = (
            str(client_id) if client_id is not None else str(train_dataset.site_id)
        )
        self.chunk_size = int(self.train_configs.get("gwas_chunk_size", 256))
        self.hit_threshold = float(
            self.train_configs.get("hit_p_threshold", HIT_P_THRESHOLD)
        )
        self.qq_max_points = int(self.train_configs.get("qq_max_points", 250000))
        _default_out = str(self.train_dataset.data_dir.parent / "output")
        self.output_dir = Path(
            self.train_configs.get(
                "trainer_output_dirname",
                self.train_configs.get("logging_output_dirname", _default_out),
            )
        ).resolve()
        self.data_dir = self.output_dir / "data"
        self.graphs_dir = self.output_dir / "graphs"
        self.logs_dir = self.output_dir / "logs"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.model_state = {}
        self.global_state = {}
        self._has_run = False

    def load_parameters(self, params):
        self.global_state = params if isinstance(params, dict) else {}

    def get_parameters(self):
        if hasattr(self, "_train_metadata"):
            return (self.model_state, self._train_metadata)
        return self.model_state

    def train(self, **kwargs):
        if "round" in kwargs:
            self.round = kwargs["round"]
        # if self._has_run:
        #     return

        self.logger.info(
            f"{self.client_id}: loading local site data from {self.train_dataset.data_dir}"
        )
        G = read_plink1_bin(str(self.train_dataset.plink_bed), verbose=False, ref="a1")
        sample_df, variant_df, X_cov, y_bmi_gwas, y_t2d_gwas, y_bmi_eval, y_t2d_eval = (
            self._load_site_tables(G)
        )
        n_variants_total = len(variant_df)
        variant_df, G = apply_variant_scaling(variant_df, G)
        if len(variant_df) < n_variants_total:
            self.logger.info(
                f"{self.client_id}: Variant_Scaling applied: using {len(variant_df)} / {n_variants_total} variants"
            )

        bmi_df = self._run_bmi_gwas(G, variant_df, X_cov, y_bmi_gwas)
        t2d_df = self._run_t2d_gwas(G, variant_df, X_cov, y_t2d_gwas)
        pgs_metrics = self._run_local_pgs(
            G, sample_df[["FID", "IID"]], X_cov, y_bmi_eval, y_t2d_eval, bmi_df, t2d_df
        )

        bmi_path = self.data_dir / f"{self.client_id}_local_gwas_bmi.csv.gz"
        t2d_path = self.data_dir / f"{self.client_id}_local_gwas_t2d.csv.gz"
        hits_path = self.data_dir / f"{self.client_id}_local_gwas_hits.csv"
        metrics_path = self.data_dir / f"{self.client_id}_local_pgs_metrics.csv"

        bmi_df.to_csv(bmi_path, index=False)
        t2d_df.to_csv(t2d_path, index=False)
        self._write_hits_table(bmi_df, t2d_df, hits_path)
        pgs_metrics.to_csv(metrics_path, index=False)

        _plot_manhattan(
            bmi_df,
            "BMI",
            self.hit_threshold,
            self.graphs_dir / f"{self.client_id}_local_gwas_bmi_manhattan.png",
            label=self.client_id,
        )
        _plot_qq(
            bmi_df["P"].to_numpy(dtype=np.float64),
            "BMI",
            self.qq_max_points,
            self.graphs_dir / f"{self.client_id}_local_gwas_bmi_qq.png",
        )
        _plot_manhattan(
            t2d_df,
            "T2D",
            self.hit_threshold,
            self.graphs_dir / f"{self.client_id}_local_gwas_t2d_manhattan.png",
            label=self.client_id,
        )
        _plot_qq(
            t2d_df["P"].to_numpy(dtype=np.float64),
            "T2D",
            self.qq_max_points,
            self.graphs_dir / f"{self.client_id}_local_gwas_t2d_qq.png",
        )

        _variant_meta = json.dumps(
            {
                "CHR": variant_df["CHR"].tolist(),
                "SNP": variant_df["SNP"].tolist(),
                "BP": variant_df["BP"].tolist(),
                "EA": variant_df["EA"].tolist(),
                "NEA": variant_df["NEA"].tolist(),
            }
        ).encode("utf-8")

        self.model_state = {
            "bmi_beta": torch.from_numpy(bmi_df["BETA"].to_numpy(dtype=np.float64)),
            "bmi_se": torch.from_numpy(bmi_df["SE"].to_numpy(dtype=np.float64)),
            "t2d_beta": torch.from_numpy(t2d_df["BETA"].to_numpy(dtype=np.float64)),
            "t2d_se": torch.from_numpy(t2d_df["SE"].to_numpy(dtype=np.float64)),
            "maf": torch.from_numpy(bmi_df["MAF"].to_numpy(dtype=np.float64)),
            "gwas_n": torch.tensor([len(y_bmi_gwas)], dtype=torch.int64),
            "eval_n": torch.tensor([len(y_bmi_eval)], dtype=torch.int64),
            "local_bmi_r2": torch.tensor(
                [float(pgs_metrics.loc[0, "VALUE"])], dtype=torch.float64
            ),
            "local_t2d_auc": torch.tensor(
                [float(pgs_metrics.loc[1, "VALUE"])], dtype=torch.float64
            ),
            "variant_meta": torch.frombuffer(
                bytearray(_variant_meta), dtype=torch.uint8
            ),
        }
        self._train_metadata = {
            "round": self.round,
            "Local BMI PGS R²": round(float(pgs_metrics.loc[0, "VALUE"]), 4),
            "Local T2D PGS AUROC": round(float(pgs_metrics.loc[1, "VALUE"]), 4),
            "GWAS Sample Size": len(y_bmi_gwas),
            "PGS Eval Sample Size": len(y_bmi_eval),
            "Num Variants": len(variant_df),
        }

        self._has_run = True
        self.round += 1
        self.logger.info(f"{self.client_id}: local GWAS/PGS complete.")

    def _load_site_tables(self, G):
        fam = pd.DataFrame(
            {
                "FID": G.fid.values.astype(str),
                "IID": G.iid.values.astype(str),
            }
        )
        pheno_gwas = pd.read_csv(self.train_dataset.pheno_gwas)
        pheno_eval = pd.read_csv(self.train_dataset.pheno_eval)
        cov = pd.read_csv(self.train_dataset.covariates)

        sample_df = (
            fam.merge(pheno_gwas, on=["FID", "IID"], how="left", validate="one_to_one")
            .rename(columns={"T2D": "T2D_gwas", "BMI": "BMI_gwas"})
            .merge(pheno_eval, on=["FID", "IID"], how="left", validate="one_to_one")
            .rename(columns={"T2D": "T2D_eval", "BMI": "BMI_eval"})
            .merge(cov, on=["FID", "IID"], how="left", validate="one_to_one")
        )
        required_cols = ["BMI_gwas", "T2D_gwas", "BMI_eval", "T2D_eval", "age", "sex"]
        missing = sample_df[required_cols].isna().sum()
        if missing.any():
            raise ValueError(
                f"{self.client_id}: missing local phenotype/covariate values: {missing.to_dict()}"
            )

        age = sample_df["age"].to_numpy(dtype=np.float64)
        age_sd = age.std(ddof=0)
        if age_sd == 0:
            raise ValueError(f"{self.client_id}: cannot standardize age with zero SD.")
        age_z = (age - age.mean()) / age_sd
        X_cov = np.column_stack(
            [
                np.ones(len(sample_df), dtype=np.float64),
                age_z,
                sample_df["sex"].to_numpy(dtype=np.float64),
            ]
        )
        variant_df = pd.DataFrame(
            {
                "CHR": _normalize_chr(pd.Series(G.chrom.values)),
                "SNP": pd.Series(G.snp.values).astype(str),
                "BP": pd.Series(G.pos.values).astype(np.int64),
                "EA": pd.Series(G.a1.values).astype(str),
                "NEA": pd.Series(G.a0.values).astype(str),
            }
        )
        return (
            sample_df,
            variant_df,
            X_cov,
            sample_df["BMI_gwas"].to_numpy(dtype=np.float64),
            sample_df["T2D_gwas"].to_numpy(dtype=np.float64),
            sample_df["BMI_eval"].to_numpy(dtype=np.float64),
            sample_df["T2D_eval"].to_numpy(dtype=np.float64),
        )

    def _run_bmi_gwas(self, G, variant_df, X_cov, y_bmi):
        self.logger.info(f"{self.client_id}: running local BMI GWAS")
        n_samples, n_cov = X_cov.shape
        df = n_samples - n_cov - 1
        y_model = get_linear_regression(fit_intercept=False)
        y_model.fit(X_cov, y_bmi)
        y_res = y_bmi - y_model.predict(X_cov)

        out_chunks = []
        n_variants = len(variant_df)
        for start in range(0, n_variants, self.chunk_size):
            end = min(start + self.chunk_size, n_variants)
            if start == 0 or start % (20 * self.chunk_size) == 0:
                self.logger.info(
                    f"{self.client_id}: BMI GWAS chunk {start + 1}-{end} / {n_variants}"
                )

            G_chunk, maf = _load_genotype_chunk(G, start, end)
            g_model = get_linear_regression(fit_intercept=False)
            g_model.fit(X_cov, G_chunk)
            G_res = G_chunk - g_model.predict(X_cov)

            ss_g = np.einsum("ij,ij->j", G_res, G_res)
            beta = np.divide(
                G_res.T @ y_res,
                ss_g,
                out=np.full(end - start, np.nan, dtype=np.float64),
                where=ss_g > 0,
            )
            resid = y_res[:, None] - G_res * beta[None, :]
            sigma2 = np.einsum("ij,ij->j", resid, resid) / df
            se = np.sqrt(
                np.divide(
                    sigma2,
                    ss_g,
                    out=np.full(end - start, np.nan, dtype=np.float64),
                    where=ss_g > 0,
                )
            )
            stat = np.divide(
                beta,
                se,
                out=np.zeros(end - start, dtype=np.float64),
                where=np.isfinite(se) & (se > 0),
            )
            p_value = np.clip(
                2.0 * t.sf(np.abs(stat), df=df),
                np.finfo(np.float64).tiny,
                1.0,
            )

            chunk_df = variant_df.iloc[start:end].copy()
            chunk_df["TRAIT"] = "BMI"
            chunk_df["BETA"] = beta
            chunk_df["SE"] = se
            chunk_df["STAT"] = stat
            chunk_df["P"] = p_value
            chunk_df["MAF"] = maf
            chunk_df["N"] = n_samples
            out_chunks.append(chunk_df)

        return pd.concat(out_chunks, ignore_index=True)

    def _run_t2d_gwas(self, G, variant_df, X_cov, y_t2d):
        self.logger.info(f"{self.client_id}: running local T2D GWAS")
        null_model = _fit_binary_model(X_cov, y_t2d)
        mu = np.clip(null_model.predict_proba(X_cov)[:, 1], 1e-8, 1.0 - 1e-8)
        w = mu * (1.0 - mu)
        sqrt_w = np.sqrt(w)
        X_cov_w = X_cov * sqrt_w[:, None]
        score_resid = y_t2d - mu
        n_samples = X_cov.shape[0]

        out_chunks = []
        n_variants = len(variant_df)
        for start in range(0, n_variants, self.chunk_size):
            end = min(start + self.chunk_size, n_variants)
            if start == 0 or start % (20 * self.chunk_size) == 0:
                self.logger.info(
                    f"{self.client_id}: T2D GWAS chunk {start + 1}-{end} / {n_variants}"
                )

            G_chunk, maf = _load_genotype_chunk(G, start, end)
            g_model = get_linear_regression(fit_intercept=False)
            g_model.fit(X_cov_w, G_chunk * sqrt_w[:, None])
            G_res = G_chunk - g_model.predict(X_cov)

            info = np.einsum("ij,i,ij->j", G_res, w, G_res)
            beta = np.divide(
                G_res.T @ score_resid,
                info,
                out=np.full(end - start, np.nan, dtype=np.float64),
                where=info > 0,
            )
            se = np.sqrt(
                np.divide(
                    1.0,
                    info,
                    out=np.full(end - start, np.nan, dtype=np.float64),
                    where=info > 0,
                )
            )
            stat = np.divide(
                beta,
                se,
                out=np.zeros(end - start, dtype=np.float64),
                where=np.isfinite(se) & (se > 0),
            )
            p_value = np.clip(
                chi2.sf(stat * stat, df=1), np.finfo(np.float64).tiny, 1.0
            )

            chunk_df = variant_df.iloc[start:end].copy()
            chunk_df["TRAIT"] = "T2D"
            chunk_df["BETA"] = beta
            chunk_df["SE"] = se
            chunk_df["STAT"] = stat
            chunk_df["OR"] = np.exp(np.clip(beta, -50, 50))
            chunk_df["P"] = p_value
            chunk_df["MAF"] = maf
            chunk_df["N"] = n_samples
            chunk_df["TEST"] = "score"
            out_chunks.append(chunk_df)

        return pd.concat(out_chunks, ignore_index=True)

    def _run_local_pgs(
        self, G, sample_df, X_cov, y_bmi_eval, y_t2d_eval, bmi_df, t2d_df
    ):
        self.logger.info(
            f"{self.client_id}: scoring local BMI/T2D PGS in one genotype pass"
        )
        beta_matrix = np.column_stack(
            [
                bmi_df["BETA"].fillna(0.0).to_numpy(dtype=np.float64),
                t2d_df["BETA"].fillna(0.0).to_numpy(dtype=np.float64),
            ]
        )
        scores = np.zeros((G.sizes["sample"], 2), dtype=np.float64)
        n_variants = G.sizes["variant"]

        for start in range(0, n_variants, self.chunk_size):
            end = min(start + self.chunk_size, n_variants)
            weights = beta_matrix[start:end, :]
            if not np.any(weights):
                continue
            G_chunk, _ = _load_genotype_chunk(G, start, end)
            scores += G_chunk @ weights

        score_sd = scores.std(axis=0, ddof=0)
        if np.any(score_sd == 0):
            raise ValueError(f"{self.client_id}: local PGS has zero variance.")
        scores = (scores - scores.mean(axis=0)[None, :]) / score_sd[None, :]

        pgs_bmi = scores[:, 0]
        pgs_t2d = scores[:, 1]

        X_bmi = np.column_stack([X_cov, pgs_bmi])
        bmi_model = get_linear_regression(fit_intercept=False)
        bmi_model.fit(X_bmi, y_bmi_eval)
        y_bmi_pred = bmi_model.predict(X_bmi)
        bmi_r2 = float(r2_score(y_bmi_eval, y_bmi_pred))

        X_t2d = np.column_stack([X_cov, pgs_t2d])
        t2d_model = _fit_binary_model(X_t2d, y_t2d_eval)
        y_t2d_prob = t2d_model.predict_proba(X_t2d)[:, 1]
        t2d_auc = float(roc_auc_score(y_t2d_eval, y_t2d_prob))

        pgs_df = sample_df.copy()
        pgs_df["local_pgs_BMI"] = pgs_bmi
        pgs_df["local_pgs_T2D"] = pgs_t2d
        pgs_df["BMI_observed"] = y_bmi_eval
        pgs_df["BMI_predicted"] = y_bmi_pred
        pgs_df["T2D_observed"] = y_t2d_eval
        pgs_df["T2D_probability"] = y_t2d_prob
        pgs_df.to_csv(
            self.data_dir / f"{self.client_id}_local_pgs_scores.csv", index=False
        )

        _plot_bmi_scatter(
            y_bmi_eval,
            y_bmi_pred,
            bmi_r2,
            self.graphs_dir / f"{self.client_id}_local_pgs_bmi_scatter.png",
        )
        _plot_t2d_roc(
            y_t2d_eval,
            y_t2d_prob,
            t2d_auc,
            self.graphs_dir / f"{self.client_id}_local_pgs_t2d_roc.png",
        )

        return pd.DataFrame(
            [
                {
                    "CLIENT_ID": self.client_id,
                    "TRAIT": "BMI",
                    "METRIC": "R2",
                    "VALUE": bmi_r2,
                    "N_EVAL": len(y_bmi_eval),
                    "N_GWAS": len(sample_df),
                },
                {
                    "CLIENT_ID": self.client_id,
                    "TRAIT": "T2D",
                    "METRIC": "AUROC",
                    "VALUE": t2d_auc,
                    "N_EVAL": len(y_t2d_eval),
                    "N_GWAS": len(sample_df),
                },
            ]
        )

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
