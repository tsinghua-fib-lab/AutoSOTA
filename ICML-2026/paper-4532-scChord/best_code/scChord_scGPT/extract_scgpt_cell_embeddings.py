#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract cell-level scGPT embeddings and save aligned arrays for downstream training.

This script is intended to run in the scGPT conda environment.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import scanpy as sc


def _resolve_gene_col(adata, requested):
    if requested != "auto":
        if requested == "index":
            return "index"
        if requested not in adata.var.columns:
            raise ValueError(f"Requested gene_col '{requested}' not found in adata.var")
        return requested

    if "feature_name" in adata.var.columns:
        return "feature_name"
    return "index"


def _import_embed_data(scgpt_main_path: Path):
    if str(scgpt_main_path) not in sys.path:
        sys.path.insert(0, str(scgpt_main_path))
    from scgpt.tasks.cell_emb import embed_data  # pylint: disable=import-error

    return embed_data


def parse_args():
    parser = argparse.ArgumentParser(description="Extract cell-level scGPT embeddings")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Input AnnData (.h5ad) file",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./third_party/scGPT",
        help="Directory containing scGPT args.json / vocab.json / best_model.pt",
    )
    parser.add_argument(
        "--scgpt_main_path",
        type=str,
        default="./third_party/scGPT/scGPT-main",
        help="Path to scGPT-main repository root (for importing scgpt package)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output .npz file path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size used by scGPT embedding inference",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1200,
        help="Maximum token length for each cell",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device passed to scGPT embedding API",
    )
    parser.add_argument(
        "--gene_col",
        type=str,
        default="auto",
        help="Gene column in adata.var. Use auto / index / feature_name / custom column",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    env_name = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
    print(f"[env] CONDA_DEFAULT_ENV={env_name}")

    data_path = Path(args.data_path).resolve()
    output_path = Path(args.output_path).resolve()
    model_dir = Path(args.model_dir).resolve()
    scgpt_main_path = Path(args.scgpt_main_path).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] reading {data_path}")
    adata = sc.read_h5ad(data_path)
    gene_col = _resolve_gene_col(adata, args.gene_col)
    print(f"[config] gene_col={gene_col}, n_cells={adata.n_obs}, n_genes={adata.n_vars}")

    embed_data = _import_embed_data(scgpt_main_path)

    print("[run] extracting X_scGPT cell embeddings")
    adata_with_emb = embed_data(
        adata_or_file=adata,
        model_dir=model_dir,
        gene_col=gene_col,
        max_length=args.max_length,
        batch_size=args.batch_size,
        obs_to_save=None,
        device=args.device,
        use_fast_transformer=True,
        return_new_adata=False,
    )

    if "X_scGPT" not in adata_with_emb.obsm:
        raise RuntimeError("X_scGPT not found in adata.obsm after embedding extraction")

    emb = np.asarray(adata_with_emb.obsm["X_scGPT"], dtype=np.float32)
    obs_names = adata_with_emb.obs_names.to_numpy(dtype=str)

    if emb.shape[0] != len(obs_names):
        raise RuntimeError("Embedding row count does not match obs_names length")

    np.savez_compressed(
        output_path,
        embeddings=emb,
        obs_names=obs_names,
        data_path=str(data_path),
        model_dir=str(model_dir),
        gene_col=gene_col,
    )

    print(f"[save] {output_path}")
    print(f"[shape] embeddings={emb.shape}, obs_names={obs_names.shape}")


if __name__ == "__main__":
    main()
