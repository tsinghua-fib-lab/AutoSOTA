import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd

from cellbridge.cci.lr_pairs_config import get_lr_pairs
from cellbridge.utils.typing import as_index, ensure_upper_var

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PairTable:
    """Generic interaction pairs (UPPERCASE gene symbols)."""

    df: pd.DataFrame  # columns: left, right (both UPPERCASE)

    @property
    def n_pairs(self) -> int:
        """Return the number of ligand-receptor pairs."""
        return int(self.df.shape[0])


def _pairs_from_gene_pairs(genes: Iterable[tuple[str, str]]) -> PairTable:
    """Build (g1,g2) pairs from a list of gene pairs (case-insensitive)."""
    g1 = pd.Index([str(x[0]).upper() for x in genes])
    g2 = pd.Index([str(x[1]).upper() for x in genes])
    df = pd.DataFrame({"left": g1, "right": g2})
    return PairTable(df)


def _detect_cols(df: pd.DataFrame) -> tuple[str, str, str, str]:
    cols = df.columns
    gs_src = (
        "genesymbol_intercell_source"
        if "genesymbol_intercell_source" in cols
        else "genesymbol_source"
    )
    gs_tgt = (
        "genesymbol_intercell_target"
        if "genesymbol_intercell_target" in cols
        else "genesymbol_target"
    )
    cat_src = (
        "category_intercell_source"
        if "category_intercell_source" in cols
        else "category_source"
    )
    cat_tgt = (
        "category_intercell_target"
        if "category_intercell_target" in cols
        else "category_target"
    )
    for c in (gs_src, gs_tgt, cat_src, cat_tgt):
        if c not in cols:
            raise KeyError(f"Expected column '{c}' not found in the OmniPath table.")
    return gs_src, gs_tgt, cat_src, cat_tgt


def load_pairs(
    organism: Literal["human", "mouse"] = "human",
    *,
    categories: tuple[str, str] = ("ligand", "receptor"),
    homophilic: bool | None = None,
) -> PairTable:
    """
    Load interaction pairs from OmniPath intercell network,
    returning standardized columns: left, right  (UPPERCASE)

    Parameters
    ----------
    categories:
        Which intercell categories to use for source and target.
    homophilic:
        If True, generate only same-gene pairs (g,g) from the chosen category set.
        If None, defaults to False.
    """
    from omnipath import interactions as op_interactions

    df = op_interactions.import_intercell_network(
        interactions_params={"organisms": organism}
    )

    gs_src, gs_tgt, cat_src, cat_tgt = _detect_cols(df)
    # Normalize categories input to sets for matching
    left_cats = {categories[0].lower()}
    right_cats = {categories[1].lower()}

    tab = (
        df.loc[
            df[cat_src].str.lower().isin(left_cats)
            & df[cat_tgt].str.lower().isin(right_cats),
            [gs_src, gs_tgt],
        ]
        .rename(columns={gs_src: "left", gs_tgt: "right"})
        .copy()
    )

    # Uppercase normalization
    tab["left"] = tab["left"].astype(str).str.upper()
    tab["right"] = tab["right"].astype(str).str.upper()

    if homophilic is None:
        homophilic = False

    if homophilic:
        # Build self-pairs (g,g) from the set of genes that appear on either side
        genes = pd.unique(pd.concat([tab["left"], tab["right"]], ignore_index=True))
        tab = pd.DataFrame({"left": genes, "right": genes}, copy=False)

    # Drop duplicates & NA
    tab = tab.dropna().drop_duplicates().reset_index(drop=True)

    return PairTable(tab)


def filter_pairs_to_genes(pairs: PairTable, genes_upper: Iterable[str]) -> PairTable:
    """Keep pairs whose genes exist in the data."""
    gset = set(map(str.upper, genes_upper))
    df = pairs.df[
        pairs.df.left.isin(gset) & pairs.df.right.isin(gset)
    ].drop_duplicates()
    return PairTable(df.reset_index(drop=True))  # type: ignore


def pair_indices(
    adata: ad.AnnData,
    pairs: PairTable,
    var_col: str = "UP",
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """
    Map pair symbols to gene column indices.
    Returns: L_idx, R_idx (length P), and Series symbol->col for debugging.
    """
    upper = ensure_upper_var(adata, var_col)
    symbols = as_index(upper).astype(str)

    # Build symbol -> column index mapping
    col_index = pd.Series(range(adata.n_vars), index=symbols)

    # Make index unique so we can reindex safely
    if col_index.index.has_duplicates:
        col_index = col_index[~col_index.index.duplicated(keep="first")]

    # Ensure pair symbols are strings (and same casing as var_col)
    left = pairs.df.left.astype(str)
    right = pairs.df.right.astype(str)

    tmp = pairs.df.assign(
        l_idx=col_index.reindex(left).values,  # type: ignore
        r_idx=col_index.reindex(right).values,  # type: ignore
    ).dropna(subset=["l_idx", "r_idx"])

    L_idx = tmp.l_idx.astype(int).to_numpy()
    R_idx = tmp.r_idx.astype(int).to_numpy()

    return L_idx, R_idx, col_index  # type: ignore


def pairs_from_adata(
    adata: ad.AnnData,
    organism: Literal["human", "mouse"] = "human",
    *,
    mode: Literal[
        "lr",
        "none",
        "liana_embryo",
        "liana_cancer",
        "liana_light",
        "channels_1_2_3",
        "channels_1_2",
        "channels_1_3",
        "channels_2_3",
        "liana_immune",
        "random_lr",
        "random",
    ] = "none",
    categories: tuple[str, str] | None = None,
    random_state: int | None = None,
) -> PairTable:
    """
    Convenience wrapper to load + filter interaction pairs from AnnData.
    """
    if categories is None and mode == "liana_embryo":
        logger.info("Loading manual LIANA Embryo pairs")
        pairs = _pairs_from_gene_pairs(get_lr_pairs("LIANA Embryo Signaling"))
        return filter_pairs_to_genes(pairs, adata.var_names)

    if categories is None and mode == "liana_cancer":
        logger.info("Loading manual LIANA Cancer pairs")
        pairs = _pairs_from_gene_pairs(get_lr_pairs("LIANA Cancer Signaling"))
        return filter_pairs_to_genes(pairs, adata.var_names)

    if categories is None and mode == "liana_light":
        logger.info("Loading LIANA Light Mouse pairs")
        pairs = _pairs_from_gene_pairs(get_lr_pairs("LIANA Light Mouse Signaling"))
        return filter_pairs_to_genes(pairs, adata.var_names)

    if categories is None and mode == "channels_1_2_3":
        logger.info("Loading manual Channels 1 2 3 pairs")
        pairs = _pairs_from_gene_pairs(get_lr_pairs("Channels 1 2 3"))
        return filter_pairs_to_genes(pairs, adata.var_names)

    if categories is None and mode == "channels_1_2":
        logger.info("Loading manual Channels 1 2 pairs")
        pairs = _pairs_from_gene_pairs(get_lr_pairs("Channels 1 2"))
        return filter_pairs_to_genes(pairs, adata.var_names)

    if categories is None and mode == "channels_1_3":
        logger.info("Loading manual Channels 1 3 pairs")
        pairs = _pairs_from_gene_pairs(get_lr_pairs("Channels 1 3"))
        return filter_pairs_to_genes(pairs, adata.var_names)

    if categories is None and mode == "channels_2_3":
        logger.info("Loading manual Channels 2 3 pairs")
        pairs = _pairs_from_gene_pairs(get_lr_pairs("Channels 2 3"))
        return filter_pairs_to_genes(pairs, adata.var_names)

    if categories is None and mode == "liana_immune":
        logger.info("Loading LIANA Immune pairs")
        pairs = _pairs_from_gene_pairs(get_lr_pairs("LIANA Immune Signaling"))
        return filter_pairs_to_genes(pairs, adata.var_names)

    # Random baseline: sample 10 random ligand/receptor pairs
    if categories is None and mode in ("random_lr", "random"):
        logger.info("Loading random baseline: 10 random ligand/receptor pairs")
        # Load OmniPath LR pairs, then filter to genes present in the AnnData
        pairs_all = load_pairs(
            organism=organism,
            categories=("ligand", "receptor"),
            homophilic=False,
        )
        pairs_all = filter_pairs_to_genes(pairs_all, adata.var_names)
        df = pairs_all.df
        if df.shape[0] == 0:
            logger.warning(
                "No LR pairs found after filtering to AnnData genes. "
                "Returning empty PairTable."
            )
            return PairTable(pd.DataFrame(columns=["left", "right"]))  # type: ignore
        # Sample up to 10 without replacement
        logger.info("Sampling among %d available LR pairs.", df.shape[0])
        n_sample = min(10, df.shape[0])
        sample_seed = 42 if random_state is None else int(random_state)
        df_sample = df.sample(n=n_sample, replace=False, random_state=sample_seed)
        logger.info(f"Sampled {n_sample} random LR pairs.")
        logger.info(df_sample)
        return PairTable(df_sample.reset_index(drop=True))

    # Handle OmniPath-based modes
    if categories is None:
        if mode == "lr":
            categories = ("ligand", "receptor")
            homophilic = False
        elif mode == "none":
            # We return a dummy Pair table as we won't need it anyway.
            return PairTable(pd.DataFrame(columns=["left", "right"]))  # type: ignore
        else:
            raise ValueError(f"Unknown mode: {mode}")
    else:
        homophilic = None  # let load_pairs decide default for custom categories

    pairs = load_pairs(organism=organism, categories=categories, homophilic=homophilic)
    pairs = filter_pairs_to_genes(pairs, adata.var_names)
    return pairs
