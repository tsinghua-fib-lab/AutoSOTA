import anndata as ad


def _split_h5ad(
    A: ad.AnnData, domain_key: str, x_label, y_label
) -> tuple[ad.AnnData, ad.AnnData]:
    X = A[A.obs[domain_key] == x_label].copy()
    Y = A[A.obs[domain_key] == y_label].copy()
    if X.n_obs == 0 or Y.n_obs == 0:
        raise ValueError(
            f"Empty split: X has {X.n_obs} cells, Y has {Y.n_obs}. "
            f"Check domain_key='{domain_key}', x_label='{x_label}', y_label={y_label}"
        )
    return X, Y
