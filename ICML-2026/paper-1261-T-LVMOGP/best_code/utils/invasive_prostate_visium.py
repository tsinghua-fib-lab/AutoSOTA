import ast
from types import SimpleNamespace
import torch
import scanpy as sc


def load_ipv_data(ipv_folder: str = "./data/ST/invasive_prostate_visium/"):
    # read ST data
    adata_origin = sc.read_visium(
        path = ipv_folder,
        count_file = 'Visium_FFPE_Human_Prostate_Cancer_filtered_feature_bc_matrix.h5',
        load_images = True
    )
    adata_origin.var_names_make_unique()
    adata_origin.var["mt"] = adata_origin.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata_origin, qc_vars=["mt"], inplace=True)

    sc.pp.filter_cells(adata_origin, min_counts=2000)
    sc.pp.filter_cells(adata_origin, max_counts=35000)

    adata_origin = adata_origin[adata_origin.obs["pct_counts_mt"] < 20].copy()
    # print(f"#cells after MT filter: {adata_origin.n_obs}")

    sc.pp.filter_genes(adata_origin, min_cells=10)

    # Pick highly variable genes
    adata_tmp = adata_origin.copy()

    sc.pp.normalize_total(adata_tmp, inplace=True)
    sc.pp.log1p(adata_tmp)
    sc.pp.highly_variable_genes(adata_tmp, flavor='seurat', n_top_genes=5000, inplace=True)

    adata_pick = adata_origin[:, adata_tmp.var.highly_variable].copy()

    return adata_pick


def parse_config(config_file_path: str):
    config_dict = {}

    with open(config_file_path, 'r') as f:
        for line in f:
            # 1. Clean whitespace and skip empty lines
            line = line.strip()
            if not line:
                continue

            # 2. Remove the trailing semicolon if present
            if line.endswith(';'):
                line = line[:-1]

            # 3. Split into Key and Value
            if '=' in line:
                key, value_str = line.split('=', 1)
                key = key.strip()
                value_str = value_str.strip()

                # 4. Attempt to convert types automatically
                try:
                    # This handles int, float, bool (True/False), lists [], and None
                    val = ast.literal_eval(value_str)
                except (ValueError, SyntaxError):
                    # If eval fails (e.g., for "FCNet" or "Gaussian"), treat it as a string
                    val = value_str

                config_dict[key] = val

    return config_dict


def load_model_ipv(results_folder: str):
    from models.dkl_lvmogp_ipv import dkl_lvmogp_ipv

    config_file_path = results_folder + "/configs.txt"
    model_file_path = results_folder + "/model.pt"
    config_dict = parse_config(config_file_path)

    args = SimpleNamespace(**config_dict)
    jitter = 1e-6

    assert args.qH_type == "Gaussian"

    if args.neural_network_type == "FCNet":
        model = dkl_lvmogp_ipv(
            D_H=args.D_H, M=args.M, qH_mean_field=args.qH_mean_field, whitening=args.whitening,
            tighter_elbo=args.tighter_elbo, qU_type=args.qU_type,
            neural_network_type="FCNet", out_dim=args.out_dim, hidden_dims=args.hidden_dims, num_blocks=None,
            spectral_norm=args.spectral_norm, sn_ub=args.sn_ub, jitter=jitter, use_cache_for_svgp=args.use_cache_for_svgp,
            k_m=args.k_m, scale_factor=args.scale_factor,
        )
    elif args.neural_network_type == "ResNet":
        model = dkl_lvmogp_ipv(
            D_H=args.D_H, M=args.M, qH_mean_field=args.qH_mean_field, whitening=args.whitening,
            tighter_elbo=args.tighter_elbo, qU_type=args.qU_type,
            neural_network_type="ResNet", out_dim=None, hidden_dims=None, num_blocks=args.num_blocks,
            spectral_norm=args.spectral_norm, sn_ub=args.sn_ub, jitter=jitter, use_cache_for_svgp=args.use_cache_for_svgp,
            k_m=args.k_m, scale_factor=args.scale_factor,
        )
    elif args.neural_network_type == "Identity":
        model = dkl_lvmogp_ipv(
            D_H=args.D_H, M=args.M, qH_mean_field=args.qH_mean_field, whitening=args.whitening,
            tighter_elbo=args.tighter_elbo, qU_type=args.qU_type,
            neural_network_type="Identity", out_dim=None, hidden_dims=None, num_blocks=None,
            spectral_norm=None, sn_ub=None, jitter=jitter, use_cache_for_svgp=args.use_cache_for_svgp,
            k_m=args.k_m, scale_factor=args.scale_factor,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load state dict
    state_dict = torch.load(model_file_path, map_location=device, weights_only=True)

    # Apply weights to the model
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print("Model loaded successfully!")

    return model


def k_means_clustering(qH_means: torch.Tensor, k: int = 6):
    from sklearn.cluster import KMeans
    # qH_means: [P, D_H], after clustering, return cluster assignments: [P]
    data_np = qH_means.detach().cpu().numpy()

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_np)

    labels_np = kmeans.labels_

    labels = torch.from_numpy(labels_np).to(device=qH_means.device, dtype=torch.long)

    return labels


def add_group_signature_score(adata, gene_list, score_name='my_genes_avg'):
    import numpy as np
    import scipy
    tmp_adata = adata[:, gene_list].copy()
    sc.pp.normalize_total(tmp_adata, target_sum=1e4)
    sc.pp.log1p(tmp_adata)
    # axis=1 for mean across genes, provides a score per cell
    if scipy.sparse.issparse(tmp_adata.X):
        mean_expression = tmp_adata.X.mean(axis=1) #  np.matrix
        mean_expression = np.array(mean_expression).flatten() # 1D data array
    else:
        mean_expression = np.mean(tmp_adata.X, axis=1)
    adata.obs[score_name] = mean_expression
    return adata


if __name__ == "__main__":
    results_folder = "./results/ipv/dkl_lvmogp/FCNet_Gaussian_no_sn/1227_01_20_15"
    model = load_model_ipv(results_folder)
    qH_means = model.qH.mean_qH
    print("Latent Variables have shape: ", qH_means.shape)
    labels = k_means_clustering(qH_means, k=6)
    print("Cluster assignments have shape: ", labels.shape)
    print(labels[:20])

