import scanpy as sc
import anndata as ad
import torch
import scarches as sca
import matplotlib.pyplot as plt
import numpy as np
import scvi as scv
import pandas as pd

# import os
# os.chdir('../')
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=UserWarning)

from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from scipy.io import mmread, mmwrite

train_id = "24"
test_id = "23"
data_path = "../dataset/"
train_data_path = data_path + "dataset" + train_id + "_adata.h5ad" #path to training data
test_data_path = data_path + "dataset" + test_id + "_adata.h5ad" #path to test data
output_path = "../Results/" #path to results

adata_ref = sc.read(train_data_path)
adata_query = sc.read(test_data_path)

genes_inter = adata_ref.var_names.intersection(adata_query.var_names)
adata_ref = adata_ref[:,genes_inter].copy()
adata_query = adata_query[:,genes_inter].copy()

adata_ref.obs["batch"] = "train"
adata_query.obs["batch"] = "test"

# put matrix of zeros for protein expression (considered missing)
pro_exp = adata_ref.obsm["protein_expression"]
data = np.zeros((adata_query.n_obs, pro_exp.shape[1]))
adata_query.obsm["protein_expression"] = pd.DataFrame(columns=pro_exp.columns, index=adata_query.obs_names, data = data)

adata_full = ad.concat([adata_ref, adata_query])

batch = ['train']*adata_ref.shape[0]+['test']*adata_query.shape[0]

sc.pp.highly_variable_genes(
    adata_ref,
    n_top_genes=4000,
    flavor="seurat_v3",
    batch_key="batch",
    subset=True,
)

adata_query = adata_query[:, adata_ref.var_names].copy()

sca.models.TOTALVI.setup_anndata(
    adata_ref,
    batch_key="batch",
    protein_expression_obsm_key="protein_expression"
)

arches_params = dict(
    use_layer_norm="both",
    use_batch_norm="none",
)
vae_ref = sca.models.TOTALVI(
    adata_ref, 
    **arches_params
)

vae_ref.train()

adata_ref.obsm["X_totalVI"] = vae_ref.get_latent_representation()
sc.pp.neighbors(adata_ref, use_rep="X_totalVI")
sc.tl.umap(adata_ref, min_dist=0.4)

# sc.pl.umap(
#     adata_ref,
#     color=["batch"],
#     frameon=False,
#     ncols=1,
#     title="Reference"
# )

dir_path = "../Results/scArches/Twobatches/saved_model/"+"dataset"+train_id+"to"+test_id+"/"
vae_ref.save(dir_path, overwrite=True)

vae_q = sca.models.TOTALVI.load_query_data(
    adata_query, 
    dir_path, 
    freeze_expression=True
)

vae_q.train(200, plan_kwargs=dict(weight_decay=0.0))

adata_query.obsm["X_totalVI"] = vae_q.get_latent_representation()
sc.pp.neighbors(adata_query, use_rep="X_totalVI")
sc.tl.umap(adata_query, min_dist=0.4)

_, imputed_proteins = vae_q.get_normalized_expression(
    adata_query,
    n_samples=25,
    return_mean=True,
    transform_batch=["train"],
)

adata_query.obs = pd.concat([adata_query.obs, imputed_proteins], axis=1)

imputed_proteins.to_csv(output_path+test_id+"_from_"+train_id+"_scArches1.csv")