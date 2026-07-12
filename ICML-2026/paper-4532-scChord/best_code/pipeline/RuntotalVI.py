import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scvi

train_id = "24"
test_id = "23"
data_path = "../dataset/"
# File naming with "dataset" prefix
train_data_path = f"{data_path}dataset{train_id}_adata.h5ad"  # path to training data
test_data_path = f"{data_path}dataset{test_id}_adata.h5ad"    # path to test data
output_path = "../Results/" #path to results

adata_batch1 = sc.read(train_data_path)
adata_batch2 = sc.read(test_data_path)

batch_idx = 1
adata = ad.concat([adata_batch1, adata_batch2],axis=0)
adata.obs_keys

modality = ['batch1']*adata_batch1.shape[0]+['batch2']*adata_batch2.shape[0]
adata.obs['batch']=modality
# batch = adata.obs.batch.values.ravel()
# adata.obs.batch

batch = modality

batch_set = ['batch1', 'batch2']
hold_out_batch = batch_set[batch_idx]
held_out_proteins = adata.obsm["protein_expression"][[batch[i] == hold_out_batch for i in range(len(batch))]].copy()
adata.obsm["protein_expression"].loc[[batch[i] == hold_out_batch for i in range(len(batch))]] = np.zeros_like(adata.obsm["protein_expression"][[batch[i] == hold_out_batch for i in range(len(batch))]])

sc.pp.highly_variable_genes(
    adata,
    batch_key="batch",
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=True
)

scvi.model.TOTALVI.setup_anndata(adata, batch_key="batch", protein_expression_obsm_key="protein_expression")

model = scvi.model.TOTALVI(
    adata,
    latent_distribution="normal",
    n_layers_decoder=2
)
model.train()

adata.obsm["X_totalVI"] = model.get_latent_representation()
adata.obsm["protein_fg_prob"] = model.get_protein_foreground_probability(transform_batch=batch_set[int(batch_idx==0)])
_, protein_means = model.get_normalized_expression(
    n_samples=25,
    transform_batch=batch_set[int(batch_idx==0)],
    include_protein_background=True,
    sample_protein_mixing=False,
    return_mean=True,
)

protein = protein_means.loc[[batch[i]==hold_out_batch for i in range(len(batch))]]

protein.to_csv(output_path+test_id+"_from_"+train_id+"_totalVI.csv")
