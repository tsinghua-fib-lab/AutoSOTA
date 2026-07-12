import logging
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import scanpy as sc

from sklearn.decomposition import TruncatedSVD
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge

logging.basicConfig(level=logging.INFO)

# load datasets
train_id = "24"
test_id = "23"
data_path = "../dataset/"

train_data_path = f"{data_path}dataset{train_id}_adata.h5ad"  # path to training data
test_data_path = f"{data_path}dataset{test_id}_adata.h5ad"    # path to test data
output_path = "../Results/" #path to results

logging.info('Reading data files...')
input_train_mod1 = sc.read(train_data_path)

input_test_mod1 = sc.read(test_data_path)

genes_inter = input_train_mod1.var_names.intersection(input_test_mod1.var_names)
input_train_mod1 = input_train_mod1[:,genes_inter].copy()
input_test_mod1 = input_test_mod1[:,genes_inter].copy()

input_train_mod1.obs['batch'] = pd.Categorical(len(input_train_mod1.obs_names)*['batch1'])
input_test_mod1.obs['batch'] = pd.Categorical(len(input_test_mod1.obs_names)*['batch2'])

pred_dimx = input_test_mod1.shape[0] # Number of cells
pred_dimy = input_train_mod1.obsm['protein_expression'].shape[1]  # Number of proteins

feature_obs = input_train_mod1.obs
# gs_obs = input_train_mod2.obs
gs_obs = input_train_mod1.obs

batches = input_train_mod1.obs.batch.unique().tolist()
batch_len = len(batches)

index = input_test_mod1.obsm['protein_expression'].index
columns = input_train_mod1.obsm['protein_expression'].columns
# ad.

input_train = ad.concat(
    {"train": input_train_mod1, "test": input_test_mod1},
    axis=0,
    join="outer",
    label="group",
    fill_value=0,
    index_unique="-"
)

logging.info('Determine parameters by the modalities')
# mod1_type = input_train_mod1.var.feature_types[0]
# mod1_type = mod1_type.upper()
# mod2_type = input_train_mod2.var.feature_types[0]
# mod2_type = mod2_type.upper()
mod1_type = "GEX"
mod2_type = "ADT"
n_comp_dict = {
        ("GEX", "ADT"): (300, 70, 10, 0.2),
        ("ADT", "GEX"): (None, 50, 10, 0.2),
        ("GEX", "ATAC"): (1000, 50, 10, 0.1),
        ("ATAC", "GEX"): (100, 70, 10, 0.1)
        }
logging.info(f"{mod1_type}, {mod2_type}")
n_mod1, n_mod2, scale, alpha = n_comp_dict[(mod1_type, mod2_type)]
logging.info(f"{n_mod1}, {n_mod2}, {scale}, {alpha}")

# Do PCA on the input data
logging.info('Models using the Truncated SVD to reduce the dimension')

if n_mod1 is not None and n_mod1 < input_train.shape[1]:
    embedder_mod1 = TruncatedSVD(n_components=n_mod1)
    mod1_pca = embedder_mod1.fit_transform(input_train.X).astype(np.float32)
    train_matrix = mod1_pca[input_train.obs['group'] == 'train']
    test_matrix = mod1_pca[input_train.obs['group'] == 'test']
else:
    train_matrix = input_train_mod1.to_df().values.astype(np.float32)
    test_matrix = input_test_mod1.to_df().values.astype(np.float32)

# if n_mod2 is not None and n_mod2 < input_train_mod2.shape[1]:
#     embedder_mod2 = TruncatedSVD(n_components=n_mod2)
#     train_gs = embedder_mod2.fit_transform(input_train_mod2.X).astype(np.float32)
# else:
#     train_gs = input_train_mod2.to_df().values.astype(np.float32)
if n_mod2 is not None and n_mod2 < input_train_mod1.obsm['protein_expression'].shape[1]:
    embedder_mod2 = TruncatedSVD(n_components=n_mod2)
    train_gs = embedder_mod2.fit_transform(input_train_mod1.obsm['protein_expression'].values).astype(np.float32)
else:
    embedder_mod2 = None 
    train_gs = input_train_mod1.obsm['protein_expression'].values.astype(np.float32)

del input_train
del input_train_mod1
# del input_train_mod2
del input_test_mod1

logging.info('Running normalization ...')
train_sd = np.std(train_matrix, axis=1).reshape(-1, 1)
train_sd[train_sd == 0] = 1
train_norm = (train_matrix - np.mean(train_matrix, axis=1).reshape(-1, 1)) / train_sd
train_norm = train_norm.astype(np.float32)
del train_matrix

test_sd = np.std(test_matrix, axis=1).reshape(-1, 1)
test_sd[test_sd == 0] = 1
test_norm = (test_matrix - np.mean(test_matrix, axis=1).reshape(-1, 1)) / test_sd
test_norm = test_norm.astype(np.float32)
del test_matrix

logging.info('Running KRR model ...')
y_pred = np.zeros((pred_dimx, pred_dimy), dtype=np.float32)
np.random.seed(1000)

for _ in range(5):
    np.random.shuffle(batches)
    for batch in [batches[:batch_len//2], batches[batch_len//2:]]:
        # for passing the test
        if not batch:
            batch = [batches[0]]

        logging.info(batch)
        kernel = RBF(length_scale = scale)
        krr = KernelRidge(alpha=alpha, kernel=kernel)
        logging.info('Fitting KRR ... ')
        krr.fit(train_norm[feature_obs.batch.isin(batch)], 
                train_gs[gs_obs.batch.isin(batch)])
        # y_pred += (krr.predict(test_norm) @ embedder_mod2.components_)
        if embedder_mod2 is not None:
            y_pred += (krr.predict(test_norm).dot(embedder_mod2.components_))
        else:
            y_pred += krr.predict(test_norm)

np.clip(y_pred, a_min=0, a_max=None, out=y_pred)
if mod2_type == "ATAC":
    np.clip(y_pred, a_min=0, a_max=1, out=y_pred)

logging.info('Storing annotated data...')
y_pred /= 10
protein_pred = pd.DataFrame(y_pred,index = index,columns = columns)
protein_pred.to_csv(output_path+test_id+"_from_"+train_id+"_Dengkw1.csv")

# # Store as sparse matrix to be efficient. Note that this might require
# # different classifiers/embedders before-hand. Not every class is able
# # to support such data structures.
# y_pred = csc_matrix(y_pred)

# logging.info("Generate anndata object ...")
# adata = ad.AnnData(
#     X=y_pred,
#     obs=obs,
#     var=var,
#     uns={
#         'dataset_id': dataset_id,
#         'method_id': meta['functionality_name'],
#     },
# )

# logging.info('Storing annotated data...')
# adata.write_h5ad(par['output'], compression = "gzip")
