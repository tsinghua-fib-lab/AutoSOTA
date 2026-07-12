import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse
import h5py
import scanpy as sc

import tensorflow as tf
import tensorflow_probability as tfp

# from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LeakyReLU
# from tensorflow.keras.utils import Progbar

tfd = tfp.distributions

import matplotlib.pyplot as plt
from types import SimpleNamespace
from sklearn.model_selection import train_test_split

train_id = "24"
test_id = "23"

data_path = "./data/" #path to training data and test data
path_root = "./results/" #path to results

train = sc.read_h5ad(data_path+"dataset"+train_id+"_adata.h5ad")

sc.pp.filter_cells(train, min_genes=1)
sc.pp.filter_genes(train, min_cells=1)

adata = train.copy()

sc.pp.normalize_total(train, target_sum=1e4)
sc.pp.log1p(train)
sc.pp.highly_variable_genes(train,n_top_genes = 4000)

train = adata[:, train.var.highly_variable]
print(train.shape)
X = train.X
Y = train.obsm['protein_expression'].values
gene_names = train.var_names
ADT_names = train.obsm['protein_expression'].keys()

dim_input_arr = [len(gene_names), len(ADT_names)]
config = {
    'dim_input_arr': dim_input_arr,
    'dimensions':[256],
    'dim_latent':32,
    'dim_block': np.array(dim_input_arr),
    # 'dist_block':['NB','NB'],
    'dist_block':['Gaussian', 'Gaussian'],
    'dim_block_enc':np.array([256, 128]),
    'dim_block_dec':np.array([256, 128]),
    'dim_block_embed':np.array([32, 16]),

    'block_names':np.array(['rna', 'adt']),
    'uni_block_names':np.array(['rna','adt']),

    'beta_kl':1.,
    'beta_unobs':2./3.,
    'beta_modal':np.array([0.15,0.85]),
    'beta_reverse':0.5,

    "p_feat" : 0.2,
    "p_modal" : np.ones(2)/2,
}
config = SimpleNamespace(**config)
n_samples = 50


# preprocess
X = np.log(X/np.sum(X, axis=1, keepdims=True)*1e4+1.)
#X = np.log(X/np.sum(X, axis=1)*1e4+1.)
Y = np.log(Y/np.sum(Y, axis=1, keepdims=True)*1e4+1.)

# data spliting
data = np.c_[X, Y]
# data_norm = np.exp(data)-1


from functools import partial
from scVAEIT.VAEIT import scVAEIT
model = scVAEIT(config, data)
del X, Y, data

model.train(
    valid=False, num_epoch=300, batch_size=512, save_every_epoch=50,
    verbose=True, checkpoint_dir=path_root+'checkpoint/'
)

test = sc.read_h5ad(data_path+"dataset"+test_id+"_adata.h5ad")
test = test[:,train.var_names]

test_cells = test.obs_names
X = test.X
Y = test.obsm['protein_expression'].values
X = np.log(X/np.sum(X, axis=1, keepdims=True)*1e4+1.)
#X = np.log(X/np.sum(X, axis=1)*1e4+1.)
Y = np.log(Y/np.sum(Y, axis=1, keepdims=True)*1e4+1.)

# data spliting
test = np.c_[X, np.zeros(Y.shape)]
X_test = X
Y_test = Y
del X, Y

dataset_test = tf.data.Dataset.from_tensor_slices((
    test,
    model.cat_enc.transform(np.zeros((test.shape[0],1))).toarray().astype(np.float32),
    np.zeros(test.shape[0]).astype(np.int32)
)).batch(512).prefetch(tf.data.experimental.AUTOTUNE)

mask_adt = np.zeros((1,np.sum(model.vae.config.dim_input_arr)), dtype=np.float32)
mask_adt[:,model.vae.config.dim_input_arr[0]:] = -1.
recon = model.vae.get_recon(dataset_test, mask_adt)
X_hat = recon[:, :model.vae.config.dim_input_arr[0]] #get predicted mRNA data
Y_hat = recon[:, model.vae.config.dim_input_arr[0]:] #get predicted protein data

_df = pd.DataFrame({
    'ADT':ADT_names,
    'Pearson r':[np.corrcoef(Y_hat[:,i], Y_test[:,i])[0,1] for i in np.arange(Y_hat.shape[1])],
    'Spearman r':[scipy.stats.spearmanr(Y_hat[:,i], Y_test[:,i])[0] for i in np.arange(Y_hat.shape[1])],
    'MSE':np.mean((Y_hat-Y_test)**2, axis=0)
})
print(np.quantile(_df['Pearson r'].values, [0.,0.5,1.0]))
print(np.quantile(_df['Spearman r'].values, [0.,0.5,1.0]))
print(np.quantile(_df['MSE'].values, [0.,0.5,1.0]))
_df.to_csv(path_root+test_id+'_from_'+train_id+'_index_scVAEIT.csv')

results = pd.DataFrame(data = Y_hat,index = test_cells,columns = train.obsm['protein_expression'].columns)
results.to_csv(path_root+test_id+'_from_'+train_id+'_scVAEIT1.csv')

