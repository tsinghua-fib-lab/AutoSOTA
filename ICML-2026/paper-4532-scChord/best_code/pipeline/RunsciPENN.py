import numpy as np
from matplotlib import pyplot
from copy import deepcopy
from time import time
from math import ceil
from scipy.stats import spearmanr, gamma, poisson
from anndata import AnnData, read_h5ad
import scanpy as sc
from scanpy import read
import pandas as pd
from scipy.io import mmread
from sciPENN.sciPENN_API import sciPENN_API
from numpy import intersect1d, setdiff1d, quantile, unique, asarray, zeros
import scanpy as sc

train_id = "24"
test_id = "23"
data_path = "../dataset/"
train_data_path = data_path + "dataset" + train_id + "_adata.h5ad" #path to training data
test_data_path = data_path + "dataset" + test_id + "_adata.h5ad" #path to test data
output_path = "../Results/" #path to results

adata_gene = sc.read_h5ad(train_data_path)
adata_protein = AnnData(X = adata_gene.obsm['protein_expression'])

adata_gene_test = sc.read(test_data_path)
adata_protein_test = pd.DataFrame(0, columns=adata_gene_test.obsm['protein_expression'].columns, index=adata_gene_test.obsm['protein_expression'].index)
adata_protein_test = AnnData(adata_protein_test)

adata_protein_test.obs['sample'] = [1]*adata_protein_test.shape[0]

ref = set(adata_protein_test.var.index)

prots = []
for x in adata_protein.var.index:
    if x in ref:
        prots.append(x)
        
sciPENN = sciPENN_API([adata_gene], [adata_protein], adata_gene_test)

start = time()
sciPENN.train(n_epochs = 10000, ES_max = 12, decay_max = 6, 
             decay_step = 0.1, lr = 10**(-3), weights_dir = "weights_dir/"+test_id+"_from_"+train_id)
imputed_test = sciPENN.predict()
time() - start

embedding = sciPENN.embed()
sciPENN_pre = pd.DataFrame(data = imputed_test.X,index = imputed_test.obs_names,columns = imputed_test.var_names)
sciPENN_pre.to_csv(output_path+test_id+"_from_"+train_id+"_totalVI.csv")