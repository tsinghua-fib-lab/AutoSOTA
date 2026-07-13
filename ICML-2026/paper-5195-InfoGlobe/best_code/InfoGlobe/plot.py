import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


def factor_sort(adata):
    embedding = adata.obsm['markov_embedding']
    factor_matrix = adata.obsm['markov_embedding'].T
    factor_linkage = linkage(
        pdist(factor_matrix, metric='correlation'),  # 使用相关性距离
        method='ward'  # 使用ward方法
    )

    factor_order = leaves_list(factor_linkage)

    embedding_clustered = factor_matrix[factor_order, :]
    adata.obsm['markov_embedding_clustered'] = embedding_clustered.T

    factor_labels = [f'factor_{i}' for i in range(embedding.shape[1])]
    factor_df = pd.DataFrame({
        'factor_id': factor_labels,
        'cluster_order': factor_order
    }).sort_values('cluster_order')
    return adata, factor_order

def factor_plot(adata, factor_order):
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        adata.obsm['markov_embedding_clustered'],  
        cmap='coolwarm',
        center=0,
        xticklabels=[f'F{idx}' for idx in factor_order],  
        yticklabels=False,  
    )

    cell_types = adata.obs['subtype'].values
    le = LabelEncoder()
    cell_types_encoded = le.fit_transform(cell_types)

    type_boundaries = np.where(np.diff(cell_types_encoded))[0] + 0.5

    for boundary in type_boundaries:
        plt.axhline(boundary, color='black', linewidth=2)

    unique_types = le.classes_  
    type_positions = []
    for ct in unique_types:
        positions = np.where(cell_types == ct)[0]
        if len(positions) > 0:
            type_positions.append(positions.mean())

    plt.yticks(type_positions, unique_types, rotation=0)


    plt.tight_layout()
    plt.show()