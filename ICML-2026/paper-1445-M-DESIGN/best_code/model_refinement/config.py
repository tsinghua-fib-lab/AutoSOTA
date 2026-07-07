# Copyright (c) 2024-Current Jialiang Wang
# License: Apache-2.0 license

design_dimensions = {
    'node_classification': ['neigh', 'norm', 'agg', 'comb', 'l_mp', 'stage'],
    'link_prediction': ['decode', 'norm', 'agg', 'comb', 'l_mp', 'stage'],
    'graph_classification': ['decode', 'norm', 'agg', 'comb', 'l_mp', 'stage']
}

design_choice_translate = {
    'neigh': 'Intra-layer Neighbor Mechanism',
    'edge_index': 'edge_index (ordinary)',
    'edge_index_knn': 'edge_index_knn (with diffusion-based re-wiring)',
    'edge_index_2hop': 'edge_index_2hop (higher-order)',
    'edge_index_knn_rwpe': 'edge_index_knn (with random-walk structural encoding-based re-wiring)',
    'edge_index_knn_lepe': 'edge_index_knn (with Laplacian eigenvector positional encoding-based re-wiring)',
    'norm': 'Intra-layer Edge Weight',
    'degree_sys': 'systematic normalization',
    'degree_row': 'random-walk normalization',
    'fagcn_like': 'self-attention',
    'rel_rwpe': 'relative random-walk structural encoding',
    'rel_lepe': 'relative Laplacian eigenvector positional encoding',
    'agg': 'Intra-layer Aggregation',
    'add': 'add',
    'mean': 'mean',
    'max': 'max',
    'min': 'min',
    'comb': 'Intra-layer Combination',
    'concat': 'concat',
    'l_mp': 'Number of Layers',
    '4': '4',
    '6': '6',
    'stage': 'Inter-layer Aggregation',
    'skipconcat': 'concat (with skip connections)',
    'skipsum': 'sum (with skip connections)',
    'ppr_01': 'personalized PageRank 0.1 decay',
    'ppr_0.1': 'personalized PageRank 0.1 decay',
    'gpr': 'gpr (layer-wise adaptive)',
    'lstm': 'bidirectional LSTM',
    'node_adaptive': 'gating mechanism'
}

