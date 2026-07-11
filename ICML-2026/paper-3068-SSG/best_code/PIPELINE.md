# ImageNet Embedding Extraction and Graph Construction Pipeline

This document describes the scripts and usage for extracting ImageNet embeddings, clustering samples, and constructing graph data.

## Pipeline Overview

```text
inference.py    ->    cluster.py         ->    build_graph.py            ->    Training w/ Pruning
   |                      |                         |                          
[Extract embeddings]  [Class-wise KMeans]      [Build similarity graphs]       
   |                      |                         |                          
  .pkl/.json          parquet files             .graph.pkl                      
```

## 1. Embedding Extraction

### `graph_construction/inference.py`

Runs inference over ImageNet samples and extracts features from the model embedding layer.

**Main functionality:**
- Loads the ImageNet dataset, including distributed multi-GPU support.
- Runs inference with models such as ResNet and Swin Transformer.
- Extracts embedding-layer features through `capture_feature.py`.
- Saves per-sample metadata.

**Output fields:**
| Field | Description |
|------|-------------|
| `image_name` | Image path |
| `logits` | Model-output logits |
| `loss` | Cross-entropy loss |
| `feature` | Embedding vector |
| `true_class` | Ground-truth class ID |
| `pred_class` | Predicted class ID |
| `correct` | Whether the prediction is correct |

**Usage example:**
```bash
python graph_construction/inference.py \
    --data-path /path/to/imagenet \
    --model swin_t \
    --output-dir /path/to/output \
    --resume /path/to/checkpoint.pth \
    --test-only
```

## 2. Clustering

### `graph_construction/cluster.py`

Runs KMeans clustering over samples within each class using extracted embeddings.

**Main functionality:**
- Reads PKL files produced by inference.
- Groups samples by `true_class`.
- Runs KMeans clustering with Faiss.
- Saves clustering results as parquet files.

**Usage example:**
```bash
python graph_construction/cluster.py \
    --input /path/to/embeddings \
    --output /path/to/clusters
```

**Arguments:**
| Argument | Default | Description |
|---------|---------|-------------|
| `feature_column` | `"feature"` | Embedding column name |
| `n_clusters_per_class` | 25 | Number of subclusters per class |
| `max_samples_per_class` | 5000 | Maximum number of samples used per class |
| `kmeans_iter` | 25 | Number of KMeans iterations |
| `use_gpu` | False | Whether to use GPU acceleration |
| `max_rows_per_file` | 10000 | Maximum rows per parquet file |

**Output structure:**
```text
output_dir/
`-- class_{id}/
    |-- centroids.npy
    |-- cluster_mapping.json
    `-- clusters/
        |-- cluster_0000_00000.parquet
        |-- cluster_0000_00001.parquet
        `-- ...
```

## 3. Graph Construction

### `graph_construction/build_graph.py`

Builds sample-level similarity graphs from clustering results.

**Main functionality:**
- Loads clustering results from parquet files.
- Computes pairwise cosine similarity or distance between samples.
- Builds fully connected graphs.
- Saves graph data as PKL files.

**Graph data structure:**
```python
{
    'nodes': [...],           # node list
    'edges': [...],           # edge list (i, j, weight)
    'adj_matrix': csr_matrix, # sparse adjacency matrix
    'num_nodes': N,
    'avg_loss': float,
    'within_cluster_feature_variance': float,
}
```

**Usage example:**
```bash
python graph_construction/build_graph.py \
    --input /path/to/clusters \
    --output /path/to/graphs \
    --feature_col feature
```

**Related files:**
- `build_graph_fast.py`: optimized implementation using numpy matrix multiplication.

## 4. Other Related Files

| File | Description |
|------|-------------|
| `graph_construction/capture_feature.py` | Feature-extraction utility used by `inference.py` |
| `training/imagenet/train_graph_static_prob.py` | Static-probability training with graph information |
| `training/imagenet/train_online.py` | PFB online pruning training |

## 5. Typical Workflow

```bash
# Step 1: Extract embeddings
python graph_construction/inference.py \
    --data-path /path/to/imagenet \
    --model swin_t \
    --output-dir /path/to/output/embeddings \
    --resume checkpoint.pth \
    --test-only

# Step 2: Cluster samples
python graph_construction/cluster.py \
    --input /path/to/output/embeddings \
    --output /path/to/output/clusters

# Step 3: Build graphs
python graph_construction/build_graph.py \
    --input /path/to/output/clusters \
    --output /path/to/output/graphs
```

## 6. Output Files

Generated graph files (`*.graph.pkl`) are saved under the specified output directory and can be passed directly to `train_graph_static_prob.py` through the `--graph-dir` argument.
