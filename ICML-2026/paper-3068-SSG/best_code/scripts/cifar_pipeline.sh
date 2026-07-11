#!/bin/bash

# CIFAR-100 Complete Pipeline: Graph Construction + Training
# This script shows the full pipeline from embedding clustering to training

EMBEDDING_PATH="/path/to/cifar100_embeddings.pt"
CLUSTER_OUTPUT="/path/to/cifar100_clusters"
GRAPH_OUTPUT="/path/to/cifar100_graphs"
FULL_GRAPH_PATH="/path/to/cifar100_full_graph.pkl"

# ============================================
# Step 1: Cluster embeddings
# ============================================
python graph_construction/cifar_cluster.py \
    --input $EMBEDDING_PATH \
    --output $CLUSTER_OUTPUT \
    --n-clusters 10 \
    --kmeans-iter 15

# ============================================
# Step 2a: Build cluster-based graph
# ============================================
python graph_construction/cifar_build_graph.py \
    --input $CLUSTER_OUTPUT \
    --output $GRAPH_OUTPUT

# ============================================
# Step 2b: Build full graph (alternative)
# ============================================
python graph_construction/cifar_build_full_graph.py \
    --input $EMBEDDING_PATH \
    --output $FULL_GRAPH_PATH \
    --batch-size 5000

# ============================================
# Step 3: Train with PFB (using full graph)
# ============================================
python training/cifar/cifar_full_graph_prune.py \
    --graph-path $FULL_GRAPH_PATH \
    --dataloader-ratio 0.625 \
    --score-weight 1.0 \
    --model r18 \
    --num_epoch 200 \
    --batch-size 128 \
    --max-lr 0.05
