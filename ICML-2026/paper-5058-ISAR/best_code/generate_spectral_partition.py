#!/usr/bin/env python3
"""Generate spectral clustering seed partition for SCC."""

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import sys

DATASET_CC = "/datasets/bitcoinotc_dedup.cc"

def load_graph_edges(cc_file):
    edges = []
    n_nodes = 0
    with open(cc_file) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 3:
                continue
            u, v, s = int(p[0]), int(p[1]), int(p[2])
            edges.append((u-1, v-1, s))
            n_nodes = max(n_nodes, u, v)
    return n_nodes, edges

def build_signed_laplacian(n_nodes, edges):
    rows, cols, data = [], [], []
    deg = np.zeros(n_nodes)
    for u, v, s in edges:
        rows.append(u)
        cols.append(v)
        data.append(s)
        rows.append(v)
        cols.append(u)
        data.append(s)
        deg[u] += 1
        deg[v] += 1
    A = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    D = diags(deg)
    L = D - A
    return L

def spectral_clustering(L, n_clusters=400, n_eigenvectors=10):
    print("Computing {} smallest eigenvectors of signed Laplacian...".format(n_eigenvectors))
    eigenvalues, eigenvectors = eigsh(L, k=n_eigenvectors, which="SM")
    print("Smallest eigenvalues: {}...{}".format(eigenvalues[:5], eigenvalues[-3:]))
    embedding = eigenvectors.copy()
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    embedding = embedding / norms
    print("Running k-means with k={}...".format(n_clusters))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, random_state=42)
    labels = kmeans.fit_predict(embedding)
    return labels

def main():
    output_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/spectral_partition.txt"
    n_clusters = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    n_eig = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    print("Loading graph from {}...".format(DATASET_CC))
    n_nodes, edges = load_graph_edges(DATASET_CC)
    print("Graph: {} nodes, {} edges".format(n_nodes, len(edges)))
    print("Building signed Laplacian...")
    L = build_signed_laplacian(n_nodes, edges)
    labels = spectral_clustering(L, n_clusters=n_clusters, n_eigenvectors=n_eig)
    with open(output_file, "w") as f:
        for label in labels:
            f.write("{}\n".format(label))
    print("Partition written to {} ({} unique clusters)".format(output_file, len(set(labels))))

if __name__ == "__main__":
    main()
