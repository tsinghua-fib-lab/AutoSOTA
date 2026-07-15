# Federated Learning with PyTorch Geometric (PyG)

This simple example demonstrates how to perform federated learning within the `APPFL` framework on graph-structured data using PyTorch Geometric (PyG) via `Globus Compute`. The implementation focuses on node classification on the Cora citation network. Each client is assigned a subset of nodes for training, while all clients have access to the full graph structure (nodes, edges, and features). This setup enables federated node classification where clients train GNN models on their assigned nodes.

## Dataset: Cora Citation Network

**Cora** is a scientific publication citation network widely used for benchmarking GNN models.

- **Nodes**: 2,708 scientific publications
- **Edges**: 10,556 citation links (directed)
- **Features**: 1,433 bag-of-words features per node
- **Classes**: 7 research topics (Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory)
- **Task**: Multi-class node classification

**💡 Note**: The provided data scripts will download the datasets automatically when you run it for the first time.

## Installation

1. **APPFL Framework**: Follow the instructions in the main `README.md` to set up the `APPFL` framework and its dependencies.

2. **PyTorch Geometric**:
    ```bash
    # Install PyTorch Geometric
    pip install torch-geometric

    # Install optional dependencies for PyG
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
    # Replace ${TORCH} with your PyTorch version (e.g., 2.0.0) and ${CUDA} with your CUDA version (e.g., cu118)
    # For CPU-only: use cpu instead of cuda version
    ```

For more details on PyG installation, see: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Quick Start

Ensure the configuration files in `examples/resources/config_gc/pyg/` reflect the correct Globus Endpoint ID

```bash
cd examples
python globus_compute/run.py --server_config ./resources/config_gc/pyg/server_fedcompass_pyg.yaml --client_config ./resources/config_gc/pyg/clients_pyg.yaml
```
