# Data

The repository does not include datasets or checkpoints.

By default, datasets are stored under `data/` and are ignored by Git. PyTorch Geometric downloads common datasets automatically:

- TUDataset graph datasets such as `MUTAG`, `PROTEINS`, `NCI1`, `BZR`, `COX2`, `IMDB-BINARY`, and `COLLAB`
- Planetoid datasets such as `Cora`, `CiteSeer`, and `PubMed`
- Amazon datasets such as `Photo` and `Computers`

Node-level few-shot experiments first create ego-graph files:

```text
data/node_graphs/<dataset>_graph.pt
```

These files can be regenerated with `scripts/build_node_graphs.py`.
