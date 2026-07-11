# Reproducing Few-Shot Results

SCGFM uses a two-stage protocol:

1. Pretrain geometric bases on source-domain graphs.
2. Freeze the encoder and evaluate target-domain tasks with a ProtoNet classifier.

## Graph-Level Few-Shot

```bash
python scripts/pretrain_graph.py --config configs/graph/pretrain_proteins_nci1_bzr.yaml
python scripts/eval_graph_fewshot.py --config configs/graph/fewshot_cross_domain.yaml --checkpoint outputs/graph_pretrain/model.pt
```

Edit `configs/graph/fewshot_cross_domain.yaml` to choose the target dataset and the few-shot setup.

## Node-Level Few-Shot

Node tasks are converted into graph tasks by sampling PPR ego-graphs around labeled nodes.

```bash
python scripts/build_node_graphs.py --config configs/node/build_ppr_ego_graphs.yaml
python scripts/pretrain_node.py --config configs/node/pretrain_photo_computers.yaml
python scripts/eval_node_fewshot.py --config configs/node/fewshot_cross_domain.yaml --checkpoint outputs/node_pretrain/model.pt
```

## Outputs

Each evaluation writes:

- `metrics.json`: mean and standard deviation across runs
- `results.csv`: per-run accuracy and balanced accuracy
