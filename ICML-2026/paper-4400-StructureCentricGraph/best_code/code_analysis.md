# Code Analysis - SCGFM (Paper 4400)

## Evaluation Path
- Script: `scripts/eval_node_fewshot.py`
- Config: `configs/node/fewshot_cross_domain.yaml`
- Checkpoint: `outputs/node_pretrain/model.pt`
- Flow: load model → SCGFMEncoder.encode_dataset() → evaluate_fewshot()
- Metrics: `accuracy_mean`, `accuracy_std` from `outputs/node_fewshot/metrics.json`

## Training Path
- Script: `scripts/pretrain_node.py`
- Config: `configs/node/pretrain_photo_computers.yaml`
- Graph building: `scripts/build_node_graphs.py` with `configs/node/build_ppr_ego_graphs.yaml`
- Already done (graphs in `data/node_graphs/`)

## Metric Parser
- Last line of stdout: Python dict with `accuracy_mean`, `accuracy_std`, etc.
- Also saved to `outputs/node_fewshot/metrics.json`

## Key Files
| File | Role | Risk |
|------|------|------|
| `scgfm/models/geometric_bases.py` | Core model, basis learning, loss computation | SAFE to modify |
| `scgfm/training.py` | Pretraining loop, optimizer setup | SAFE to modify |
| `scgfm/encoders.py` | SCGFMEncoder for downstream eval | SAFE to modify |
| `scgfm/fewshot.py` | ProtoClassifier, evaluate_fewshot | SAFE to modify |
| `scgfm/data/graph_features.py` | Graph statistics precomputation | SAFE to modify |
| `scripts/eval_node_fewshot.py` | Eval entry point | DO NOT change metric defs |
| `configs/node/pretrain_photo_computers.yaml` | Pretrain config | SAFE to modify |
| `configs/node/fewshot_cross_domain.yaml` | Eval config | SAFE to modify (hyperparams only) |

## Known Issues
1. `compute_gw_distance_for_attention` generates new random `theta` each call (no seed)
2. Train/eval τ mismatch: pretrain τ=0.3, eval τ=0.1
3. Train/eval num_projections mismatch: pretrain=50, eval=200
4. Diversity loss (hinge margin) saturates to ~0 by epoch 60
5. No LR scheduling or gradient clipping in training loop
6. Frozen encoder has no task-specific adaptation

## Data Locations
- PPR ego-graphs: `data/node_graphs/*.pt` (already built)
- Pretrained model: `outputs/node_pretrain/model.pt`
- Eval output: `outputs/node_fewshot/metrics.json`

## PyTorch 2.1.0 Compatibility Fixes (already applied)
- `torch.amp.GradScaler` → `torch.cuda.amp.GradScaler`
- `torch.autocast` → `torch.cuda.amp.autocast`
