# PSAHS Code Analysis - Paper 2891

## Evaluation Path
- `eval.py` → main entry point for DBLP-ACM evaluation
- Sets paper hyperparameters inline: epochs=300, h_threshold=1.0, start_epoch=200, rw_freq=15
- Loads data via `datasets.prepare_dblp_acm()`
- Trains MLP (attribute-only classifier), then PSAHS with GNN_adv
- Reports `acc_tgt_test` per seed + summary mean/std
- Outputs metrics to `outputs/metrics.json`

## Key Files
| File | Role | Safe to Modify |
|------|------|----------------|
| `eval.py` | Main evaluation | Yes - training loop, MLP reuse, scheduling |
| `psahs/data/datasets.py` | Data loading + graph structure adjustment | Yes - adjustment functions, edge stats |
| `main/models.py` | GNN_adv, GCN_reweight, MLP models | Yes - new backbone options, dropout |
| `main/args.py` | CLI arguments and defaults | Yes - new parameters |
| `psahs/training_utils.py` | Optimizer, metrics, logging | No - metric computation |
| `psahs/edge_stats.py` | Edge statistics computation | Yes - diagnostics |

## Metric Parser
- stdout: `[Summary] {"acc_tgt_test": "XX.XXXXX +/- Y.YYYYY"}`
- File: `outputs/metrics.json` → `{"accuracy": X, "accuracy_std": Y, "per_seed": {...}}`
- Primary metric: Accuracy (mean over 5 seeds)

## Red-Line Files (NEVER MODIFY)
- `psahs/training_utils.py`: `accuracy()`, `classification_scores()`, `ce_loss()` - metric computation
- Dataset loader splits: 60/20/20 train/val/test in `prepare_dblp_acm()`
- `/tools/record_score.sh`: scoring script
- Test data in `/repo/dataset/`

## Risky Modification Targets
- `prepare_dblp_acm()`: data splits - DO NOT TOUCH
- `adjust_graph_structure_fast_source()`: source graph preprocessing - careful
- `adjust_graph_structure_fast_target_Plabel()`: PRIMARY TARGET - pseudo-label based edge modification

## Known Bottlenecks
1. High seed-to-seed variance (best 83.45% vs mean 70.49%): MLP pseudo-labels differ per seed
2. Binary GNN==MLP agreement without confidence weighting
3. Hard h_threshold=1.0 from epoch 200 when pseudo-labels are worst
4. No degree-awareness in edge modification

## Container Paths
- Repo: `/repo`
- Dataset: `/repo/dataset/`
- Cache: `/autosota_cache/`, `/datasets/`, `/models/`
- Scores: `/autosota_artifacts/paper-2891/sota/scores.jsonl`
