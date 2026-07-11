# Code Analysis for FedSPA (Paper 3382)

## Evaluation Pipeline

1. **Entry point**: `main.py:main()` — parses args, loads CLIP model, pre-extracts features, creates clients + server, runs `server.run()`
2. **Server run**: `utils/server.py:Server.run()` — iterates `global_epochs` rounds: each round updates client prototypes, aggregates, then updates global semantic prototypes via InfoNCE contrastive loss
3. **Client update**: `utils/client.py:Client.update_prototypes()` — Tip-Adapter training (cosine affinity → cache logits + CLIP logits → cross-entropy loss)
4. **Generalization Accuracy** (GenAcc): `server.py:run()` computes `clip_logits = 100 * image_features @ global_semantic_prototypes.t()` across all clients, prints "Global accuracy based on text features: X.XX%"
5. **Personalization Accuracy** (PerAcc): `server.py:run()` prints "epoch N, global accuracy = X.XX%" at each round, where N = global_epochs (10). Last value is the personalization result after all rounds.

## Config

- `configs/ucf101.yaml`: alpha=9.0, beta=8.0, text_lr=0.0004, image_lr=0.0004, align=0.2

## Eval Command (in-container)

```bash
cd /repo && CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --root_path /datasets --cache_dir /autosota_cache/features \
  --datasets ucf101 --backbone RN --num_shots 8 --num_clients 10 \
  --partition distribution --dirichlet_alpha 0.1 --local_epochs 5 \
  --global_epochs 10 --local_epochs_server 100 --local_epochs_last 100 \
  --local_batch_size 8 --global_batch_size 8 --output_subdir repro
```

## Safe Modification Targets

| File | Function | Safe Changes |
|------|----------|-------------|
| `utils/server.py` | `compute_prototypes_from_clients()` | Prototype aggregation logic (shrinkage, refinement) |
| `utils/server.py` | `update_global_text_features()` | Loss function, optimizer config, alignment objective |
| `utils/server.py` | `run()` | Round-level logic, EMA, scheduling |
| `utils/client.py` | `update_prototypes()` | Training stability (gradient clipping, scheduler), adapter config |
| `configs/ucf101.yaml` | - | Hyperparameter values |
| `main.py` | `main()` | SEED (only for multi-seed), args defaults |

## Risky Files (DO NOT MODIFY)

| File | Reason |
|------|--------|
| `utils/utils.py` | `cls_acc`, `cls_acc_2`, `pre_load_features`, `distribution_label_skew_split_consistency` — evaluation and data partition |
| `datasets/ucf101.py` | Dataset loading, classnames, templates |
| `/tools/record_score.sh` | Score recording |
| Test data/splits in `/datasets/` | Immutable |

## Cache Paths

- Features: `/autosota_cache/features/ucf101/` (train_8_RN_{f,l}.pt, test_RN_{f,l}.pt)
- Dataset: `/datasets/ucf101/UCF-101-midframes/`
- Split JSON: `/datasets/ucf101/split_zhou_UCF101.json`
- CLIP weights: auto-downloaded to cache

## Metric Parsing

- GenAcc: grep "Global accuracy based on text features: " → parse float
- PerAcc: grep "epoch 10, global accuracy = " → parse float
