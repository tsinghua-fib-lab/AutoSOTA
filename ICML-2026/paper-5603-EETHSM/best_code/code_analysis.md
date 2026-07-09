# Paper 5603 SOTA Preparation Repair — Code Analysis

## Original Preparation Failure

The preparation failed because:
1. **git not installed**: The container image `autosota/paper-5603:reproduced` does not include git.
2. **Apt-get proxy failure**: apt-get failed with 502 Bad Gateway when using the container's proxy settings (`http://172.17.0.1:17890`). The proxy was not correctly routing to Ubuntu archive mirrors.
3. **Docker auth policy**: The first `docker run` attempt with `--network host` was rejected by administrative policy. The second attempt without `--network host` succeeded.

## Repair Applied

1. **Container**: Using existing container `autosota_sota_paper_5603` (10dd9f373506).
2. **Git installation**: `apt-get install git` without proxy variables. Direct connection to archive.ubuntu.com works.
3. **Git repo**: Initialized at `/repo`, baseline commit created with `_baseline` tag.
4. **record_score.sh**: Copied from host to `/tools/record_score.sh`.
5. **Artifacts directory**: `/autosota_artifacts/paper-5603/sota/` is writable.

## Corrected In-Container Evaluation Command

```bash
cd /repo/micro_hf && python3 main.py \
  --train_task var-copy --eval_task var-copy \
  --layer1 SSM --layer2 TF \
  --hidden_size 4 --window 100 --heads 1 --state_dim 1 \
  --sequence_length 100 --lr 1e-2 --epochs 4 \
  --num_examples 1000 --num_vocab 26 --num_numbers 5 \
  --train_batch_size 8 --eval_batch_size 8 --num_eval_examples 100 \
  --min_train_length 97 --max_train_length 98 \
  --min_eval_length 97 --max_eval_length 98 \
  --print True 2>&1 | grep -A1 '^Char$' | tail -1 | tr -d '[]'
```

## Baseline Verification

- Model: 872 parameters (SSM→TF hybrid, hidden_size=4, 2 layers)
- Single-run Char accuracy: **0.06136** (baseline iteration 0)
- Reproduction 11-run average: **0.086** (from manifest, CI [0.084, 0.0873])
- Single-run variance is expected; 11-run average is the statistically reliable metric

## Optimization Targets

Safe optimization targets (all preserve eval protocol):
1. **Curriculum learning** (IDEA-001): Train on progressively longer sequences
2. **Focal loss** (IDEA-002): Replace CE with focal loss (γ=2.0)
3. **Multi-head attention** (IDEA-003): heads=1→2 for parallel attention patterns
4. **Auxiliary loss** (IDEA-004): Predict number token positions from SSM output
5. **Label smoothing** (IDEA-005): α=0.1
6. **Attention entropy regularization** (IDEA-006): λ=0.01
7. **Learned attention bias** (IDEA-007): Content-dependent bias for number tokens
8. **use_cache=False** (IDEA-008): Memory optimization for training
9. **Seed setting** (IDEA-009): Multi-seed evaluation
10. **Device fix** (IDEA-010): Fix hardcoded 'cuda' device
11. **MambaBlock attention_mask fix** (IDEA-011): Consistent mask passing
12. **LR sweep + cosine schedule** (IDEA-012): Better hyperparameters

## Key Files

- `/repo/micro_hf/main.py` — Main training/eval entry point
- `/repo/micro_hf/train_utils.py` — Training loop, loss function, scheduler
- `/repo/micro_hf/models/hybrid.py` — SSM→TF hybrid model definition
- `/repo/micro_hf/models/rope.py` — GPTNeoXAttention (TF attention layer)
- `/repo/micro_hf/models/mamba.py` — MambaBlock (SSM layer)
- `/repo/micro_hf/generate.py` — Synthetic data generation
- `/repo/micro_hf/data_utils.py` — Data loading utilities
