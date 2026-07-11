# SOTA Preparation Repair — Paper 3068

## Failure Diagnosis

The SOTA preparation phase failed because the container lacked the `git` binary. The orchestrator
tried `apt-get install -y -qq git` but the apt repositories were unreachable through the HTTP proxy
configured inside the container. The conda channel was also unreachable.

## Repair Applied

1. **Git wrapper**: Installed Python package `dulwich` (pure-Python Git implementation) via pip
   (Tsinghua mirror was reachable). Created `/usr/local/bin/git` as a Python wrapper that maps
   common git commands (init, config, add, commit, tag, rev-parse, log, stash) to dulwich porcelain
   and plumbing APIs.

2. **Baseline initialization**: Ran the standard preparation sequence:
   - `git config --global --add safe.directory /repo`
   - `git rev-parse --git-dir` (verified existing `.git` directory)
   - `git config user.name optimizer` / `git config user.email opt@local`
   - `git add -A`
   - `git commit -m "optimization baseline" --allow-empty`
   - `git tag -f _baseline`

3. **Record score script**: Copied `record_score.sh` to `/tools/record_score.sh`.

## Verified Baseline

- **Command**: `python3 training/cifar/cifar_full_graph_prune.py --graph-path /datasets/cifar100_full_graph_v2.pkl --dataloader-ratio 0.7 --score-weight 1.0 --model r18 --num_epoch 200 --batch-size 128 --max-lr 0.05`
- **Result**: Acc = 78.62% (matches manifest reproduction)
- **From existing output**: `r18-0.002-ratio0.7-score1.0-epoch200-bs128-2026-07-09_18-08-35_cifar100_full_graph_prune.json`
- **GPU**: 2× NVIDIA A100-SXM4-80GB (devices 2,3 mapped as 0,1)
- **Training time**: ~957 seconds (~16 minutes) for 200 epochs

## Available Resources

### Container paths
- **Repo**: `/repo` — Data-Selection-on-Graphs at commit `017e302c`
- **Graph**: `/datasets/cifar100_full_graph_v2.pkl` (1.2MB, built from 50-epoch ResNet-18 embeddings)
- **Embeddings**: `/datasets/cifar100_embeddings_v2.pt`
- **Dataset**: `/datasets/cifar100/` (CIFAR-100, 169MB)
- **Models**: `/models/torchvision/resnet18-f37072fd.pth`
- **Cache**: `/autosota_cache/`

### Key source files
- `training/cifar/cifar_full_graph_prune.py` — Main training script with UGIES pruning
- `training/cifar/cifar_full_graph_dataloader.py` — CIFAR dataloader with pruning logic
- `scripts/build_graph_gpu.py` — GPU-accelerated graph construction
- `scripts/extract_embeddings_from_checkpoint.py` — Embedding extraction

## Safe Optimization Targets

All modifications should stay within `/repo/training/cifar/` and `/repo/scripts/`.
Do not modify dataset, evaluation protocol, or metric definitions.

### High-confidence targets
1. **CODE-01**: Fix edge score sign direction in `cifar_full_graph_dataloader.py:204`
2. **ALGO-01**: Replace entropy with EL2N in `compute_uncertainty_metrics()`
3. **ALGO-06**: Cosine annealing with linear warmup replacing OneCycleLR
4. **ALGO-03**: Dynamic score-weight annealing
5. **CODE-04**: Gradient clipping

### Medium-confidence targets
6. **ALGO-04**: Mixup augmentation
7. **CODE-02**: Pruning ratio annealing from ImageNet
8. **PARAM-01**: Score-weight × delta grid sweep

### Lower-confidence targets
9. **ALGO-02**: k-NN sparse graph construction
10. **CODE-03**: Better embeddings for graph
11. **ALGO-05**: Per-class adaptive pruning
12. **PARAM-02**: Extended training to 300 epochs

## Constraints
- Primary metric: Acc (higher is better)
- Baseline: 78.62%
- Paper target: 78.9%
- Metric upper bound: 78.94%
- Max 12 iterations, target 6+
- Evaluation timeout: 60 minutes
