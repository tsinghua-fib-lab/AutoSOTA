# MoCo-EA SOTA Preparation Repair Analysis

## Original Failure

The orchestrator failed during the preparation step because:
1. The preparation script tried to install `git` via `apt-get` with HTTP proxy `http://127.0.0.1:7890` configured, but the proxy was unreachable
2. When retrying with `docker run` to create a new container, the first attempt used `--network host` which was rejected by the Docker authorization plugin
3. The second `docker run` succeeded (without `--network host`) using proxy `http://172.17.0.1:17890`, but `apt-get` still failed with 502 Bad Gateway errors

## Repair Applied

1. **Network fix**: Ran `apt-get` without proxy (`unset HTTP_PROXY HTTPS_PROXY...`), which succeeded
2. **Git installation**: `apt-get install -y git` (without proxy)
3. **num_workers fix**: Changed `num_workers=2` and `num_workers=4` to `num_workers=0` in `experiments/common.py` to avoid NFS multiprocessing stale file handle issues
4. **Tools setup**: Created `/tools/record_score.sh` from host copy
5. **Artifacts directories**: Verified `/autosota_artifacts/paper-3374/sota/` is writable

## Verified Baseline

The reproduction results file `evolutionary_comparison_20260710_070029.json` matches the manifest baseline:
- Succ. rate: 100.0% (manifest: 100.0%)
- Avg. gen.: 1.3±0.6 (manifest: 1.3±0.6)
- Avg. queries: 478±202 (manifest: 478±202)
- Avg. time: 1.77±0.76s (manifest: 1.77±0.76s)

Model: ResNet-18 trained from scratch, 94.90% accuracy (manifest: 95.1%)

## Corrected Evaluation Command

```bash
cd /repo
python3 run_experiment.py \
  --experiment evolutionary \
  --dataset cifar10 \
  --data-root /datasets/CIFAR10 \
  --checkpoint /models/resnet18_cifar10_best.pth \
  --output-dir /repo/results \
  --seed 42
```

## Optimization Results

The dominant optimization is PGD warm-start population initialization (ALGO-01):
- PGD-2 iterations provides the optimal tradeoff: 141 avg queries (70% reduction), 0.46s avg time (74% reduction), 100% success rate maintained

See `/autosota_artifacts/paper-3374/sota/scores.jsonl` for all iteration details.
