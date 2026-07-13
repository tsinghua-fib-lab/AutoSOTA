# Code Analysis — Paper 4989 (GADL) SOTA Preparation Repair

## Original Preparation Failure

The orchestrator's preparation step failed because:
1. Git was not installed in the container (`git: command not found`)
2. The apt-based git installation failed due to disk space exhaustion (`No space left on device`)
3. The overlay filesystem was at 100% capacity (200G/200G), preventing new file creation

## Repair Steps

1. **Disk cleanup**: Removed apt lists, apt archives cache, pip cache, conda packages, and __pycache__ directories to reclaim ~85MB
2. **Git install**: Copied `/usr/bin/git` from host to container (`docker cp /usr/bin/git`)
3. **Git init**: Initialized git repo at `/repo`, set safe.directory, created baseline commit and `_baseline` tag
4. **Tools**: Copied `record_score.sh` to `/tools/record_score.sh`
5. **Baseline verification**: Ran the eval command and confirmed metrics within expected range

## Corrected In-Container Evaluation Command

```bash
cd /repo && python3 main_GADL.py --dataset 'Douban Online_Offline' --device 0 --data_path /repo/data
```

## Baseline Metrics

| Metric | Manifest | Repaired Run | Delta |
|--------|----------|-------------|-------|
| Hit@1  | 95.17%   | 94.10%      | -1.07 |
| Hit@5  | 99.91%   | 100.00%     | +0.09 |
| Hit@10 | 100.00%  | 100.00%     | 0.00  |
| Hit@50 | 100.00%  | 100.00%     | 0.00  |

The Hit@1 drop from 95.17% to 94.10% is within normal stochastic variation for this model (reproduction notes indicate 94-96% range across runs).

## Safe Optimization Targets

- FM loss decay schedule (`main_GADL.py:100`)
- Precision consistency (`main_GADL.py:123`)
- Input feature augmentation with spectral encodings (`main_GADL.py:31`)
- GConv_tide activation (`model.py:74`)
- Hyperparameter tuning (epochs, hidden_dim, GCN layers, lr, k)
- Gated fusion of high/low-pass embeddings (`model.py:306`)
- MLP functional map (`model.py:Encoder_GAE_FM`)
- MMD auxiliary loss
- Contrastive loss

## Remaining Risks

1. **Disk space**: Only ~85MB free; large checkpoints or logs could exhaust space
2. **Stochastic variance**: The model shows run-to-run variance of ~1-2% in Hit@1
