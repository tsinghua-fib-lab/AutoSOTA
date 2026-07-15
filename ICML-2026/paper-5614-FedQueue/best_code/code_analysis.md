# SOTA Preparation Repair — Paper 5614 (FedQueue)

## Original Failure

The SOTA preparation script failed because:
1. git was not installed in the `autosota/paper-5614:reproduced` Docker image.
2. apt-get failed with 502 Bad Gateway through the container proxy.
3. The preparation script's `set -e` caused immediate exit when git command was not found.

## Repair Applied

1. git installed via `apt-get install -y git`.
2. Git repo initialized with baseline commit and _baseline tag.
3. /tools/record_score.sh copied from host into container.
4. /autosota_artifacts/paper-5614/sota/ directory created.

## Corrected Evaluation Command

Inside container autosota_sota_paper_5614:
```
cd /repo/simulation && python3 run_fedqueue_v4.py --config sim_mnist_v4.yaml --output output/fedqueue_v4_results.json
```

## Baseline Verification

| Metric | Manifest | This Run | Match |
|--------|----------|----------|-------|
| Max-A | 98.74% | 98.65% | within noise |
| Time-to-A* | 9.5s | 11.1s | runtime variance |
| #Ek | 5595 | 5595 | exact match |

Baseline is confirmed reproducible.

## Code Structure

- simulation/run_fedqueue_v4.py — Main simulation entry point
- simulation/sim_mnist_v4.yaml — Configuration (hyperparameters)
- simulation/cnn.py — CNN model (2 conv layers, 2 FC layers, no BN)
- simulation/celoss.py — Cross-entropy loss wrapper

## Safe Optimization Targets

- train_client() in run_fedqueue_v4.py — optimizer, LR, proximal loss, gradient clipping, MixUp
- cnn.py — model architecture (BatchNorm, skip connections)
- celoss.py — loss function (label smoothing)
- sim_mnist_v4.yaml — hyperparameters

## Red Lines

- Do not change evaluation protocol, test data, dataset splits, or metrics
- Do not modify evaluate() function
- Do not hard-code predictions or metrics
