# Code Analysis - Paper 4550: OVLR

## Evaluation Path
- Script: /repo/reproduce_table7.py
- Eval command uses CUDA_VISIBLE_DEVICES=0,1
- Timeout: 15 minutes

## Key Files
- reproduce_table7.py - Main training/eval script (SAFE to modify)
- ovlr/estimator.py - OVLR gradient estimator (SAFE to modify)
- ovlr/noise.py - Noise generators (SAFE to modify)

## Metric Parser
- From stdout: Best test accuracy (overall): XX.XX%
- From JSON: results_table7.json -> final_test_accuracy field
- Time from JSON: total_time_s

## Baseline
- Accuracy: 69.23%, Time: 299.8s
- Best warmup peak: 72.66%, Best OVLR: 70.66%
- Gap from peak to final: -3.43% (key optimization target)

## Safe Modification Targets
1. reproduce_table7.py - training loop, optimizer, scheduler, warmup
2. ovlr/estimator.py - gradient estimation logic
3. ovlr/noise.py - noise distributions
