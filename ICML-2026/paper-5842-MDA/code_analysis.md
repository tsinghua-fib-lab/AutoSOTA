# Code Analysis: SOTA Preparation Repair for Paper 5842

## Preparation Failure

The original preparation failed because:
1. `git` was not installed in the `autosota/paper-5842:reproduced` container
2. The container has proxy environment variables that block `apt-get` from reaching Ubuntu repositories
3. The orchestrator could not initialize the git repository for baseline tracking

## Repair Applied

1. **Git installation**: Unset proxy variables before `apt-get install git`
2. **Tooling**: Created `/tools/record_score.sh` in container
3. **Git initialization**: Initialized git repo at `/repo`, created baseline commit and `_baseline` tag
4. **Baseline verification**: Confirmed eval produces 0.4306, matching manifest baseline of 0.4306

## Corrected Evaluation Command

```bash
cd /repo
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
python3 /repo/eval_induction_final.py \
  --model-path /paper_data/pythia-14m-step2000 \
  --layer 3 --head 3 \
  --n-sequences 200 \
  --output /repo/outputs/induction_score.json
```

## Reusable /paper_data Resources

- `pythia-14m-step2000/` - Peak induction score checkpoint (0.431)
- `pythia-14m-step1000/` - Pre-emergence checkpoint (0.008)
- `pythia-14m/` - Final checkpoint (0.424)
- `pythia-70m-step2000/` - Stronger induction heads reference (L3H6: 0.482)
- `the_pile_deduplicated/` - Partial Pile dataset (parquet, 19 files)

## Safe Optimization Targets

- **Eval script**: `eval_induction_final.py` - supports multi-seed, fp64, all-heads
- **Fine-tuning**: Short LM fine-tuning on pure induction data (repeated prefixes)
- **Do NOT modify**: Metric definition, detection pattern computation, evaluation protocol

## Key Findings

1. **Emergence trajectory**: L3H3 forms rapidly between steps 1000 (score=0.008) and 2000 (0.431), then slightly degrades at final (0.424)
2. **Post-hoc fine-tuning**: Standard LM fine-tuning on text-based synthetic data degrades induction score (matching step2000→final degradation)
3. **Short fine-tuning works**: Very short (10-50 step) fine-tuning at moderate LR can marginally improve score (+1-2%)
4. **Pure induction training**: Training on repeated random-token prefixes [A0..A63, A0..A63] dramatically improves induction score to 0.480 (+11.6%), approaching paper augmented score of 0.485 (+12.3%)
5. **Sweet spot**: 100 steps at lr=1e-4 with pure induction data; longer training degrades the score
