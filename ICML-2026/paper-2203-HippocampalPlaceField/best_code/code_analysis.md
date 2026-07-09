# Code Analysis: Paper 2203 - HIPE

## Evaluation Path
- Script: scripts/train_exp2_wikifull.py
- Model: OLMo/olmo/model.py
- Config: OLMo/olmo/config.py
- Validation: Every eval_interval steps, computes CE on WikiText-103 val set
- Metric: Last VAL PPL printed to stdout, logged to output_dir/log.txt

## Key Architecture
- 20M: d=256, n_heads=8, n_layers=8, mlp_ratio=8
- Bipartite RoPE: Layers 0-3 standard RoPE, layers 4-7 HIPE sigma=50
- HIPE: Gaussian decay exp(-sigma^2 * freq^2 / 2)

## Safe Modification Targets
1. train_exp2_wikifull.py: Add CLI flags (--decay_func, --warmup_ratio, --wsd, etc.)
2. model.py:ModelConfig: Add new config fields
3. model.py:OLMoBlock.build: Wire new dispatch paths
4. model.py:ScaledRotaryEmbedding: Already supports multiple decay functions

## Red Lines
- No change to eval metric, dataset splits, tokenizer
- No hard-coded predictions
