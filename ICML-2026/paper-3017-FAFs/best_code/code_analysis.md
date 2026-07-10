# Code Analysis for Paper 3017 — FAF SOTA Optimization

## Evaluation Path

- Entry: main.py
- Model: model_faf.py -> FAFMLP class
- Parsing: parse.py -> parse_method() and parser_add_main_args()
- Feature computation: aggregation.py -> aggregate_faf_features()
- Data loading: dataset.py -> load_dataset()
- Evaluation: eval.py -> evaluate() function
- Logging: logger.py -> Logger class

## Metric Parsing

Output format: "Highest Test: XX.XX ± Y.YY" where XX.XX is mean test accuracy across 5 runs.
Parsed from logger.py print_statistics() at: print(f"Highest Test: {r.mean():.2f} ± {r.std():.2f}")

## Current Configuration (Baseline)

--model faf --dataset pubmed --lr 0.01 --local_layers 4 --hidden_channels 64
--mlp_layers 2 --weight_decay 0 --dropout 0.7 --rand_split_class --valid_num 500
--test_num 1000 --seed 123 --device 0 --runs 5 --ln --mean_agg --std_agg
--epochs 2500 --data_dir ./data/ --display_step 250

Baseline: 81.80

## Safe Modification Targets

1. main.py epoch loop: Add LR scheduler, mixup, SWA
2. model_faf.py: Add FeatureGate, modify normalization
3. aggregation.py: Feature augmentation (ns_agg already implemented)
4. parse.py: New CLI arguments
5. New post-processing: C and S label propagation

## Risky Files (Do Not Modify)

- data_utils.py: eval_acc, splits
- dataset.py: data loading
- eval.py: evaluate function (metric computation)
- ./data/: test data

## Key Observations

1. No LR scheduler — constant LR=0.01 for all 2500 epochs
2. Gradient clipping coded but not enabled (needs --clip_grad)
3. Both LN and BN implemented; baseline uses LN
4. save_model requires --save_model flag; baseline does not save
5. Full-batch training (all ~19K nodes per forward pass)
6. Training set: 60 nodes (20/class), validation 500, test 1000
7. FAF features: 4500 dims (500 original + 4x500 mean + 4x500 std)
8. aggregation_other.py has ns_agg, q_agg, bin_agg, sim_agg, ka_agg implemented
