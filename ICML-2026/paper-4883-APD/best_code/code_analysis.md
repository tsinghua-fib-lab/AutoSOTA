# AutoPrune Code Analysis (Paper 4883)

## Evaluation Path
- **Script**: `prune/prune.py` (main entry point)
- **Command**: `python -u prune/prune.py --sparsity_ratio 0.5 --model /models/llama7b --cache /models --dataset wikitext2 --nsamples 128 --seqlen 2048 --seed 0 --device cuda:0`
- **Metric**: WikiText-2 perplexity, parsed from stdout `ppl <float>`
- **Eval function**: `lib/eval.py::eval_ppl_wikitext()` — standard perplexity via CrossEntropyLoss

## Core Algorithm (AutoPrune)
- **File**: `prune/prune.py`
- **Scoring formula** (lines 75-81): `scores = w.abs() * sqrt(am + ar) / row_norms`
  - `am` = mean absolute activation per input feature
  - `ar` = sqrt(mean squared activation) = RMS activation
  - `row_norms` = L1 norm of weight rows
- **Hooks**: Forward hooks on all Linear layers accumulate `sum_abs` and `sum_sq`

## Related Implementations
| File | Algorithm | Status |
|------|-----------|--------|
| `prune/prune_wanda.py` | Wanda (L2 only, no row-norm) | Working |
| `prune/prune_sparse.py` | SparseGPT (Hessian + OBS) | Working |
| `prune/prune_magnitude.py` | Magnitude pruning | Working |
| `prune/prune_skew_layer.py` | SDSA layer allocation | Bug: `re.match` → needs `re.search` |
| `prune/prune_skew_wanda_layer.py` | SDSA + Wanda | Same bug |
| `prune/prune_skew_sparse_layer.py` | SDSA + SparseGPT | Same bug |
| `prune/prune_wanda_owl_layer.py` | OWL allocation | Bug: `m in outliers.keys()` checks Match object |

## Safe Modification Targets
1. **`prune/prune.py:_hook_accumulate`** (lines 43-65): Change activation statistics accumulation
2. **`prune/prune.py` scoring** (lines 72-81): Replace AutoPrune formula with Wanda/SparseGPT/etc.
3. **`prune/prune.py` post-pruning** (line 89-92): Add energy compensation
4. **CLI arguments**: `--nsamples`, `--seqlen`, `--dataset` are safe to change
5. **Calibration data**: Switch from wikitext2 to c4 (already supported in `lib/data.py`)

## Red-Line Protection
- **Do NOT modify**: `lib/eval.py::eval_ppl_wikitext()`, test data splits, metric computation
- **Do NOT modify**: `check_sparsity()` validation function
- **Safe to modify**: scoring formula, hook behavior, calibration data source, hyperparameters

## Data Resources
- **Model**: `/models/llama7b` (huggyllama/llama-7b, pre-downloaded)
- **Datasets**: WikiText-2 (auto-downloaded via HF datasets), C4 (available)
- **Cache**: `/autosota_cache`, `/datasets`, `/models`
- **No `/paper_data` mount**

## Baseline
- **Perplexity**: 7.1116 at 50% sparsity
- **Paper reference**: 7.12 (within CI [7.11, 7.22])
- **Commit**: `b47696e`
