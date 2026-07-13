# OOD Evaluation - Simplified

Test how well the routing system generalizes to unseen query categories.

## Quick Start

### Using the Bash Script (Recommended)

```bash
# Run OOD evaluation on MMLU-Pro (default, all 10 models)
bash ood_evaluation/run_ood_experiment.sh

# Quick demo with 1 model (fast!)
bash ood_evaluation/run_ood_experiment.sh --quick

# Test on a different category
bash ood_evaluation/run_ood_experiment.sh --category "lighteval/MATH/all"
```

### Using Python Directly

```bash
# Run OOD evaluation on MMLU-Pro (default)
python ood_evaluation/run_ood.py

# Quick demo with 1 model (fast!)
python ood_evaluation/run_ood.py --quick

# Test on a different category
python ood_evaluation/run_ood.py --category "lighteval/MATH/all"

# Specify custom model pool
python ood_evaluation/run_ood.py \
    --model "GLM-4.5-Air" 0.85 "data/GLM-4.5-Air.csv" \
    --model "Llama-3.2-3B-Instruct" 0.02 "data/Llama-3.2-3B-Instruct.csv"
```

That's it! The script will:
1. Load data and split by category (train on 19, test on 1)
2. Train R2-Router with **Ridge regression (alpha=10.0)** - same as IID for consistent generalization
3. Train baselines (MIRT, NIRT, CARROT-KNN, CARROT-Linear)
4. Evaluate routing performance
5. Generate plots and metrics
6. Save results to `./comparison_results/ood_evaluation/`

**Important**: R2-Router now uses Ridge regression instead of LinearRegression to prevent overfitting
on training categories. This improves OOD accuracy from ~53% to ~75%, making it competitive with CARROT baselines.

**Checkpoint Loading**: The script automatically loads existing checkpoints instead of retraining. On subsequent runs,
it will load pre-trained models instantly (~10 seconds vs 30-60 minutes). To force retraining, delete the checkpoint directory.

## What is OOD Evaluation?

**Out-of-Distribution (OOD)** evaluation tests generalization:
- **Train**: On 19 categories (e.g., all except MMLU-Pro)
- **Test**: On 1 held-out category (e.g., MMLU-Pro only)
- **Goal**: See if the router works on unseen query types

## Results

The script outputs:
- **Metrics CSV**: Peak accuracy, AUDC, QNC for each method
- **Curves CSV**: Full cost-performance curves
- **Plot PNG**: Visual comparison of methods

### Key Metrics
- **Peak Accuracy**: Best performance achieved (higher is better)
- **AUDC**: Area under cost-performance curve (higher is better)
- **QNC**: Cost to reach 90% of peak accuracy (lower is better)

## File Structure

### Essential Files (use these)
```
ood_evaluation/
├── run_ood_experiment.sh      # ⭐ Bash wrapper - recommended!
├── run_ood.py                 # ⭐ Main Python script
├── ood_dataset_manager.py     # Manages OOD splits
├── map_and_split_data.py      # Creates category splits (run once)
├── category_splits/           # Pre-computed splits
│   ├── ood_splits.pkl        # Train/test indices per category
│   └── embeddings/           # Category-specific embeddings
├── checkpoints/               # OOD-trained model checkpoints (organized by test category)
│   ├── TIGER-Lab_MMLU-Pro/   # Checkpoints when MMLU-Pro is held out
│   │   ├── GLM_4_5_Air_ridge_alpha10.0/   # R2-Router predictor (Ridge, trained on other 19)
│   │   ├── GLM_4_6_ridge_alpha10.0/       # R2-Router predictor
│   │   ├── ... (10 R2-Router models total)
│   │   ├── carrot_knn/       # CARROT-KNN baseline
│   │   ├── carrot_linear/    # CARROT-Linear baseline
│   │   ├── irt_mirt/         # MIRT baseline
│   │   └── irt_nirt/         # NIRT baseline
│   ├── lighteval_MATH_all/   # Checkpoints when MATH is held out
│   │   └── ... (same structure, different train/test split)
│   └── ... (one subdirectory per test category)
└── ../comparison_results/ood_evaluation/  # Output directory
```

**Important**: OOD checkpoints are **separate** from main checkpoints:
- `checkpoints/` (root) → Trained on ALL 30,968 queries (IID evaluation)
- `ood_evaluation/checkpoints/` → Trained on 19 categories only (OOD evaluation)

### Legacy Files (can be ignored/deleted)
```
├── explore_dataset.py         # One-time data exploration
├── analyze_categories.py      # One-time category analysis
├── verify_alignment.py        # One-time alignment check
├── train_ood.py              # Replaced by run_ood.py
├── evaluate_ood.py           # Replaced by run_ood.py
├── eval_mmlu_pro.py          # Replaced by run_ood.py
├── quick_demo.py             # Replaced by run_ood.py --quick
├── *.csv                     # Exploration outputs
├── *.md (except this one)    # Old documentation
└── alignment_verification.*  # Verification outputs
```

## Available Categories

20 categories from SPROUT dataset:
- **TIGER-Lab/MMLU-Pro** (8,264 queries) - Default test category
- **openhermes/teknium** (13,670 queries) - Largest category
- **lighteval/MATH/all** (5,122 queries) - Math problems
- **Idavidrein/gpqa/gpqa_extended** (384 queries)
- And 16 more...

## Advanced Usage

### Custom Output Directory
```bash
python ood_evaluation/run_ood.py --output ./my_results
```

### Test Multiple Categories
```bash
for cat in "TIGER-Lab/MMLU-Pro" "lighteval/MATH/all"; do
    python ood_evaluation/run_ood.py --category "$cat"
done
```

### Modify Model Pool

Edit `run_ood.py` and change the `MODELS` dictionary:
```python
MODELS = {
    "GLM_4_5_Air": {"name": "GLM-4.5-Air", "csv": "data/GLM-4.5-Air.csv", "size": 0.85},
    # Add or remove models here
}
```

## Troubleshooting

### "Category not found" error
Check available categories:
```python
from ood_dataset_manager import get_all_categories
cats = get_all_categories("./ood_evaluation/category_splits/ood_splits.pkl")
print(cats)
```

### "CSV not found" error
Make sure the CSV file exists in `data/` directory and path matches in `MODELS` dict.

### Out of memory
Use `--quick` mode or edit `run_ood.py` to use fewer models.

## Bash Script Features

The `run_ood_experiment.sh` script provides:
- **LLM Pool Configuration**: Centralized model pool definition (similar to `run_experiment.sh`)
- **Hyperparameter Management**: Enforces Ridge(alpha=10.0) to match IID training
- **Checkpoint Checking**: Skips retraining if checkpoints already exist
- **Category Selection**: Easy switching between test categories
- **Quick Mode**: Fast testing with just 1 model

Edit the `LLM_POOL` array in the script to customize your model pool.

## How It Works

1. **Data Split**: For test category C, train on all queries NOT in C, test on queries in C
2. **Training**: Train R2-Router predictors and baselines on training split (no data leakage!)
3. **Routing**: For each query, select best (model, token_limit) based on predictions
4. **Evaluation**: Compare routing accuracy vs cost across methods

**For detailed verification of OOD evaluation correctness, see [OOD_EVALUATION_VERIFIED.md](OOD_EVALUATION_VERIFIED.md)**

## Next Steps

After running OOD evaluation:
1. Compare OOD vs IID performance (run `python results.py` for IID)
2. Identify hardest categories (lowest OOD accuracy)
3. Analyze failure modes on specific categories
4. Consider category-aware routing strategies
