# Synthetic Controlled Study for Disagreement-Aware Evaluation

Evaluate how different aggregation methods recover the true ranking of AI agents when human annotations are noisy, biased, or ambiguous.

## Installation

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate        # Linux/Mac
# or
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running Ablation Studies

### 1. Adversarial Fraction (0%, 10%, 20%, 30%, 40%)
```bash
python scripts/run_ablation.py configs/ablation_adversarial/*.yaml
python scripts/generate_plots.py results/comparison_TIMESTAMP/
```

### 2. Strict Annotator Fraction (0% → 40%)
```bash
python scripts/run_ablation.py configs/ablation_strict/*.yaml
python scripts/generate_plots.py results/comparison_TIMESTAMP/
```

### 3. Lenient Annotator Fraction (0% → 40%)
```bash
python scripts/run_ablation.py configs/ablation_lenient/*.yaml
python scripts/generate_plots.py results/comparison_TIMESTAMP/
```

### 4. Hard Item Probability (0.0, 0.1, 0.2)
```bash
python scripts/run_ablation.py configs/ablation_hard_prob/*.yaml
python scripts/generate_plots.py results/comparison_TIMESTAMP/
```

### 5. Labels Per Item (3, 5, 7, 9) - with stability
```bash
python scripts/run_ablation.py configs/ablation_labels/*.yaml --compute-stability
python scripts/generate_plots.py results/comparison_TIMESTAMP/
```

### 6. Agent Quality Gaps (wide vs tight)
```bash
python scripts/run_ablation.py configs/ablation_gaps/*.yaml
python scripts/generate_plots.py results/comparison_TIMESTAMP/
```

## Command Line Options

```bash
python scripts/run_ablation.py [configs...] [options]

Options:
  --n-repetitions N     Number of repetitions per config (default: 100)
  --compute-stability   Compute stability metrics (slower)
  --workers N           Number of parallel workers (default: 1)
  --no-raw              Don't save raw data (saves disk space)
  --quiet               Suppress progress output
  --output-dir DIR      Output directory (default: results)
```

## Parallel Processing

Use multiple CPU cores for faster runs:

```bash
# Use 4 parallel workers
python scripts/run_ablation.py configs/ablation_adversarial/*.yaml --workers 4

```

## Quick Test Run

```bash
# Fast test with 5 repetitions
python scripts/run_ablation.py configs/ablation_adversarial/*.yaml --n-repetitions 5
```

## Output Files

Each run saves to `results/comparison_TIMESTAMP/`:

## Generated Plots

`generate_plots.py` creates:
- `combined.png` - MSE and Kendall τ side by side
- `mse_comparison.png` - MSE bar chart
- `tau_comparison.png` - Kendall τ bar chart
- `mse_vs_[param].png` - MSE line plot (if parameter varies)
- `tau_vs_[param].png` - Kendall τ line plot (if parameter varies)
- `stability_vs_[param].png` - Stability plot (if computed)

## Methods Compared

1. **Majority Vote (MV)**: Simple mode of annotations
3. **Posterior Expected Credit (PEC)**: Soft aggregation using posteriors

## Metrics

- **MSE**: Score estimation error vs ground truth
- **Kendall's τ**: Ranking correlation with true ranking
- **Stability**: Ranking consistency under annotator subsampling

## Config Parameters

| Parameter | What it controls | Default |
|-----------|------------------|---------|
| `n_agents` | Number of agents to evaluate | 6 |
| `n_items` | Items per agent | 500 |
| `agent_qualities` | Quality scores per agent | [0.85, 0.80, 0.70, 0.55, 0.35, 0.20] |
| `n_annotators` | Total annotator pool size | 30 |
| `labels_per_item` | Annotations per item | 5 |
| `annotator_distribution` | Mix of annotator types | normal:18, strict:6, lenient:4, adversarial:2 |
| `hard_item_prob` | Fraction of ambiguous items | 0.2 |