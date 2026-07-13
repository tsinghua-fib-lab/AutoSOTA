# UniRouter Experiments: Running Comparison Evaluations

This guide explains how to run experiments comparing **Original UniRouter** vs **Uni-R2** with dynamic model addition.

---

## Quick Start

```bash
# Run the full experiment pipeline
bash unirouter/run_unirouter_experiment.sh
```

This will:
1. Train Original UniRouter on initial 3 models (GLM-4.5-Air, Llama-3.2-3B, Qwen3-0.6B)
2. Train Uni-R2 on the same 3 models
3. **Dynamically add 2 new models** (Qwen2.5-Math-7B, Llama-3.1-70B) WITHOUT retraining
4. Compare routing performance across all methods
5. Generate cost-performance curves and visualizations

**Results saved to**: `unirouter/results/`

---

## What Gets Evaluated

### Original UniRouter
- **Input**: Query embedding
- **Output**: Best LLM (always with unlimited tokens)
- **Features**: Per-cluster error rates Ψ(h) ∈ R^K
- **Routing**: argmin [error + λ × cost]
- **Validation Cost**: 500 API calls per model

### Uni-R2
- **Input**: Query embedding
- **Output**: Best (LLM, token_budget) pair
- **Features**: Per-cluster, per-budget quality matrix Ψ(h) ∈ R^(K×B)
- **Routing**: argmax [quality - λ × cost]
- **Validation Cost**: 4,000 API calls per model (8 budgets × 500 queries)

### Dynamic Model Addition
Both methods support adding new models WITHOUT retraining:
- Run new model on validation set
- Register features (instant!)
- Router can now route to the new model

---

## Customizing the Experiment

### 1. Modify LLM Pools

Edit `unirouter/run_unirouter_experiment.sh`:

```bash
# Initial models (will be used for training)
INITIAL_POOL=(
    "GPT-3.5|10|data/gpt3.5.csv"
    "GPT-4|50|data/gpt4.csv"
    "Claude-2|25|data/claude2.csv"
)

# NEW models (will be added dynamically)
NEW_MODELS=(
    "Claude-3|30|data/claude3.csv"
    "Gemini-Pro|40|data/gemini.csv"
)
```

**Format**: `"ModelName|SizeInBillions|PathToCSV"`

### 2. Configure Validation Set

```bash
VAL_SIZE=500                # Number of validation queries
N_CLUSTERS=100              # Number of semantic clusters
USE_CLUSTERING=true         # Use clustering (recommended)
```

**Guidelines**:
- `VAL_SIZE=200`: Fast, less accurate
- `VAL_SIZE=500`: Balanced (recommended)
- `VAL_SIZE=1000`: Slow, more accurate

- `N_CLUSTERS=50`: Coarse-grained routing
- `N_CLUSTERS=100`: Balanced (recommended)
- `N_CLUSTERS=200`: Fine-grained routing

### 3. Configure Token Budgets

**IMPORTANT**: Token budgets must match what's available in your CSV files!

For the R2-Bench dataset, the available token limits are:
```bash
# Full R2-Bench dataset budgets (16 limits)
TOKEN_BUDGETS="10,20,30,40,50,80,100,150,200,300,500,800,1200,2000,4000,9999"

# Sparse subset (faster validation, less granular)
TOKEN_BUDGETS="50,100,200,500,1200,4000,9999"

# Dense subset (recommended - 8 budgets)
TOKEN_BUDGETS="50,100,200,400,800,1600,3200,9999"
```

**Note**:
- `9999` = unlimited tokens (maps to `unlimited_score` column in CSV)
- Each token budget must have corresponding `{budget}_score` and `{budget}_count` columns
- Check your CSV headers with: `head -1 data/your-model.csv | tr ',' '\n' | grep score`

### 4. Configure Cost-Performance Tradeoff

```bash
LAMBDA_MIN=0         # Quality-only (no cost awareness)
LAMBDA_MAX=1e-3      # Strong cost awareness
LAMBDA_STEPS=100     # Number of points on curve
```

---

## Understanding the Results

### Metrics CSV (`comparison_metrics.csv`)

```csv
method,peak_accuracy,AUDC,QNC
Original UniRouter (Initial),0.8200,0.7100,0.4500
Original UniRouter (Expanded),0.8450,0.7600,0.3800
Uni-R2 (Initial),0.8200,0.7800,0.2200
Uni-R2 (Expanded),0.8450,0.8300,0.1500
```

**Metrics explained**:
- **Peak Accuracy**: Maximum quality achieved (higher is better)
- **AUDC**: Area Under Deferral Curve (higher is better)
- **QNC**: Query-Normalized Cost at 90% peak (lower is better)

**Key Insights from Example**:
- Both methods reach same peak accuracy (0.845 after expansion)
- Uni-R2 has **higher AUDC** (0.83 vs 0.76) → Better cost-quality tradeoff
- Uni-R2 has **lower QNC** (0.15 vs 0.38) → Achieves 90% quality at lower cost

### Curves CSV (`comparison_curves.csv`)

Contains full cost-performance curves for plotting:
```csv
method,lambda,cost,performance
Original UniRouter (Initial),0.0,1250.5,0.8200
Original UniRouter (Initial),1e-5,980.3,0.8150
...
```

### Visualization (`comparison_plot.png`)

Shows cost-performance Pareto frontier:
- **X-axis**: Cost (tokens × model size)
- **Y-axis**: Average quality score
- **Higher and left = better** (high quality, low cost)

**What to look for**:
- Uni-R2 curve should dominate Original UniRouter (higher for same cost)
- Expanded pool should dominate initial pool
- More token budgets → smoother curve

---

## Advanced Usage

### Run Evaluation Script Directly

```bash
python unirouter/eval_compare.py \
    --initial-model "GLM-4.5-Air" 0.85 "data/GLM-4.5-Air.csv" \
    --initial-model "Llama-3.2-3B-Instruct" 0.02 "data/Llama-3.2-3B-Instruct.csv" \
    --new-model "Qwen2.5-Math-7B-Instruct" 0.35 "data/Qwen2.5-Math-7B-Instruct.csv" \
    --val-size 500 \
    --n-clusters 100 \
    --token-budgets "50,100,200,400,800,1600,3200,9999" \
    --checkpoint-dir "./unirouter/checkpoints" \
    --output-dir "./unirouter/results"
```

### Test Without New Models

```bash
# Edit run_unirouter_experiment.sh and set:
NEW_MODELS=()  # Empty array

# Then run:
bash unirouter/run_unirouter_experiment.sh
```

This will only evaluate initial pool (useful for debugging).

### Use Pre-computed Checkpoints

If you've already run the experiment once, subsequent runs will load checkpoints instead of recomputing features:

```
Checkpoints found:
  ✓ Original UniRouter: ./unirouter/checkpoints/original_unirouter.pkl
  ✓ Uni-R2: ./unirouter/checkpoints/uni_r2_feature_matrix.pkl
```

To force recomputation, delete checkpoint files:
```bash
rm -rf unirouter/checkpoints/*.pkl
```

---

## Expected Runtime

**With default configuration** (3 initial + 2 new models, 500 validation, 100 clusters, 8 budgets):

| Step | Time | Notes |
|------|------|-------|
| Validation set creation | ~5 sec | K-means clustering |
| Original UniRouter (5 models) | ~2 sec | Simple feature computation |
| Uni-R2 (5 models) | ~10 sec | More features (8 budgets) |
| Evaluation (100 lambda points) | ~30 sec | Route 6,000 test queries |
| **Total** | **~1 min** | With checkpoints: ~30 sec |

**Scaling**:
- 2× models → 2× registration time (no change in routing time)
- 2× validation size → 2× registration time
- 2× token budgets → 2× Uni-R2 registration time (Original unchanged)
- 2× test queries → 2× evaluation time

---

## Troubleshooting

### Issue: Missing CSV files

```
⚠️  Skipping Qwen2.5-Math-7B-Instruct: CSV not found at data/Qwen2.5-Math-7B-Instruct.csv
```

**Solution**: Check that CSV path is correct and file exists:
```bash
ls -lh data/Qwen2.5-Math-7B-Instruct.csv
```

### Issue: Import errors

```
ModuleNotFoundError: No module named 'unirouter'
```

**Solution**: Run from project root:
```bash
cd /home/jiaq/Research/Code/router
bash unirouter/run_unirouter_experiment.sh
```

### Issue: Validation set too small

```
ValueError: Sample size 500 > population size 300
```

**Solution**: Reduce `VAL_SIZE` to match available training data:
```bash
VAL_SIZE=200  # Or smaller
```

### Issue: Memory error

```
MemoryError: Unable to allocate array
```

**Solution**: Reduce number of models or validation size:
```bash
VAL_SIZE=200
N_CLUSTERS=50
```

---

## File Structure

```
unirouter/
├── run_unirouter_experiment.sh    # Main experiment script
├── eval_compare.py                # Comparison evaluation
├── unirouter_original.py          # Original UniRouter implementation
├── uni_r2.py                    # Uni-R2 implementation
├── UNI_CORE_README.md             # Uni-R2 documentation
├── COMPARISON.md                  # Detailed comparison
├── EXPERIMENT_README.md           # This file
├── checkpoints/                   # Saved model features
│   ├── original_unirouter.pkl
│   └── uni_r2_feature_matrix.pkl
└── results/                       # Evaluation outputs
    ├── comparison_metrics.csv
    ├── comparison_curves.csv
    └── comparison_plot.png
```

---

## Reproducing Paper Results

To reproduce results from the UniRouter paper:

### 1. Setup model pool matching paper

```bash
# Edit run_unirouter_experiment.sh
INITIAL_POOL=(
    "GPT-3.5|10|data/gpt3.5.csv"
    "GPT-4|50|data/gpt4.csv"
    "Claude-2|25|data/claude2.csv"
    "Llama-2-70B|70|data/llama2-70b.csv"
)

NEW_MODELS=(
    "Claude-3|30|data/claude3.csv"
)
```

### 2. Use paper hyperparameters

```bash
VAL_SIZE=500
N_CLUSTERS=100
USE_CLUSTERING=true
TOKEN_BUDGETS="50,100,200,400,800,1600,3200,9999"
```

### 3. Run experiment

```bash
bash unirouter/run_unirouter_experiment.sh
```

### 4. Compare metrics

Check `unirouter/results/comparison_metrics.csv` against paper Table 2.

---

## Citation

If you use this code, please cite:

```bibtex
@article{unirouter2025,
  title={Universal Model Routing for Efficient LLM Inference},
  author={Jitkrittum et al.},
  journal={arXiv preprint arXiv:2502.08773},
  year={2025}
}
```

For Uni-R2 (our extension combining UniRouter + R2-Router):

```bibtex
@article{unicore2025,
  title={Uni-R2: Universal Cost-aware Routing with Token Budget Optimization},
  author={Your Name},
  journal={TBD},
  year={2025}
}
```

---

## Next Steps

1. **Run baseline experiment**: `bash unirouter/run_unirouter_experiment.sh`
2. **Customize model pools**: Edit `INITIAL_POOL` and `NEW_MODELS`
3. **Tune hyperparameters**: Adjust `VAL_SIZE`, `N_CLUSTERS`, `TOKEN_BUDGETS`
4. **Analyze results**: Check `unirouter/results/` for outputs
5. **Compare with R2-Router**: See `COMPARISON.md` for detailed analysis

For questions or issues, see:
- `UNI_CORE_README.md`: Uni-R2 architecture and implementation
- `COMPARISON.md`: Original UniRouter vs Uni-R2 comparison
- `../CLAUDE.md`: Full project documentation
