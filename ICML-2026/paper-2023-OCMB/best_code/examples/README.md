# OCMB Examples

This directory contains example scripts to reproduce key experiments from the OCMB paper:
**"Global Directional Priors with Local Statistical Validation"** (ICML 2026).

## Quick Start

For a simple introduction to using OCMB, see the main `example.py` in the parent directory:
```bash
cd ..
python example.py
```

## Reproducing Paper Results

The examples in this directory demonstrate how to reproduce the main experimental results from the paper.

### 1. Scale-Free and ER Graph Experiments (Table 2)

Reproduces the main comparison table showing OCMB's performance on synthetic graphs:

```bash
python 1_synthetic_graphs.py
```

This script:
- Generates scale-free and Erdős-Rényi graphs with d=100 nodes, n=1000 samples
- Compares OCMB against PC, IAMB, and other baselines
- Reports F1, SHD, and CI test counts
- Expected F1 for OCMB: ~0.55-0.65 on scale-free, ~0.25 on ER

**Runtime**: ~5-10 minutes (depends on whether GPU is available for CaPS backbone)

### 2. K Sensitivity Analysis (Table 9, Figure 8)

Reproduces the hyperparameter sensitivity analysis showing OCMB's robustness to K:

```bash
python 2_k_sensitivity.py
```

This script:
- Sweeps K/d ratios from 0.02 to 0.40
- Shows F1 varies by <5% across the range
- Demonstrates OCMB's insensitivity to K choice
- Expected: F1 plateau across K/d ∈ [0.05, 0.20]

**Runtime**: ~10-15 minutes

### 3. Runtime and Efficiency Comparison (Figure 7)

Reproduces the runtime comparison showing OCMB is 2× faster than IAMB:

```bash
python 3_runtime_comparison.py
```

This script:
- Compares runtime across different graph sizes
- Shows CI test reduction (65% fewer than IAMB)
- Expected: OCMB ~6-15s for d=100, IAMB ~15-30s

**Runtime**: ~20-30 minutes (multiple runs for accurate timing)

### 4. Ordering Error Sensitivity (Figure 10)

Reproduces the robustness analysis with corrupted orderings:

```bash
python 4_ordering_robustness.py
```

This script:
- Injects 0%, 10%, 20%, 30%, 40%, 50% ordering errors
- Shows graceful degradation
- At 50% error: F1 drops by ~33%, but Parent Recall stays at 0.65
- Expected: F1 ≈ 0.84 at 0% error, ≈ 0.56 at 50% error (for K=1)

**Runtime**: ~15-20 minutes

## Requirements

All examples require the OCMB package to be installed. Additional requirements:

- numpy
- pandas
- matplotlib (for plotting)
- scikit-learn (for evaluation metrics)

For GPU acceleration (optional but recommended):
- PyTorch with CUDA support

## Understanding the Results

### Key Metrics

- **F1**: Harmonic mean of precision and recall (higher is better)
- **SHD**: Structural Hamming Distance (lower is better)
- **#CI Tests**: Number of conditional independence tests (lower = more efficient)
- **Parent Recall (covPa)**: Percentage of true parents captured in candidate set

### Expected Performance

Based on the paper (Table 2, Section 4.2):

| Method | F1 (SF) | SHD (SF) | #CI Tests | Runtime |
|--------|---------|----------|-----------|---------|
| OCMB(caps) | 0.558 | 31.7 | 1,980 | 6.8s |
| IAMB | 0.483 | 49.0 | 5,699 | 14.5s |
| PC | 0.663 | 16.3 | 2,072 | 1.4s |

Note: PC has better F1 but assumes homogeneous graphs (ER-like). OCMB excels on scale-free graphs where PC struggles.

### OCMB's Key Advantages

1. **Efficiency**: 65% fewer CI tests than IAMB (Figure 7)
2. **Robustness**: Insensitive to K parameter across K/d ∈ [0.05, 0.20] (Table 9)
3. **Graceful degradation**: Handles ordering errors well (Figure 10)
4. **Scale-free performance**: Strong on heterogeneous graphs where PC fails

## Troubleshooting

### GPU Memory Issues

If you encounter CUDA out-of-memory errors with CaPS backbone, try:
```python
# In the script, change:
backbone='caps'  # to:
backbone='random'  # or use CPU
device='cpu'
```

### Long Runtimes

To get faster results during testing:
```python
# Reduce the number of seeds:
seeds = [42]  # instead of [42, 123, 456, ...]

# Or reduce dimension:
d = 50  # instead of 100
```

### Import Errors

Make sure OCMB is in your Python path:
```bash
export PYTHONPATH=/data/Vortex-Causal/OCMB:$PYTHONPATH
# Or install in development mode:
pip install -e /data/Vortex-Causal/OCMB
```

## Citation

If you use OCMB in your research, please cite:

```bibtex
@inproceedings{ocmb2026,
  title={Global Directional Priors with Local Statistical Validation},
  author={[Authors]},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## Contact

For questions or issues, please open an issue on the GitHub repository.
