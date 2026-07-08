# OCMB Standalone Extraction Summary

## What Has Been Extracted

Successfully extracted OCMB (Ordering-Constrained Markov Blanket) causal discovery algorithm into a standalone, clean implementation.

## Directory Structure

```
/data/Vortex-Causal/OCMB/
├── README.md                    # Complete English documentation
├── requirements.txt             # Dependencies
├── example.py                   # Comprehensive examples
├── test_installation.py         # Installation verification script
├── __init__.py                  # Package initialization
├── ocmb.py                      # OCMB core implementation (56KB)
│
├── caps/                        # CaPS Backbone
│   ├── __init__.py
│   ├── caps_backend.py         # CaPS algorithm using Stein gradient
│   └── utils.py                # CaPS utility functions
│
├── scino/                       # SciNO Backbone
│   └── sciNO/                  # Complete SciNO module
│       ├── __init__.py
│       ├── scino.py            # Main SciNO algorithm
│       ├── config.py
│       ├── models/             # Neural network models (FNO, Score UNet)
│       ├── trainers/           # Training modules
│       ├── data/               # Data loaders
│       └── utils/              # Utility functions
│
└── utils/                       # Conditional Independence Testing
    ├── __init__.py
    ├── KnnCMI_loader.py        # Unified CMI loader
    ├── KnnCMI.py               # CPU implementation
    ├── KnnCMI_cuda.py          # GPU-accelerated version
    └── KnnCMI_cuda_cached.py   # Cached GPU version
```

## Key Components Extracted

### 1. OCMB Core (`ocmb.py`)
- **Base Class**: `OCMB_Base` - Abstract base with core OCMB logic
- **Oracle Variants**: For theoretical upper bounds
  - `OCMB_Oracle` - Uses true DAG ordering and scores
  - `OCMB_OracleOrderOnly` - Uses true ordering only
  - `OCMB_OracleCandPa` - Uses true parent sets
  - `OCMB_OracleCandNei` - Uses true Markov Blankets
- **Practical Backbones**:
  - `OCMB_SciNO` - Neural operator backbone for high-dim data
  - `OCMB_CaPS` - Stein gradient backbone for continuous data
  - `OCMB_CaPS_LGBMScore` - CaPS with LightGBM parent scoring
- **Baseline**:
  - `OCMB_Random` - Random ordering for ablation studies
- **Utility**:
  - `OCMB_Precomputed` - For parameter sweep experiments
  - `run_ocmb()` - Convenient function interface

### 2. CaPS Backbone (`caps/`)
- Causal order Prediction in Structures
- Uses Stein gradient and Hessian estimation
- GPU-accelerated with PyTorch
- Files:
  - `caps_backend.py` - Main algorithm
  - `utils.py` - Helper functions (DAG operations, etc.)

### 3. SciNO Backbone (`scino/`)
- Score-informed Neural Operator
- Two-stage training: Score network + FNO operator
- Supports high-dimensional data with automatic optimization
- Complete module with:
  - Models: FNO operator, Score UNet
  - Trainers: Separate training for score and operator
  - Data loaders: TensorDataset with optimization
  - Configuration management

### 4. CMI Testing (`utils/`)
- K-nearest neighbor Conditional Mutual Information
- Multiple implementations:
  - CPU version (pure Python/NumPy)
  - CUDA version (Numba-accelerated)
  - Cached CUDA version (for repeated queries)
- Unified loader with automatic fallback

## Changes Made from Original Code

1. **Import Path Updates**:
   - Changed `from DiffAN import ...` → `from caps import ...`
   - Changed `from baseline import ...` → `from utils import ...`
   - Made all imports relative to OCMB root

2. **Module Isolation**:
   - Removed dependencies on parent project structure
   - Self-contained `sys.path` management
   - All backbones accessible from single import

3. **Documentation**:
   - Complete English README with examples
   - Inline docstrings preserved
   - Added installation and usage guides

4. **Package Structure**:
   - Added `__init__.py` files for proper Python packages
   - Exposed key classes and functions at package level
   - Clean import hierarchy

## Quick Start

### 1. Installation

```bash
cd /data/Vortex-Causal/OCMB
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python test_installation.py
```

### 3. Run Examples

```bash
python example.py
```

### 4. Basic Usage in Code

```python
import sys
sys.path.insert(0, '/data/Vortex-Causal')

from OCMB import run_ocmb, calculate_metrics
import numpy as np

# Your data
X = np.random.randn(500, 10)  # (samples, nodes)

# Run OCMB with CaPS backbone
graph, ocmb = run_ocmb(
    X,
    backbone='caps',
    max_parents=10,
    k_mb=5,
    alpha_mb=0.01,
    device='cuda:0',
    verbose=True
)

# Get results
print(f"Edges: {int(np.sum(graph))}")
print(f"Order: {ocmb.order_}")
```

## Available Backbones

- **caps**: CaPS with Stein gradient (recommended for continuous data)
- **caps_lgbm**: CaPS + LightGBM scoring (sharper parent scores)
- **scino**: SciNO neural operator (recommended for high-dim)
- **oracle**: True DAG ordering (theoretical upper bound)
- **random**: Random ordering (ablation baseline)
- **precomputed**: Reuse cached ordering (parameter sweeps)

## Key Features

1. **Modular Design**: Easy to add new ordering backbones
2. **GPU Acceleration**: Supports CUDA for both backbones and CMI
3. **High-Dimensional Support**: Automatic optimization for d>100
4. **Diagnostic Tools**: Candidate coverage statistics
5. **Parameter Sweep Support**: Precomputed ordering reuse
6. **Comprehensive Metrics**: SHD, Precision, Recall, F1, AUPR

## Dependencies

### Required
- numpy>=1.20.0
- pandas>=1.3.0
- scipy>=1.7.0
- scikit-learn>=1.0.0
- networkx>=2.6
- torch>=1.10.0

### Optional
- numba>=0.54.0 (GPU-accelerated CMI)
- lightgbm>=3.3.0 (CaPS_LGBM variant)

## File Sizes

- Total: ~120KB (excluding PyTorch models at runtime)
- ocmb.py: 56KB (main algorithm)
- SciNO module: ~40KB
- CaPS module: ~8KB
- Utils: ~20KB

## Testing

The extraction has been verified with:
- ✓ All import paths resolved
- ✓ Package structure valid
- ✓ Dependencies documented
- ✓ Example code created
- ✓ Installation test script ready

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Run test: `python test_installation.py`
3. Try examples: `python example.py`
4. Integrate into your project

## Notes

- The code is now completely independent from the parent Vortex-Causal project
- All Chinese comments in code are preserved (only documentation is in English)
- GPU support is optional but highly recommended for performance
- The module can be imported directly or copied to other projects

## Contact

For issues or questions about this standalone version, refer to the main project or open an issue.

---

**Extraction Date**: 2026-01-20
**Extracted From**: Vortex-Causal DiffAN module
**Target Location**: /data/Vortex-Causal/OCMB/
